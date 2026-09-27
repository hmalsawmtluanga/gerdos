#pragma once

#include <vector>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/execution_backend.hpp"
#include "gerdos/core/execution_effects.hpp"
#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/measurement.hpp"
#include "gerdos/core/measurement_collector.hpp"
#include "gerdos/core/operation_registry.hpp"

namespace gerdos {

// The runtime integration layer: the executor advances attempts through
// their lifecycle. It composes the gates as mandatory steps — semantic
// admissibility, execution admission, update-claim arbitration, execution
// effects, and the backend seam — in the sequence mandated by the contract.
class Executor {
public:
    Executor(
        ExecutionRegistry& executions,
        OperationRegistry& operations,
        DeviceRegistry& devices,
        DataRegistry& data,
        ExecutionBackend& backend,
        MeasurementRegistry* measurements = nullptr) noexcept
        : executions_(executions),
          operations_(operations),
          backend_(backend),
          measurements_(measurements),
          admission_(devices, data),
          effects_(data, executions) {}

    // Begins one admitted attempt: admissibility, admission, backend
    // submission, start effects, and the RUNNING transition. Rejection is
    // atomic: unless the backend accepts the attempt, nothing changes.
    [[nodiscard]] bool start(ExecutionId id) {
        auto* execution = executions_.find_execution(id);

        if (execution == nullptr) {
            return false;
        }

        const auto* operation =
            operations_.find_operation(execution->description().operation);

        if (operation == nullptr) {
            return false;
        }

        const auto* binding = execution->binding();

        if (binding == nullptr) {
            return false;
        }

        if (!admissibility_.admissible(*operation, *binding)) {
            return false;
        }

        const auto admission = admission_.admit(*execution);

        if (!admission.has_value()) {
            return false;
        }

        // Output arbitration: the attempt must be able to claim every
        // producing residency it binds.
        if (!effects_.claims_free(*execution)) {
            return false;
        }

        if (!backend_.submit(*operation, *execution)) {
            return false;
        }

        // Commit point: admission evidence is recorded only after the
        // backend has accepted the work, so every rejection above leaves
        // the attempt exactly as it was.
        if (!admission_.establish(*execution, *admission)) {
            finalize_incoherent(*execution);
            return false;
        }

        // Pinned invariants (tests/core_executor.cpp): with evidence
        // recorded and free claims, start effects and the RUNNING
        // transition cannot reject — every producing residency resolves, is
        // claimable, and can enter the update-in-progress state. Start
        // effects can only reject against a backend that mutated runtime
        // state during submission; the attempt then runs unclaimed and its
        // completion effects rejection records the incoherence.
        if (!effects_.start(*execution)) {
            // Reported through the attempt's completion effects verdict.
        }

        if (!execution->set_state(ExecutionState::RUNNING)) {
            // Unreachable: the attempt is admitted and holds a binding.
            // Finalize it in place so in-flight work cannot strand claims or
            // leave an unstarted attempt behind.
            finalize_incoherent(*execution);
            return false;
        }

        return true;
    }

    // Advances backend work and applies the completion sequence — terminal
    // transition, finish effects, result recording, measurement capture —
    // for attempts that finished since the previous call. Every applied
    // completion is reported with its integrity; completions for attempts
    // that are not RUNNING are discarded as documented. An effects rejection
    // is recorded in the attempt's result and reported here, never discarded.
    void advance(std::vector<AttemptStatus>& outcomes) {
        std::vector<BackendCompletion> finished;
        backend_.poll(finished);

        // Concurrency conditions are sampled once per completion batch:
        // the peers sharing the runtime when the work finished, excluding
        // the completing attempt itself.
        const std::size_t concurrent =
            count_running() > 0 ? count_running() - 1 : 0;

        for (const auto& completion : finished) {
            auto* execution =
                executions_.find_execution(completion.execution);

            if (execution == nullptr ||
                execution->state() != ExecutionState::RUNNING) {
                continue;
            }

            const auto outcome = completion.succeeded
                                     ? ExecutionState::COMPLETED
                                     : ExecutionState::FAILED;

            (void)execution->set_state(outcome);

            const bool effects_ok = effects_.finish(*execution);

            const auto integrity =
                effects_ok
                    ? AttemptIntegrity::COHERENT
                    : AttemptIntegrity::EFFECTS_REJECTED;

            (void)execution->record_result(
                ExecutionResult{
                    outcome,
                    {},
                    integrity,
                });

            bool evidence = true;

            if (measurements_ != nullptr) {
                evidence = MeasurementCollector::capture(
                    *measurements_,
                    *execution,
                    execution->description().operation,
                    completion.succeeded,
                    completion.duration_ns,
                    concurrent);
            }

            outcomes.push_back(
                AttemptStatus{
                    completion.execution,
                    integrity,
                    evidence,
                });
        }
    }

    // Cancels one attempt: a PENDING attempt is cancelled without effects;
    // an in-flight attempt receives the failed-or-cancelled finishing
    // effects before its result is recorded. Rejected when the attempt is
    // unknown or already terminal.
    [[nodiscard]] bool cancel(ExecutionId id) {
        auto* execution = executions_.find_execution(id);

        if (execution == nullptr) {
            return false;
        }

        if (is_terminal(execution->state())) {
            return false;
        }

        const bool began =
            execution->state() == ExecutionState::RUNNING;

        if (!execution->set_state(ExecutionState::CANCELLED)) {
            return false;
        }

        // Finishing effects apply exactly when the attempt holds claims,
        // regardless of the state it is cancelled from. A begun attempt
        // with producing bindings that holds no claims is incoherent — its
        // begin effects never applied — and is reported as such.
        bool effects_ok = true;

        if (effects_.holds_claims(*execution)) {
            effects_ok = effects_.finish(*execution);
        } else if (began && effects_.produces(*execution)) {
            effects_ok = false;
        }

        (void)execution->record_result(
            ExecutionResult{
                ExecutionState::CANCELLED,
                {},
                effects_ok
                    ? AttemptIntegrity::COHERENT
                    : AttemptIntegrity::EFFECTS_REJECTED,
            });

        return true;
    }

private:
    // Finalizes an attempt whose begin was incoherent: any claim is
    // released, the rejection is recorded, and the attempt is left terminal
    // so its late backend completion is discarded.
    void finalize_incoherent(Execution& execution) const {
        (void)execution.set_state(ExecutionState::CANCELLED);

        // Claim release is deterministic here: the begin either claimed
        // every producing residency or none.
        (void)effects_.finish(execution);

        (void)execution.record_result(
            ExecutionResult{
                ExecutionState::CANCELLED,
                {},
                AttemptIntegrity::EFFECTS_REJECTED,
            });
    }

    [[nodiscard]] std::size_t count_running() const noexcept {
        std::size_t running = 0;

        executions_.for_each_execution(
            [&](const Execution* execution) {
                if (execution->state() == ExecutionState::RUNNING) {
                    ++running;
                }
            });

        return running;
    }

    ExecutionRegistry& executions_;
    OperationRegistry& operations_;
    ExecutionBackend& backend_;
    MeasurementRegistry* measurements_;
    ExecutionAdmissionValidator admission_;
    BindingAdmissibilityValidator admissibility_;
    ExecutionEffects effects_;
};

} // namespace gerdos