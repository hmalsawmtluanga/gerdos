#pragma once

#include <unordered_set>
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

// The runtime integration layer: the executor is the only component that
// advances an attempt through its lifecycle. It composes the gates as law —
// semantic admissibility, execution admission, execution effects, and the
// backend seam — in the sequence mandated by the execution effects contract.
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
          effects_(data) {}

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

        if (!admission_.admit(*execution)) {
            return false;
        }

        if (!backend_.submit(*operation, *execution)) {
            return false;
        }

        // Pinned invariants (tests/core_executor.cpp): given admission,
        // start effects and the RUNNING transition cannot reject — every
        // producing residency resolves and can enter the update-in-progress
        // state, and a PENDING attempt with a binding may enter RUNNING. If
        // either ever rejects, the attempt is in flight but incoherent: the
        // rejection is remembered and reported with its completion instead of
        // being discarded.
        const bool coherent = effects_.start(*execution) &&
                              execution->set_state(ExecutionState::RUNNING);

        if (!coherent) {
            incoherent_.insert(id);
        }

        return true;
    }

    // Advances backend work and applies the completion sequence — terminal
    // transition, finish effects, result recording, measurement capture —
    // for attempts that finished since the previous call. Every completion
    // is reported with its integrity; an effects rejection is recorded in
    // the attempt's result and reported here, never discarded.
    void advance(std::vector<AttemptStatus>& outcomes) {
        std::vector<BackendCompletion> finished;
        backend_.poll(finished);

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
            const bool started_coherent =
                incoherent_.erase(completion.execution) == 0;

            const auto integrity =
                effects_ok && started_coherent
                    ? AttemptIntegrity::COHERENT
                    : AttemptIntegrity::EFFECTS_REJECTED;

            (void)execution->record_result(
                ExecutionResult{
                    outcome,
                    {},
                    integrity,
                });

            if (measurements_ != nullptr) {
                const auto* operation = operations_.find_operation(
                    execution->description().operation);

                if (operation != nullptr) {
                    MeasurementCollector::capture(
                        *measurements_,
                        *execution,
                        *operation,
                        completion.duration_ns,
                        count_running());
                }
            }

            outcomes.push_back(
                AttemptStatus{
                    completion.execution,
                    integrity,
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

        const bool started =
            execution->state() == ExecutionState::RUNNING;

        if (is_terminal(execution->state())) {
            return false;
        }

        if (!execution->set_state(ExecutionState::CANCELLED)) {
            return false;
        }

        const bool effects_ok =
            !started || effects_.finish(*execution);

        const bool started_coherent = incoherent_.erase(id) == 0;

        (void)execution->record_result(
            ExecutionResult{
                ExecutionState::CANCELLED,
                {},
                effects_ok && started_coherent
                    ? AttemptIntegrity::COHERENT
                    : AttemptIntegrity::EFFECTS_REJECTED,
            });

        return true;
    }

private:
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
    std::unordered_set<ExecutionId> incoherent_;
    ExecutionBackend& backend_;
    MeasurementRegistry* measurements_;
    ExecutionAdmissionValidator admission_;
    BindingAdmissibilityValidator admissibility_;
    ExecutionEffects effects_;
};

} // namespace gerdos