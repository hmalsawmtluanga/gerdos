#pragma once

#include <vector>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/execution_backend.hpp"
#include "gerdos/core/execution_effects.hpp"
#include "gerdos/core/execution_registry.hpp"
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
        ExecutionBackend& backend) noexcept
        : executions_(executions),
          operations_(operations),
          backend_(backend),
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

        // Given admission succeeded, start effects and the RUNNING
        // transition cannot be rejected: every producing residency resolves
        // and can enter the update-in-progress state, and a PENDING attempt
        // with a binding may enter RUNNING.
        (void)effects_.start(*execution);
        (void)execution->set_state(ExecutionState::RUNNING);

        return true;
    }

    // Advances backend work and applies the completion sequence — terminal
    // transition, finish effects, result recording — for attempts that
    // finished since the previous call.
    void advance(std::vector<ExecutionId>& completed) {
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
            (void)effects_.finish(*execution);
            (void)execution->record_result(
                ExecutionResult{
                    outcome,
                    {},
                });

            completed.push_back(completion.execution);
        }
    }

private:
    ExecutionRegistry& executions_;
    OperationRegistry& operations_;
    ExecutionBackend& backend_;
    ExecutionAdmissionValidator admission_;
    BindingAdmissibilityValidator admissibility_;
    ExecutionEffects effects_;
};

} // namespace gerdos