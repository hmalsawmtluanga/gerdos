#include "test_check.hpp"

#include "gerdos/core/execution.hpp"
#include "gerdos/core/physical_binding.hpp"

int main() {
    using namespace gerdos;

    const ExecutionDescription description{
        ExecutionId{100},
        OperationId{42},
    };

    Execution execution(description);

    GERDOS_CHECK(execution.description().id == ExecutionId{100});
    GERDOS_CHECK(execution.description().operation == OperationId{42});
    GERDOS_CHECK(execution.state() == ExecutionState::PENDING);

    GERDOS_CHECK(execution.bind(PhysicalBinding{}));
    GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(execution.state() == ExecutionState::RUNNING);

    GERDOS_CHECK(execution.set_state(ExecutionState::COMPLETED));
    GERDOS_CHECK(execution.state() == ExecutionState::COMPLETED);

    GERDOS_CHECK(!execution.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(execution.state() == ExecutionState::COMPLETED);

    // A terminal execution records its result exactly once.
    GERDOS_CHECK(!execution.has_result());
    GERDOS_CHECK(execution.result() == nullptr);

    GERDOS_CHECK(
        execution.record_result(
            ExecutionResult{ExecutionState::COMPLETED, {}}));
    GERDOS_CHECK(execution.has_result());
    GERDOS_CHECK(
        execution.result()->outcome == ExecutionState::COMPLETED);
    GERDOS_CHECK(
        execution.result()->integrity == AttemptIntegrity::COHERENT);

    // Result recording is rejected before a terminal state is reached and
    // for an outcome other than the one reached.
    Execution failed_attempt{
        ExecutionDescription{
            ExecutionId{101},
            OperationId{42},
        }};

    GERDOS_CHECK(
        !failed_attempt.record_result(
            ExecutionResult{ExecutionState::FAILED, "too early"}));
    GERDOS_CHECK(!failed_attempt.has_result());

    GERDOS_CHECK(failed_attempt.bind(PhysicalBinding{}));
    GERDOS_CHECK(failed_attempt.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(failed_attempt.set_state(ExecutionState::FAILED));

    GERDOS_CHECK(
        !failed_attempt.record_result(
            ExecutionResult{ExecutionState::COMPLETED, "mismatched"}));
    GERDOS_CHECK(!failed_attempt.has_result());

    GERDOS_CHECK(
        failed_attempt.record_result(
            ExecutionResult{ExecutionState::FAILED, "device reset"}));

    // The result is immutable after recording.
    GERDOS_CHECK(
        !failed_attempt.record_result(
            ExecutionResult{ExecutionState::FAILED, "rewritten"}));
    GERDOS_CHECK(failed_attempt.result()->detail == "device reset");
    GERDOS_CHECK(failed_attempt.result()->outcome == ExecutionState::FAILED);

    // An attempt whose effects were rejected records its incoherence.
    Execution incoherent_attempt{
        ExecutionDescription{
            ExecutionId{103},
            OperationId{42},
        }};

    GERDOS_CHECK(incoherent_attempt.bind(PhysicalBinding{}));
    GERDOS_CHECK(incoherent_attempt.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(incoherent_attempt.set_state(ExecutionState::COMPLETED));

    GERDOS_CHECK(
        incoherent_attempt.record_result(
            ExecutionResult{
                ExecutionState::COMPLETED,
                {},
                AttemptIntegrity::EFFECTS_REJECTED,
            }));

    GERDOS_CHECK(
        incoherent_attempt.result()->integrity ==
        AttemptIntegrity::EFFECTS_REJECTED);

    // A cancelled attempt records its own result.
    Execution cancelled_attempt{
        ExecutionDescription{
            ExecutionId{102},
            OperationId{42},
        }};

    GERDOS_CHECK(
        cancelled_attempt.set_state(ExecutionState::CANCELLED));

    GERDOS_CHECK(
        cancelled_attempt.record_result(
            ExecutionResult{ExecutionState::CANCELLED, {}}));

    GERDOS_CHECK(
        cancelled_attempt.result()->outcome ==
        ExecutionState::CANCELLED);

    return 0;
}
