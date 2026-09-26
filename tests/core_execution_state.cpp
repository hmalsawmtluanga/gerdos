#include "test_check.hpp"

#include "gerdos/core/execution_state.hpp"

int main() {
    using namespace gerdos;

    GERDOS_CHECK(!is_terminal(ExecutionState::PENDING));
    GERDOS_CHECK(!is_terminal(ExecutionState::RUNNING));
    GERDOS_CHECK(is_terminal(ExecutionState::COMPLETED));
    GERDOS_CHECK(is_terminal(ExecutionState::FAILED));
    GERDOS_CHECK(is_terminal(ExecutionState::CANCELLED));

    GERDOS_CHECK(can_transition(
        ExecutionState::PENDING,
        ExecutionState::PENDING));

    GERDOS_CHECK(can_transition(
        ExecutionState::PENDING,
        ExecutionState::RUNNING));

    GERDOS_CHECK(can_transition(
        ExecutionState::PENDING,
        ExecutionState::CANCELLED));

    GERDOS_CHECK(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::RUNNING));

    GERDOS_CHECK(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::COMPLETED));

    GERDOS_CHECK(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::FAILED));

    GERDOS_CHECK(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::CANCELLED));

    GERDOS_CHECK(!can_transition(
        ExecutionState::RUNNING,
        ExecutionState::PENDING));

    GERDOS_CHECK(!can_transition(
        ExecutionState::PENDING,
        ExecutionState::COMPLETED));

    GERDOS_CHECK(!can_transition(
        ExecutionState::PENDING,
        ExecutionState::FAILED));

    GERDOS_CHECK(!can_transition(
        ExecutionState::COMPLETED,
        ExecutionState::PENDING));

    GERDOS_CHECK(!can_transition(
        ExecutionState::COMPLETED,
        ExecutionState::RUNNING));

    GERDOS_CHECK(!can_transition(
        ExecutionState::COMPLETED,
        ExecutionState::FAILED));

    GERDOS_CHECK(!can_transition(
        ExecutionState::FAILED,
        ExecutionState::PENDING));

    GERDOS_CHECK(!can_transition(
        ExecutionState::FAILED,
        ExecutionState::RUNNING));

    GERDOS_CHECK(!can_transition(
        ExecutionState::FAILED,
        ExecutionState::COMPLETED));

    GERDOS_CHECK(!can_transition(
        ExecutionState::CANCELLED,
        ExecutionState::PENDING));

    GERDOS_CHECK(!can_transition(
        ExecutionState::CANCELLED,
        ExecutionState::RUNNING));

    GERDOS_CHECK(!can_transition(
        ExecutionState::CANCELLED,
        ExecutionState::COMPLETED));

    return 0;
}
