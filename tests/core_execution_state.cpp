#include <cassert>

#include "gerdos/core/execution_state.hpp"

int main() {
    using namespace gerdos;

    assert(!is_terminal(ExecutionState::PENDING));
    assert(!is_terminal(ExecutionState::RUNNING));
    assert(is_terminal(ExecutionState::COMPLETED));
    assert(is_terminal(ExecutionState::FAILED));
    assert(is_terminal(ExecutionState::CANCELLED));

    assert(can_transition(
        ExecutionState::PENDING,
        ExecutionState::PENDING));

    assert(can_transition(
        ExecutionState::PENDING,
        ExecutionState::RUNNING));

    assert(can_transition(
        ExecutionState::PENDING,
        ExecutionState::CANCELLED));

    assert(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::RUNNING));

    assert(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::COMPLETED));

    assert(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::FAILED));

    assert(can_transition(
        ExecutionState::RUNNING,
        ExecutionState::CANCELLED));

    assert(!can_transition(
        ExecutionState::RUNNING,
        ExecutionState::PENDING));

    assert(!can_transition(
        ExecutionState::PENDING,
        ExecutionState::COMPLETED));

    assert(!can_transition(
        ExecutionState::PENDING,
        ExecutionState::FAILED));

    assert(!can_transition(
        ExecutionState::COMPLETED,
        ExecutionState::PENDING));

    assert(!can_transition(
        ExecutionState::COMPLETED,
        ExecutionState::RUNNING));

    assert(!can_transition(
        ExecutionState::COMPLETED,
        ExecutionState::FAILED));

    assert(!can_transition(
        ExecutionState::FAILED,
        ExecutionState::PENDING));

    assert(!can_transition(
        ExecutionState::FAILED,
        ExecutionState::RUNNING));

    assert(!can_transition(
        ExecutionState::FAILED,
        ExecutionState::COMPLETED));

    assert(!can_transition(
        ExecutionState::CANCELLED,
        ExecutionState::PENDING));

    assert(!can_transition(
        ExecutionState::CANCELLED,
        ExecutionState::RUNNING));

    assert(!can_transition(
        ExecutionState::CANCELLED,
        ExecutionState::COMPLETED));

    return 0;
}
