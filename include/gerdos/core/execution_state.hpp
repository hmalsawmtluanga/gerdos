#pragma once

namespace gerdos {

enum class ExecutionState {
    PENDING,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED,
};

[[nodiscard]] constexpr bool is_terminal(
    ExecutionState state) noexcept {
    return state == ExecutionState::COMPLETED ||
           state == ExecutionState::FAILED ||
           state == ExecutionState::CANCELLED;
}

[[nodiscard]] constexpr bool can_transition(
    ExecutionState from,
    ExecutionState to) noexcept {
    if (from == to) {
        return true;
    }

    switch (from) {
    case ExecutionState::PENDING:
        return to == ExecutionState::RUNNING ||
               to == ExecutionState::CANCELLED;

    case ExecutionState::RUNNING:
        return to == ExecutionState::COMPLETED ||
               to == ExecutionState::FAILED ||
               to == ExecutionState::CANCELLED;

    case ExecutionState::COMPLETED:
    case ExecutionState::FAILED:
    case ExecutionState::CANCELLED:
        return false;
    }

    return false;
}

} // namespace gerdos
