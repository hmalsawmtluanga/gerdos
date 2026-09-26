#pragma once

#include <utility>

#include "gerdos/core/execution_state.hpp"
#include "gerdos/core/ids.hpp"

namespace gerdos {

struct ExecutionDescription {
    ExecutionId id;
    OperationId operation;
};

class Execution {
public:
    explicit Execution(ExecutionDescription description)
        : description_(std::move(description)) {}

    Execution(const Execution&) = delete;
    Execution& operator=(const Execution&) = delete;
    Execution(Execution&&) noexcept = default;
    Execution& operator=(Execution&&) noexcept = delete;

    [[nodiscard]] const ExecutionDescription&
    description() const noexcept {
        return description_;
    }

    [[nodiscard]] ExecutionState state() const noexcept {
        return state_;
    }

    [[nodiscard]] bool set_state(ExecutionState state) noexcept {
        if (!can_transition(state_, state)) {
            return false;
        }

        state_ = state;
        return true;
    }

private:
    ExecutionDescription description_;
    ExecutionState state_{ExecutionState::PENDING};
};

} // namespace gerdos
