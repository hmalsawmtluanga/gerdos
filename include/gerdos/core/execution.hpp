#pragma once

#include <optional>
#include <string>
#include <utility>

#include "gerdos/core/execution_state.hpp"
#include "gerdos/core/ids.hpp"
#include "gerdos/core/physical_binding.hpp"

namespace gerdos {

struct ExecutionDescription {
    ExecutionId id;
    OperationId operation;
};

// The result of one attempt. The outcome identifies the terminal state the
// attempt reached; the detail is an opaque non-semantic diagnostic. The
// result carries no measurements, output data, or backend state.
struct ExecutionResult {
    ExecutionState outcome;
    std::string detail;
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

    [[nodiscard]] bool has_binding() const noexcept {
        return binding_.has_value();
    }

    [[nodiscard]] const PhysicalBinding* binding() const noexcept {
        if (!binding_.has_value()) {
            return nullptr;
        }

        return &*binding_;
    }

    [[nodiscard]] bool bind(PhysicalBinding binding) {
        if (binding_.has_value()) {
            return false;
        }

        if (state_ != ExecutionState::PENDING) {
            return false;
        }

        binding_ = std::move(binding);
        return true;
    }

    [[nodiscard]] bool record_result(ExecutionResult result) noexcept {
        if (result_.has_value()) {
            return false;
        }

        if (!is_terminal(state_)) {
            return false;
        }

        if (result.outcome != state_) {
            return false;
        }

        result_ = std::move(result);
        return true;
    }

    [[nodiscard]] bool has_result() const noexcept {
        return result_.has_value();
    }

    [[nodiscard]] const ExecutionResult* result() const noexcept {
        if (!result_.has_value()) {
            return nullptr;
        }

        return &*result_;
    }

    [[nodiscard]] bool set_state(ExecutionState state) noexcept {
        if (state == ExecutionState::RUNNING && !binding_.has_value()) {
            return false;
        }

        if (!can_transition(state_, state)) {
            return false;
        }

        state_ = state;
        return true;
    }

private:
    ExecutionDescription description_;
    ExecutionState state_{ExecutionState::PENDING};
    std::optional<PhysicalBinding> binding_;
    std::optional<ExecutionResult> result_;
};

} // namespace gerdos
