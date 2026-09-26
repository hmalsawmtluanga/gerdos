#pragma once

#include <optional>
#include <utility>

#include "gerdos/core/execution_state.hpp"
#include "gerdos/core/ids.hpp"
#include "gerdos/core/physical_binding.hpp"

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
};

} // namespace gerdos
