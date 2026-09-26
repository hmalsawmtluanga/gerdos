#pragma once

#include <vector>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/execution.hpp"

namespace gerdos {

// Execution effects: the residency state changes an attempt applies to the
// physical representations it produces. Effects apply only to data bindings
// in producing roles; consuming bindings are never modified. Application is
// transactional: a call either applies all of its state changes or leaves
// every residency unchanged and reports rejection.
class ExecutionEffects {
public:
    explicit ExecutionEffects(DataRegistry& data_registry) noexcept
        : data_registry_(data_registry) {}

    // Marks every producing residency update-in-progress. Idempotent.
    // Rejected when the attempt is terminal, when a producing residency
    // cannot be resolved, or when a binding role is outside the role domain.
    [[nodiscard]] bool start(const Execution& execution) const {
        if (is_terminal(execution.state())) {
            return false;
        }

        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return true;
        }

        return apply(*binding, DataResidencyState::TRANSFERRING);
    }

    // Applies the outcome effects to producing residencies whose update is in
    // progress: a completed attempt makes them VALID; a failed or cancelled
    // attempt makes them UNAVAILABLE. Single-shot: finishing effects require
    // every producing residency to still be update-in-progress.
    [[nodiscard]] bool finish(const Execution& execution) const {
        if (!is_terminal(execution.state())) {
            return false;
        }

        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return true;
        }

        const auto target = execution.state() == ExecutionState::COMPLETED
                                ? DataResidencyState::VALID
                                : DataResidencyState::UNAVAILABLE;

        return apply(*binding, target, true);
    }

private:
    [[nodiscard]] DataResidency* resolve(
        DataResidencyRef ref) const noexcept {
        auto* data = data_registry_.find_data(ref.data);

        if (data == nullptr) {
            return nullptr;
        }

        return data->find_residency(ref.residency);
    }

    // Transactional application: validate every producing entry first, then
    // apply. With require_updating set, every producing residency must still
    // be in the update-in-progress state.
    [[nodiscard]] bool apply(
        const PhysicalBinding& binding,
        DataResidencyState target,
        bool require_updating = false) const {
        std::vector<DataResidency*> producing;

        for (const auto& data_binding : binding.data) {
            const auto role = data_binding.role;

            if (is_consuming(role)) {
                continue;
            }

            if (!is_producing(role)) {
                return false;
            }

            auto* residency = resolve(data_binding.residency);

            if (residency == nullptr) {
                return false;
            }

            if (require_updating &&
                residency->state() != DataResidencyState::TRANSFERRING) {
                return false;
            }

            if (!can_transition(residency->state(), target)) {
                return false;
            }

            producing.push_back(residency);
        }

        for (auto* residency : producing) {
            (void)residency->set_state(target);
        }

        return true;
    }

    DataRegistry& data_registry_;
};

} // namespace gerdos