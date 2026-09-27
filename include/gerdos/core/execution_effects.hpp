#pragma once

#include <vector>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/execution.hpp"
#include "gerdos/core/execution_registry.hpp"

namespace gerdos {

// Execution effects: the residency state changes an attempt applies to the
// physical representations it produces. Effects apply only to data bindings
// in producing roles; consuming bindings are never modified. Application is
// transactional: a call either applies all of its state changes or leaves
// every residency unchanged and reports rejection. No step allocates, so no
// call can fail after a mutation has begun.
//
// An update-in-progress is owned by exactly one attempt: starting effects
// claim the producing residencies for the attempt, finishing effects apply
// only to residencies the attempt claims, and a claim held by a terminal
// attempt is stale: a later attempt may take it over, which releases the
// stale owner's remaining claims so that none of them can wedge a
// representation.
class ExecutionEffects {
public:
    ExecutionEffects(
        DataRegistry& data_registry,
        const ExecutionRegistry& executions) noexcept
        : data_registry_(data_registry),
          executions_(executions) {}

    // Whether the attempt can claim every producing residency it binds:
    // none of them is claimed by another live attempt.
    [[nodiscard]] bool claims_free(const Execution& execution) const
        noexcept {
        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return true;
        }

        for (const auto& data_binding : binding->data) {
            const auto role = data_binding.role;

            if (is_consuming(role)) {
                continue;
            }

            if (!is_producing(role)) {
                return false;
            }

            const auto* residency = resolve(data_binding.residency);

            if (residency == nullptr) {
                return false;
            }

            if (claimed_by_other(
                    *residency,
                    execution.description().id)) {
                return false;
            }
        }

        return true;
    }

    // Whether the attempt currently claims any producing residency.
    [[nodiscard]] bool holds_claims(const Execution& execution) const
        noexcept {
        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return false;
        }

        for (const auto& data_binding : binding->data) {
            const auto role = data_binding.role;

            if (is_consuming(role)) {
                continue;
            }

            const auto* residency = resolve(data_binding.residency);

            if (residency != nullptr &&
                residency->update_owner() == execution.description().id) {
                return true;
            }
        }

        return false;
    }

    // Whether the attempt binds any producing residency at all.
    [[nodiscard]] bool produces(const Execution& execution) const
        noexcept {
        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return false;
        }

        for (const auto& data_binding : binding->data) {
            if (is_producing(data_binding.role)) {
                return true;
            }
        }

        return false;
    }

    // Claims every producing residency for the attempt. Idempotent for the
    // claiming attempt. Rejected when the attempt is terminal, when a
    // producing residency cannot be resolved, or when another live attempt
    // holds a claim. Taking over a stale claim releases the stale owner's
    // remaining claims.
    [[nodiscard]] bool start(const Execution& execution) const {
        if (is_terminal(execution.state())) {
            return false;
        }

        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return true;
        }

        return apply_claim(*binding, execution.description().id);
    }

    // Applies the outcome effects to producing residencies claimed by the
    // attempt: a completed attempt makes them VALID; a failed or cancelled
    // attempt makes them UNAVAILABLE. Single-shot: the claim must still be
    // held and the update still in progress.
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

        return apply_outcome(*binding, execution.description().id, target);
    }

    // Releases every claim held by an attempt whose update can no longer
    // complete, so that its remaining representations cannot be wedged.
    // This is the explicit-release path of the claim contract: claims are
    // released by finishing, by takeover, or here.
    void release_claims_of(ExecutionId owner) const noexcept {
        data_registry_.for_each_data(
            [&](Data* data) {
                data->for_each_residency(
                    [&](DataResidency* residency) {
                        if (residency->update_owner() == owner) {
                            residency->set_update_owner(ExecutionId{});
                        }
                    });
            });
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

    [[nodiscard]] bool live(ExecutionId owner) const noexcept {
        const auto* attempt = executions_.find_execution(owner);

        return attempt != nullptr && !is_terminal(attempt->state());
    }

    [[nodiscard]] bool claimed_by_other(
        const DataResidency& residency,
        ExecutionId id) const noexcept {
        const auto owner = residency.update_owner();

        return owner.valid() && owner != id && live(owner);
    }

    [[nodiscard]] bool apply_claim(
        const PhysicalBinding& binding,
        ExecutionId id) const {
        for (const auto& data_binding : binding.data) {
            const auto role = data_binding.role;

            if (is_consuming(role)) {
                continue;
            }

            if (!is_producing(role)) {
                return false;
            }

            const auto* residency = resolve(data_binding.residency);

            if (residency == nullptr) {
                return false;
            }

            if (claimed_by_other(*residency, id)) {
                return false;
            }
        }

        for (const auto& data_binding : binding.data) {
            if (!is_producing(data_binding.role)) {
                continue;
            }

            auto* residency = resolve(data_binding.residency);
            const auto stale_owner = residency->update_owner();

            if (stale_owner.valid() && stale_owner != id &&
                !live(stale_owner)) {
                release_claims_of(stale_owner);
            }

            (void)residency->set_state(DataResidencyState::TRANSFERRING);
            residency->set_update_owner(id);
        }

        return true;
    }

    [[nodiscard]] bool apply_outcome(
        const PhysicalBinding& binding,
        ExecutionId id,
        DataResidencyState target) const {
        for (const auto& data_binding : binding.data) {
            const auto role = data_binding.role;

            if (is_consuming(role)) {
                continue;
            }

            if (!is_producing(role)) {
                return false;
            }

            const auto* residency = resolve(data_binding.residency);

            if (residency == nullptr) {
                return false;
            }

            if (residency->state() != DataResidencyState::TRANSFERRING ||
                residency->update_owner() != id) {
                return false;
            }
        }

        // Setting the state outside the update-in-progress state releases
        // the claim. The target transition is legal from the
        // update-in-progress state for both outcomes.
        for (const auto& data_binding : binding.data) {
            if (!is_producing(data_binding.role)) {
                continue;
            }

            auto* residency = resolve(data_binding.residency);
            (void)residency->set_state(target);
        }

        return true;
    }

    DataRegistry& data_registry_;
    const ExecutionRegistry& executions_;
};

} // namespace gerdos