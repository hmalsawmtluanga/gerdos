#pragma once

#include <optional>

#include "gerdos/core/data.hpp"
#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/execution.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"
#include "gerdos/core/physical_binding_validation.hpp"
#include "gerdos/core/resource.hpp"

namespace gerdos {

// Admission evidence identifies the Execution that the admission gate found
// to be an executable physical realization. Evidence can be produced only by
// ExecutionAdmissionValidator and is established against the runtime state
// observed at admission time.
class ExecutionAdmission {
public:
    [[nodiscard]] ExecutionId execution() const noexcept {
        return execution_;
    }

private:
    explicit ExecutionAdmission(ExecutionId execution) noexcept
        : execution_(execution) {}

    ExecutionId execution_{};

    friend class ExecutionAdmissionValidator;
};

class ExecutionAdmissionValidator {
public:
    ExecutionAdmissionValidator(
        const DeviceRegistry& devices,
        const DataRegistry& data_registry) noexcept
        : devices_(devices),
          resolver_(devices, data_registry) {}

    // The admission verdict is observational: it does not mutate the
    // attempt, its binding, or any registry. The evidence token identifies
    // what was admitted; recording the evidence on the attempt is a separate
    // step (establish), performed at the commit point.
    [[nodiscard]] std::optional<ExecutionAdmission> admit(
        const Execution& execution) const {
        if (execution.state() != ExecutionState::PENDING) {
            return std::nullopt;
        }

        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return std::nullopt;
        }

        if (binding->data.empty() && binding->resources.empty()) {
            return std::nullopt;
        }

        if (!validator_.validate(*binding)) {
            return std::nullopt;
        }

        const auto resolution = resolver_.resolve(*binding);

        if (!resolution.fully_resolved()) {
            return std::nullopt;
        }

        if (!runtime_usable(resolution)) {
            return std::nullopt;
        }

        return ExecutionAdmission{execution.description().id};
    }

    // Records admission evidence on the attempt: the single step that
    // enables its RUNNING transition. Only evidence produced by this gate
    // for this attempt is accepted, and only once.
    [[nodiscard]] bool establish(
        Execution& execution,
        const ExecutionAdmission& admission) const {
        if (admission.execution() != execution.description().id) {
            return false;
        }

        return execution.mark_admitted();
    }

private:
    [[nodiscard]] bool resource_available(
        ResourceRef ref) const noexcept {
        const auto* device = devices_.find_device(ref.device);

        if (device == nullptr) {
            return false;
        }

        const auto* resource = device->find_resource(ref.resource);

        return resource != nullptr && resource->available();
    }

    // Effective runtime usability: every referenced Resource is available,
    // and every residency in a consuming role is usable. Residencies in
    // producing roles are not required to be usable because a representation
    // that is about to be created or rewritten is not yet usable.
    [[nodiscard]] bool runtime_usable(
        const BindingResolution& resolution) const noexcept {
        for (const auto& data_binding : resolution.data) {
            const auto* residency = data_binding.resolved;

            if (!resource_available(residency->description().resource)) {
                return false;
            }

            if (is_consuming(data_binding.role) && !residency->usable()) {
                return false;
            }
        }

        for (const auto& resource_binding : resolution.resources) {
            const auto* resource = resource_binding.resolved;

            if (!resource->available()) {
                return false;
            }

            if (!role_coherent(
                    resource_binding.role,
                    resource->description().kind)) {
                return false;
            }
        }

        return true;
    }

    // Binding roles name mechanisms: a COMPUTE binding references a compute
    // resource and a TRANSFER binding references a transfer resource. Memory
    // and storage resources are places, reached through residencies, and are
    // never bound as mechanisms.
    [[nodiscard]] static constexpr bool role_coherent(
        ResourceBindingRole role,
        ResourceKind kind) noexcept {
        switch (role) {
        case ResourceBindingRole::COMPUTE:
            return kind == ResourceKind::COMPUTE;
        case ResourceBindingRole::TRANSFER:
            return kind == ResourceKind::TRANSFER;
        default:
            return false;
        }
    }

    PhysicalBindingValidator validator_;
    const DeviceRegistry& devices_;
    BindingResolver resolver_;
};

} // namespace gerdos