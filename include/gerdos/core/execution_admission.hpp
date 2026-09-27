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

    // Establishes admission evidence on the attempt and returns the
    // evidence token. The attempt's RUNNING transition is gated on this
    // evidence.
    [[nodiscard]] std::optional<ExecutionAdmission> admit(
        Execution& execution) const {
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

        execution.mark_admitted();

        return ExecutionAdmission{execution.description().id};
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
            if (!resource_binding.resolved->available()) {
                return false;
            }
        }

        return true;
    }

    PhysicalBindingValidator validator_;
    const DeviceRegistry& devices_;
    BindingResolver resolver_;
};

} // namespace gerdos