#pragma once

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/physical_binding.hpp"
#include "gerdos/core/physical_binding_resolution.hpp"

namespace gerdos {

class BindingResolver {
public:
    BindingResolver(
        const DeviceRegistry& devices,
        const DataRegistry& data_registry) noexcept
        : devices_(devices),
          data_registry_(data_registry) {}

    [[nodiscard]] BindingResolution resolve(
        const PhysicalBinding& binding) const {
        BindingResolution resolution;

        resolution.data.reserve(binding.data.size());
        resolution.resources.reserve(binding.resources.size());

        for (const auto& data_binding : binding.data) {
            resolution.data.push_back(
                ResolvedDataBinding{
                    data_binding.role,
                    data_binding.residency,
                    resolve_residency(data_binding.residency),
                });
        }

        for (const auto& resource_binding : binding.resources) {
            resolution.resources.push_back(
                ResolvedResourceBinding{
                    resource_binding.role,
                    resource_binding.resource,
                    resolve_resource(resource_binding.resource),
                });
        }

        return resolution;
    }

private:
    [[nodiscard]] const Resource* resolve_resource(
        ResourceRef ref) const noexcept {
        const auto* device = devices_.find_device(ref.device);

        if (device == nullptr) {
            return nullptr;
        }

        return device->find_resource(ref.resource);
    }

    [[nodiscard]] const DataResidency* resolve_residency(
        DataResidencyRef ref) const noexcept {
        const auto* data = data_registry_.find_data(ref.data);

        if (data == nullptr) {
            return nullptr;
        }

        return data->find_residency(ref.residency);
    }

    const DeviceRegistry& devices_;
    const DataRegistry& data_registry_;
};

} // namespace gerdos
