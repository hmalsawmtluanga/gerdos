#include <cassert>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/physical_binding.hpp"
#include "gerdos/core/physical_binding_resolution.hpp"

int main() {
    using namespace gerdos;

    DeviceRegistry devices;
    DataRegistry data_registry;

    auto* device = devices.create_device(
        DeviceDescription{
            DeviceId{100},
            "Synthetic Device",
        });
    assert(device != nullptr);

    assert(device->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{200},
                DeviceId{100},
                ResourceKind::COMPUTE,
                "Synthetic Compute",
            },
        }));

    assert(device->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{201},
                DeviceId{100},
                ResourceKind::TRANSFER,
                "Synthetic Transfer",
            },
        }));

    auto* data = data_registry.create_data(
        DataDescription{
            DataId{300},
            "Synthetic Data",
        });
    assert(data != nullptr);

    assert(data->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{400},
                DataId{300},
                ResourceRef{
                    DeviceId{100},
                    ResourceId{200},
                },
                "primary",
            },
        }));

    // The result contract itself is valid for an empty binding.
    {
        BindingResolution resolution;
        assert(resolution.fully_resolved());
    }

    // Valid identities must be representable in the resolution result.
    {
        BindingResolution resolution;

        resolution.data.push_back(
            ResolvedDataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{400},
                },
                data->find_residency(DataResidencyId{400}),
            });

        resolution.resources.push_back(
            ResolvedResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{200},
                },
                device->find_resource(ResourceId{200}),
            });

        assert(resolution.fully_resolved());
        assert(resolution.data.size() == 1);
        assert(resolution.resources.size() == 1);
        assert(resolution.data.front().role == DataBindingRole::INPUT);
        assert(resolution.resources.front().role ==
               ResourceBindingRole::COMPUTE);
        assert(resolution.data.front().resolved != nullptr);
        assert(resolution.resources.front().resolved != nullptr);
    }

    // Any unresolved entry makes the complete resolution incomplete.
    {
        BindingResolution resolution;

        resolution.resources.push_back(
            ResolvedResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{999},
                },
                nullptr,
            });

        assert(!resolution.fully_resolved());
    }

    return 0;
}
