#include <cassert>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/physical_binding.hpp"
#include "gerdos/core/physical_binding_resolution.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"

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

    BindingResolver resolver(devices, data_registry);

    // Empty bindings are fully resolved by the result contract.
    {
        const PhysicalBinding binding;
        const auto resolution = resolver.resolve(binding);

        assert(resolution.data.empty());
        assert(resolution.resources.empty());
        assert(resolution.fully_resolved());
    }

    // Existing resources and residencies resolve through their owning
    // registries, while roles and original references are preserved.
    {
        PhysicalBinding binding;

        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{400},
                },
            });

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{200},
                },
            });

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{201},
                },
            });

        const auto resolution = resolver.resolve(binding);

        assert(resolution.fully_resolved());

        assert(resolution.data.size() == 1);
        assert(resolution.data.front().role == DataBindingRole::INPUT);
        const DataResidencyRef expected_residency{
            DataId{300},
            DataResidencyId{400},
        };
        assert(resolution.data.front().residency == expected_residency);
        assert(resolution.data.front().resolved ==
               data->find_residency(DataResidencyId{400}));

        assert(resolution.resources.size() == 2);

        assert(resolution.resources[0].role ==
               ResourceBindingRole::COMPUTE);
        const ResourceRef expected_compute_resource{
            DeviceId{100},
            ResourceId{200},
        };
        assert(resolution.resources[0].resource ==
               expected_compute_resource);
        assert(resolution.resources[0].resolved ==
               device->find_resource(ResourceId{200}));

        assert(resolution.resources[1].role ==
               ResourceBindingRole::TRANSFER);
        const ResourceRef expected_transfer_resource{
            DeviceId{100},
            ResourceId{201},
        };
        assert(resolution.resources[1].resource ==
               expected_transfer_resource);
        assert(resolution.resources[1].resolved ==
               device->find_resource(ResourceId{201}));
    }

    // A missing device leaves the resource unresolved.
    {
        PhysicalBinding binding;

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{999},
                    ResourceId{200},
                },
            });

        const auto resolution = resolver.resolve(binding);

        assert(resolution.resources.size() == 1);
        assert(resolution.resources.front().resolved == nullptr);
        assert(!resolution.fully_resolved());
    }

    // An existing device with a missing resource leaves the resource
    // unresolved.
    {
        PhysicalBinding binding;

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{999},
                },
            });

        const auto resolution = resolver.resolve(binding);

        assert(resolution.resources.size() == 1);
        assert(resolution.resources.front().resolved == nullptr);
        assert(!resolution.fully_resolved());
    }

    // A missing DataId leaves the residency unresolved.
    {
        PhysicalBinding binding;

        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{999},
                    DataResidencyId{400},
                },
            });

        const auto resolution = resolver.resolve(binding);

        assert(resolution.data.size() == 1);
        assert(resolution.data.front().resolved == nullptr);
        assert(!resolution.fully_resolved());
    }

    // An existing DataId with a missing residency remains unresolved.
    {
        PhysicalBinding binding;

        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{999},
                },
            });

        const auto resolution = resolver.resolve(binding);

        assert(resolution.data.size() == 1);
        assert(resolution.data.front().resolved == nullptr);
        assert(!resolution.fully_resolved());
    }

    // Removing a previously bound resource does not rewrite the binding;
    // a fresh resolution simply becomes unresolved.
    {
        PhysicalBinding binding;

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{200},
                },
            });

        const auto before_removal = resolver.resolve(binding);
        assert(before_removal.resources.front().resolved != nullptr);

        assert(device->remove_resource(ResourceId{200}));

        const auto after_removal = resolver.resolve(binding);
        assert(after_removal.resources.front().resolved == nullptr);
        assert(!after_removal.fully_resolved());
    }

    // Removing a previously bound residency has the same semantics.
    {
        PhysicalBinding binding;

        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{400},
                },
            });

        const auto before_removal = resolver.resolve(binding);
        assert(before_removal.data.front().resolved != nullptr);

        assert(data->remove_residency(DataResidencyId{400}));

        const auto after_removal = resolver.resolve(binding);
        assert(after_removal.data.front().resolved == nullptr);
        assert(!after_removal.fully_resolved());
    }

    return 0;
}
