#include <cassert>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "gerdos/core/device.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/ids.hpp"
#include "gerdos/core/resource.hpp"

int main() {
    using namespace gerdos;

    static_assert(!std::is_copy_constructible_v<Resource>);
    static_assert(!std::is_copy_assignable_v<Resource>);
    static_assert(std::is_move_constructible_v<Resource>);
    static_assert(!std::is_move_assignable_v<Resource>);

    static_assert(!std::is_copy_constructible_v<Device>);
    static_assert(!std::is_copy_assignable_v<Device>);
    static_assert(!std::is_move_constructible_v<Device>);
    static_assert(!std::is_move_assignable_v<Device>);

    static_assert(!std::is_copy_constructible_v<DeviceRegistry>);
    static_assert(!std::is_copy_assignable_v<DeviceRegistry>);
    static_assert(!std::is_move_constructible_v<DeviceRegistry>);
    static_assert(!std::is_move_assignable_v<DeviceRegistry>);

    const DeviceId device_a{1};
    const DeviceId device_b{1};
    const DeviceId device_c{2};

    assert(!DeviceId{}.valid());
    assert(device_a.valid());

    assert(device_a == device_b);
    assert(device_a != device_c);

    const ResourceId resource_a{1};
    const ResourceId resource_b{1};

    assert(resource_a.valid());
    assert(resource_a.value() == 1);
    assert(resource_a == resource_b);

    std::unordered_map<DeviceId, int> devices;
    devices.emplace(device_a, 42);

    assert(devices.at(device_b) == 42);

    std::unordered_map<ResourceId, int> resources;
    resources.emplace(resource_a, 99);

    assert(resources.at(resource_b) == 99);

    const ResourceDescription resource_description{
        resource_a,
        device_a,
        ResourceKind::COMPUTE,
        "test-compute",
    };

    Resource resource{resource_description};

    assert(resource.description().id == resource_a);
    assert(resource.description().owner == device_a);
    assert(resource.description().kind == ResourceKind::COMPUTE);
    assert(resource.description().name == "test-compute");

    assert(
        resource.availability() == ResourceAvailability::INITIALIZING);
    assert(!resource.available());

    resource.set_availability(ResourceAvailability::AVAILABLE);

    assert(
        resource.availability() == ResourceAvailability::AVAILABLE);
    assert(resource.available());
    assert(is_available(ResourceAvailability::AVAILABLE));
    assert(!is_available(ResourceAvailability::FAILED));

    const DeviceDescription device_description{
        device_a,
        "test-device",
    };

    Device device{device_description};

    assert(device.description().id == device_a);
    assert(device.description().name == "test-device");

    Resource owned_resource{
        ResourceDescription{
            ResourceId{10},
            device_a,
            ResourceKind::COMPUTE,
            "owned-compute",
        },
    };

    assert(device.resource_count() == 0);
    assert(device.add_resource(std::move(owned_resource)));
    assert(device.resource_count() == 1);

    Resource* borrowed_resource = device.find_resource(ResourceId{10});

    assert(borrowed_resource != nullptr);
    assert(borrowed_resource->description().id == ResourceId{10});
    assert(borrowed_resource->description().owner == device_a);
    assert(
        borrowed_resource->description().kind == ResourceKind::COMPUTE);

    Resource duplicate_resource{
        ResourceDescription{
            ResourceId{10},
            device_a,
            ResourceKind::COMPUTE,
            "duplicate-id",
        },
    };

    assert(!device.add_resource(std::move(duplicate_resource)));
    assert(
        duplicate_resource.description().name == "duplicate-id");

    Resource wrong_owner_resource{
        ResourceDescription{
            ResourceId{11},
            device_c,
            ResourceKind::MEMORY,
            "wrong-owner",
        },
    };

    assert(!device.add_resource(std::move(wrong_owner_resource)));
    assert(
        wrong_owner_resource.description().name == "wrong-owner");
    assert(device.resource_count() == 1);
    assert(device.find_resource(ResourceId{11}) == nullptr);

    assert(device.remove_resource(ResourceId{10}));
    assert(device.resource_count() == 0);
    assert(device.find_resource(ResourceId{10}) == nullptr);
    assert(!device.remove_resource(ResourceId{10}));

    Resource invalid_id_resource{
        ResourceDescription{
            ResourceId{},
            device_a,
            ResourceKind::STORAGE,
            "invalid-id",
        },
    };

    assert(!device.add_resource(std::move(invalid_id_resource)));

    resource.set_availability(ResourceAvailability::DRAINING);

    assert(
        resource.availability() == ResourceAvailability::DRAINING);
    assert(!resource.available());

    DeviceRegistry registry;

    assert(registry.device_count() == 0);

    assert(
        registry.create_device(
            DeviceDescription{
                DeviceId{},
                "invalid-device",
            }) == nullptr);

    Device* device_one = registry.create_device(
        DeviceDescription{
            DeviceId{100},
            "device-one",
        });

    assert(device_one != nullptr);
    assert(device_one->description().id == DeviceId{100});
    assert(device_one->description().name == "device-one");
    assert(registry.device_count() == 1);

    assert(registry.find_device(DeviceId{100}) == device_one);
    assert(registry.find_device(DeviceId{101}) == nullptr);

    assert(
        registry.create_device(
            DeviceDescription{
                DeviceId{100},
                "duplicate-device",
            }) == nullptr);

    assert(registry.device_count() == 1);
    assert(registry.find_device(DeviceId{100}) == device_one);

    Resource registry_resource{
        ResourceDescription{
            ResourceId{1000},
            DeviceId{100},
            ResourceKind::COMPUTE,
            "registry-owned-compute",
        },
    };

    assert(device_one->add_resource(std::move(registry_resource)));
    assert(device_one->resource_count() == 1);

    assert(registry.remove_device(DeviceId{100}));
    assert(registry.device_count() == 0);
    assert(registry.find_device(DeviceId{100}) == nullptr);

    assert(!registry.remove_device(DeviceId{100}));

    assert(
        registry.create_device(
            DeviceDescription{
                DeviceId{100},
                "reused-device-id",
            }) == nullptr);

    Device* device_two = registry.create_device(
        DeviceDescription{
            DeviceId{200},
            "device-two",
        });

    assert(device_two != nullptr);
    assert(registry.device_count() == 1);
    assert(registry.find_device(DeviceId{200}) == device_two);

    return 0;
}
