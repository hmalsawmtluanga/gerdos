#include <cassert>
#include <unordered_map>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/resource.hpp"

int main() {
    using namespace gerdos;

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

    resource.set_availability(ResourceAvailability::DRAINING);

    assert(
        resource.availability() == ResourceAvailability::DRAINING);
    assert(!resource.available());

    return 0;
}
