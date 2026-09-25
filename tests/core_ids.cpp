#include <cassert>
#include <unordered_map>

#include "gerdos/core/ids.hpp"

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

    return 0;
}
