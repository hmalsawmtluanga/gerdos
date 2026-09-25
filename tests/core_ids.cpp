#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "gerdos/core/data.hpp"
#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/ids.hpp"
#include "gerdos/core/resource.hpp"
#include "gerdos/core/topology.hpp"

namespace {

void check(bool condition, const char* expression) {
    if (!condition) {
        std::fprintf(stderr, "TEST FAILURE: %s\n", expression);
        std::abort();
    }
}

} // namespace

#define GERDOS_CHECK(condition) check((condition), #condition)

int main() {
    using namespace gerdos;

    static_assert(!std::is_copy_constructible_v<DataResidency>);
    static_assert(!std::is_copy_assignable_v<DataResidency>);
    static_assert(std::is_move_constructible_v<DataResidency>);
    static_assert(!std::is_move_assignable_v<DataResidency>);

    static_assert(!std::is_copy_constructible_v<Data>);
    static_assert(!std::is_copy_assignable_v<Data>);
    static_assert(!std::is_move_constructible_v<Data>);
    static_assert(!std::is_move_assignable_v<Data>);

    static_assert(!std::is_copy_constructible_v<DataRegistry>);
    static_assert(!std::is_copy_assignable_v<DataRegistry>);
    static_assert(!std::is_move_constructible_v<DataRegistry>);
    static_assert(!std::is_move_assignable_v<DataRegistry>);

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

    static_assert(!std::is_copy_constructible_v<TopologyLink>);
    static_assert(!std::is_copy_assignable_v<TopologyLink>);
    static_assert(std::is_move_constructible_v<TopologyLink>);
    static_assert(!std::is_move_assignable_v<TopologyLink>);

    static_assert(!std::is_copy_constructible_v<Topology>);
    static_assert(!std::is_copy_assignable_v<Topology>);
    static_assert(!std::is_move_constructible_v<Topology>);
    static_assert(!std::is_move_assignable_v<Topology>);

    const DeviceId device_a{1};
    const DeviceId device_b{1};
    const DeviceId device_c{2};

    GERDOS_CHECK(!DeviceId{}.valid());
    GERDOS_CHECK(device_a.valid());

    GERDOS_CHECK(device_a == device_b);
    GERDOS_CHECK(device_a != device_c);

    const ResourceId resource_a{1};
    const ResourceId resource_b{1};

    GERDOS_CHECK(resource_a.valid());
    GERDOS_CHECK(resource_a.value() == 1);
    GERDOS_CHECK(resource_a == resource_b);

    std::unordered_map<DeviceId, int> devices;
    devices.emplace(device_a, 42);

    GERDOS_CHECK(devices.at(device_b) == 42);

    std::unordered_map<ResourceId, int> resources;
    resources.emplace(resource_a, 99);

    GERDOS_CHECK(resources.at(resource_b) == 99);

    const ResourceDescription resource_description{
        resource_a,
        device_a,
        ResourceKind::COMPUTE,
        "test-compute",
    };

    Resource resource{resource_description};

    GERDOS_CHECK(resource.description().id == resource_a);
    GERDOS_CHECK(resource.description().owner == device_a);
    GERDOS_CHECK(resource.description().kind == ResourceKind::COMPUTE);
    GERDOS_CHECK(resource.description().name == "test-compute");

    GERDOS_CHECK(
        resource.availability() == ResourceAvailability::INITIALIZING);
    GERDOS_CHECK(!resource.available());

    resource.set_availability(ResourceAvailability::AVAILABLE);

    GERDOS_CHECK(
        resource.availability() == ResourceAvailability::AVAILABLE);
    GERDOS_CHECK(resource.available());
    GERDOS_CHECK(is_available(ResourceAvailability::AVAILABLE));
    GERDOS_CHECK(!is_available(ResourceAvailability::FAILED));

    const DeviceDescription device_description{
        device_a,
        "test-device",
    };

    Device device{device_description};

    GERDOS_CHECK(device.description().id == device_a);
    GERDOS_CHECK(device.description().name == "test-device");

    Resource owned_resource{
        ResourceDescription{
            ResourceId{10},
            device_a,
            ResourceKind::COMPUTE,
            "owned-compute",
        },
    };

    GERDOS_CHECK(device.resource_count() == 0);
    GERDOS_CHECK(device.add_resource(std::move(owned_resource)));
    GERDOS_CHECK(device.resource_count() == 1);

    Resource* borrowed_resource = device.find_resource(ResourceId{10});

    GERDOS_CHECK(borrowed_resource != nullptr);
    GERDOS_CHECK(borrowed_resource->description().id == ResourceId{10});
    GERDOS_CHECK(borrowed_resource->description().owner == device_a);
    GERDOS_CHECK(
        borrowed_resource->description().kind == ResourceKind::COMPUTE);

    std::size_t resource_enumerated = 0;
    device.for_each_resource(
        [&](Resource* resource) {
            GERDOS_CHECK(resource != nullptr);
            GERDOS_CHECK(resource->description().id == ResourceId{10});
            ++resource_enumerated;
        });
    GERDOS_CHECK(resource_enumerated == 1);

    const Device& const_device = device;
    std::size_t const_resource_enumerated = 0;
    const_device.for_each_resource(
        [&](const Resource* resource) {
            GERDOS_CHECK(resource != nullptr);
            GERDOS_CHECK(resource->description().id == ResourceId{10});
            ++const_resource_enumerated;
        });
    GERDOS_CHECK(const_resource_enumerated == 1);

    Resource duplicate_resource{
        ResourceDescription{
            ResourceId{10},
            device_a,
            ResourceKind::COMPUTE,
            "duplicate-id",
        },
    };

    GERDOS_CHECK(!device.add_resource(std::move(duplicate_resource)));
    GERDOS_CHECK(
        duplicate_resource.description().name == "duplicate-id");

    Resource wrong_owner_resource{
        ResourceDescription{
            ResourceId{11},
            device_c,
            ResourceKind::MEMORY,
            "wrong-owner",
        },
    };

    GERDOS_CHECK(!device.add_resource(std::move(wrong_owner_resource)));
    GERDOS_CHECK(
        wrong_owner_resource.description().name == "wrong-owner");
    GERDOS_CHECK(device.resource_count() == 1);
    GERDOS_CHECK(device.find_resource(ResourceId{11}) == nullptr);

    GERDOS_CHECK(device.remove_resource(ResourceId{10}));
    GERDOS_CHECK(device.resource_count() == 0);
    GERDOS_CHECK(device.find_resource(ResourceId{10}) == nullptr);
    GERDOS_CHECK(!device.remove_resource(ResourceId{10}));

    Resource reused_resource{
        ResourceDescription{
            ResourceId{10},
            device_a,
            ResourceKind::MEMORY,
            "reused-resource-id",
        },
    };

    GERDOS_CHECK(!device.add_resource(std::move(reused_resource)));
    GERDOS_CHECK(
        reused_resource.description().name == "reused-resource-id");
    GERDOS_CHECK(device.resource_count() == 0);

    Resource invalid_id_resource{
        ResourceDescription{
            ResourceId{},
            device_a,
            ResourceKind::STORAGE,
            "invalid-id",
        },
    };

    GERDOS_CHECK(!device.add_resource(std::move(invalid_id_resource)));

    resource.set_availability(ResourceAvailability::DRAINING);

    GERDOS_CHECK(
        resource.availability() == ResourceAvailability::DRAINING);
    GERDOS_CHECK(!resource.available());

    DeviceRegistry registry;

    GERDOS_CHECK(registry.device_count() == 0);

    GERDOS_CHECK(
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

    GERDOS_CHECK(device_one != nullptr);
    GERDOS_CHECK(device_one->description().id == DeviceId{100});
    GERDOS_CHECK(device_one->description().name == "device-one");
    GERDOS_CHECK(registry.device_count() == 1);

    GERDOS_CHECK(registry.find_device(DeviceId{100}) == device_one);
    GERDOS_CHECK(registry.find_device(DeviceId{101}) == nullptr);

    std::size_t registry_enumerated = 0;
    registry.for_each_device(
        [&](Device* device) {
            GERDOS_CHECK(device != nullptr);
            GERDOS_CHECK(device->description().id == DeviceId{100});
            ++registry_enumerated;
        });
    GERDOS_CHECK(registry_enumerated == 1);

    const DeviceRegistry& const_registry = registry;
    std::size_t const_registry_enumerated = 0;
    const_registry.for_each_device(
        [&](const Device* device) {
            GERDOS_CHECK(device != nullptr);
            GERDOS_CHECK(device->description().id == DeviceId{100});
            ++const_registry_enumerated;
        });
    GERDOS_CHECK(const_registry_enumerated == 1);

    GERDOS_CHECK(
        registry.create_device(
            DeviceDescription{
                DeviceId{100},
                "duplicate-device",
            }) == nullptr);

    GERDOS_CHECK(registry.device_count() == 1);
    GERDOS_CHECK(registry.find_device(DeviceId{100}) == device_one);

    Resource registry_resource{
        ResourceDescription{
            ResourceId{1000},
            DeviceId{100},
            ResourceKind::COMPUTE,
            "registry-owned-compute",
        },
    };

    GERDOS_CHECK(device_one->add_resource(std::move(registry_resource)));
    GERDOS_CHECK(device_one->resource_count() == 1);

    GERDOS_CHECK(registry.remove_device(DeviceId{100}));
    GERDOS_CHECK(registry.device_count() == 0);
    GERDOS_CHECK(registry.find_device(DeviceId{100}) == nullptr);

    GERDOS_CHECK(!registry.remove_device(DeviceId{100}));

    GERDOS_CHECK(
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

    GERDOS_CHECK(device_two != nullptr);
    GERDOS_CHECK(registry.device_count() == 1);
    GERDOS_CHECK(registry.find_device(DeviceId{200}) == device_two);

    Topology topology;

    GERDOS_CHECK(topology.link_count() == 0);

    TopologyLink invalid_link{
        TopologyLinkDescription{
            TopologyLinkId{},
            TopologyEndpoint::device_endpoint(DeviceId{200}),
            TopologyEndpoint::device_endpoint(DeviceId{201}),
            TopologyLinkDirection::BIDIRECTIONAL,
        },
    };

    GERDOS_CHECK(!topology.add_link(std::move(invalid_link)));

    TopologyLink invalid_endpoint_link{
        TopologyLinkDescription{
            TopologyLinkId{3},
            TopologyEndpoint::device_endpoint(DeviceId{}),
            TopologyEndpoint::device_endpoint(DeviceId{201}),
            TopologyLinkDirection::DIRECTED,
        },
    };

    GERDOS_CHECK(
        !topology.add_link(std::move(invalid_endpoint_link)));

    TopologyLink invalid_resource_endpoint_link{
        TopologyLinkDescription{
            TopologyLinkId{4},
            TopologyEndpoint::resource_endpoint(
                DeviceId{200},
                ResourceId{}),
            TopologyEndpoint::resource_endpoint(
                DeviceId{200},
                ResourceId{2001}),
            TopologyLinkDirection::DIRECTED,
        },
    };

    GERDOS_CHECK(
        !topology.add_link(
            std::move(invalid_resource_endpoint_link)));

    TopologyLink device_link{
        TopologyLinkDescription{
            TopologyLinkId{1},
            TopologyEndpoint::device_endpoint(DeviceId{200}),
            TopologyEndpoint::device_endpoint(DeviceId{201}),
            TopologyLinkDirection::BIDIRECTIONAL,
        },
    };

    GERDOS_CHECK(topology.add_link(std::move(device_link)));
    GERDOS_CHECK(topology.link_count() == 1);

    const TopologyLink* borrowed_link =
        topology.find_link(TopologyLinkId{1});

    GERDOS_CHECK(borrowed_link != nullptr);
    GERDOS_CHECK(
        borrowed_link->description().id == TopologyLinkId{1});
    GERDOS_CHECK(
        borrowed_link->description().source.kind ==
        TopologyEndpointKind::DEVICE);
    GERDOS_CHECK(
        borrowed_link->description().source.device == DeviceId{200});
    GERDOS_CHECK(
        borrowed_link->description().destination.device == DeviceId{201});
    GERDOS_CHECK(
        borrowed_link->description().direction ==
        TopologyLinkDirection::BIDIRECTIONAL);

    std::size_t links_enumerated = 0;
    topology.for_each_link(
        [&](TopologyLink* link) {
            GERDOS_CHECK(link != nullptr);
            GERDOS_CHECK(link->description().id == TopologyLinkId{1});
            ++links_enumerated;
        });
    GERDOS_CHECK(links_enumerated == 1);

    const Topology& const_topology = topology;
    std::size_t const_links_enumerated = 0;
    const_topology.for_each_link(
        [&](const TopologyLink* link) {
            GERDOS_CHECK(link != nullptr);
            GERDOS_CHECK(link->description().id == TopologyLinkId{1});
            ++const_links_enumerated;
        });
    GERDOS_CHECK(const_links_enumerated == 1);

    TopologyLink resource_link{
        TopologyLinkDescription{
            TopologyLinkId{2},
            TopologyEndpoint::resource_endpoint(
                DeviceId{200},
                ResourceId{2000}),
            TopologyEndpoint::resource_endpoint(
                DeviceId{200},
                ResourceId{2001}),
            TopologyLinkDirection::DIRECTED,
        },
    };

    GERDOS_CHECK(topology.add_link(std::move(resource_link)));
    GERDOS_CHECK(topology.link_count() == 2);

    GERDOS_CHECK(topology.remove_link(TopologyLinkId{1}));
    GERDOS_CHECK(topology.link_count() == 1);
    GERDOS_CHECK(topology.find_link(TopologyLinkId{1}) == nullptr);
    GERDOS_CHECK(!topology.remove_link(TopologyLinkId{1}));

    TopologyLink reused_link{
        TopologyLinkDescription{
            TopologyLinkId{1},
            TopologyEndpoint::device_endpoint(DeviceId{200}),
            TopologyEndpoint::device_endpoint(DeviceId{201}),
            TopologyLinkDirection::DIRECTED,
        },
    };

    GERDOS_CHECK(!topology.add_link(std::move(reused_link)));
    GERDOS_CHECK(topology.link_count() == 1);

    // Data / residency identity and ownership tests.
    GERDOS_CHECK(
        !is_usable(DataResidencyState::UNAVAILABLE));
    GERDOS_CHECK(
        !is_usable(DataResidencyState::STALE));
    GERDOS_CHECK(
        !is_usable(DataResidencyState::TRANSFERRING));
    GERDOS_CHECK(
        is_usable(DataResidencyState::VALID));

    DataResidency residency{
        DataResidencyDescription{
            DataResidencyId{1},
            DataId{100},
            ResourceId{2000},
            "opaque-representation",
        },
    };

    GERDOS_CHECK(
        residency.state() == DataResidencyState::UNAVAILABLE);
    GERDOS_CHECK(!residency.usable());

    residency.set_state(DataResidencyState::VALID);
    GERDOS_CHECK(
        residency.state() == DataResidencyState::VALID);
    GERDOS_CHECK(residency.usable());

    residency.set_state(DataResidencyState::STALE);
    GERDOS_CHECK(!residency.usable());

    residency.set_state(DataResidencyState::TRANSFERRING);
    GERDOS_CHECK(!residency.usable());

    Data data{
        DataDescription{
            DataId{100},
            "logical-data",
        },
    };

    GERDOS_CHECK(data.description().id == DataId{100});
    GERDOS_CHECK(data.description().name == "logical-data");
    GERDOS_CHECK(data.residency_count() == 0);

    DataResidency invalid_residency_id{
        DataResidencyDescription{
            DataResidencyId{},
            DataId{100},
            ResourceId{2000},
            "representation",
        },
    };

    GERDOS_CHECK(
        !data.add_residency(std::move(invalid_residency_id)));
    GERDOS_CHECK(data.residency_count() == 0);

    DataResidency wrong_data_residency{
        DataResidencyDescription{
            DataResidencyId{2},
            DataId{101},
            ResourceId{2000},
            "representation",
        },
    };

    GERDOS_CHECK(
        !data.add_residency(std::move(wrong_data_residency)));
    GERDOS_CHECK(data.residency_count() == 0);

    DataResidency invalid_resource_residency{
        DataResidencyDescription{
            DataResidencyId{3},
            DataId{100},
            ResourceId{},
            "representation",
        },
    };

    GERDOS_CHECK(
        !data.add_residency(std::move(invalid_resource_residency)));
    GERDOS_CHECK(data.residency_count() == 0);

    DataResidency residency_a{
        DataResidencyDescription{
            DataResidencyId{10},
            DataId{100},
            ResourceId{2000},
            "representation-a",
        },
    };

    GERDOS_CHECK(data.add_residency(std::move(residency_a)));
    GERDOS_CHECK(data.residency_count() == 1);

    DataResidency* borrowed_residency =
        data.find_residency(DataResidencyId{10});

    GERDOS_CHECK(borrowed_residency != nullptr);
    GERDOS_CHECK(
        borrowed_residency->description().id ==
        DataResidencyId{10});
    GERDOS_CHECK(
        borrowed_residency->description().data ==
        DataId{100});
    GERDOS_CHECK(
        borrowed_residency->description().resource ==
        ResourceId{2000});
    GERDOS_CHECK(
        borrowed_residency->description().representation ==
        "representation-a");
    GERDOS_CHECK(
        borrowed_residency->state() ==
        DataResidencyState::UNAVAILABLE);

    borrowed_residency->set_state(DataResidencyState::VALID);
    GERDOS_CHECK(borrowed_residency->usable());

    const Data& const_data = data;
    const DataResidency* const_borrowed_residency =
        const_data.find_residency(DataResidencyId{10});

    GERDOS_CHECK(const_borrowed_residency != nullptr);
    GERDOS_CHECK(const_borrowed_residency->usable());

    DataResidency residency_b{
        DataResidencyDescription{
            DataResidencyId{11},
            DataId{100},
            ResourceId{2001},
            "representation-b",
        },
    };

    GERDOS_CHECK(data.add_residency(std::move(residency_b)));
    GERDOS_CHECK(data.residency_count() == 2);

    std::size_t residencies_enumerated = 0;
    data.for_each_residency(
        [&](DataResidency* entry) {
            GERDOS_CHECK(entry != nullptr);
            GERDOS_CHECK(entry->description().data == DataId{100});
            ++residencies_enumerated;
        });
    GERDOS_CHECK(residencies_enumerated == 2);

    std::size_t const_residencies_enumerated = 0;
    const_data.for_each_residency(
        [&](const DataResidency* entry) {
            GERDOS_CHECK(entry != nullptr);
            GERDOS_CHECK(entry->description().data == DataId{100});
            ++const_residencies_enumerated;
        });
    GERDOS_CHECK(const_residencies_enumerated == 2);

    DataResidency duplicate_residency{
        DataResidencyDescription{
            DataResidencyId{10},
            DataId{100},
            ResourceId{2002},
            "duplicate-id",
        },
    };

    GERDOS_CHECK(
        !data.add_residency(std::move(duplicate_residency)));
    GERDOS_CHECK(data.residency_count() == 2);

    GERDOS_CHECK(
        data.remove_residency(DataResidencyId{10}));
    GERDOS_CHECK(data.residency_count() == 1);
    GERDOS_CHECK(
        data.find_residency(DataResidencyId{10}) == nullptr);
    GERDOS_CHECK(
        !data.remove_residency(DataResidencyId{10}));

    DataResidency reused_residency{
        DataResidencyDescription{
            DataResidencyId{10},
            DataId{100},
            ResourceId{2003},
            "reused-id",
        },
    };

    GERDOS_CHECK(
        !data.add_residency(std::move(reused_residency)));
    GERDOS_CHECK(data.residency_count() == 1);

    DataResidency surviving_residency{
        DataResidencyDescription{
            DataResidencyId{12},
            DataId{100},
            ResourceId{2004},
            "surviving",
        },
    };

    GERDOS_CHECK(
        data.add_residency(std::move(surviving_residency)));
    GERDOS_CHECK(data.residency_count() == 2);

    DataRegistry data_registry;

    GERDOS_CHECK(data_registry.data_count() == 0);

    GERDOS_CHECK(
        data_registry.create_data(
            DataDescription{
                DataId{},
                "invalid",
            }) == nullptr);

    Data* data_a =
        data_registry.create_data(
            DataDescription{
                DataId{500},
                "data-a",
            });

    GERDOS_CHECK(data_a != nullptr);
    GERDOS_CHECK(data_registry.data_count() == 1);
    GERDOS_CHECK(data_registry.find_data(DataId{500}) == data_a);

    const DataRegistry& const_data_registry = data_registry;
    const Data* const_data_a =
        const_data_registry.find_data(DataId{500});

    GERDOS_CHECK(const_data_a == data_a);
    GERDOS_CHECK(
        const_data_a->description().name == "data-a");

    GERDOS_CHECK(
        data_registry.create_data(
            DataDescription{
                DataId{500},
                "duplicate",
            }) == nullptr);
    GERDOS_CHECK(data_registry.data_count() == 1);

    Data* data_b =
        data_registry.create_data(
            DataDescription{
                DataId{501},
                "data-b",
            });

    GERDOS_CHECK(data_b != nullptr);
    GERDOS_CHECK(data_registry.data_count() == 2);

    std::size_t data_enumerated = 0;
    data_registry.for_each_data(
        [&](Data* entry) {
            GERDOS_CHECK(entry != nullptr);
            GERDOS_CHECK(entry->description().id.valid());
            ++data_enumerated;
        });
    GERDOS_CHECK(data_enumerated == 2);

    std::size_t const_data_enumerated = 0;
    const_data_registry.for_each_data(
        [&](const Data* entry) {
            GERDOS_CHECK(entry != nullptr);
            GERDOS_CHECK(entry->description().id.valid());
            ++const_data_enumerated;
        });
    GERDOS_CHECK(const_data_enumerated == 2);

    GERDOS_CHECK(
        data_registry.remove_data(DataId{500}));
    GERDOS_CHECK(data_registry.data_count() == 1);
    GERDOS_CHECK(
        data_registry.find_data(DataId{500}) == nullptr);
    GERDOS_CHECK(
        !data_registry.remove_data(DataId{500}));

    GERDOS_CHECK(
        data_registry.create_data(
            DataDescription{
                DataId{500},
                "reused-data",
            }) == nullptr);
    GERDOS_CHECK(data_registry.data_count() == 1);

    GERDOS_CHECK(
        data_registry.find_data(DataId{501}) == data_b);

    return 0;
}
