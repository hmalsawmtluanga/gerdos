#include <cassert>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/topology.hpp"

int main() {
    using namespace gerdos;

    // ---------------------------------------------------------------------
    // 1. Synthetic heterogeneous topology
    // ---------------------------------------------------------------------

    DeviceRegistry devices;

    Device* host = devices.create_device(
        DeviceDescription{
            DeviceId{100},
            "host",
        });

    Device* accelerator = devices.create_device(
        DeviceDescription{
            DeviceId{200},
            "accelerator",
        });

    assert(host != nullptr);
    assert(accelerator != nullptr);

    assert(host->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{100},
                DeviceId{100},
                ResourceKind::MEMORY,
                "host-ram",
            },
        }));

    assert(host->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{101},
                DeviceId{100},
                ResourceKind::STORAGE,
                "nvme",
            },
        }));

    assert(accelerator->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{200},
                DeviceId{200},
                ResourceKind::MEMORY,
                "device-memory",
            },
        }));

    assert(accelerator->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{201},
                DeviceId{200},
                ResourceKind::COMPUTE,
                "accelerator-compute",
            },
        }));

    assert(accelerator->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{202},
                DeviceId{200},
                ResourceKind::TRANSFER,
                "accelerator-transfer",
            },
        }));

    assert(host->resource_count() == 2);
    assert(accelerator->resource_count() == 3);

    // ---------------------------------------------------------------------
    // 2. Topology is separate from Resource identity
    // ---------------------------------------------------------------------

    Topology topology;

    assert(topology.add_link(
        TopologyLink{
            TopologyLinkDescription{
                TopologyLinkId{1},
                TopologyEndpoint::resource_endpoint(
                    DeviceId{100},
                    ResourceId{101}),
                TopologyEndpoint::resource_endpoint(
                    DeviceId{100},
                    ResourceId{100}),
                TopologyLinkDirection::DIRECTED,
            },
        }));

    assert(topology.add_link(
        TopologyLink{
            TopologyLinkDescription{
                TopologyLinkId{2},
                TopologyEndpoint::resource_endpoint(
                    DeviceId{100},
                    ResourceId{100}),
                TopologyEndpoint::resource_endpoint(
                    DeviceId{200},
                    ResourceId{200}),
                TopologyLinkDirection::DIRECTED,
            },
        }));

    assert(topology.add_link(
        TopologyLink{
            TopologyLinkDescription{
                TopologyLinkId{3},
                TopologyEndpoint::resource_endpoint(
                    DeviceId{200},
                    ResourceId{200}),
                TopologyEndpoint::resource_endpoint(
                    DeviceId{200},
                    ResourceId{201}),
                TopologyLinkDirection::DIRECTED,
            },
        }));

    assert(topology.link_count() == 3);

    // ---------------------------------------------------------------------
    // 3. One logical Data object with multiple physical residencies
    // ---------------------------------------------------------------------

    DataRegistry data_registry;

    Data* data = data_registry.create_data(
        DataDescription{
            DataId{500},
            "synthetic-workload-data",
        });

    assert(data != nullptr);

    assert(data->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5000},
                DataId{500},
                ResourceRef{
                    DeviceId{100},
                    ResourceId{101},
                },
                "storage",
            },
        }));

    assert(data->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5001},
                DataId{500},
                ResourceRef{
                    DeviceId{100},
                    ResourceId{100},
                },
                "host",
            },
        }));

    assert(data->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5002},
                DataId{500},
                ResourceRef{
                    DeviceId{200},
                    ResourceId{200},
                },
                "compute",
            },
        }));

    assert(data->residency_count() == 3);

    // Multiple representations of one logical Data coexist.
    assert(data->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5003},
                DataId{500},
                ResourceRef{
                    DeviceId{200},
                    ResourceId{200},
                },
                "alternate-compute",
            },
        }));

    assert(data->residency_count() == 4);

    // The residency identity remains qualified by its Device.
    assert(
        data->find_residency(DataResidencyId{5002})
            ->description()
            .resource ==
        (ResourceRef{
            DeviceId{200},
            ResourceId{200},
        }));

    assert(
        data->find_residency(DataResidencyId{5003})
            ->description()
            .resource ==
        (ResourceRef{
            DeviceId{200},
            ResourceId{200},
        }));

    assert(
        data->find_residency(DataResidencyId{5002})
            ->description()
            .representation == "compute");

    assert(
        data->find_residency(DataResidencyId{5003})
            ->description()
            .representation == "alternate-compute");

    // ---------------------------------------------------------------------
    // 4. Residency state pressure
    // ---------------------------------------------------------------------

    auto* storage_residency =
        data->find_residency(DataResidencyId{5000});

    auto* host_residency =
        data->find_residency(DataResidencyId{5001});

    auto* device_residency =
        data->find_residency(DataResidencyId{5002});

    assert(storage_residency != nullptr);
    assert(host_residency != nullptr);
    assert(device_residency != nullptr);

    assert(storage_residency->set_state(DataResidencyState::VALID));
    assert(host_residency->set_state(DataResidencyState::VALID));
    assert(device_residency->set_state(DataResidencyState::VALID));

    // A residency can be in TRANSFERRING state while another usable
    // residency still exists.
    assert(device_residency->set_state(
        DataResidencyState::TRANSFERRING));

    assert(!device_residency->usable());
    assert(host_residency->usable());

    // Another residency can become unavailable independently.
    assert(storage_residency->set_state(DataResidencyState::STALE));
    assert(storage_residency->set_state(
        DataResidencyState::UNAVAILABLE));

    assert(!storage_residency->usable());
    assert(host_residency->usable());

    // ---------------------------------------------------------------------
    // 5. Resource availability is distinct from residency state
    // ---------------------------------------------------------------------

    Resource* device_memory =
        accelerator->find_resource(ResourceId{200});

    assert(device_memory != nullptr);
    assert(!device_memory->available());

    device_memory->set_availability(ResourceAvailability::AVAILABLE);
    assert(device_memory->available());

    // A resource can become unavailable while its Data residency record
    // still exists.
    device_memory->set_availability(ResourceAvailability::DRAINING);
    assert(!device_memory->available());

    assert(
        data->find_residency(DataResidencyId{5002}) ==
        device_residency);

    assert(
        device_residency->state() ==
        DataResidencyState::TRANSFERRING);

    // ---------------------------------------------------------------------
    // 6. Declarative Operation
    // ---------------------------------------------------------------------

    OperationRegistry operations;

    Operation* operation = operations.create_operation(
        OperationDescription{
            OperationId{700},
            {DataId{500}},
            {DataId{500}},
            {},
        });

    assert(operation != nullptr);
    assert(operation->description().inputs.size() == 1);
    assert(operation->description().inputs[0] == DataId{500});
    assert(operation->description().outputs.size() == 1);
    assert(operation->description().outputs[0] == DataId{500});

    // Operation contains no selected ResourceRef or DataResidencyId.
    assert(operation->description().dependencies.empty());

    // ---------------------------------------------------------------------
    // 7. Multiple concrete attempts of the same Operation
    // ---------------------------------------------------------------------

    ExecutionRegistry executions;

    Execution* failed_attempt = executions.create_execution(
        ExecutionDescription{
            ExecutionId{800},
            OperationId{700},
        });

    Execution* retry_attempt = executions.create_execution(
        ExecutionDescription{
            ExecutionId{801},
            OperationId{700},
        });

    assert(failed_attempt != nullptr);
    assert(retry_attempt != nullptr);
    assert(executions.execution_count() == 2);

    assert(
        failed_attempt->description().operation ==
        retry_attempt->description().operation);

    assert(
        failed_attempt->description().id !=
        retry_attempt->description().id);

    assert(failed_attempt->set_state(ExecutionState::RUNNING));
    assert(failed_attempt->set_state(ExecutionState::FAILED));

    assert(retry_attempt->state() == ExecutionState::PENDING);
    assert(retry_attempt->set_state(ExecutionState::RUNNING));

    // ---------------------------------------------------------------------
    // 8. Identity boundaries remain intact
    // ---------------------------------------------------------------------

    assert(operation->description().id == OperationId{700});
    assert(data->description().id == DataId{500});

    assert(
        failed_attempt->description().operation ==
        operation->description().id);

    assert(
        retry_attempt->description().operation ==
        operation->description().id);

    assert(
        device_residency->description().data ==
        data->description().id);

    assert(
        device_residency->description().resource ==
        (ResourceRef{
            DeviceId{200},
            ResourceId{200},
        }));

    return 0;
}
