#include <cassert>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/physical_binding.hpp"
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
    // 7. Physical binding is orthogonal to Execution lifecycle
    // ---------------------------------------------------------------------

    ExecutionRegistry executions;

    Execution* unbound_cancelled = executions.create_execution(
        ExecutionDescription{
            ExecutionId{802},
            OperationId{700},
        });

    assert(unbound_cancelled != nullptr);
    assert(unbound_cancelled->state() == ExecutionState::PENDING);
    assert(!unbound_cancelled->has_binding());
    assert(unbound_cancelled->binding() == nullptr);

    // A PENDING execution cannot enter RUNNING without a physical binding.
    assert(!unbound_cancelled->set_state(ExecutionState::RUNNING));

    // Cancellation does not require a physical binding.
    assert(unbound_cancelled->set_state(ExecutionState::CANCELLED));
    assert(!unbound_cancelled->has_binding());

    // ---------------------------------------------------------------------
    // 8. Binding is established once and becomes immutable
    // ---------------------------------------------------------------------

    PhysicalBinding binding_a{
        {
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{500},
                    DataResidencyId{5001},
                },
            },
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{
                    DataId{500},
                    DataResidencyId{5002},
                },
            },
        },
        {
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{200},
                    ResourceId{201},
                },
            },
        },
    };

    Execution* failed_attempt = executions.create_execution(
        ExecutionDescription{
            ExecutionId{800},
            OperationId{700},
        });

    assert(failed_attempt != nullptr);
    assert(!failed_attempt->has_binding());

    assert(failed_attempt->bind(std::move(binding_a)));
    assert(failed_attempt->has_binding());
    assert(failed_attempt->binding() != nullptr);
    assert(failed_attempt->binding()->data.size() == 2);
    assert(failed_attempt->binding()->resources.size() == 1);

    assert(
        failed_attempt->binding()->data[0].role ==
        DataBindingRole::INPUT);

    assert(
        failed_attempt->binding()->data[0].residency ==
        (DataResidencyRef{
            DataId{500},
            DataResidencyId{5001},
        }));

    assert(
        failed_attempt->binding()->data[1].role ==
        DataBindingRole::OUTPUT);

    assert(
        failed_attempt->binding()->data[1].residency ==
        (DataResidencyRef{
            DataId{500},
            DataResidencyId{5002},
        }));

    assert(
        failed_attempt->binding()->resources[0].role ==
        ResourceBindingRole::COMPUTE);

    assert(
        failed_attempt->binding()->resources[0].resource ==
        (ResourceRef{
            DeviceId{200},
            ResourceId{201},
        }));

    // A physical binding may be established only once.
    PhysicalBinding replacement_binding{
        {
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{500},
                    DataResidencyId{5000},
                },
            },
        },
        {
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{200},
                    ResourceId{202},
                },
            },
        },
    };

    assert(!failed_attempt->bind(std::move(replacement_binding)));

    // The original binding remains unchanged.
    assert(failed_attempt->binding()->data.size() == 2);
    assert(failed_attempt->binding()->resources.size() == 1);
    assert(
        failed_attempt->binding()->resources[0].role ==
        ResourceBindingRole::COMPUTE);

    // A bound PENDING execution may enter RUNNING.
    assert(failed_attempt->set_state(ExecutionState::RUNNING));

    // Binding remains immutable while RUNNING.
    PhysicalBinding running_replacement{
        {},
        {
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{200},
                    ResourceId{202},
                },
            },
        },
    };

    assert(!failed_attempt->bind(std::move(running_replacement)));
    assert(
        failed_attempt->binding()->resources[0].role ==
        ResourceBindingRole::COMPUTE);

    // A failed attempt retains its historical physical binding.
    assert(failed_attempt->set_state(ExecutionState::FAILED));
    assert(failed_attempt->has_binding());

    PhysicalBinding failed_replacement{
        {},
        {
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{200},
                    ResourceId{201},
                },
            },
        },
    };

    assert(!failed_attempt->bind(std::move(failed_replacement)));
    assert(failed_attempt->has_binding());

    // ---------------------------------------------------------------------
    // 9. Retry is a new Execution with an independent physical binding
    // ---------------------------------------------------------------------

    PhysicalBinding binding_b{
        {
            DataBinding{
                DataBindingRole::SOURCE,
                DataResidencyRef{
                    DataId{500},
                    DataResidencyId{5001},
                },
            },
            DataBinding{
                DataBindingRole::DESTINATION,
                DataResidencyRef{
                    DataId{500},
                    DataResidencyId{5002},
                },
            },
        },
        {
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{200},
                    ResourceId{202},
                },
            },
        },
    };

    Execution* retry_attempt = executions.create_execution(
        ExecutionDescription{
            ExecutionId{801},
            OperationId{700},
        });

    assert(retry_attempt != nullptr);
    assert(retry_attempt->state() == ExecutionState::PENDING);
    assert(!retry_attempt->has_binding());

    assert(retry_attempt->bind(std::move(binding_b)));
    assert(retry_attempt->has_binding());
    assert(retry_attempt->set_state(ExecutionState::RUNNING));

    assert(
        failed_attempt->description().operation ==
        retry_attempt->description().operation);

    assert(
        failed_attempt->description().id !=
        retry_attempt->description().id);

    // The retry has a different physical realization.
    assert(
        retry_attempt->binding()->data[0].role ==
        DataBindingRole::SOURCE);

    assert(
        retry_attempt->binding()->resources[0].role ==
        ResourceBindingRole::TRANSFER);

    assert(
        retry_attempt->binding()->resources[0].resource ==
        (ResourceRef{
            DeviceId{200},
            ResourceId{202},
        }));

    // The failed attempt retains its original compute binding.
    assert(
        failed_attempt->binding()->resources[0].role ==
        ResourceBindingRole::COMPUTE);

    assert(
        failed_attempt->binding()->resources[0].resource ==
        (ResourceRef{
            DeviceId{200},
            ResourceId{201},
        }));

    // ---------------------------------------------------------------------
    // 10. Terminal executions cannot acquire a physical binding
    // ---------------------------------------------------------------------

    Execution* completed_attempt = executions.create_execution(
        ExecutionDescription{
            ExecutionId{803},
            OperationId{700},
        });

    assert(completed_attempt != nullptr);

    assert(completed_attempt->bind(PhysicalBinding{}));
    assert(completed_attempt->set_state(ExecutionState::RUNNING));
    assert(completed_attempt->set_state(ExecutionState::COMPLETED));

    assert(
        !completed_attempt->bind(
            PhysicalBinding{
                {},
                {
                    ResourceBinding{
                        ResourceBindingRole::COMPUTE,
                        ResourceRef{
                            DeviceId{200},
                            ResourceId{201},
                        },
                    },
                },
            }));

    Execution* cancelled_attempt = executions.create_execution(
        ExecutionDescription{
            ExecutionId{804},
            OperationId{700},
        });

    assert(cancelled_attempt != nullptr);
    assert(cancelled_attempt->bind(PhysicalBinding{}));
    assert(cancelled_attempt->set_state(ExecutionState::CANCELLED));

    assert(
        !cancelled_attempt->bind(
            PhysicalBinding{
                {},
                {
                    ResourceBinding{
                        ResourceBindingRole::TRANSFER,
                        ResourceRef{
                            DeviceId{200},
                            ResourceId{202},
                        },
                    },
                },
            }));

    // ---------------------------------------------------------------------
    // 11. Identity boundaries remain intact
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

    // Binding references do not rewrite logical or runtime identities.
    assert(
        failed_attempt->binding()->data[0].residency ==
        (DataResidencyRef{
            DataId{500},
            DataResidencyId{5001},
        }));

    assert(
        failed_attempt->binding()->resources[0].resource ==
        (ResourceRef{
            DeviceId{200},
            ResourceId{201},
        }));

    return 0;
}
