#include "test_check.hpp"

#include <vector>

#include "gerdos/core/executor.hpp"
#include "gerdos/sim/simulated_backend.hpp"

int main() {
    using namespace gerdos;

    // ---------------------------------------------------------------------
    // 1. A synthetic heterogeneous machine
    //
    //     host (100)                 accelerator (200)
    //      +-- storage  100 nvme      +-- memory  200 device-memory
    //      +-- memory   101 ram       +-- compute 201 compute
    //                                 +-- transfer 202 copy-engine
    // ---------------------------------------------------------------------

    DeviceRegistry devices;
    DataRegistry data;
    OperationRegistry operations;
    ExecutionRegistry executions;
    SimulatedBackend backend{1};

    auto* host = devices.create_device(
        DeviceDescription{
            DeviceId{100},
            "host",
        });

    auto* accelerator = devices.create_device(
        DeviceDescription{
            DeviceId{200},
            "accelerator",
        });

    GERDOS_CHECK(host != nullptr);
    GERDOS_CHECK(accelerator != nullptr);

    GERDOS_CHECK(host->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{100},
                DeviceId{100},
                ResourceKind::STORAGE,
                "nvme",
            },
        }));

    GERDOS_CHECK(host->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{101},
                DeviceId{100},
                ResourceKind::MEMORY,
                "ram",
            },
        }));

    GERDOS_CHECK(accelerator->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{200},
                DeviceId{200},
                ResourceKind::MEMORY,
                "device-memory",
            },
        }));

    GERDOS_CHECK(accelerator->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{201},
                DeviceId{200},
                ResourceKind::COMPUTE,
                "compute",
            },
        }));

    GERDOS_CHECK(accelerator->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{202},
                DeviceId{200},
                ResourceKind::TRANSFER,
                "copy-engine",
            },
        }));

    host->find_resource(ResourceId{100})
        ->set_availability(ResourceAvailability::AVAILABLE);
    host->find_resource(ResourceId{101})
        ->set_availability(ResourceAvailability::AVAILABLE);

    accelerator->find_resource(ResourceId{200})
        ->set_availability(ResourceAvailability::AVAILABLE);
    accelerator->find_resource(ResourceId{201})
        ->set_availability(ResourceAvailability::AVAILABLE);
    accelerator->find_resource(ResourceId{202})
        ->set_availability(ResourceAvailability::AVAILABLE);

    // ---------------------------------------------------------------------
    // 2. Weights on disk, an activation already on the accelerator
    // ---------------------------------------------------------------------

    auto* weights = data.create_data(
        DataDescription{
            DataId{500},
            "weights",
        });

    GERDOS_CHECK(weights->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5000},
                DataId{500},
                ResourceRef{
                    DeviceId{100},
                    ResourceId{100},
                },
                "disk",
            },
        }));

    GERDOS_CHECK(weights->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5001},
                DataId{500},
                ResourceRef{
                    DeviceId{100},
                    ResourceId{101},
                },
                "staged",
            },
        }));

    GERDOS_CHECK(weights->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5002},
                DataId{500},
                ResourceRef{
                    DeviceId{200},
                    ResourceId{200},
                },
                "resident",
            },
        }));

    auto* disk = weights->find_residency(DataResidencyId{5000});
    auto* staged = weights->find_residency(DataResidencyId{5001});
    auto* resident = weights->find_residency(DataResidencyId{5002});

    GERDOS_CHECK(disk->set_state(DataResidencyState::VALID));

    auto* activation = data.create_data(
        DataDescription{
            DataId{501},
            "activation",
        });

    GERDOS_CHECK(activation->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5100},
                DataId{501},
                ResourceRef{
                    DeviceId{200},
                    ResourceId{200},
                },
                "device",
            },
        }));

    auto* activation_device =
        activation->find_residency(DataResidencyId{5100});
    GERDOS_CHECK(activation_device->set_state(DataResidencyState::VALID));

    auto* output = data.create_data(
        DataDescription{
            DataId{502},
            "output",
        });

    GERDOS_CHECK(output->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{5200},
                DataId{502},
                ResourceRef{
                    DeviceId{200},
                    ResourceId{200},
                },
                "device",
            },
        }));

    auto* output_device = output->find_residency(DataResidencyId{5200});

    // ---------------------------------------------------------------------
    // 3. Work: two movement steps and one compute step
    // ---------------------------------------------------------------------

    GERDOS_CHECK(
        operations.create_operation(
            OperationDescription{
                OperationId{800},
                {DataId{500}},
                {DataId{500}},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            }) != nullptr);

    GERDOS_CHECK(
        operations.create_operation(
            OperationDescription{
                OperationId{801},
                {DataId{500}},
                {DataId{500}},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            }) != nullptr);

    GERDOS_CHECK(
        operations.create_operation(
            OperationDescription{
                OperationId{802},
                {DataId{501}},
                {DataId{502}},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        1,
                    },
                },
            }) != nullptr);

    Executor executor(
        executions,
        operations,
        devices,
        data,
        backend);

    // Movement one: nvme -> ram.
    auto* stage = executions.create_execution(
        ExecutionDescription{
            ExecutionId{900},
            OperationId{800},
        });

    PhysicalBinding stage_binding;
    stage_binding.data.push_back(
        DataBinding{
            DataBindingRole::SOURCE,
            DataResidencyRef{
                DataId{500},
                DataResidencyId{5000},
            },
        });

    stage_binding.data.push_back(
        DataBinding{
            DataBindingRole::DESTINATION,
            DataResidencyRef{
                DataId{500},
                DataResidencyId{5001},
            },
        });

    stage_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::TRANSFER,
            ResourceRef{
                DeviceId{100},
                ResourceId{101},
            },
        });

    GERDOS_CHECK(stage->bind(std::move(stage_binding)));

    // Movement two: ram -> device memory.
    auto* install = executions.create_execution(
        ExecutionDescription{
            ExecutionId{901},
            OperationId{801},
        });

    PhysicalBinding install_binding;
    install_binding.data.push_back(
        DataBinding{
            DataBindingRole::SOURCE,
            DataResidencyRef{
                DataId{500},
                DataResidencyId{5001},
            },
        });

    install_binding.data.push_back(
        DataBinding{
            DataBindingRole::DESTINATION,
            DataResidencyRef{
                DataId{500},
                DataResidencyId{5002},
            },
        });

    install_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::TRANSFER,
            ResourceRef{
                DeviceId{200},
                ResourceId{202},
            },
        });

    GERDOS_CHECK(install->bind(std::move(install_binding)));

    // Compute: activation -> output on the accelerator.
    auto* compute = executions.create_execution(
        ExecutionDescription{
            ExecutionId{902},
            OperationId{802},
        });

    PhysicalBinding compute_binding;
    compute_binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{501},
                DataResidencyId{5100},
            },
        });

    compute_binding.data.push_back(
        DataBinding{
            DataBindingRole::OUTPUT,
            DataResidencyRef{
                DataId{502},
                DataResidencyId{5200},
            },
        });

    compute_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            ResourceRef{
                DeviceId{200},
                ResourceId{201},
            },
        });

    GERDOS_CHECK(compute->bind(std::move(compute_binding)));

    backend.set_work(ExecutionId{900}, 4);
    backend.set_work(ExecutionId{901}, 3);
    backend.set_work(ExecutionId{902}, 6);

    // ---------------------------------------------------------------------
    // 4. The dependency is enforced by the state model: the second movement
    //    cannot consume a representation that does not exist yet
    // ---------------------------------------------------------------------

    GERDOS_CHECK(!executor.start(ExecutionId{901}));
    GERDOS_CHECK(install->state() == ExecutionState::PENDING);

    // ---------------------------------------------------------------------
    // 5. The run: staging and compute overlap
    // ---------------------------------------------------------------------

    std::vector<AttemptStatus> trace;

    GERDOS_CHECK(executor.start(ExecutionId{900}));
    GERDOS_CHECK(executor.start(ExecutionId{902}));

    GERDOS_CHECK(stage->state() == ExecutionState::RUNNING);
    GERDOS_CHECK(compute->state() == ExecutionState::RUNNING);
    GERDOS_CHECK(staged->state() == DataResidencyState::TRANSFERRING);
    GERDOS_CHECK(output_device->state() == DataResidencyState::TRANSFERRING);

    // Steps 1-4: staging progresses while compute runs alongside it.
    for (int step = 0; step < 3; ++step) {
        executor.advance(trace);
        GERDOS_CHECK(trace.empty());
    }

    executor.advance(trace);

    // Step 4: staging completed; the source copy remains usable.
    GERDOS_CHECK(trace.size() == 1);
    GERDOS_CHECK(trace.front().execution == ExecutionId{900});
    GERDOS_CHECK(stage->state() == ExecutionState::COMPLETED);
    GERDOS_CHECK(staged->state() == DataResidencyState::VALID);
    GERDOS_CHECK(disk->state() == DataResidencyState::VALID);
    GERDOS_CHECK(compute->state() == ExecutionState::RUNNING);

    // The staged representation now exists, so installation can begin.
    GERDOS_CHECK(executor.start(ExecutionId{901}));
    GERDOS_CHECK(install->state() == ExecutionState::RUNNING);
    GERDOS_CHECK(resident->state() == DataResidencyState::TRANSFERRING);

    // Steps 5-6: compute finishes while installation is still moving.
    trace.clear();
    executor.advance(trace);
    GERDOS_CHECK(trace.empty());

    executor.advance(trace);
    GERDOS_CHECK(trace.size() == 1);
    GERDOS_CHECK(trace.front().execution == ExecutionId{902});
    GERDOS_CHECK(compute->state() == ExecutionState::COMPLETED);
    GERDOS_CHECK(output_device->state() == DataResidencyState::VALID);
    GERDOS_CHECK(install->state() == ExecutionState::RUNNING);
    GERDOS_CHECK(resident->state() == DataResidencyState::TRANSFERRING);

    // Step 7: installation completes; the weights are now resident.
    trace.clear();
    executor.advance(trace);
    GERDOS_CHECK(trace.size() == 1);
    GERDOS_CHECK(trace.front().execution == ExecutionId{901});
    GERDOS_CHECK(install->state() == ExecutionState::COMPLETED);
    GERDOS_CHECK(resident->state() == DataResidencyState::VALID);

    // The complete schedule, in deterministic order.
    GERDOS_CHECK(backend.elapsed_steps() == 7);

    // Logical identity survived the whole run: one Data object with three
    // representations, two of them created by this run.
    GERDOS_CHECK(weights->residency_count() == 3);
    GERDOS_CHECK(disk->usable());
    GERDOS_CHECK(staged->usable());
    GERDOS_CHECK(resident->usable());

    return 0;
}