#include "test_check.hpp"

#include <vector>

#include "gerdos/core/executor.hpp"
#include "gerdos/sim/simulated_backend.hpp"

namespace {

using namespace gerdos;

struct Fixture {
    DeviceRegistry devices;
    DataRegistry data;
    OperationRegistry operations;
    ExecutionRegistry executions;

    DataResidency* input = nullptr;
    DataResidency* output_a = nullptr;
    DataResidency* output_b = nullptr;

    Fixture() {
        auto* device = devices.create_device(
            DeviceDescription{
                DeviceId{100},
                "Synthetic Device",
            });

        (void)device->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{200},
                    DeviceId{100},
                    ResourceKind::COMPUTE,
                    "compute",
                },
            });

        (void)device->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{201},
                    DeviceId{100},
                    ResourceKind::MEMORY,
                    "memory",
                },
            });

        (void)device->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{202},
                    DeviceId{100},
                    ResourceKind::TRANSFER,
                    "transfer",
                },
            });

        (void)device->find_resource(ResourceId{200})
            ->set_availability(ResourceAvailability::AVAILABLE);

        (void)device->find_resource(ResourceId{201})
            ->set_availability(ResourceAvailability::AVAILABLE);

        (void)device->find_resource(ResourceId{202})
            ->set_availability(ResourceAvailability::AVAILABLE);

        auto* data_a = data.create_data(
            DataDescription{
                DataId{300},
                "input",
            });

        (void)data_a->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{400},
                    DataId{300},
                    ResourceRef{
                        DeviceId{100},
                        ResourceId{201},
                    },
                    "input",
                },
            });

        input = data_a->find_residency(DataResidencyId{400});
        (void)input->set_state(DataResidencyState::VALID);

        auto* data_b = data.create_data(
            DataDescription{
                DataId{301},
                "output-a",
            });

        (void)data_b->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{410},
                    DataId{301},
                    ResourceRef{
                        DeviceId{100},
                        ResourceId{201},
                    },
                    "output-a",
                },
            });

        output_a = data_b->find_residency(DataResidencyId{410});

        auto* data_c = data.create_data(
            DataDescription{
                DataId{302},
                "output-b",
            });

        (void)data_c->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{420},
                    DataId{302},
                    ResourceRef{
                        DeviceId{100},
                        ResourceId{201},
                    },
                    "output-b",
                },
            });

        output_b = data_c->find_residency(DataResidencyId{420});

        (void)operations.create_operation(
            OperationDescription{
                OperationId{700},
                {DataId{300}},
                {DataId{301}},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        1,
                    },
                },
            });

        (void)operations.create_operation(
            OperationDescription{
                OperationId{701},
                {DataId{300}},
                {DataId{301}},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            });

        (void)operations.create_operation(
            OperationDescription{
                OperationId{702},
                {DataId{300}},
                {DataId{302}},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        1,
                    },
                },
            });
    }
};

PhysicalBinding compute_binding(DataId output, DataResidencyId residency) {
    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{300},
                DataResidencyId{400},
            },
        });

    binding.data.push_back(
        DataBinding{
            DataBindingRole::OUTPUT,
            DataResidencyRef{
                output,
                residency,
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

    return binding;
}

} // namespace

int main() {
    using namespace gerdos;

    // ---------------------------------------------------------------------
    // 1. The complete lifecycle for a successful attempt
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{3};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* execution = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{800},
                OperationId{700},
            });

        GERDOS_CHECK(
            execution->bind(compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(executor.start(ExecutionId{800}));

        GERDOS_CHECK(execution->state() == ExecutionState::RUNNING);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::TRANSFERRING);
        GERDOS_CHECK(fixture.input->state() == DataResidencyState::VALID);

        std::vector<ExecutionId> completed;
        executor.advance(completed);
        GERDOS_CHECK(completed.empty());

        executor.advance(completed);
        GERDOS_CHECK(completed.empty());

        executor.advance(completed);
        GERDOS_CHECK(completed.size() == 1);
        GERDOS_CHECK(completed.front() == ExecutionId{800});

        GERDOS_CHECK(execution->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(fixture.output_a->state() == DataResidencyState::VALID);
        GERDOS_CHECK(execution->has_result());
        GERDOS_CHECK(
            execution->result()->outcome == ExecutionState::COMPLETED);

        // Completions are reported once.
        completed.clear();
        executor.advance(completed);
        GERDOS_CHECK(completed.empty());
    }

    // ---------------------------------------------------------------------
    // 2. A failed attempt leaves its output unusable and records why not
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{1};
        backend.set_failure(ExecutionId{801});

        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* execution = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{801},
                OperationId{700},
            });

        GERDOS_CHECK(
            execution->bind(compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(executor.start(ExecutionId{801}));

        std::vector<ExecutionId> completed;
        executor.advance(completed);

        GERDOS_CHECK(completed.size() == 1);
        GERDOS_CHECK(execution->state() == ExecutionState::FAILED);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(
            execution->result()->outcome == ExecutionState::FAILED);
    }

    // ---------------------------------------------------------------------
    // 3. Rejection is atomic: gate verdicts change nothing
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{1};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        // Semantically inadmissible: no resource binding satisfies the
        // COMPUTE requirement.
        auto* inadmissible = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{802},
                OperationId{700},
            });

        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{400},
                },
            });

        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{
                    DataId{301},
                    DataResidencyId{410},
                },
            });

        GERDOS_CHECK(inadmissible->bind(std::move(binding)));
        GERDOS_CHECK(!executor.start(ExecutionId{802}));

        GERDOS_CHECK(inadmissible->state() == ExecutionState::PENDING);
        GERDOS_CHECK(inadmissible->has_binding());
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(backend.in_flight_count() == 0);

        // Admission rejection: the input residency is not usable.
        GERDOS_CHECK(
            fixture.input->set_state(DataResidencyState::STALE));

        auto* unusable_input = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{803},
                OperationId{700},
            });

        GERDOS_CHECK(
            unusable_input->bind(
                compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(!executor.start(ExecutionId{803}));
        GERDOS_CHECK(unusable_input->state() == ExecutionState::PENDING);
        GERDOS_CHECK(backend.in_flight_count() == 0);

        // Restoring usability follows the state machine: a stale
        // representation becomes usable again only through an update.
        GERDOS_CHECK(
            fixture.input->set_state(DataResidencyState::TRANSFERRING));
        GERDOS_CHECK(fixture.input->set_state(DataResidencyState::VALID));

        // Unknown identities are rejected.
        GERDOS_CHECK(!executor.start(ExecutionId{999}));

        auto* unknown_operation = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{804},
                OperationId{999},
            });

        GERDOS_CHECK(
            unknown_operation->bind(
                compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(!executor.start(ExecutionId{804}));
        GERDOS_CHECK(unknown_operation->state() == ExecutionState::PENDING);
    }

    // ---------------------------------------------------------------------
    // 4. Backend rejection leaves the attempt untouched
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{1};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* execution = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{805},
                OperationId{700},
            });

        const auto binding =
            compute_binding(DataId{301}, DataResidencyId{410});
        GERDOS_CHECK(execution->bind(binding));

        // Occupy the backend slot directly, then attempt to start.
        GERDOS_CHECK(
            backend.submit(
                *fixture.operations.find_operation(OperationId{700}),
                *execution));

        GERDOS_CHECK(!executor.start(ExecutionId{805}));

        // Nothing changed: no effects, no state transition.
        GERDOS_CHECK(execution->state() == ExecutionState::PENDING);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::UNAVAILABLE);
    }

    // ---------------------------------------------------------------------
    // 5. Overlapping attempts progress concurrently
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{1};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        // A movement attempt over the transfer resource.
        auto* movement = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{810},
                OperationId{701},
            });

        PhysicalBinding movement_binding;
        movement_binding.data.push_back(
            DataBinding{
                DataBindingRole::SOURCE,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{400},
                },
            });

        movement_binding.data.push_back(
            DataBinding{
                DataBindingRole::DESTINATION,
                DataResidencyRef{
                    DataId{301},
                    DataResidencyId{410},
                },
            });

        movement_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{202},
                },
            });

        GERDOS_CHECK(movement->bind(std::move(movement_binding)));

        // A compute attempt running alongside it.
        auto* compute = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{811},
                OperationId{702},
            });

        GERDOS_CHECK(
            compute->bind(compute_binding(DataId{302}, DataResidencyId{420})));

        backend.set_work(ExecutionId{810}, 5);
        backend.set_work(ExecutionId{811}, 3);

        GERDOS_CHECK(executor.start(ExecutionId{810}));
        GERDOS_CHECK(executor.start(ExecutionId{811}));
        GERDOS_CHECK(backend.in_flight_count() == 2);

        std::vector<ExecutionId> completed;
        executor.advance(completed);
        executor.advance(completed);
        GERDOS_CHECK(completed.empty());

        // The compute attempt finishes first while the movement continues.
        executor.advance(completed);
        GERDOS_CHECK(completed.size() == 1);
        GERDOS_CHECK(completed.front() == ExecutionId{811});
        GERDOS_CHECK(compute->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(fixture.output_b->state() == DataResidencyState::VALID);
        GERDOS_CHECK(movement->state() == ExecutionState::RUNNING);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::TRANSFERRING);

        // The movement completes later.
        completed.clear();
        executor.advance(completed);
        GERDOS_CHECK(completed.empty());

        executor.advance(completed);
        GERDOS_CHECK(completed.size() == 1);
        GERDOS_CHECK(completed.front() == ExecutionId{810});
        GERDOS_CHECK(movement->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(fixture.output_a->state() == DataResidencyState::VALID);

        // The source representation remained usable throughout.
        GERDOS_CHECK(fixture.input->state() == DataResidencyState::VALID);
    }

    // ---------------------------------------------------------------------
    // 6. Completions leave measured evidence
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{3};
        MeasurementRegistry measurements;

        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend,
            &measurements);

        auto* execution = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{806},
                OperationId{700},
            });

        GERDOS_CHECK(
            execution->bind(
                compute_binding(DataId{301}, DataResidencyId{410})));

        backend.set_work(ExecutionId{806}, 3);
        GERDOS_CHECK(executor.start(ExecutionId{806}));

        std::vector<ExecutionId> completed;
        for (int step = 0; step < 3; ++step) {
            executor.advance(completed);
        }

        GERDOS_CHECK(completed.size() == 1);
        GERDOS_CHECK(measurements.count() == 1);

        const auto summary = measurements.summarize(
            ResourceRef{
                DeviceId{100},
                ResourceId{200},
            },
            MeasurementQuantity::DURATION_NS);

        GERDOS_CHECK(summary.observations == 1);
        GERDOS_CHECK(
            summary.latest_value ==
            3 * SimulatedBackend::ns_per_step);
        GERDOS_CHECK(summary.latest.valid());
    }

    return 0;
}