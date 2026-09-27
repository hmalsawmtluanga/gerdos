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

        (void)operations.create_operation(
            OperationDescription{
                OperationId{703},
                {DataId{300}},
                {DataId{301}, DataId{302}},
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

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());

        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());

        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(outcomes.front().execution == ExecutionId{800});

        GERDOS_CHECK(execution->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(fixture.output_a->state() == DataResidencyState::VALID);
        GERDOS_CHECK(execution->has_result());
        GERDOS_CHECK(
            execution->result()->outcome == ExecutionState::COMPLETED);

        // Completions are reported once.
        outcomes.clear();
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());
    }

    // ---------------------------------------------------------------------
    // 2. A failed attempt leaves its output unusable and records the
    //    failure outcome
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

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        GERDOS_CHECK(outcomes.size() == 1);
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
        GERDOS_CHECK(!inadmissible->admitted());

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

        // Nothing changed: no effects, no state transition, and no
        // admission evidence — the rejected begin leaves no gate verdict
        // behind.
        GERDOS_CHECK(!execution->admitted());
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

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());

        // The compute attempt finishes first while the movement continues.
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(outcomes.front().execution == ExecutionId{811});
        GERDOS_CHECK(compute->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(fixture.output_b->state() == DataResidencyState::VALID);
        GERDOS_CHECK(movement->state() == ExecutionState::RUNNING);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::TRANSFERRING);

        // The movement completes later.
        outcomes.clear();
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());

        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(outcomes.front().execution == ExecutionId{810});
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

        std::vector<AttemptStatus> outcomes;
        for (int step = 0; step < 3; ++step) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
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

    // ---------------------------------------------------------------------
    // 7. In-flight objects cannot be removed underneath an attempt
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{4};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        // A two-output attempt: both producing residencies must survive to
        // the completion effects.
        auto* dual = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{820},
                OperationId{703},
            });

        PhysicalBinding dual_binding;
        dual_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{400},
                },
            });

        dual_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{
                    DataId{301},
                    DataResidencyId{410},
                },
            });

        dual_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{
                    DataId{302},
                    DataResidencyId{420},
                },
            });

        dual_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{100},
                    ResourceId{200},
                },
            });

        GERDOS_CHECK(dual->bind(std::move(dual_binding)));
        GERDOS_CHECK(executor.start(ExecutionId{820}));

        // The mid-attempt removal route is closed at the ownership layer.
        GERDOS_CHECK(!fixture.data.remove_data(DataId{301}));

        auto* output_owner = fixture.data.find_data(DataId{301});
        GERDOS_CHECK(output_owner != nullptr);
        GERDOS_CHECK(
            !output_owner->remove_residency(DataResidencyId{410}));

        // The attempt is equally unremovable while in flight.
        GERDOS_CHECK(
            !fixture.executions.remove_execution(ExecutionId{820}));

        std::vector<AttemptStatus> outcomes;
        for (int step = 0; step < 4; ++step) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(
            outcomes.front().integrity == AttemptIntegrity::COHERENT);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::VALID);
        GERDOS_CHECK(
            fixture.output_b->state() == DataResidencyState::VALID);

        // Terminal attempts become removable again.
        GERDOS_CHECK(
            fixture.executions.remove_execution(ExecutionId{820}));
    }

    // ---------------------------------------------------------------------
    // 8. A rejected completion effect is reported, never discarded
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{2};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* execution = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{821},
                OperationId{700},
            });

        GERDOS_CHECK(
            execution->bind(
                compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(executor.start(ExecutionId{821}));

        // Interference: the producing residency's update state is resolved
        // externally mid-attempt, so the finishing effects cannot apply.
        GERDOS_CHECK(
            fixture.output_a->set_state(DataResidencyState::VALID));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());

        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(outcomes.front().execution == ExecutionId{821});

        // The disagreement is part of the attempt's history.
        GERDOS_CHECK(
            outcomes.front().integrity ==
            AttemptIntegrity::EFFECTS_REJECTED);
        GERDOS_CHECK(execution->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(
            execution->result()->integrity ==
            AttemptIntegrity::EFFECTS_REJECTED);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::VALID);
    }

    // ---------------------------------------------------------------------
    // 9. Cancellation applies the documented effects mapping
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{4};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        // An in-flight attempt receives failed-or-cancelled finishing
        // effects before its result is recorded.
        auto* in_flight = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{822},
                OperationId{700},
            });

        GERDOS_CHECK(
            in_flight->bind(
                compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(executor.start(ExecutionId{822}));
        GERDOS_CHECK(executor.cancel(ExecutionId{822}));

        GERDOS_CHECK(in_flight->state() == ExecutionState::CANCELLED);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(in_flight->has_result());
        GERDOS_CHECK(
            in_flight->result()->outcome == ExecutionState::CANCELLED);
        GERDOS_CHECK(
            in_flight->result()->integrity == AttemptIntegrity::COHERENT);

        // A PENDING attempt is cancelled without effects.
        auto* pending = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{823},
                OperationId{700},
            });

        GERDOS_CHECK(
            pending->bind(
                compute_binding(DataId{302}, DataResidencyId{420})));

        GERDOS_CHECK(executor.cancel(ExecutionId{823}));
        GERDOS_CHECK(pending->state() == ExecutionState::CANCELLED);
        GERDOS_CHECK(
            fixture.output_b->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(pending->has_result());

        // Unknown and already-terminal attempts cannot be cancelled.
        GERDOS_CHECK(!executor.cancel(ExecutionId{999}));
        GERDOS_CHECK(!executor.cancel(ExecutionId{823}));

        // The backend's late completion of a cancelled attempt is discarded.
        std::vector<AttemptStatus> outcomes;
        for (int step = 0; step < 4; ++step) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.empty());
    }

    // ---------------------------------------------------------------------
    // 10. Two attempts cannot update the same representation at once
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{4};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* first = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{830},
                OperationId{700},
            });

        auto* second = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{831},
                OperationId{700},
            });

        GERDOS_CHECK(
            first->bind(compute_binding(DataId{301}, DataResidencyId{410})));

        GERDOS_CHECK(
            second->bind(compute_binding(DataId{301}, DataResidencyId{410})));

        GERDOS_CHECK(executor.start(ExecutionId{830}));
        GERDOS_CHECK(
            fixture.output_a->update_owner() == ExecutionId{830});

        // The second attempt is refused atomically: nothing about it
        // changes and no work is submitted for it.
        GERDOS_CHECK(!executor.start(ExecutionId{831}));
        GERDOS_CHECK(second->state() == ExecutionState::PENDING);
        GERDOS_CHECK(!second->has_result());
        GERDOS_CHECK(backend.in_flight_count() == 1);
        GERDOS_CHECK(
            fixture.output_a->update_owner() == ExecutionId{830});

        // Once the owner completes and releases the claim, the rewrite
        // may proceed.
        std::vector<AttemptStatus> outcomes;
        for (int step = 0; step < 4; ++step) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(
            outcomes.front().integrity == AttemptIntegrity::COHERENT);
        GERDOS_CHECK(!fixture.output_a->update_owner().valid());

        GERDOS_CHECK(executor.start(ExecutionId{831}));
        GERDOS_CHECK(
            fixture.output_a->update_owner() == ExecutionId{831});

        outcomes.clear();
        for (int step = 0; step < 4; ++step) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(outcomes.front().execution == ExecutionId{831});
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::VALID);
    }

    // ---------------------------------------------------------------------
    // 11. Evidence carries outcome and symmetric contention conditions
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{1};
        MeasurementRegistry measurements;

        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend,
            &measurements);

        backend.set_failure(ExecutionId{840});
        backend.set_work(ExecutionId{840}, 2);
        backend.set_work(ExecutionId{841}, 2);

        auto* first = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{840},
                OperationId{700},
            });

        auto* second = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{841},
                OperationId{702},
            });

        GERDOS_CHECK(
            first->bind(compute_binding(DataId{301}, DataResidencyId{410})));

        GERDOS_CHECK(
            second->bind(compute_binding(DataId{302}, DataResidencyId{420})));

        GERDOS_CHECK(executor.start(ExecutionId{840}));
        GERDOS_CHECK(executor.start(ExecutionId{841}));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());

        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.size() == 2);
        GERDOS_CHECK(outcomes.front().evidence);
        GERDOS_CHECK(outcomes.back().evidence);

        GERDOS_CHECK(measurements.count() == 2);

        // Co-completing attempts observe identical contention conditions:
        // one peer each, regardless of processing order.
        const auto summary = measurements.summarize(
            ResourceRef{
                DeviceId{100},
                ResourceId{200},
            },
            MeasurementQuantity::DURATION_NS);

        GERDOS_CHECK(summary.observations == 2);
        GERDOS_CHECK(summary.succeeded_observations == 1);
        GERDOS_CHECK(
            summary.total_value ==
            2 * 2 * SimulatedBackend::ns_per_step);

        measurements.for_each(
            ResourceRef{
                DeviceId{100},
                ResourceId{200},
            },
            MeasurementQuantity::DURATION_NS,
            [&](const MeasurementRecord& record) {
                GERDOS_CHECK(
                    record.observation.conditions.concurrent_attempts == 1);
                GERDOS_CHECK(record.observation.value == 2 * SimulatedBackend::ns_per_step);
            });
    }

    // ---------------------------------------------------------------------
    // 12. Evidence survives removal of the attempted Operation
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{2};
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
                ExecutionId{842},
                OperationId{700},
            });

        GERDOS_CHECK(
            execution->bind(
                compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(executor.start(ExecutionId{842}));

        // The attempted Operation disappears while the work is in flight.
        GERDOS_CHECK(fixture.operations.remove_operation(OperationId{700}));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.empty());

        executor.advance(outcomes);
        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(outcomes.front().evidence);

        // The evidence is still recorded, with the attempted identity.
        GERDOS_CHECK(measurements.count() == 1);

        std::size_t seen = 0;
        measurements.for_each(
            ResourceRef{
                DeviceId{100},
                ResourceId{200},
            },
            MeasurementQuantity::DURATION_NS,
            [&](const MeasurementRecord& record) {
                ++seen;
                GERDOS_CHECK(
                    record.observation.operation == OperationId{700});
                GERDOS_CHECK(
                    record.observation.attempt == ExecutionId{842});
            });

        GERDOS_CHECK(seen == 1);
    }

    // ---------------------------------------------------------------------
    // 13. A failed attempt is retried by a later Execution
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{1};
        backend.set_failure(ExecutionId{850});

        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* first = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{850},
                OperationId{700},
            });

        GERDOS_CHECK(
            first->bind(compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(executor.start(ExecutionId{850}));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(first->state() == ExecutionState::FAILED);
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::UNAVAILABLE);

        // The failed attempt released its claim, so a later attempt may
        // retry the same work on the same representation.
        backend.set_work(ExecutionId{851}, 1);

        auto* retry = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{851},
                OperationId{700},
            });

        GERDOS_CHECK(
            retry->bind(compute_binding(DataId{301}, DataResidencyId{410})));
        GERDOS_CHECK(executor.start(ExecutionId{851}));

        outcomes.clear();
        executor.advance(outcomes);

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(outcomes.front().execution == ExecutionId{851});
        GERDOS_CHECK(outcomes.front().integrity == AttemptIntegrity::COHERENT);
        GERDOS_CHECK(retry->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(fixture.output_a->state() == DataResidencyState::VALID);

        // The failed attempt keeps its own history.
        GERDOS_CHECK(first->has_result());
        GERDOS_CHECK(first->result()->outcome == ExecutionState::FAILED);
        GERDOS_CHECK(first->binding() != nullptr);
    }

    // ---------------------------------------------------------------------
    // 14. A hostile backend cannot leave evidence or claims behind
    // ---------------------------------------------------------------------

    {
        Fixture fixture;

        // A backend that mutates runtime state during submission: it claims
        // the producing residency on behalf of another live attempt.
        auto* intruder = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{860},
                OperationId{700},
            });

        class HostileBackend final : public ExecutionBackend {
        public:
            HostileBackend(DataResidency* target, ExecutionId intruder)
                : target_(target), intruder_(intruder) {}

            [[nodiscard]] bool submit(
                const Operation&,
                const Execution&) override {
                (void)target_->set_state(DataResidencyState::TRANSFERRING);
                target_->set_update_owner(intruder_);
                return true;
            }

            void poll(std::vector<BackendCompletion>& completed) override {
                if (pending_) {
                    completed.push_back(
                        BackendCompletion{
                            pending_.value(),
                            true,
                            1'000'000,
                        });

                    pending_.reset();
                }
            }

            void complete(ExecutionId id) {
                pending_ = id;
            }

        private:
            DataResidency* target_;
            ExecutionId intruder_;
            std::optional<ExecutionId> pending_;
        };

        HostileBackend backend(
            fixture.output_a,
            intruder->description().id);

        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* victim = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{861},
                OperationId{700},
            });

        GERDOS_CHECK(
            victim->bind(compute_binding(DataId{301}, DataResidencyId{410})));

        // The attempt begins (submission succeeded) but its start effects
        // are rejected by the mutation: it runs unclaimed and the
        // incoherence is reported with its completion, never silent.
        GERDOS_CHECK(executor.start(ExecutionId{861}));
        GERDOS_CHECK(victim->state() == ExecutionState::RUNNING);

        backend.complete(ExecutionId{861});

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(
            outcomes.front().integrity ==
            AttemptIntegrity::EFFECTS_REJECTED);
        GERDOS_CHECK(victim->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(
            victim->result()->integrity ==
            AttemptIntegrity::EFFECTS_REJECTED);
    }

    // ---------------------------------------------------------------------
    // 15. Cancellation releases claims from any state
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        SimulatedBackend backend{4};
        Executor executor(
            fixture.executions,
            fixture.operations,
            fixture.devices,
            fixture.data,
            backend);

        auto* execution = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{862},
                OperationId{700},
            });

        GERDOS_CHECK(
            execution->bind(
                compute_binding(DataId{301}, DataResidencyId{410})));

        // An attempt holding claims while still PENDING — the shape a
        // failed begin leaves behind — is still finalizable.
        ExecutionEffects effects(fixture.data, fixture.executions);
        GERDOS_CHECK(effects.start(*execution));
        GERDOS_CHECK(
            fixture.output_a->update_owner() == ExecutionId{862});

        GERDOS_CHECK(executor.cancel(ExecutionId{862}));
        GERDOS_CHECK(!fixture.output_a->update_owner().valid());
        GERDOS_CHECK(
            fixture.output_a->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(execution->has_result());
        GERDOS_CHECK(
            execution->result()->integrity == AttemptIntegrity::COHERENT);
    }

    return 0;
}