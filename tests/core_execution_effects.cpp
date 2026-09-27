#include "test_check.hpp"

#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/execution_effects.hpp"

namespace {

using namespace gerdos;

struct Fixture {
    DeviceRegistry devices;
    DataRegistry data_registry;
    ExecutionRegistry executions;
    ExecutionAdmissionValidator admission{devices, data_registry};
    Data* data = nullptr;
    DataResidency* input = nullptr;
    DataResidency* output = nullptr;

    Fixture() {
        auto* device = devices.create_device(
            DeviceDescription{
                DeviceId{100},
                "Fixture Device",
            });

        (void)device->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{200},
                    DeviceId{100},
                    ResourceKind::MEMORY,
                    "input-memory",
                },
            });

        (void)device->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{201},
                    DeviceId{100},
                    ResourceKind::MEMORY,
                    "output-memory",
                },
            });

        device->find_resource(ResourceId{200})
            ->set_availability(ResourceAvailability::AVAILABLE);
        device->find_resource(ResourceId{201})
            ->set_availability(ResourceAvailability::AVAILABLE);

        data = data_registry.create_data(
            DataDescription{
                DataId{300},
                "Data",
            });

        (void)data->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{400},
                    DataId{300},
                    ResourceRef{
                        DeviceId{100},
                        ResourceId{200},
                    },
                    "input",
                },
            });

        (void)data->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{401},
                    DataId{300},
                    ResourceRef{
                        DeviceId{100},
                        ResourceId{201},
                    },
                    "output",
                },
            });

        input = data->find_residency(DataResidencyId{400});
        output = data->find_residency(DataResidencyId{401});

        (void)input->set_state(DataResidencyState::VALID);
    }

    // Admission is mechanism: attempts reach RUNNING only through it.
    bool start_attempt(Execution& execution) {
        return admission.admit(execution).has_value() &&
               execution.set_state(ExecutionState::RUNNING);
    }
};

Execution bound_execution(ExecutionId id) {
    Execution execution{
        ExecutionDescription{
            id,
            OperationId{600},
        }};

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
                DataId{300},
                DataResidencyId{401},
            },
        });

    (void)execution.bind(std::move(binding));

    return execution;
}

} // namespace

int main() {
    using namespace gerdos;

    // ---------------------------------------------------------------------
    // 1. Starting marks producing residencies update-in-progress only
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);
        auto execution = bound_execution(ExecutionId{500});

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::TRANSFERRING);

        // Consuming residencies are never modified.
        GERDOS_CHECK(
            fixture.input->state() == DataResidencyState::VALID);

        // Starting effects are idempotent.
        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::TRANSFERRING);
    }

    // ---------------------------------------------------------------------
    // 2. Completed attempts make producing residencies usable
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);
        auto execution = bound_execution(ExecutionId{501});

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(fixture.start_attempt(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::COMPLETED));
        GERDOS_CHECK(effects.finish(execution));

        GERDOS_CHECK(fixture.output->state() == DataResidencyState::VALID);
        GERDOS_CHECK(fixture.input->state() == DataResidencyState::VALID);

        // Finishing effects are single-shot.
        GERDOS_CHECK(!effects.finish(execution));
    }

    // ---------------------------------------------------------------------
    // 3. Failed and cancelled attempts leave producing residencies
    //    existing but unusable
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);
        auto execution = bound_execution(ExecutionId{502});

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(fixture.start_attempt(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::FAILED));
        GERDOS_CHECK(effects.finish(execution));

        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(fixture.input->state() == DataResidencyState::VALID);
    }

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);
        auto execution = bound_execution(ExecutionId{503});

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(fixture.start_attempt(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::CANCELLED));
        GERDOS_CHECK(effects.finish(execution));

        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::UNAVAILABLE);
    }

    // ---------------------------------------------------------------------
    // 4. Finishing effects require the update to have started
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);
        auto execution = bound_execution(ExecutionId{504});

        // The rewrite scenario: an existing usable representation would be
        // destroyed by an unwarranted failure effect.
        GERDOS_CHECK(
            fixture.output->set_state(DataResidencyState::VALID));

        GERDOS_CHECK(fixture.start_attempt(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::COMPLETED));
        GERDOS_CHECK(!effects.finish(execution));

        // Nothing changed: the never-started representation keeps its state.
        GERDOS_CHECK(fixture.output->state() == DataResidencyState::VALID);
    }

    // ---------------------------------------------------------------------
    // 5. Lifecycle phase requirements are enforced
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);
        auto execution = bound_execution(ExecutionId{505});

        GERDOS_CHECK(fixture.start_attempt(execution));
        GERDOS_CHECK(effects.start(execution));

        // Finishing effects require a terminal attempt.
        GERDOS_CHECK(!effects.finish(execution));
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::TRANSFERRING);

        // Starting effects are rejected once the attempt is terminal.
        GERDOS_CHECK(execution.set_state(ExecutionState::COMPLETED));
        GERDOS_CHECK(!effects.start(execution));
        GERDOS_CHECK(effects.finish(execution));
    }

    // ---------------------------------------------------------------------
    // 6. Application is transactional
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);

        // A second producing residency that cannot be resolved rejects the
        // whole call: the resolvable one stays unchanged.
        Execution execution{
            ExecutionDescription{
                ExecutionId{506},
                OperationId{600},
            }};

        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{401},
                },
            });

        binding.data.push_back(
            DataBinding{
                DataBindingRole::DESTINATION,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{999},
                },
            });

        (void)execution.bind(std::move(binding));

        GERDOS_CHECK(!effects.start(execution));
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::UNAVAILABLE);
    }

    // ---------------------------------------------------------------------
    // 7. Role domain violations are rejected
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);

        Execution execution{
            ExecutionDescription{
                ExecutionId{507},
                OperationId{600},
            }};

        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                static_cast<DataBindingRole>(255),
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{401},
                },
            });

        (void)execution.bind(std::move(binding));

        GERDOS_CHECK(!effects.start(execution));
        GERDOS_CHECK(!effects.finish(execution));
    }

    // ---------------------------------------------------------------------
    // 8. Attempts without producing bindings have no effects
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);

        Execution execution{
            ExecutionDescription{
                ExecutionId{508},
                OperationId{600},
            }};

        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{300},
                    DataResidencyId{400},
                },
            });

        (void)execution.bind(std::move(binding));

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(fixture.start_attempt(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::CANCELLED));
        GERDOS_CHECK(effects.finish(execution));
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(fixture.input->state() == DataResidencyState::VALID);
    }

    // ---------------------------------------------------------------------
    // 9. An update-in-progress is owned by exactly one attempt
    // ---------------------------------------------------------------------

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry, fixture.executions);

        auto* first = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{520},
                OperationId{600},
            });

        auto* second = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{521},
                OperationId{600},
            });

        auto* third = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{522},
                OperationId{600},
            });

        auto* fourth = fixture.executions.create_execution(
            ExecutionDescription{
                ExecutionId{523},
                OperationId{600},
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
                    DataId{300},
                    DataResidencyId{401},
                },
            });

        GERDOS_CHECK(first->bind(binding));
        GERDOS_CHECK(second->bind(binding));
        GERDOS_CHECK(third->bind(binding));
        GERDOS_CHECK(fourth->bind(binding));

        // The first attempt claims the producing residency.
        GERDOS_CHECK(effects.claims_free(*first));
        GERDOS_CHECK(effects.start(*first));
        GERDOS_CHECK(
            fixture.output->update_owner() == ExecutionId{520});

        // A live foreign claim blocks the second attempt.
        GERDOS_CHECK(!effects.claims_free(*second));
        GERDOS_CHECK(!effects.start(*second));
        GERDOS_CHECK(
            fixture.output->update_owner() == ExecutionId{520});

        // The second attempt cannot finish the first attempt's claim.
        GERDOS_CHECK(fixture.start_attempt(*second));
        GERDOS_CHECK(second->set_state(ExecutionState::COMPLETED));
        GERDOS_CHECK(!effects.finish(*second));
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::TRANSFERRING);

        // The owner finishes and releases the claim.
        GERDOS_CHECK(fixture.start_attempt(*first));
        GERDOS_CHECK(first->set_state(ExecutionState::COMPLETED));
        GERDOS_CHECK(effects.finish(*first));
        GERDOS_CHECK(!fixture.output->update_owner().valid());
        GERDOS_CHECK(fixture.output->state() == DataResidencyState::VALID);

        // A released claim is free for the next attempt.
        GERDOS_CHECK(effects.claims_free(*third));
        GERDOS_CHECK(effects.start(*third));
        GERDOS_CHECK(
            fixture.output->update_owner() == ExecutionId{522});

        // An attempt that terminates without applying its finishing
        // effects leaves a stale claim; a later attempt takes it over.
        GERDOS_CHECK(fixture.start_attempt(*third));
        GERDOS_CHECK(third->set_state(ExecutionState::CANCELLED));
        GERDOS_CHECK(
            fixture.output->update_owner() == ExecutionId{522});

        GERDOS_CHECK(effects.claims_free(*fourth));
        GERDOS_CHECK(effects.start(*fourth));
        GERDOS_CHECK(
            fixture.output->update_owner() == ExecutionId{523});
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::TRANSFERRING);
    }

    return 0;
}