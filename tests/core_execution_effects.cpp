#include "test_check.hpp"

#include "gerdos/core/execution_effects.hpp"

namespace {

using namespace gerdos;

struct Fixture {
    DataRegistry data_registry;
    Data* data = nullptr;
    DataResidency* input = nullptr;
    DataResidency* output = nullptr;

    Fixture() {
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
        ExecutionEffects effects(fixture.data_registry);
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
        ExecutionEffects effects(fixture.data_registry);
        auto execution = bound_execution(ExecutionId{501});

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
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
        ExecutionEffects effects(fixture.data_registry);
        auto execution = bound_execution(ExecutionId{502});

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
        GERDOS_CHECK(execution.set_state(ExecutionState::FAILED));
        GERDOS_CHECK(effects.finish(execution));

        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(fixture.input->state() == DataResidencyState::VALID);
    }

    {
        Fixture fixture;
        ExecutionEffects effects(fixture.data_registry);
        auto execution = bound_execution(ExecutionId{503});

        GERDOS_CHECK(effects.start(execution));
        GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
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
        ExecutionEffects effects(fixture.data_registry);
        auto execution = bound_execution(ExecutionId{504});

        // The rewrite scenario: an existing usable representation would be
        // destroyed by an unwarranted failure effect.
        GERDOS_CHECK(
            fixture.output->set_state(DataResidencyState::VALID));

        GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
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
        ExecutionEffects effects(fixture.data_registry);
        auto execution = bound_execution(ExecutionId{505});

        GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
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
        ExecutionEffects effects(fixture.data_registry);

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
        ExecutionEffects effects(fixture.data_registry);

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
        ExecutionEffects effects(fixture.data_registry);

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
        GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
        GERDOS_CHECK(execution.set_state(ExecutionState::CANCELLED));
        GERDOS_CHECK(effects.finish(execution));
        GERDOS_CHECK(
            fixture.output->state() == DataResidencyState::UNAVAILABLE);
        GERDOS_CHECK(fixture.input->state() == DataResidencyState::VALID);
    }

    return 0;
}