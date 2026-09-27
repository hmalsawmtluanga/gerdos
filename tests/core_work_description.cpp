#include "test_check.hpp"

#include <optional>
#include <vector>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"
#include "gerdos/core/physical_binding_validation.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/cpu/cpu_backend.hpp"

namespace {

using namespace gerdos;

struct Machine {
    DeviceRegistry devices;
    DataRegistry data;
    OperationRegistry operations;
    ExecutionRegistry executions;

    Machine() {
        auto* host = devices.create_device(
            DeviceDescription{DeviceId{100}, "host"});

        (void)host->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{101},
                    DeviceId{100},
                    ResourceKind::MEMORY,
                    "ram",
                },
            });

        host->find_resource(ResourceId{101})->set_availability(
            ResourceAvailability::AVAILABLE);

        (void)host->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{102},
                    DeviceId{100},
                    ResourceKind::COMPUTE,
                    "compute",
                },
            });

        host->find_resource(ResourceId{102})->set_availability(
            ResourceAvailability::AVAILABLE);

        auto* input = data.create_data(
            DataDescription{DataId{500}, "input"});

        (void)input->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{5001},
                    DataId{500},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "input",
                },
            });

        (void)input->find_residency(DataResidencyId{5001})
            ->set_state(DataResidencyState::VALID);

        auto* output = data.create_data(
            DataDescription{DataId{501}, "output"});

        (void)output->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{5101},
                    DataId{501},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "output",
                },
            });
    }
};

} // namespace

int main() {
    using namespace gerdos;

    Machine machine;

    // ---------------------------------------------------------------------
    // 1. The declared algebra computes exactly what it says
    // ---------------------------------------------------------------------

    {
        OperationDescription description{
            OperationId{800},
            {DataId{500}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            WorkDescription{4, 2, 2.0f, 3.0f, 4.0f},
        };

        (void)machine.operations.create_operation(description);

        CpuBackend backend;

        Executor executor(
            machine.executions,
            machine.operations,
            machine.devices,
            machine.data,
            backend);

        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{500}, DataResidencyId{5001}},
            });

        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{501}, DataResidencyId{5101}},
            });

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{900}, OperationId{800}});

        GERDOS_CHECK(execution->bind(binding));
        GERDOS_CHECK(executor.start(ExecutionId{900}));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        while (outcomes.empty()) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);

        // dst starts at 1.0 and src is 1.0: pass one yields
        // 1*2 + 1*3 + 4 = 9; pass two yields 9*2 + 1*3 + 4 = 25.
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{501}, DataResidencyId{5101}},
                0) == 25.0f);
    }

    // ---------------------------------------------------------------------
    // 2. Exact copying is the special case of the same algebra
    // ---------------------------------------------------------------------

    {
        OperationDescription description{
            OperationId{801},
            {DataId{500}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            WorkDescription{4, 1, 0.0f, 1.0f, 0.0f},
        };

        (void)machine.operations.create_operation(description);

        CpuBackend backend;

        Executor executor(
            machine.executions,
            machine.operations,
            machine.devices,
            machine.data,
            backend);

        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{500}, DataResidencyId{5001}},
            });

        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{501}, DataResidencyId{5101}},
            });

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{901}, OperationId{801}});

        GERDOS_CHECK(execution->bind(binding));
        GERDOS_CHECK(executor.start(ExecutionId{901}));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        while (outcomes.empty()) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(backend.allocations_equal(
            DataResidencyRef{DataId{500}, DataResidencyId{5001}},
            DataResidencyRef{DataId{501}, DataResidencyId{5101}}));
    }

    // ---------------------------------------------------------------------
    // 3. Gates are blind to the work
    // ---------------------------------------------------------------------

    {
        OperationDescription quiet{
            OperationId{802},
            {DataId{500}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            WorkDescription{},
        };

        // The same realization with absurd work: enormous sizes, hostile
        // scales. Every gate must judge it identically.
        auto loud = quiet;
        loud.id = OperationId{803};
        loud.work = WorkDescription{
            ~std::size_t{0},
            ~std::size_t{0},
            -440.0f,
            1e30f,
            -1e30f,
        };

        const Operation quiet_operation{quiet};
        const Operation loud_operation{loud};

        PhysicalBindingValidator validator;
        BindingResolver resolver(machine.devices, machine.data);
        BindingAdmissibilityValidator admissibility;
        ExecutionAdmissionValidator admission(
            machine.devices,
            machine.data);

        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{500}, DataResidencyId{5001}},
            });

        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{501}, DataResidencyId{5101}},
            });

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        GERDOS_CHECK(validator.validate(binding));
        GERDOS_CHECK(resolver.resolve(binding).fully_resolved());
        GERDOS_CHECK(
            admissibility.admissible(quiet_operation, binding) ==
            admissibility.admissible(loud_operation, binding));
        GERDOS_CHECK(
            admissibility.admissible(quiet_operation, binding));

        Execution quiet_attempt{
            ExecutionDescription{ExecutionId{902}, OperationId{802}}};
        Execution loud_attempt{
            ExecutionDescription{ExecutionId{903}, OperationId{803}}};

        GERDOS_CHECK(quiet_attempt.bind(binding));
        GERDOS_CHECK(loud_attempt.bind(binding));

        GERDOS_CHECK(
            admission.admit(quiet_attempt).has_value() ==
            admission.admit(loud_attempt).has_value());
    }

    return 0;
}