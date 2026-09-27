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

    // ---------------------------------------------------------------------
    // 4. The seam rejects work that is not well-formed
    // ---------------------------------------------------------------------

    {
        CpuBackend backend;

        OperationDescription hostile{
            OperationId{804},
            {DataId{500}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            // Sizing that wraps elements * sizeof(float) into an
            // undersized allocation.
            WorkDescription{(std::size_t{1} << 61) + 1, 1, 0.0f, 1.0f, 0.0f},
        };

        const Operation hostile_operation{hostile};

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

        Execution hostile_attempt{
            ExecutionDescription{ExecutionId{904}, OperationId{804}}};

        GERDOS_CHECK(hostile_attempt.bind(binding));

        // Rejected without touching an allocation.
        GERDOS_CHECK(
            !backend.submit(hostile_operation, hostile_attempt));
        GERDOS_CHECK(backend.allocation_count() == 0);

        // Zero work is not executable work: no vacuous success, no
        // "infinitely fast" evidence.
        OperationDescription empty{
            OperationId{805},
            {DataId{500}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            WorkDescription{},
        };

        const Operation empty_operation{empty};

        Execution empty_attempt{
            ExecutionDescription{ExecutionId{905}, OperationId{805}}};

        GERDOS_CHECK(empty_attempt.bind(binding));
        GERDOS_CHECK(!backend.submit(empty_operation, empty_attempt));
        GERDOS_CHECK(backend.allocation_count() == 0);
    }

    // ---------------------------------------------------------------------
    // 5. In-place aliasing is the iterated form
    // ---------------------------------------------------------------------

    {
        Machine machine;

        (void)machine.data.find_data(DataId{501})->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{5102},
                    DataId{501},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "sole",
                },
            });

        // An in-place update consumes a usable record.
        (void)machine.data.find_data(DataId{501})
            ->find_residency(DataResidencyId{5102})
            ->set_state(DataResidencyState::VALID);

        // One record consumed and produced: each pass composes over the
        // previous result. dst starts at 1.0 with dst = dst*1 + dst*2:
        // 1 -> 3 -> 9.
        OperationDescription in_place{
            OperationId{806},
            {DataId{501}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            WorkDescription{4, 2, 1.0f, 2.0f, 0.0f},
        };

        (void)machine.operations.create_operation(in_place);

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
                DataResidencyRef{DataId{501}, DataResidencyId{5102}},
            });

        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{501}, DataResidencyId{5102}},
            });

        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{906}, OperationId{806}});

        GERDOS_CHECK(execution->bind(binding));
        GERDOS_CHECK(executor.start(ExecutionId{906}));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        while (outcomes.empty()) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{501}, DataResidencyId{5102}},
                0) == 9.0f);
    }

    // ---------------------------------------------------------------------
    // 6. Resizing a representation while work is in flight is safe
    // ---------------------------------------------------------------------

    {
        Machine machine;

        OperationDescription slow{
            OperationId{807},
            {DataId{500}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            WorkDescription{1 << 20, 64, 0.5f, 1.5f, 0.0f},
        };

        OperationDescription quick{
            OperationId{808},
            {DataId{500}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            },
            WorkDescription{4, 1, 0.0f, 1.0f, 0.0f},
        };

        (void)machine.operations.create_operation(slow);
        (void)machine.operations.create_operation(quick);

        CpuBackend backend;

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

        const Operation slow_operation{slow};
        const Operation quick_operation{quick};

        Execution slow_attempt{
            ExecutionDescription{ExecutionId{907}, OperationId{807}}};
        Execution quick_attempt{
            ExecutionDescription{ExecutionId{908}, OperationId{808}}};

        GERDOS_CHECK(slow_attempt.bind(binding));
        GERDOS_CHECK(quick_attempt.bind(binding));

        // The second attempt resizes the representations the first is
        // still working on. Shared ownership keeps the first attempt's
        // buffers alive; both complete coherently.
        GERDOS_CHECK(backend.submit(slow_operation, slow_attempt));
        GERDOS_CHECK(backend.submit(quick_operation, quick_attempt));

        std::vector<BackendCompletion> completed;

        while (completed.size() < 2) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.size() == 2);

        for (const auto& completion : completed) {
            GERDOS_CHECK(completion.succeeded);
        }

        // The mapping now holds the quick attempt's storage, with its own
        // semantics: an exact copy of the input.
        GERDOS_CHECK(backend.allocation_count() == 2);
        GERDOS_CHECK(
            backend.allocation_bytes(
                DataResidencyRef{DataId{501}, DataResidencyId{5101}}) == 4);
        GERDOS_CHECK(backend.allocations_equal(
            DataResidencyRef{DataId{500}, DataResidencyId{5001}},
            DataResidencyRef{DataId{501}, DataResidencyId{5101}}));
    }

    // ---------------------------------------------------------------------
    // 7. The reduction form computes exactly what it says
    // ---------------------------------------------------------------------

    {
        Machine machine;

        auto* source = machine.data.create_data(
            DataDescription{DataId{700}, "source"});
        (void)source->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7001},
                    DataId{700},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "source",
                },
            });

        auto* partial = machine.data.create_data(
            DataDescription{DataId{701}, "partial"});
        (void)partial->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7002},
                    DataId{701},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "partial",
                },
            });

        auto* product = machine.data.create_data(
            DataDescription{DataId{702}, "product"});
        (void)product->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7003},
                    DataId{702},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "product",
                },
            });

        CpuBackend backend;

        // Sum of six 1.0 elements: dst[0] = 0*1 + 1*6 + 0 = 6.
        OperationDescription reduce{
            OperationId{810},
            {DataId{700}},
            {DataId{701}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
        };

        const Operation reduce_operation{reduce};
        Execution reduce_attempt{
            ExecutionDescription{ExecutionId{910}, OperationId{810}}};

        PhysicalBinding reduce_binding;
        reduce_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
            });
        reduce_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
            });
        reduce_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        GERDOS_CHECK(reduce_attempt.bind(reduce_binding));
        GERDOS_CHECK(backend.submit(reduce_operation, reduce_attempt));

        std::vector<BackendCompletion> completed;

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
                0) == 6.0f);

        // -----------------------------------------------------------------
        // 8. The matrix form computes exactly what it says
        // -----------------------------------------------------------------

        // The partial record now holds [6,1,1,1,1,1] — a non-uniform
        // 2x3 operand that distinguishes real matrix semantics from any
        // plausible fake. B is the uniform source record.
        OperationDescription matmul{
            OperationId{811},
            {DataId{701}, DataId{700}},
            {DataId{702}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{
                0, 1, 0.0f, 1.0f, 0.0f, WorkForm::MATRIX_PRODUCT, 2, 3, 2},
        };

        const Operation matmul_operation{matmul};
        Execution matmul_attempt{
            ExecutionDescription{ExecutionId{911}, OperationId{811}}};

        PhysicalBinding matmul_binding;
        matmul_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
            });
        matmul_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
            });
        matmul_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
            });
        matmul_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        GERDOS_CHECK(matmul_attempt.bind(matmul_binding));
        GERDOS_CHECK(backend.submit(matmul_operation, matmul_attempt));
        completed.clear();

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);

        // A = [[6,1,1],[1,1,1]], B = ones(3,2):
        // C = [[8,8],[3,3]] exactly.
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                0) == 8.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                1) == 8.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                2) == 3.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                3) == 3.0f);
    }

    return 0;
}