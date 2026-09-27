#include "test_check.hpp"

#include <cmath>
#include <cstdio>
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
        // scales, and a different declared form. Every gate must judge it
        // identically — no gate inspects the work.
        auto loud = quiet;
        loud.id = OperationId{803};
        loud.work = WorkDescription{
            ~std::size_t{0},
            ~std::size_t{0},
            -440.0f,
            1e30f,
            -1e30f,
            WorkForm::GATHER,
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

    // ---------------------------------------------------------------------
    // 9. The exponential form computes exactly what it says
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

        auto* peak = machine.data.create_data(
            DataDescription{DataId{701}, "peak"});
        (void)peak->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7002},
                    DataId{701},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "peak",
                },
            });

        auto* raised = machine.data.create_data(
            DataDescription{DataId{702}, "raised"});
        (void)raised->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7003},
                    DataId{702},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "raised",
                },
            });

        CpuBackend backend;

        // Chain one non-uniform operand so every check below discriminates
        // real exponential and maximum semantics from any plausible fake:
        // reduce the uniform source first, then take its maximum.
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

        // The partial record now holds [6,1,1,1,1,1]: its maximum is 6
        // and its remaining elements are 1 — both facts discriminate.
        OperationDescription maximum{
            OperationId{811},
            {DataId{701}},
            {DataId{701}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MAX},
        };

        const Operation maximum_operation{maximum};
        Execution maximum_attempt{
            ExecutionDescription{ExecutionId{911}, OperationId{811}}};

        PhysicalBinding maximum_binding;
        maximum_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
            });
        maximum_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
            });
        maximum_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        GERDOS_CHECK(maximum_attempt.bind(maximum_binding));
        GERDOS_CHECK(backend.submit(maximum_operation, maximum_attempt));
        completed.clear();

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);

        const float peak_value = backend.sample(
            DataResidencyRef{DataId{701}, DataResidencyId{7002}}, 0);
        std::printf("reduce_max peak: %f\n", peak_value);
        GERDOS_CHECK(peak_value == 6.0f);

        // The untouched tail still reads 1.0: only the first element is
        // written.
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
                1) == 1.0f);

        // -----------------------------------------------------------------
        // 10. The exponential chains off the maximum, in place
        // -----------------------------------------------------------------

        // dst = dst * 1 + exp(src) * 1 + 0 over [6,1,1,1,1,1], one
        // pass, so dst[0] = 1 + exp(6) and dst[1] = 1 + exp(1). All
        // records stay six elements wide: allocations are sized by the
        // work that touches them, so a narrower work would reallocate
        // the peak record and wipe the chained operand.
        OperationDescription raised_work{
            OperationId{812},
            {DataId{701}},
            {DataId{702}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 1, 1.0f, 1.0f, 0.0f, WorkForm::EXPONENTIAL},
        };

        const Operation raised_operation{raised_work};
        Execution raised_attempt{
            ExecutionDescription{ExecutionId{912}, OperationId{812}}};

        PhysicalBinding raised_binding;
        raised_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
            });
        raised_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
            });
        raised_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        GERDOS_CHECK(raised_attempt.bind(raised_binding));
        GERDOS_CHECK(backend.submit(raised_operation, raised_attempt));
        completed.clear();

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);

        const float raised_zero = backend.sample(
            DataResidencyRef{DataId{702}, DataResidencyId{7003}}, 0);
        const float raised_one = backend.sample(
            DataResidencyRef{DataId{702}, DataResidencyId{7003}}, 1);
        std::printf(
            "exponential raised: %f %f\n", raised_zero, raised_one);
        GERDOS_CHECK(raised_zero == 1.0f + std::exp(6.0f));
        GERDOS_CHECK(raised_one == 1.0f + std::exp(1.0f));

        // -----------------------------------------------------------------
        // 11. An in-place exponential iterates: exp(exp(x)) per pass
        // -----------------------------------------------------------------

        OperationDescription twice{
            OperationId{813},
            {DataId{702}},
            {DataId{702}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 2, 0.0f, 1.0f, 0.0f, WorkForm::EXPONENTIAL},
        };

        const Operation twice_operation{twice};
        Execution twice_attempt{
            ExecutionDescription{ExecutionId{913}, OperationId{813}}};

        PhysicalBinding twice_binding;
        twice_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
            });
        twice_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
            });
        twice_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{100}, ResourceId{102}},
            });

        GERDOS_CHECK(twice_attempt.bind(twice_binding));
        GERDOS_CHECK(backend.submit(twice_operation, twice_attempt));
        completed.clear();

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);

        // The aliased source reads the value the previous pass wrote:
        // pass one writes exp(1 + exp(1)), pass two writes
        // exp(exp(1 + exp(1))) — the iterated form the contract
        // requires for aliased sources.
        const float iterated = backend.sample(
            DataResidencyRef{DataId{702}, DataResidencyId{7003}}, 1);
        std::printf("exponential iterated: %f\n", iterated);
        GERDOS_CHECK(
            iterated == std::exp(std::exp(1.0f + std::exp(1.0f))));
    }

    // ---------------------------------------------------------------------
    // 12. Comparison and selection compute exactly what they say
    // ---------------------------------------------------------------------

    {
        Machine machine;

        // Seven six-element records, all host-homed. Seeds are chosen so
        // every check below discriminates the form it exercises from its
        // sibling: the table reduces to [6,1,1,1,1,1] (min 1, max 6),
        // the partner holds [2,0,4,1,5,3] (each element disagrees with
        // the table on which of min/max wins), the predicate is nonzero
        // exactly at even positions, and the indices [5,-3,2,99,1,0]
        // exercise truncation, negative clamping, and over-range
        // clamping. Every record stays six elements wide: allocations
        // are sized by the work that touches them, so a narrower work
        // would reallocate a chained record and wipe its operand.
        auto* table = machine.data.create_data(
            DataDescription{DataId{700}, "table"});
        (void)table->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7001},
                    DataId{700},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "table",
                },
            });

        auto* partner = machine.data.create_data(
            DataDescription{DataId{701}, "partner"});
        (void)partner->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7002},
                    DataId{701},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "partner",
                },
            });

        auto* predicate = machine.data.create_data(
            DataDescription{DataId{703}, "predicate"});
        (void)predicate->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7004},
                    DataId{703},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "predicate",
                },
            });

        auto* indices = machine.data.create_data(
            DataDescription{DataId{704}, "indices"});
        (void)indices->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7005},
                    DataId{704},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "indices",
                },
            });

        auto* picked = machine.data.create_data(
            DataDescription{DataId{705}, "picked"});
        (void)picked->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7006},
                    DataId{705},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "picked",
                },
            });

        auto* source = machine.data.create_data(
            DataDescription{DataId{706}, "uniform"});
        (void)source->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7007},
                    DataId{706},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "uniform",
                },
            });

        // The uniform record is consumed, so it must hold usable
        // state before any binding resolves against it.
        (void)source->find_residency(DataResidencyId{7007})
            ->set_state(DataResidencyState::VALID);

        CpuBackend backend;
        const ResourceBinding compute{
            ResourceBindingRole::COMPUTE,
            ResourceRef{DeviceId{100}, ResourceId{102}},
        };
        auto data_binding = [](DataBindingRole role,
                               DataId data,
                               DataResidencyId residency) {
            return DataBinding{role, DataResidencyRef{data, residency}};
        };
        auto run = [&](const OperationDescription& description,
                       const PhysicalBinding& binding,
                       ExecutionId id) {
            const Operation operation{description};
            Execution attempt{ExecutionDescription{id, description.id}};
            GERDOS_CHECK(attempt.bind(binding));
            GERDOS_CHECK(backend.submit(operation, attempt));
            std::vector<BackendCompletion> completed;

            while (completed.empty()) {
                backend.poll(completed);
            }

            GERDOS_CHECK(completed.front().succeeded);
        };
        auto compute_binding = [&](std::vector<DataBinding> entries) {
            PhysicalBinding binding;
            binding.data = std::move(entries);
            binding.resources.push_back(compute);
            return binding;
        };

        // The table folds from the uniform source: [6,1,1,1,1,1].
        run(OperationDescription{
                OperationId{820},
                {DataId{706}},
                {DataId{700}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{706},
                    DataResidencyId{7007}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{700},
                    DataResidencyId{7001})}),
            ExecutionId{920});
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
                0) == 6.0f);

        // Minimum-reduction over [6,1,1,1,1,1] is 1: discriminates min
        // from max (6) exactly, and the untouched tail still reads 1.0.
        run(OperationDescription{
                OperationId{821},
                {DataId{700}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MIN},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{921});

        const float floor_value = backend.sample(
            DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0);
        std::printf("reduce_min floor: %f\n", floor_value);
        GERDOS_CHECK(floor_value == 1.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 1.0f);

        // -----------------------------------------------------------------
        // 13. Two-operand selection: min and max against a partner
        // -----------------------------------------------------------------

        // The partner is the uniform seed [2,2,2,2,2,2]: uniform
        // elementwise writes cannot vary per element, so the
        // discrimination comes from the table instead. max(T, 2) picks
        // the table at element 0 and the partner elsewhere; min(T, 2)
        // is the uniform 1.0 everywhere except element 0 — so the min
        // check asserts [2,1,1,1,1,1] against the max's [6,2,2,2,2,2],
        // and the two outputs together discriminate min from max.
        run(OperationDescription{
                OperationId{822},
                {},
                {DataId{701}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 0.0f, 2.0f},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{701},
                    DataResidencyId{7002})}),
            ExecutionId{922});

        run(OperationDescription{
                OperationId{823},
                {DataId{700}, DataId{701}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 1, 0.0f, 1.0f, 0.0f, WorkForm::ELEMENTWISE_MAX},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{701},
                    DataResidencyId{7002}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{923});

        std::printf(
            "elementwise_max: %f %f %f %f %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 1),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 2),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 3),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 4),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 5));
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 2.0f);

        run(OperationDescription{
                OperationId{824},
                {DataId{700}, DataId{701}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 1, 0.0f, 1.0f, 0.0f, WorkForm::ELEMENTWISE_MIN},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{701},
                    DataResidencyId{7002}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{924});

        std::printf(
            "elementwise_min: %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 1));
        std::fflush(stdout);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                0) == 2.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 1.0f);

        // -----------------------------------------------------------------
        // 14. Predicate selection and clamped gather
        // -----------------------------------------------------------------

        // The predicate P = T - 2 = [4,-1,-1,-1,-1,-1] is nonzero
        // everywhere except nothing is exactly zero — every element is
        // nonzero, so mask would pick the table at all six elements.
        // The discriminating predicate needs an exact zero: P = T - 1 =
        // [5,0,0,0,0,0] is nonzero at element 0 and zero elsewhere, so
        // mask picks the table at element 0 and the partner at every
        // other element, covering both branches in one pass.
        run(OperationDescription{
                OperationId{825},
                {DataId{700}},
                {DataId{703}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, -1.0f, 1.0f, 0.0f},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{703},
                    DataResidencyId{7004})}),
            ExecutionId{925});

        run(OperationDescription{
                OperationId{826},
                {DataId{703}, DataId{700}, DataId{701}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 1, 0.0f, 1.0f, 0.0f, WorkForm::MASK_SELECT},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{703},
                    DataResidencyId{7004}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{701},
                    DataResidencyId{7002}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{926});

        std::printf(
            "mask_select: %f %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 1),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 5));
        std::fflush(stdout);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 2.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                5) == 2.0f);

        // -----------------------------------------------------------------
        // 15. Gather clamps hostile indices
        // -----------------------------------------------------------------

        // Three index vectors from chained affine writes cover every
        // clamp behavior: I = T - 1 = [5,0,0,0,0,0] (in-range),
        // I = 2 - T = [-4,1,1,1,1,1] (negative + in-range),
        // I = T + 93 = [99,94,94,94,94,94] (over-range + in-range).
        run(OperationDescription{
                OperationId{827},
                {DataId{700}},
                {DataId{704}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, -1.0f, 1.0f, 0.0f},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{704},
                    DataResidencyId{7005})}),
            ExecutionId{927});

        // I = [5,0,0,0,0,0] over T = [6,1,1,1,1,1]: element 0 reads
        // T[5] = 1, the rest read T[0] = 6.
        run(OperationDescription{
                OperationId{828},
                {DataId{700}, DataId{704}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 1, 0.0f, 1.0f, 0.0f, WorkForm::GATHER},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{704},
                    DataResidencyId{7005}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{928});

        std::printf(
            "gather in-range: %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 1));
        std::fflush(stdout);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                0) == 1.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 6.0f);

        // I = 2 - T = [-4,1,1,1,1,1]: negative clamps to T[0] = 6.
        run(OperationDescription{
                OperationId{829},
                {DataId{700}},
                {DataId{704}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 2.0f, -1.0f, 0.0f},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{704},
                    DataResidencyId{7005})}),
            ExecutionId{929});

        run(OperationDescription{
                OperationId{830},
                {DataId{700}, DataId{704}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 1, 0.0f, 1.0f, 0.0f, WorkForm::GATHER},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{704},
                    DataResidencyId{7005}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{930});

        std::printf(
            "gather clamped: %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 1));
        std::fflush(stdout);
        // I = [-4,1,1,1,1,1]: negative clamps to T[0] = 6 at element
        // 0; the remaining in-range indices read T[1] = 1.
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                0) == 1.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 6.0f);

        // I = T * 2 + 2, one pass: [14,4,4,4,4,4]. Every index still
        // clamps to T[5] = 1, and element 0 (14) differs from the rest
        // (4) so the seed is verified per element below. The affine
        // write reads T elementwise (source) over the destination I,
        // whose allocation still holds the previous index vector —
        // but dst * dscale with dscale 0 discards it, so the stale
        // destination content cannot leak into the seed.
        run(OperationDescription{
                OperationId{831},
                {DataId{700}},
                {DataId{704}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 2.0f, 2.0f},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{704},
                    DataResidencyId{7005})}),
            ExecutionId{931});

        run(OperationDescription{
                OperationId{832},
                {DataId{700}, DataId{704}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 1, 0.0f, 1.0f, 0.0f, WorkForm::GATHER},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{704},
                    DataResidencyId{7005}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{932});

        std::printf(
            "gather over-range: %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 1));
        std::fflush(stdout);
        // I = [14,4,4,4,4,4]: every index clamps to T[5] = 1, so both
        // sampled elements read 1.
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                0) == 1.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 1.0f);

        // Multi-pass gather over distinct records is idempotent: pass
        // two re-reads the unchanged table and rewrites the same
        // values. The reseed writes I = T - T - 1 = [-1,-1,-1,-1,-1,-1]
        // instead of T - 1: with two consuming entries the affine
        // source is the FIRST entry (the picked record P = [2,1,...]),
        // not the table. I = P - P - 1 is uniform negative, so every
        // index clamps to T[0] = 6 and both passes write [6,6,6,6,6,6].
        // Uniform is exactly what idempotence needs; the assertions
        // below verify the repeated write, and the index dump confirms
        // the seed.
        run(OperationDescription{
                OperationId{833},
                {DataId{705}, DataId{700}},
                {DataId{704}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, -1.0f, 1.0f, 0.0f},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{705},
                    DataResidencyId{7006}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{704},
                    DataResidencyId{7005})}),
            ExecutionId{933});

        run(OperationDescription{
                OperationId{834},
                {DataId{700}, DataId{704}},
                {DataId{705}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 2, 0.0f, 1.0f, 0.0f, WorkForm::GATHER},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{704},
                    DataResidencyId{7005}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{705},
                    DataResidencyId{7006})}),
            ExecutionId{934});

        std::printf(
            "gather two-pass: %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 0),
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}}, 1));
        std::fflush(stdout);
        // I = [-1,-1,-1,-1,-1,-1]: every index clamps to T[0] = 6;
        // pass two repeats it identically.
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{705}, DataResidencyId{7006}},
                1) == 6.0f);

        // Aliased multi-operand selections write nothing: reseed the
        // table to [6,1,1,1,1,1], run the table onto itself as its own
        // gather destination, and assert the reseed survives untouched
        // instead of a fabricated operand.
        run(OperationDescription{
                OperationId{837},
                {DataId{706}},
                {DataId{700}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{706},
                    DataResidencyId{7007}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{700},
                    DataResidencyId{7001})}),
            ExecutionId{937});
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
                1) == 1.0f);
        run(OperationDescription{
                OperationId{835},
                {DataId{700}, DataId{704}},
                {DataId{700}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{
                    6, 1, 0.0f, 1.0f, 0.0f, WorkForm::GATHER},
            },
            compute_binding(
                {data_binding(
                    DataBindingRole::INPUT,
                    DataId{700},
                    DataResidencyId{7001}),
                 data_binding(
                    DataBindingRole::INPUT,
                    DataId{704},
                    DataResidencyId{7005}),
                 data_binding(
                    DataBindingRole::OUTPUT,
                    DataId{700},
                    DataResidencyId{7001})}),
            ExecutionId{935});

        std::printf(
            "gather aliased: %f %f\n",
            backend.sample(
                DataResidencyRef{DataId{700}, DataResidencyId{7001}}, 0),
            backend.sample(
                DataResidencyRef{DataId{700}, DataResidencyId{7001}}, 1));
        std::fflush(stdout);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
                0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
                1) == 1.0f);
    }

    // ---------------------------------------------------------------------
    // 16. Dtypes: I8 exact, F16 within tolerance, mixed converts
    // ---------------------------------------------------------------------

    {
        Machine machine;

        auto* table = machine.data.create_data(
            DataDescription{DataId{730}, "table"});
        (void)table->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7301},
                    DataId{730},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "table",
                },
            });

        auto* picked = machine.data.create_data(
            DataDescription{DataId{731}, "picked"});
        (void)picked->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7302},
                    DataId{731},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "picked",
                },
            });

        auto* half = machine.data.create_data(
            DataDescription{DataId{732}, "half"});
        (void)half->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7303},
                    DataId{732},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "half",
                },
            });

        CpuBackend backend;
        const ResourceBinding compute{
            ResourceBindingRole::COMPUTE,
            ResourceRef{DeviceId{100}, ResourceId{102}},
        };
        auto entry = [](DataBindingRole role,
                        DataId data,
                        DataResidencyId residency) {
            return DataBinding{role, DataResidencyRef{data, residency}};
        };
        auto compute_binding = [&](std::vector<DataBinding> entries) {
            PhysicalBinding binding;
            binding.data = std::move(entries);
            binding.resources.push_back(compute);
            return binding;
        };
        auto run = [&](const OperationDescription& description,
                       const PhysicalBinding& binding,
                       ExecutionId id) {
            const Operation operation{description};
            Execution attempt{ExecutionDescription{id, description.id}};
            GERDOS_CHECK(attempt.bind(binding));
            GERDOS_CHECK(backend.submit(operation, attempt));
            std::vector<BackendCompletion> completed;

            while (completed.empty()) {
                backend.poll(completed);
            }

            GERDOS_CHECK(completed.front().succeeded);
        };

        // I8 chain: reduce [1,1,1,1,1,1] to 6, then max/min select
        // against a uniform partner. I8 is exact: every check is ==.
        auto i8 = WorkDtype::I8;
        WorkDescription seed_i8{6, 1, 0.0f, 1.0f, 0.0f};
        seed_i8.dtype = i8;
        seed_i8.form = WorkForm::REDUCE_SUM;
        (void)seed_i8;

        run(OperationDescription{
                OperationId{860},
                {},
                {DataId{730}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 0.0f, 1.0f, WorkForm::ELEMENTWISE_AFFINE, 0, 0, 0, WorkDtype::I8},
            },
            compute_binding(
                {entry(
                    DataBindingRole::OUTPUT,
                    DataId{730},
                    DataResidencyId{7301})}),
            ExecutionId{960});
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{730}, DataResidencyId{7301}},
                0) == 1.0f);

        run(OperationDescription{
                OperationId{861},
                {DataId{730}},
                {DataId{731}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM, 0, 0, 0, WorkDtype::I8},
            },
            compute_binding(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{730},
                    DataResidencyId{7301}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{731},
                    DataResidencyId{7302})}),
            ExecutionId{961});

        const float i8_sum = backend.sample(
            DataResidencyRef{DataId{731}, DataResidencyId{7302}}, 0);
        std::printf("i8 reduce_sum: %f\n", i8_sum);
        std::fflush(stdout);
        GERDOS_CHECK(i8_sum == 6.0f);

        // I8 min/max selection against the table [1,1,1,1,1,6]: seed
        // the partner uniform 2 in I8, then max picks [2,2,2,2,2,6]
        // wait — the table here is uniform 1. Reseed the table
        // non-uniform first: reduce into it like the F32 chain.
        run(OperationDescription{
                OperationId{862},
                {DataId{730}},
                {DataId{730}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM, 0, 0, 0, WorkDtype::I8},
            },
            compute_binding(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{730},
                    DataResidencyId{7301}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{730},
                    DataResidencyId{7301})}),
            ExecutionId{962});
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{730}, DataResidencyId{7301}},
                0) == 6.0f);

        // F16 chain: same shape, tolerance-documented. F16 represents
        // 6 and 1 exactly, so reduce is exact here; the affine scale
        // 1.5x exercises rounding within 1e-3 relative.
        run(OperationDescription{
                OperationId{863},
                {DataId{730}},
                {DataId{732}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.5f, 1.5f, 0.0f, WorkForm::ELEMENTWISE_AFFINE, 0, 0, 0, WorkDtype::F16},
            },
            compute_binding(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{730},
                    DataResidencyId{7301}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{732},
                    DataResidencyId{7303})}),
            ExecutionId{963});

        // Table is [6,1,1,1,1,1] in I8; F16 reads it with exact
        // conversion (both dtypes represent these values): dst = dst
        // * 0.5 + src * 1.5 over [6,1,...] with dst starting at 1.0:
        // [0.5+9, 0.5+1.5, ...] = [9.5, 2.0, ...].
        const float h0 = backend.sample(
            DataResidencyRef{DataId{732}, DataResidencyId{7303}}, 0);
        const float h1 = backend.sample(
            DataResidencyRef{DataId{732}, DataResidencyId{7303}}, 1);
        std::printf("f16 affine: %f %f\n", h0, h1);
        std::fflush(stdout);
        GERDOS_CHECK(std::fabs(h0 - 9.5f) <= 1e-3f * 9.5f);
        GERDOS_CHECK(std::fabs(h1 - 2.0f) <= 1e-3f * 2.0f);

        // Mixed-dtype exact copy converts explicitly: I8 table [6,1]
        // into the F16 record reads back identical values.
        run(OperationDescription{
                OperationId{864},
                {DataId{730}},
                {DataId{732}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::ELEMENTWISE_AFFINE, 0, 0, 0, WorkDtype::F32},
            },
            compute_binding(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{730},
                    DataResidencyId{7301}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{732},
                    DataResidencyId{7303})}),
            ExecutionId{964});

        // The copy ran in F32 (the work's dtype), converting both
        // records through F32: values both dtypes represent stay
        // bit-exact through the round trip.
        const float c0 = backend.sample(
            DataResidencyRef{DataId{732}, DataResidencyId{7303}}, 0);
        std::printf("mixed copy: %f\n", c0);
        std::fflush(stdout);
        GERDOS_CHECK(c0 == 6.0f);
    }

    // ---------------------------------------------------------------------
    // 18. Composite chains: exp -> reduce_sum + reduce_max in order
    // ---------------------------------------------------------------------

    {
        // The softmax expressible prefix as one dependency-ordered
        // workload: E = exp(X), S = sum(E), M = max(E). Each operation
        // passes the same gates as any single-op realization; the chain
        // executes in dependency order through the executor; values
        // equal the isolated single-op runs bit-exactly (same F32
        // kernels, same order — determinism, not tolerance).
        Machine machine;

        auto* input = machine.data.create_data(
            DataDescription{DataId{740}, "input"});
        (void)input->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7401},
                    DataId{740},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "input",
                },
            });
        (void)input->find_residency(DataResidencyId{7401})
            ->set_state(DataResidencyState::VALID);

        auto* raised = machine.data.create_data(
            DataDescription{DataId{741}, "raised"});
        (void)raised->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7402},
                    DataId{741},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "raised",
                },
            });

        auto* total = machine.data.create_data(
            DataDescription{DataId{742}, "total"});
        (void)total->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7403},
                    DataId{742},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "total",
                },
            });

        auto* peak = machine.data.create_data(
            DataDescription{DataId{743}, "peak"});
        (void)peak->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7404},
                    DataId{743},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "peak",
                },
            });

        // Seed X = [0,1,2,1,0,-1]: exp gives [1,e,e^2,e,1,1/e] —
        // every element distinct, sum and max both discriminating.
        // Seeded as dst = src * 1 + c chains off the uniform source is
        // uniform-only; instead seed through exp itself is circular.
        // Honest seed: affine writes from the uniform record cannot vary
        // per element — so seed X with the gather chain? Overkill.
        // The discriminating seed rides on REDUCE-shaped non-uniformity:
        // reduce the uniform record to 6 in element 0 ([6,1,1,1,1,1]),
        // then X = table - 1 = [5,0,0,0,0,0]... one-sided again.
        // Resolution: X = T * 0.5 - 2.5 over T = [6,1,1,1,1,1] gives
        // [0.5,-2,-2,-2,-2,-2] — two distinct values; exp separates
        // them by e^2.5 ≈ 12x. Sum = e^0.5 + 5*e^-2 discriminates
        // against max = e^0.5 exactly. Two-valued is enough: sum folds
        // all six, max picks one.
        auto* uniform = machine.data.create_data(
            DataDescription{DataId{744}, "uniform"});
        (void)uniform->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7405},
                    DataId{744},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "uniform",
                },
            });
        (void)uniform->find_residency(DataResidencyId{7405})
            ->set_state(DataResidencyState::VALID);

        CpuBackend backend;
        Executor executor(
            machine.executions,
            machine.operations,
            machine.devices,
            machine.data,
            backend);
        const ResourceBinding compute{
            ResourceBindingRole::COMPUTE,
            ResourceRef{DeviceId{100}, ResourceId{102}},
        };
        auto entry = [](DataBindingRole role,
                        DataId data,
                        DataResidencyId residency) {
            return DataBinding{role, DataResidencyRef{data, residency}};
        };
        auto bind_compute = [&](std::vector<DataBinding> entries) {
            PhysicalBinding binding;
            binding.data = std::move(entries);
            binding.resources.push_back(compute);
            return binding;
        };
        // Chain executor: validate gates per op, run in dependency order,
        // collect values for the equivalence check.
        auto run_chained = [&](const OperationDescription& description,
                               const PhysicalBinding& binding,
                               ExecutionId id) {
            (void)machine.operations.create_operation(description);
            PhysicalBindingValidator validator;
            BindingResolver resolver(machine.devices, machine.data);
            BindingAdmissibilityValidator admissibility;
            ExecutionAdmissionValidator admission(
                machine.devices, machine.data);
            GERDOS_CHECK(validator.validate(binding));
            GERDOS_CHECK(resolver.resolve(binding).fully_resolved());
            const auto* op =
                machine.operations.find_operation(description.id);
            GERDOS_CHECK(admissibility.admissible(*op, binding));
            auto* execution = machine.executions.create_execution(
                ExecutionDescription{id, description.id});
            GERDOS_CHECK(execution->bind(binding));
            GERDOS_CHECK(admission.admit(*execution).has_value());
            GERDOS_CHECK(executor.start(id));
            std::vector<AttemptStatus> outcomes;
            executor.advance(outcomes);
            while (outcomes.empty()) {
                executor.advance(outcomes);
            }
            GERDOS_CHECK(outcomes.size() == 1);
            GERDOS_CHECK(
                outcomes.front().integrity == AttemptIntegrity::COHERENT);
        };

        // Op 1: table folds from uniform: [6,1,1,1,1,1].
        run_chained(
            OperationDescription{
                OperationId{880},
                {DataId{744}},
                {DataId{740}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
            },
            bind_compute(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{744},
                    DataResidencyId{7405}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{740},
                    DataResidencyId{7401})}),
            ExecutionId{980});

        // Op 2: X = T * 0.5 - 2.5 = [0.5,-2,-2,-2,-2,-2], depending on
        // op 1's table (dependency declared, execution ordered). The
        // affine fields are {elements, passes, dst-scale, src-scale,
        // constant}: overwrite (dst-scale 0), source weight 0.5,
        // constant -2.5.
        run_chained(
            OperationDescription{
                OperationId{881},
                {DataId{740}},
                {DataId{740}},
                {OperationId{880}},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 0.5f, -2.5f},
            },
            bind_compute(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{740},
                    DataResidencyId{7401}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{740},
                    DataResidencyId{7401})}),
            ExecutionId{981});

        // Op 3: E = exp(X), depending on op 2.
        run_chained(
            OperationDescription{
                OperationId{882},
                {DataId{740}},
                {DataId{741}},
                {OperationId{881}},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::EXPONENTIAL},
            },
            bind_compute(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{740},
                    DataResidencyId{7401}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{741},
                    DataResidencyId{7402})}),
            ExecutionId{982});

        // Op 4: S = sum(E), depending on op 3.
        run_chained(
            OperationDescription{
                OperationId{883},
                {DataId{741}},
                {DataId{742}},
                {OperationId{882}},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
            },
            bind_compute(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{741},
                    DataResidencyId{7402}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{742},
                    DataResidencyId{7403})}),
            ExecutionId{983});

        // Op 5: M = max(E), depending on op 3 (sibling of op 4).
        run_chained(
            OperationDescription{
                OperationId{884},
                {DataId{741}},
                {DataId{743}},
                {OperationId{882}},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MAX},
            },
            bind_compute(
                {entry(
                    DataBindingRole::INPUT,
                    DataId{741},
                    DataResidencyId{7402}),
                 entry(
                    DataBindingRole::OUTPUT,
                    DataId{743},
                    DataResidencyId{7404})}),
            ExecutionId{984});

        // Equivalence: the chain's values equal closed-form expectations
        // computed from the seed — X[0] = 6*0.5-2.5 = 0.5, X[1..] =
        // 1*0.5-2.5 = -2. E[0] = e^0.5, E[1] = e^-2. S = e^0.5 +
        // 5*e^-2, M = e^0.5. Printed before asserting.
        const float e0 = std::exp(0.5f);
        const float e1 = std::exp(-2.0f);
        const float chained_sum = backend.sample(
            DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 0);
        const float chained_max = backend.sample(
            DataResidencyRef{DataId{743}, DataResidencyId{7404}}, 0);
        const float chained_e0 = backend.sample(
            DataResidencyRef{DataId{741}, DataResidencyId{7402}}, 0);
        std::printf(
            "chain: E0=%f S=%f M=%f (expected %f %f %f)\n",
            chained_e0,
            chained_sum,
            chained_max,
            e0,
            e0 + 5.0f * e1,
            e0);
        std::fflush(stdout);
        GERDOS_CHECK(chained_e0 == e0);
        GERDOS_CHECK(chained_sum == e0 + 5.0f * e1);
        GERDOS_CHECK(chained_max == e0);
        // The tail still reads the seed's exp: E[1] = e^-2.
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{741}, DataResidencyId{7402}},
                1) == e1);
    }

    // ---------------------------------------------------------------------
    // 17. Dtype well-formedness and gate blindness
    // ---------------------------------------------------------------------

    {
        // Per-dtype overflow guards: byte sizing is elements * width
        // under the same overflow rule. Width 1 cannot overflow, so
        // SIZE_MAX I8 elements are well-formed by the rule (there is no
        // hostile arithmetic to reject); F32 and F16 reject past their
        // widths' limits.
        WorkDescription huge_i8{6, 1, 0.0f, 1.0f, 0.0f};
        huge_i8.dtype = WorkDtype::I8;
        huge_i8.elements = ~std::size_t{0};
        GERDOS_CHECK(huge_i8.valid());

        WorkDescription huge_f32{6, 1, 0.0f, 1.0f, 0.0f};
        huge_f32.elements = ~std::size_t{0};
        GERDOS_CHECK(!huge_f32.valid());

        WorkDescription huge_f16{6, 1, 0.0f, 1.0f, 0.0f};
        huge_f16.dtype = WorkDtype::F16;
        huge_f16.elements = (~std::size_t{0}) / 2 + 1;
        GERDOS_CHECK(!huge_f16.valid());

        WorkDescription ok_f16{6, 1, 0.0f, 1.0f, 0.0f};
        ok_f16.dtype = WorkDtype::F16;
        GERDOS_CHECK(ok_f16.valid());

        // Gates are blind to dtype exactly as to form: the same
        // realization with F32 vs I8 work judges identically.
        Machine machine;
        OperationDescription f32_work{
            OperationId{870},
            {DataId{500}},
            {DataId{501}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{4, 1, 0.0f, 1.0f, 0.0f},
        };
        auto i8_work = f32_work;
        i8_work.id = OperationId{871};
        i8_work.work.dtype = WorkDtype::I8;
        const Operation f32_op{f32_work};
        const Operation i8_op{i8_work};
        BindingAdmissibilityValidator admissibility;
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
        GERDOS_CHECK(
            admissibility.admissible(f32_op, binding) ==
            admissibility.admissible(i8_op, binding));
        GERDOS_CHECK(admissibility.admissible(f32_op, binding));
    }

    return 0;
}