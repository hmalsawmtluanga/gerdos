#include "test_check.hpp"

#include <string>
#include <vector>

#include "gerdos/adapters/model_adapter.hpp"
#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"
#include "gerdos/core/physical_binding_validation.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/cpu/cpu_backend.hpp"

namespace {

using namespace gerdos;
using gerdos::adapters::Adaptation;
using gerdos::adapters::adapt;
using gerdos::adapters::ModelOp;
using gerdos::adapters::ModelStep;

} // namespace

int main() {
    // ---------------------------------------------------------------------
    // 1. Model semantics translate into the work vocabulary
    // ---------------------------------------------------------------------

    const std::vector<ModelStep> fragment{
        ModelStep{ModelOp::LINEAR, 701, 700, 0, 702, 2, 3, 2},
        ModelStep{ModelOp::SOFTMAX, 702, 0, 702},
        ModelStep{ModelOp::ATTENTION, 701, 700, 702},
        ModelStep{ModelOp::REDUCE_SUM, 700, 0, 0, 701, 6},
        ModelStep{ModelOp::EXPONENTIAL, 700, 0, 0, 701, 6},
        ModelStep{ModelOp::REDUCE_MAX, 700, 0, 0, 701, 6},
        ModelStep{ModelOp::REDUCE_MIN, 700, 0, 0, 701, 6},
        ModelStep{ModelOp::ELEMENTWISE_MIN, 700, 701, 0, 702, 6},
        ModelStep{ModelOp::ELEMENTWISE_MAX, 700, 701, 0, 702, 6},
        ModelStep{ModelOp::MASK_SELECT, 700, 701, 702, 703, 6},
        ModelStep{ModelOp::GATHER, 700, 701, 0, 702, 6},
        ModelStep{ModelOp::MOVE, 700, 0, 0, 0, 6},
        ModelStep{ModelOp::REDUCE_MEAN, 700, 0, 0, 701, 6},
        ModelStep{ModelOp::RESIDUAL_ADD, 700, 701, 0, 702, 6},
        ModelStep{ModelOp::DIVIDE, 700, 701, 0, 702, 6},
        ModelStep{ModelOp::LAYER_NORM, 700, 0, 0, 701, 6},
    };

    const Adaptation adaptation = adapt("decoder fragment", fragment);

    GERDOS_CHECK(adaptation.workload.name == "decoder fragment");
    GERDOS_CHECK(adaptation.workload.operations.size() == 11);

    // Fail-closed translation: what the algebra cannot express exactly is
    // refused by name, never approximated silently.
    GERDOS_CHECK(adaptation.refused.size() == 5);
    GERDOS_CHECK(adaptation.refused[0].find("SOFTMAX") == 0);
    GERDOS_CHECK(adaptation.refused[1].find("ATTENTION") == 0);
    GERDOS_CHECK(adaptation.refused[2].find("RESIDUAL_ADD") == 0);
    GERDOS_CHECK(adaptation.refused[3].find("DIVIDE") == 0);
    GERDOS_CHECK(adaptation.refused[4].find("LAYER_NORM") == 0);

    const auto& linear = adaptation.workload.operations[0];
    GERDOS_CHECK(
        linear.work.form == WorkForm::MATRIX_PRODUCT);
    GERDOS_CHECK(linear.work.rows == 2);
    GERDOS_CHECK(linear.work.inner == 3);
    GERDOS_CHECK(linear.work.columns == 2);
    GERDOS_CHECK(linear.inputs.size() == 2);
    GERDOS_CHECK(linear.outputs.size() == 1);

    const auto& reduce = adaptation.workload.operations[1];
    GERDOS_CHECK(reduce.work.form == WorkForm::REDUCE_SUM);
    GERDOS_CHECK(reduce.work.elements == 6);

    const auto& raised = adaptation.workload.operations[2];
    GERDOS_CHECK(raised.work.form == WorkForm::EXPONENTIAL);
    GERDOS_CHECK(raised.work.elements == 6);

    const auto& peak = adaptation.workload.operations[3];
    GERDOS_CHECK(peak.work.form == WorkForm::REDUCE_MAX);
    GERDOS_CHECK(peak.work.elements == 6);

    const auto& floor = adaptation.workload.operations[4];
    GERDOS_CHECK(floor.work.form == WorkForm::REDUCE_MIN);
    GERDOS_CHECK(floor.work.elements == 6);

    const auto& emin = adaptation.workload.operations[5];
    GERDOS_CHECK(emin.work.form == WorkForm::ELEMENTWISE_MIN);
    GERDOS_CHECK(emin.work.elements == 6);
    GERDOS_CHECK(emin.inputs.size() == 2);

    const auto& emax = adaptation.workload.operations[6];
    GERDOS_CHECK(emax.work.form == WorkForm::ELEMENTWISE_MAX);
    GERDOS_CHECK(emax.work.elements == 6);
    GERDOS_CHECK(emax.inputs.size() == 2);

    const auto& mask = adaptation.workload.operations[7];
    GERDOS_CHECK(mask.work.form == WorkForm::MASK_SELECT);
    GERDOS_CHECK(mask.work.elements == 6);
    GERDOS_CHECK(mask.inputs.size() == 3);

    const auto& gather = adaptation.workload.operations[8];
    GERDOS_CHECK(gather.work.form == WorkForm::GATHER);
    GERDOS_CHECK(gather.work.elements == 6);
    GERDOS_CHECK(gather.inputs.size() == 2);

    const auto& move = adaptation.workload.operations[9];
    GERDOS_CHECK(
        move.work.form == WorkForm::ELEMENTWISE_AFFINE);
    GERDOS_CHECK(move.work.source_scale == 1.0f);
    GERDOS_CHECK(move.work.destination_scale == 0.0f);
    GERDOS_CHECK(move.resource_requirements[0].role ==
                  ResourceBindingRole::TRANSFER);

    // REDUCE_MEAN folds through the affine wrapper with source_scale
    // 1/N: over six 1.0 elements the mean is exactly 1.0.
    const auto& mean = adaptation.workload.operations[10];
    GERDOS_CHECK(mean.work.form == WorkForm::REDUCE_SUM);
    GERDOS_CHECK(mean.work.elements == 6);
    GERDOS_CHECK(mean.work.source_scale == 1.0f / 6.0f);
    GERDOS_CHECK(mean.work.destination_scale == 0.0f);

    // ---------------------------------------------------------------------
    // 1b. Dtype carriage: steps run in their declared dtype
    // ---------------------------------------------------------------------

    {
        using adapters::ModelStep;
        std::vector<ModelStep> typed{
            ModelStep{ModelOp::REDUCE_SUM, 700, 0, 0, 701, 6},
            ModelStep{ModelOp::ELEMENTWISE_MIN, 700, 701, 0, 702, 6},
        };
        typed[0].dtype = WorkDtype::I8;
        typed[1].dtype = WorkDtype::F16;
        const Adaptation carried = adapt("typed", typed);
        GERDOS_CHECK(carried.refused.empty());
        GERDOS_CHECK(carried.workload.operations.size() == 2);
        GERDOS_CHECK(
            carried.workload.operations[0].work.dtype == WorkDtype::I8);
        GERDOS_CHECK(
            carried.workload.operations[1].work.dtype == WorkDtype::F16);
        // Untyped steps default to F32.
        GERDOS_CHECK(
            adaptation.workload.operations[1].work.dtype == WorkDtype::F32);
    }

    // ---------------------------------------------------------------------
    // 2. A translated step executes on real hardware
    // ---------------------------------------------------------------------

    {
        DeviceRegistry devices;
        DataRegistry data;
        OperationRegistry operations;
        ExecutionRegistry executions;

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

        auto* partial = data.create_data(
            DataDescription{DataId{701}, "partial"});
        (void)partial->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{702},
                    DataId{701},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "partial",
                },
            });

        auto* source = data.create_data(
            DataDescription{DataId{700}, "source"});
        (void)source->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{701},
                    DataId{700},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "source",
                },
            });

        // Consuming representations must be usable.
        (void)source->find_residency(DataResidencyId{701})
            ->set_state(DataResidencyState::VALID);

        auto* product = data.create_data(
            DataDescription{DataId{702}, "product"});
        (void)product->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{703},
                    DataId{702},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "product",
                },
            });

        // Seed the non-uniform operand through the algebra itself:
        // reduce the uniform source into the partial record first.
        const std::vector<ModelStep> seeding{
            ModelStep{ModelOp::REDUCE_SUM, 700, 0, 0, 701, 6},
            ModelStep{ModelOp::LINEAR, 701, 700, 0, 702, 2, 3, 2},
        };

        const Adaptation plan = adapt("projection", seeding);
        GERDOS_CHECK(plan.refused.empty());
        GERDOS_CHECK(plan.workload.operations.size() == 2);

        CpuBackend backend;
        Executor executor(
            executions, operations, devices, data, backend);

        for (const auto& description : plan.workload.operations) {
            (void)operations.create_operation(description);

            // The adapter's declared work drives the binding shape.
            PhysicalBinding binding;

            for (const auto& input : description.inputs) {
                binding.data.push_back(
                    DataBinding{
                        DataBindingRole::INPUT,
                        DataResidencyRef{
                            input,
                            DataResidencyId{input.value() + 1},
                        },
                    });
            }

            for (const auto& output : description.outputs) {
                binding.data.push_back(
                    DataBinding{
                        DataBindingRole::OUTPUT,
                        DataResidencyRef{
                            output,
                            DataResidencyId{output.value() + 1},
                        },
                    });
            }

            binding.resources.push_back(
                ResourceBinding{
                    ResourceBindingRole::COMPUTE,
                    ResourceRef{DeviceId{100}, ResourceId{102}},
                });

            auto* execution = executions.create_execution(
                ExecutionDescription{
                    ExecutionId{1000 + description.id.value()},
                    description.id,
                });

            GERDOS_CHECK(execution->bind(binding));
            PhysicalBindingValidator validator;
            BindingResolver resolver(devices, data);
            BindingAdmissibilityValidator admissibility;
            ExecutionAdmissionValidator admission(devices, data);
            // The translated step passes the same gates as any other
            // realization before the executor accepts it.
            GERDOS_CHECK(validator.validate(binding));
            GERDOS_CHECK(resolver.resolve(binding).fully_resolved());
            GERDOS_CHECK(
                admissibility.admissible(
                    *operations.find_operation(description.id),
                    binding));
            GERDOS_CHECK(executor.start(execution->description().id));

            std::vector<AttemptStatus> outcomes;
            executor.advance(outcomes);

            while (outcomes.empty()) {
                executor.advance(outcomes);
            }

            GERDOS_CHECK(outcomes.size() == 1);
            GERDOS_CHECK(
                outcomes.front().integrity ==
                AttemptIntegrity::COHERENT);
        }

        // The translated projection computed exactly what the model
        // semantics asked for: A = [[6,1,1],[1,1,1]], B = ones(3,2),
        // C = [[8,8],[3,3]].
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{703}},
                0) == 8.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{703}},
                2) == 3.0f);
    }

    // ---------------------------------------------------------------------
    // 3. The expressible softmax prefix chains; SOFTMAX stays refused
    // ---------------------------------------------------------------------

    {
        // SOFTMAX as a composite is refused — but its expressible
        // prefix (EXPONENTIAL + REDUCE_SUM) adapts to chained steps
        // that execute in dependency order with exact values. The
        // refusal names the missing divide; the prefix proves the
        // chain machinery the full composite will ride on.
        const std::vector<ModelStep> prefix{
            ModelStep{ModelOp::EXPONENTIAL, 700, 0, 0, 701, 6},
            ModelStep{ModelOp::REDUCE_SUM, 701, 0, 0, 702, 6},
            ModelStep{ModelOp::SOFTMAX, 700, 0, 0, 701, 6},
        };
        const Adaptation chain = adapt("softmax prefix", prefix);
        GERDOS_CHECK(chain.workload.operations.size() == 2);
        GERDOS_CHECK(chain.refused.size() == 1);
        GERDOS_CHECK(chain.refused[0].find("SOFTMAX") == 0);
        GERDOS_CHECK(
            chain.refused[0].find("divide-by-sum") != std::string::npos);
        GERDOS_CHECK(
            chain.workload.operations[0].work.form == WorkForm::EXPONENTIAL);
        GERDOS_CHECK(
            chain.workload.operations[1].work.form == WorkForm::REDUCE_SUM);
        // The prefix chain declares the data dependency: reduce consumes
        // what exp produces.
        GERDOS_CHECK(chain.workload.operations[0].outputs[0] == DataId{701});
        GERDOS_CHECK(chain.workload.operations[1].inputs[0] == DataId{701});
    }

    return 0;
}