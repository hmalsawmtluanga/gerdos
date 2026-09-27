#pragma once

#include <string>
#include <vector>

#include "gerdos/core/operation.hpp"
#include "gerdos/core/workload.hpp"

namespace gerdos::adapters {

// The model-semantic vocabulary of the adapter layer. This is the one place
// model terms are allowed to exist: adapters translate them into the core's
// generic work vocabulary, and the core never sees them.
enum class ModelOp {
    // C = A x B, a projection of two operands into a product.
    LINEAR,
    // dst = dst * scale + operand * operand_scale + constant.
    AFFINE,
    // dst[0] accumulates the sum of an operand.
    REDUCE_SUM,
    // dst transforms the elementwise exponential of an operand.
    EXPONENTIAL,
    // dst[0] accumulates the maximum of an operand.
    REDUCE_MAX,
    // dst[0] accumulates the minimum of an operand.
    REDUCE_MIN,
    // dst selects the elementwise minimum of two operands.
    ELEMENTWISE_MIN,
    // dst selects the elementwise maximum of two operands.
    ELEMENTWISE_MAX,
    // dst selects between two values by a nonzero predicate.
    MASK_SELECT,
    // dst gathers table values by a clamped index table.
    GATHER,
    // Move a representation to another home.
    MOVE,
    // Refused by the current algebra — kept so refusal is explicit.
    SOFTMAX,
    ATTENTION,
};

struct ModelStep {
    ModelOp op;
    std::uint64_t operand_a{0};
    std::uint64_t operand_b{0};
    std::uint64_t operand_c{0};
    std::uint64_t result{0};
    std::size_t rows{0};
    std::size_t inner{0};
    std::size_t columns{0};
    float destination_scale{1.0f};
    float source_scale{1.0f};
    float constant{0.0f};
    WorkDtype dtype{WorkDtype::F32};
};

struct Adaptation {
    Workload workload;
    std::vector<std::string> refused;
};

// Translates model-shaped steps into core work units. Translation is
// fail-closed: a step the algebra cannot express exactly is refused by
// name with its reason, never approximated silently.
[[nodiscard]] inline Adaptation adapt(
    const std::string& name,
    const std::vector<ModelStep>& steps) {
    Adaptation adaptation;
    adaptation.workload.name = name;

    std::uint64_t next_id = 4000;

    for (const auto& step : steps) {
        const std::size_t before = adaptation.workload.operations.size();
        switch (step.op) {
        case ModelOp::LINEAR: {
            if (step.rows == 0 || step.inner == 0 ||
                step.columns == 0) {
                adaptation.refused.push_back(
                    "LINEAR without a complete shape");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}, DataId{step.operand_b}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        0,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::MATRIX_PRODUCT,
                        step.rows,
                        step.inner,
                        step.columns,
                    },
                });
            break;
        }

        case ModelOp::AFFINE: {
            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}, DataId{step.operand_b}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        step.destination_scale,
                        step.source_scale,
                        step.constant,
                    },
                });
            break;
        }

        case ModelOp::REDUCE_SUM: {
            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::REDUCE_SUM,
                    },
                });
            break;
        }

        case ModelOp::EXPONENTIAL: {
            if (step.rows == 0) {
                adaptation.refused.push_back(
                    "EXPONENTIAL without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::EXPONENTIAL,
                    },
                });
            break;
        }

        case ModelOp::REDUCE_MAX: {
            if (step.rows == 0) {
                adaptation.refused.push_back(
                    "REDUCE_MAX without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::REDUCE_MAX,
                    },
                });
            break;
        }

        case ModelOp::REDUCE_MIN: {
            if (step.rows == 0) {
                adaptation.refused.push_back(
                    "REDUCE_MIN without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::REDUCE_MIN,
                    },
                });
            break;
        }

        case ModelOp::ELEMENTWISE_MIN: {
            if (step.rows == 0) {
                adaptation.refused.push_back(
                    "ELEMENTWISE_MIN without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}, DataId{step.operand_b}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::ELEMENTWISE_MIN,
                    },
                });
            break;
        }

        case ModelOp::ELEMENTWISE_MAX: {
            if (step.rows == 0) {
                adaptation.refused.push_back(
                    "ELEMENTWISE_MAX without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}, DataId{step.operand_b}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::ELEMENTWISE_MAX,
                    },
                });
            break;
        }

        case ModelOp::MASK_SELECT: {
            if (step.rows == 0) {
                adaptation.refused.push_back(
                    "MASK_SELECT without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {
                        DataId{step.operand_a},
                        DataId{step.operand_b},
                        DataId{step.operand_c},
                    },
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::MASK_SELECT,
                    },
                });
            break;
        }

        case ModelOp::GATHER: {
            if (step.rows == 0) {
                adaptation.refused.push_back(
                    "GATHER without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}, DataId{step.operand_b}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::GATHER,
                    },
                });
            break;
        }

        case ModelOp::MOVE: {
            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a}},
                    {DataId{step.operand_a}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::TRANSFER,
                            1,
                        },
                    },
                    WorkDescription{
                        step.rows,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                    },
                });
            break;
        }

        case ModelOp::SOFTMAX:
            adaptation.refused.push_back(
                "SOFTMAX: the algebra has no normalization yet — "
                "compose EXPONENTIAL with REDUCE_SUM instead");
            break;

        case ModelOp::ATTENTION:
            adaptation.refused.push_back(
                "ATTENTION: depends on SOFTMAX, which the algebra "
                "cannot express exactly");
            break;
        }

        // Dtype carriage: the emitted work (if any) runs in the step's
        // dtype. Refused steps `continue` before pushing, so only
        // accepted steps are stamped.
        if (adaptation.workload.operations.size() == before + 1) {
            adaptation.workload.operations.back().work.dtype = step.dtype;
        }
    }

    return adaptation;
}

} // namespace gerdos::adapters