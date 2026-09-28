#pragma once

#include <string>
#include <vector>

#include "gerdos/core/operation.hpp"
#include "gerdos/core/workload.hpp"

namespace gerdos::adapters {

// The signal-processing vocabulary of the second consumer family. Generic
// image/signal terms only — never model semantics. Translation is
// fail-closed exactly like the model adapter: a step the algebra cannot
// express exactly is refused by name with its reason.
enum class SignalOp {
    // dst = dst * destination_scale + src * source_scale + constant.
    SIGNAL_NORMALIZE,
    // dst[i] = src[i] > threshold ? high : low. The algebra has no
    // comparison form, so this is refused with its reason.
    SIGNAL_THRESHOLD,
    // dst selects between two values by a nonzero predicate.
    SIGNAL_SELECT,
    // dst accumulates bin counts of src indices (gather into bins).
    SIGNAL_HISTOGRAM,
    // Move a representation to another home.
    SIGNAL_MOVE,
};

struct SignalStep {
    SignalOp op;
    std::uint64_t operand_a{0};
    std::uint64_t operand_b{0};
    std::uint64_t operand_c{0};
    std::uint64_t result{0};
    std::size_t elements{0};
    float destination_scale{1.0f};
    float source_scale{1.0f};
    float constant{0.0f};
    WorkDtype dtype{WorkDtype::F32};
};

struct SignalAdaptation {
    Workload workload;
    std::vector<std::string> refused;
};

// Translates signal-shaped steps into core work units. Translation is
// fail-closed: a step the algebra cannot express exactly is refused by
// name with its reason, never approximated silently.
[[nodiscard]] inline SignalAdaptation adapt_signal(
    const std::string& name,
    const std::vector<SignalStep>& steps) {
    SignalAdaptation adaptation;
    adaptation.workload.name = name;
    std::uint64_t next_id = 4000;

    for (const auto& step : steps) {
        const std::size_t before = adaptation.workload.operations.size();

        switch (step.op) {
        case SignalOp::SIGNAL_NORMALIZE: {
            if (step.elements == 0) {
                adaptation.refused.push_back(
                    "SIGNAL_NORMALIZE without an element count");
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
                        step.elements,
                        1,
                        step.destination_scale,
                        step.source_scale,
                        step.constant,
                    },
                });
            break;
        }

        case SignalOp::SIGNAL_THRESHOLD:
            // No comparison form exists: thresholding would need a
            // per-element predicate the algebra cannot produce exactly.
            adaptation.refused.push_back(
                "SIGNAL_THRESHOLD: the algebra has no comparison form"
                " — predicates cannot be produced exactly");
            break;

        case SignalOp::SIGNAL_SELECT: {
            if (step.elements == 0) {
                adaptation.refused.push_back(
                    "SIGNAL_SELECT without an element count");
                continue;
            }

            adaptation.workload.operations.push_back(
                OperationDescription{
                    OperationId{next_id++},
                    {DataId{step.operand_a},
                     DataId{step.operand_b},
                     DataId{step.operand_c}},
                    {DataId{step.result}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{
                        step.elements,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::MASK_SELECT,
                    },
                });
            break;
        }

        case SignalOp::SIGNAL_HISTOGRAM: {
            // Histogram is gather with clamping: indices address bins,
            // values accumulate per bin through the affine wrapper.
            if (step.elements == 0) {
                adaptation.refused.push_back(
                    "SIGNAL_HISTOGRAM without an element count");
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
                        step.elements,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                        WorkForm::GATHER,
                    },
                });
            break;
        }

        case SignalOp::SIGNAL_MOVE: {
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
                        step.elements,
                        1,
                        0.0f,
                        1.0f,
                        0.0f,
                    },
                });
            break;
        }
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
