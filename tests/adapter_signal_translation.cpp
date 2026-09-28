#include "test_check.hpp"

#include <string>
#include <vector>

#include "gerdos/adapters/signal_adapter.hpp"

int main() {
    using namespace gerdos;
    using gerdos::adapters::adapt_signal;
    using gerdos::adapters::SignalAdaptation;
    using gerdos::adapters::SignalOp;
    using gerdos::adapters::SignalStep;

    // ---------------------------------------------------------------------
    // 1. Signal semantics translate into the work vocabulary
    // ---------------------------------------------------------------------

    const std::vector<SignalStep> chain{
        SignalStep{SignalOp::SIGNAL_NORMALIZE, 900, 0, 0, 901, 6, 0.0f, 0.5f, 0.25f},
        SignalStep{SignalOp::SIGNAL_THRESHOLD, 901, 0, 0, 910, 6},
        SignalStep{SignalOp::SIGNAL_SELECT, 910, 911, 912, 913, 6},
        SignalStep{SignalOp::SIGNAL_HISTOGRAM, 913, 920, 0, 921, 6},
        SignalStep{SignalOp::SIGNAL_MOVE, 901, 0, 0, 0, 6},
    };

    const SignalAdaptation adaptation = adapt_signal("signal chain", chain);

    GERDOS_CHECK(adaptation.workload.name == "signal chain");
    GERDOS_CHECK(adaptation.workload.operations.size() == 4);

    // Fail-closed: thresholding needs a comparison form that does not
    // exist — refused by name, never approximated silently.
    GERDOS_CHECK(adaptation.refused.size() == 1);
    GERDOS_CHECK(adaptation.refused[0].find("SIGNAL_THRESHOLD") == 0);

    const auto& normalize = adaptation.workload.operations[0];
    GERDOS_CHECK(normalize.work.form == WorkForm::ELEMENTWISE_AFFINE);
    GERDOS_CHECK(normalize.work.elements == 6);
    GERDOS_CHECK(normalize.work.source_scale == 0.5f);
    GERDOS_CHECK(normalize.work.constant == 0.25f);
    GERDOS_CHECK(normalize.inputs.size() == 1);

    const auto& select = adaptation.workload.operations[1];
    GERDOS_CHECK(select.work.form == WorkForm::MASK_SELECT);
    GERDOS_CHECK(select.work.elements == 6);
    GERDOS_CHECK(select.inputs.size() == 3);

    const auto& histogram = adaptation.workload.operations[2];
    GERDOS_CHECK(histogram.work.form == WorkForm::GATHER);
    GERDOS_CHECK(histogram.work.elements == 6);
    GERDOS_CHECK(histogram.inputs.size() == 2);

    const auto& move = adaptation.workload.operations[3];
    GERDOS_CHECK(move.work.form == WorkForm::ELEMENTWISE_AFFINE);
    GERDOS_CHECK(move.resource_requirements[0].role == ResourceBindingRole::TRANSFER);

    // ---------------------------------------------------------------------
    // 2. Zero counts are refused by name
    // ---------------------------------------------------------------------

    {
        const std::vector<SignalStep> empty{
            SignalStep{SignalOp::SIGNAL_NORMALIZE, 900, 0, 0, 901, 0},
            SignalStep{SignalOp::SIGNAL_SELECT, 910, 911, 912, 913, 0},
            SignalStep{SignalOp::SIGNAL_HISTOGRAM, 913, 920, 0, 921, 0},
        };
        const SignalAdaptation refused = adapt_signal("empty", empty);
        GERDOS_CHECK(refused.workload.operations.empty());
        GERDOS_CHECK(refused.refused.size() == 3);
    }

    return 0;
}
