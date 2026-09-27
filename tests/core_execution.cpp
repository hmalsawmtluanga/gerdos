#include "test_check.hpp"

#include "gerdos/core/execution.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/physical_binding.hpp"

namespace {

using namespace gerdos;

// Minimal admission fixture: one device with one available resource and one
// Data with one usable residency. Admission is mechanism, so attempts reach
// RUNNING only through it.
struct Fixture {
    DeviceRegistry devices;
    DataRegistry data;
    ExecutionAdmissionValidator admission{devices, data};

    Fixture() {
        auto* device = devices.create_device(
            DeviceDescription{
                DeviceId{1},
                "device",
            });

        (void)device->add_resource(
            Resource{
                ResourceDescription{
                    ResourceId{1},
                    DeviceId{1},
                    ResourceKind::COMPUTE,
                    "compute",
                },
            });

        device->find_resource(ResourceId{1})
            ->set_availability(ResourceAvailability::AVAILABLE);

        auto* datum = data.create_data(
            DataDescription{
                DataId{1},
                "data",
            });

        (void)datum->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{1},
                    DataId{1},
                    ResourceRef{
                        DeviceId{1},
                        ResourceId{1},
                    },
                    "input",
                },
            });

        (void)datum->find_residency(DataResidencyId{1})
            ->set_state(DataResidencyState::VALID);
    }

    [[nodiscard]] bool start_attempt(Execution& execution) {
        const auto verdict = admission.admit(execution);

        return verdict.has_value() &&
               admission.establish(execution, *verdict) &&
               execution.set_state(ExecutionState::RUNNING);
    }
};

PhysicalBinding fixture_binding() {
    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{1},
                DataResidencyId{1},
            },
        });

    return binding;
}

} // namespace

int main() {
    using namespace gerdos;

    Fixture fixture;

    const ExecutionDescription description{
        ExecutionId{100},
        OperationId{42},
    };

    Execution execution(description);

    GERDOS_CHECK(execution.description().id == ExecutionId{100});
    GERDOS_CHECK(execution.description().operation == OperationId{42});
    GERDOS_CHECK(execution.state() == ExecutionState::PENDING);
    GERDOS_CHECK(!execution.admitted());

    // The RUNNING transition is gated on admission evidence: a bound but
    // unadmitted attempt cannot start.
    GERDOS_CHECK(execution.bind(fixture_binding()));
    GERDOS_CHECK(!execution.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(execution.state() == ExecutionState::PENDING);

    GERDOS_CHECK(fixture.start_attempt(execution));
    GERDOS_CHECK(execution.admitted());
    GERDOS_CHECK(execution.state() == ExecutionState::RUNNING);

    GERDOS_CHECK(execution.set_state(ExecutionState::COMPLETED));
    GERDOS_CHECK(execution.state() == ExecutionState::COMPLETED);

    GERDOS_CHECK(!execution.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(execution.state() == ExecutionState::COMPLETED);

    // A terminal execution records its result exactly once.
    GERDOS_CHECK(!execution.has_result());
    GERDOS_CHECK(execution.result() == nullptr);

    GERDOS_CHECK(
        execution.record_result(
            ExecutionResult{ExecutionState::COMPLETED, {}}));
    GERDOS_CHECK(execution.has_result());
    GERDOS_CHECK(
        execution.result()->outcome == ExecutionState::COMPLETED);
    GERDOS_CHECK(
        execution.result()->integrity == AttemptIntegrity::COHERENT);

    // Result recording is rejected before a terminal state is reached and
    // for an outcome other than the one reached.
    Execution failed_attempt{
        ExecutionDescription{
            ExecutionId{101},
            OperationId{42},
        }};

    GERDOS_CHECK(
        !failed_attempt.record_result(
            ExecutionResult{ExecutionState::FAILED, "too early"}));
    GERDOS_CHECK(!failed_attempt.has_result());

    GERDOS_CHECK(failed_attempt.bind(fixture_binding()));
    GERDOS_CHECK(fixture.start_attempt(failed_attempt));
    GERDOS_CHECK(failed_attempt.set_state(ExecutionState::FAILED));

    GERDOS_CHECK(
        !failed_attempt.record_result(
            ExecutionResult{ExecutionState::COMPLETED, "mismatched"}));
    GERDOS_CHECK(!failed_attempt.has_result());

    GERDOS_CHECK(
        failed_attempt.record_result(
            ExecutionResult{ExecutionState::FAILED, "device reset"}));

    // The result is immutable after recording.
    GERDOS_CHECK(
        !failed_attempt.record_result(
            ExecutionResult{ExecutionState::FAILED, "rewritten"}));
    GERDOS_CHECK(failed_attempt.result()->detail == "device reset");
    GERDOS_CHECK(failed_attempt.result()->outcome == ExecutionState::FAILED);

    // An attempt whose effects were rejected records its incoherence.
    Execution incoherent_attempt{
        ExecutionDescription{
            ExecutionId{103},
            OperationId{42},
        }};

    GERDOS_CHECK(incoherent_attempt.bind(fixture_binding()));
    GERDOS_CHECK(fixture.start_attempt(incoherent_attempt));
    GERDOS_CHECK(incoherent_attempt.set_state(ExecutionState::COMPLETED));

    GERDOS_CHECK(
        incoherent_attempt.record_result(
            ExecutionResult{
                ExecutionState::COMPLETED,
                {},
                AttemptIntegrity::EFFECTS_REJECTED,
            }));

    GERDOS_CHECK(
        incoherent_attempt.result()->integrity ==
        AttemptIntegrity::EFFECTS_REJECTED);

    // A cancelled attempt records its own result.
    Execution cancelled_attempt{
        ExecutionDescription{
            ExecutionId{102},
            OperationId{42},
        }};

    GERDOS_CHECK(
        cancelled_attempt.set_state(ExecutionState::CANCELLED));

    GERDOS_CHECK(
        cancelled_attempt.record_result(
            ExecutionResult{ExecutionState::CANCELLED, {}}));

    GERDOS_CHECK(
        cancelled_attempt.result()->outcome ==
        ExecutionState::CANCELLED);

    return 0;
}