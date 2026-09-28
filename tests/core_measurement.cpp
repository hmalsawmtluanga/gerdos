#include "test_check.hpp"

#include <limits>
#include <type_traits>
#include <vector>

#include "gerdos/core/measurement_collector.hpp"

namespace {

using namespace gerdos;

const ResourceRef subject_a{DeviceId{1}, ResourceId{10}};
const ResourceRef subject_b{DeviceId{1}, ResourceId{11}};

MeasurementObservation observation(
    ResourceRef subject,
    std::uint64_t value,
    bool succeeded = true,
    std::size_t concurrent = 0) {
    return MeasurementObservation{
        MeasurementQuantity::DURATION_NS,
        subject,
        OperationId{700},
        ExecutionId{800},
        succeeded,
        value,
        MeasurementConditions{
            concurrent,
        },
    };
}

} // namespace

int main() {
    using namespace gerdos;

    // ---------------------------------------------------------------------
    // 1. The registry is an append-only evidence log
    // ---------------------------------------------------------------------

    MeasurementRegistry measurements;

    const auto first = measurements.record(observation(subject_a, 100));
    const auto second =
        measurements.record(observation(subject_a, 250, false, 2));
    const auto third = measurements.record(observation(subject_b, 40));

    GERDOS_CHECK(first.valid());
    GERDOS_CHECK(second.valid());
    GERDOS_CHECK(third.valid());
    GERDOS_CHECK(measurements.count() == 3);

    // Identifiers are allocated in observation order: identifier order is
    // observation order.
    GERDOS_CHECK(first.value() < second.value());
    GERDOS_CHECK(second.value() < third.value());

    // Structurally invalid observations are rejected without allocating.
    GERDOS_CHECK(
        !measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                ResourceRef{},
                OperationId{700},
                ExecutionId{800},
                true,
                1,
                {},
            }).valid());

    GERDOS_CHECK(
        !measurements.record(
            MeasurementObservation{
                static_cast<MeasurementQuantity>(255),
                subject_a,
                OperationId{700},
                ExecutionId{800},
                true,
                1,
                {},
            }).valid());

    GERDOS_CHECK(
        !measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                subject_a,
                OperationId{},
                ExecutionId{800},
                true,
                1,
                {},
            }).valid());

    GERDOS_CHECK(
        !measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                subject_a,
                OperationId{700},
                ExecutionId{},
                true,
                1,
                {},
            }).valid());

    GERDOS_CHECK(measurements.count() == 3);

    // The registry is evidence: it cannot be copied, moved, or reset.
    static_assert(
        !std::is_copy_constructible_v<MeasurementRegistry>);
    static_assert(!std::is_copy_assignable_v<MeasurementRegistry>);
    static_assert(
        !std::is_move_constructible_v<MeasurementRegistry>);
    static_assert(!std::is_move_assignable_v<MeasurementRegistry>);

    // ---------------------------------------------------------------------
    // 2. Queries report evidence, its support, and its outcome split
    // ---------------------------------------------------------------------

    const auto summary_a =
        measurements.summarize(subject_a, MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(summary_a.observations == 2);
    GERDOS_CHECK(summary_a.succeeded_observations == 1);
    GERDOS_CHECK(summary_a.total_value == 350);
    GERDOS_CHECK(summary_a.succeeded_total_value == 100);
    GERDOS_CHECK(summary_a.latest_value == 250);
    GERDOS_CHECK(summary_a.latest == second);

    const auto summary_b =
        measurements.summarize(subject_b, MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(summary_b.observations == 1);
    GERDOS_CHECK(summary_b.succeeded_observations == 1);
    GERDOS_CHECK(summary_b.latest == third);

    const auto summary_c =
        measurements.summarize(
            ResourceRef{
                DeviceId{9},
                ResourceId{9},
            },
            MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(summary_c.observations == 0);
    GERDOS_CHECK(!summary_c.latest.valid());

    std::size_t all = 0;
    measurements.for_each([&](const MeasurementRecord&) { ++all; });
    GERDOS_CHECK(all == 3);

    std::size_t filtered = 0;
    std::uint64_t filtered_total = 0;
    measurements.for_each(
        subject_a,
        MeasurementQuantity::DURATION_NS,
        [&](const MeasurementRecord& record) {
            ++filtered;
            filtered_total += record.observation.value;
        });

    GERDOS_CHECK(filtered == 2);
    GERDOS_CHECK(filtered_total == 350);

    // Records are immutable evidence: queries are stable and repeatable.
    const auto summary_again =
        measurements.summarize(subject_a, MeasurementQuantity::DURATION_NS);
    GERDOS_CHECK(summary_again.observations == summary_a.observations);
    GERDOS_CHECK(summary_again.total_value == summary_a.total_value);
    GERDOS_CHECK(measurements.count() == 3);

    // Totals saturate instead of wrapping.
    MeasurementRegistry saturated;
    const auto maximum = std::numeric_limits<std::uint64_t>::max();

    GERDOS_CHECK(
        saturated.record(observation(subject_a, maximum)).valid());
    GERDOS_CHECK(
        saturated.record(observation(subject_a, maximum)).valid());

    const auto saturated_summary =
        saturated.summarize(subject_a, MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(saturated_summary.observations == 2);
    GERDOS_CHECK(saturated_summary.total_value == maximum);
    GERDOS_CHECK(saturated_summary.succeeded_total_value == maximum);

    // ---------------------------------------------------------------------
    // 3. Capture observes each bound subject once per attempt
    // ---------------------------------------------------------------------

    MeasurementRegistry captured;

    Execution execution{
        ExecutionDescription{
            ExecutionId{800},
            OperationId{700},
        }};

    PhysicalBinding binding;
    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            subject_a,
        });

    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            subject_a,
        });

    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::TRANSFER,
            subject_b,
        });

    GERDOS_CHECK(execution.bind(std::move(binding)));

    GERDOS_CHECK(
        MeasurementCollector::capture(
            captured,
            execution,
            OperationId{700},
            true,
            1'000'000,
            1));

    // Two distinct subjects observed once each despite three entries.
    GERDOS_CHECK(captured.count() == 2);

    const auto captured_a =
        captured.summarize(subject_a, MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(captured_a.observations == 1);
    GERDOS_CHECK(captured_a.latest_value == 1'000'000);

    std::size_t conditions_seen = 0;
    captured.for_each(
        subject_a,
        MeasurementQuantity::DURATION_NS,
        [&](const MeasurementRecord& record) {
            conditions_seen = record.observation.conditions.concurrent_attempts;
            GERDOS_CHECK(
                record.observation.operation == OperationId{700});
            GERDOS_CHECK(
                record.observation.attempt == ExecutionId{800});
            GERDOS_CHECK(record.observation.succeeded);
        });

    GERDOS_CHECK(conditions_seen == 1);

    // An attempt without a binding produces no observations.
    Execution unbound{
        ExecutionDescription{
            ExecutionId{801},
            OperationId{700},
        }};

    GERDOS_CHECK(
        MeasurementCollector::capture(
            captured,
            unbound,
            OperationId{700},
            true,
            1,
            0));

    GERDOS_CHECK(captured.count() == 2);

    // Capture never mutates the attempt: state and binding are untouched.
    GERDOS_CHECK(execution.state() == ExecutionState::PENDING);
    GERDOS_CHECK(execution.has_binding());
    GERDOS_CHECK(execution.binding()->resources.size() == 3);

    // ---------------------------------------------------------------------
    // Rotation: a fresh registry starts empty and grows independently
    // ---------------------------------------------------------------------

    {
        // Long-running operators rotate registries per workload round
        // rather than growing one log forever: the old registry keeps
        // its records (nothing is migrated silently), the new one
        // starts empty, and planning reads only what it is handed.
        MeasurementRegistry rotated;
        GERDOS_CHECK(rotated.count() == 0);
        GERDOS_CHECK(
            rotated.record(observation(subject_a, 50)).valid());
        GERDOS_CHECK(rotated.count() == 1);
        GERDOS_CHECK(measurements.count() == 3);
    }

    return 0;
}