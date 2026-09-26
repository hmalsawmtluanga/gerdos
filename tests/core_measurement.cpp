#include "test_check.hpp"

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
    std::size_t concurrent = 0) {
    return MeasurementObservation{
        MeasurementQuantity::DURATION_NS,
        subject,
        OperationId{700},
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
    const auto second = measurements.record(observation(subject_a, 250, 2));
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
                1,
                {},
            }).valid());

    GERDOS_CHECK(
        !measurements.record(
            MeasurementObservation{
                static_cast<MeasurementQuantity>(255),
                subject_a,
                OperationId{700},
                1,
                {},
            }).valid());

    GERDOS_CHECK(measurements.count() == 3);

    // ---------------------------------------------------------------------
    // 2. Queries report evidence and its support
    // ---------------------------------------------------------------------

    const auto summary_a =
        measurements.summarize(subject_a, MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(summary_a.observations == 2);
    GERDOS_CHECK(summary_a.total_value == 350);
    GERDOS_CHECK(summary_a.latest_value == 250);
    GERDOS_CHECK(summary_a.latest == second);

    const auto summary_b =
        measurements.summarize(subject_b, MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(summary_b.observations == 1);
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

    // Records are immutable evidence: repeated queries agree.
    const auto summary_again =
        measurements.summarize(subject_a, MeasurementQuantity::DURATION_NS);
    GERDOS_CHECK(summary_again.observations == summary_a.observations);
    GERDOS_CHECK(summary_again.total_value == summary_a.total_value);

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

    const Operation operation{OperationDescription{
        OperationId{700},
        {},
        {},
        {},
        {},
    }};

    MeasurementCollector::capture(
        captured,
        execution,
        operation,
        1'000'000,
        1);

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
        });

    GERDOS_CHECK(conditions_seen == 1);

    // An attempt without a binding is not observed.
    Execution unbound{
        ExecutionDescription{
            ExecutionId{801},
            OperationId{700},
        }};

    MeasurementCollector::capture(
        captured,
        unbound,
        operation,
        1,
        0);

    GERDOS_CHECK(captured.count() == 2);

    // Capture never mutates the attempt.
    GERDOS_CHECK(execution.state() == ExecutionState::PENDING);

    return 0;
}