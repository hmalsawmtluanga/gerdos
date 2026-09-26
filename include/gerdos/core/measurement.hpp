#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gerdos/core/ids.hpp"

namespace gerdos {

// The measured quantity. The first quantity is the observed duration of one
// execution attempt in nanoseconds; further quantities are introduced
// together with the producers that can measure them.
enum class MeasurementQuantity {
    DURATION_NS,
};

// The conditions under which an observation was made. Concurrent attempts
// distinguish evidence gathered under contention from evidence gathered on
// an idle subject.
struct MeasurementConditions {
    std::size_t concurrent_attempts{0};
};

// One observation, without identity. A record is evidence: it is never
// rewritten after it is recorded.
struct MeasurementObservation {
    MeasurementQuantity quantity;
    ResourceRef subject;
    OperationId operation;
    std::uint64_t value;
    MeasurementConditions conditions;
};

// Evidence of one observation. Identifiers are allocated in observation
// order, so identifier order is observation order.
struct MeasurementRecord {
    MeasurementId id;
    MeasurementObservation observation;
};

// Query evidence by subject and quantity. Confidence is reported as the
// supporting observation count and the recency of the latest observation,
// not stored per record.
struct MeasurementSummary {
    std::uint64_t observations{0};
    std::uint64_t total_value{0};
    std::uint64_t latest_value{0};
    MeasurementId latest{};
};

// The measurement registry is an append-only evidence log. Identifiers are
// allocated in observation order and are never reused; records are immutable
// and are never removed. Recording is rejected when the observation is
// structurally invalid.
class MeasurementRegistry {
public:
    [[nodiscard]] MeasurementId record(
        const MeasurementObservation& observation) {
        if (!observation.subject.valid()) {
            return MeasurementId{};
        }

        if (!valid_quantity(observation.quantity)) {
            return MeasurementId{};
        }

        const MeasurementId id{next_id_};
        ++next_id_;

        records_.push_back(
            MeasurementRecord{
                id,
                observation,
            });

        return id;
    }

    [[nodiscard]] std::size_t count() const noexcept {
        return records_.size();
    }

    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (const auto& record : records_) {
            fn(record);
        }
    }

    template <typename Fn>
    void for_each(
        ResourceRef subject,
        MeasurementQuantity quantity,
        Fn&& fn) const {
        for (const auto& record : records_) {
            if (record.observation.subject == subject &&
                record.observation.quantity == quantity) {
                fn(record);
            }
        }
    }

    [[nodiscard]] MeasurementSummary summarize(
        ResourceRef subject,
        MeasurementQuantity quantity) const noexcept {
        MeasurementSummary summary;

        for (const auto& record : records_) {
            if (record.observation.subject != subject ||
                record.observation.quantity != quantity) {
                continue;
            }

            ++summary.observations;
            summary.total_value += record.observation.value;
            summary.latest_value = record.observation.value;
            summary.latest = record.id;
        }

        return summary;
    }

private:
    [[nodiscard]] static constexpr bool valid_quantity(
        MeasurementQuantity quantity) noexcept {
        switch (quantity) {
        case MeasurementQuantity::DURATION_NS:
            return true;
        default:
            return false;
        }
    }

    std::vector<MeasurementRecord> records_;
    std::uint64_t next_id_{1};
};

} // namespace gerdos