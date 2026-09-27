#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
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
// rewritten after it is recorded. The attempt identifies the execution the
// observation belongs to; succeeded reports whether the attempt's work
// succeeded.
struct MeasurementObservation {
    MeasurementQuantity quantity;
    ResourceRef subject;
    OperationId operation;
    ExecutionId attempt;
    bool succeeded;
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
// not stored per record; successful evidence is counted separately so that
// planning can prefer it over time-to-failure evidence.
struct MeasurementSummary {
    std::uint64_t observations{0};
    std::uint64_t succeeded_observations{0};
    std::uint64_t total_value{0};
    std::uint64_t succeeded_total_value{0};
    std::uint64_t latest_value{0};
    MeasurementId latest{};
};

// The measurement registry is an append-only evidence log. Identifiers are
// allocated in observation order and are never reused; records are immutable
// and are never removed. Recording is rejected when the observation is
// structurally invalid. Query cost is proportional to the evidence for the
// queried subject; retention of old evidence is future policy.
class MeasurementRegistry {
public:
    MeasurementRegistry() = default;

    MeasurementRegistry(const MeasurementRegistry&) = delete;
    MeasurementRegistry& operator=(const MeasurementRegistry&) = delete;
    MeasurementRegistry(MeasurementRegistry&&) = delete;
    MeasurementRegistry& operator=(MeasurementRegistry&&) = delete;

    [[nodiscard]] MeasurementId record(
        const MeasurementObservation& observation) {
        if (!observation.subject.valid() ||
            !observation.operation.valid() ||
            !observation.attempt.valid() ||
            !valid_quantity(observation.quantity)) {
            return MeasurementId{};
        }

        const MeasurementId id{next_id_};

        records_.push_back(
            MeasurementRecord{
                id,
                observation,
            });

        // The record and its index entry commit together: an index failure
        // rolls the record back so no identifier is consumed and no record
        // is left unindexed.
        try {
            subject_index_[observation.subject].push_back(
                records_.size() - 1);
        } catch (...) {
            records_.pop_back();

            const auto it = subject_index_.find(observation.subject);

            if (it != subject_index_.end() && it->second.empty()) {
                subject_index_.erase(it);
            }

            throw;
        }

        ++next_id_;

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
        for (const auto index : indices_for(subject)) {
            const auto& record = records_[index];

            if (record.observation.quantity == quantity) {
                fn(record);
            }
        }
    }

    [[nodiscard]] MeasurementSummary summarize(
        ResourceRef subject,
        MeasurementQuantity quantity) const noexcept {
        return summarize(subject, quantity, OperationId{}, false);
    }

    // Like-for-like evidence: the observations of one attempted Operation on
    // one subject. Comparisons across operations are not meaningful.
    [[nodiscard]] MeasurementSummary summarize(
        ResourceRef subject,
        MeasurementQuantity quantity,
        OperationId operation) const noexcept {
        return summarize(subject, quantity, operation, true);
    }

private:
    [[nodiscard]] MeasurementSummary summarize(
        ResourceRef subject,
        MeasurementQuantity quantity,
        OperationId operation,
        bool match_operation) const noexcept {
        MeasurementSummary summary;

        for (const auto index : indices_for(subject)) {
            const auto& record = records_[index];

            if (record.observation.quantity != quantity) {
                continue;
            }

            if (match_operation &&
                record.observation.operation != operation) {
                continue;
            }

            ++summary.observations;
            summary.total_value =
                saturating_add(summary.total_value, record.observation.value);
            summary.latest_value = record.observation.value;
            summary.latest = record.id;

            if (record.observation.succeeded) {
                ++summary.succeeded_observations;
                summary.succeeded_total_value = saturating_add(
                    summary.succeeded_total_value,
                    record.observation.value);
            }
        }

        return summary;
    }

    [[nodiscard]] static constexpr bool valid_quantity(
        MeasurementQuantity quantity) noexcept {
        switch (quantity) {
        case MeasurementQuantity::DURATION_NS:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]] static constexpr std::uint64_t saturating_add(
        std::uint64_t left,
        std::uint64_t right) noexcept {
        return right > std::numeric_limits<std::uint64_t>::max() - left
                   ? std::numeric_limits<std::uint64_t>::max()
                   : left + right;
    }

    [[nodiscard]] const std::vector<std::size_t>& indices_for(
        ResourceRef subject) const noexcept {
        static const std::vector<std::size_t> none;

        const auto it = subject_index_.find(subject);

        return it == subject_index_.end() ? none : it->second;
    }

    std::vector<MeasurementRecord> records_;
    std::unordered_map<ResourceRef, std::vector<std::size_t>> subject_index_;
    std::uint64_t next_id_{1};
};

} // namespace gerdos