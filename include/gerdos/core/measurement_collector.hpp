#pragma once

#include <cstdint>
#include <vector>

#include "gerdos/core/execution.hpp"
#include "gerdos/core/measurement.hpp"

namespace gerdos {

// Measurement capture: records one duration observation for each Resource a
// completed attempt bound, carrying the attempted Operation, the attempt's
// identity and outcome, and the observed conditions. Each subject is
// observed once per attempt. Reports whether every intended observation was
// recorded.
class MeasurementCollector {
public:
    static bool capture(
        MeasurementRegistry& measurements,
        const Execution& execution,
        OperationId operation,
        bool succeeded,
        std::uint64_t duration_ns,
        std::size_t concurrent_attempts) {
        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return true;
        }

        std::vector<ResourceRef> observed;
        bool complete = true;

        for (const auto& resource_binding : binding->resources) {
            const auto subject = resource_binding.resource;

            if (contains(observed, subject)) {
                continue;
            }

            observed.push_back(subject);

            const auto id = measurements.record(
                MeasurementObservation{
                    MeasurementQuantity::DURATION_NS,
                    subject,
                    operation,
                    execution.description().id,
                    succeeded,
                    duration_ns,
                    MeasurementConditions{
                        concurrent_attempts,
                    },
                });

            if (!id.valid()) {
                complete = false;
            }
        }

        return complete;
    }

private:
    [[nodiscard]] static bool contains(
        const std::vector<ResourceRef>& references,
        ResourceRef ref) noexcept {
        for (const auto reference : references) {
            if (reference == ref) {
                return true;
            }
        }

        return false;
    }
};

} // namespace gerdos