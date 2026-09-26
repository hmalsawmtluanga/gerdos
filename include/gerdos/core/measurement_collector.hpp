#pragma once

#include <cstdint>
#include <vector>

#include "gerdos/core/execution.hpp"
#include "gerdos/core/measurement.hpp"
#include "gerdos/core/operation.hpp"

namespace gerdos {

// Measurement capture: records one duration observation for each Resource a
// completed attempt bound, carrying the attempted Operation and the observed
// conditions. Each subject is observed once per attempt.
class MeasurementCollector {
public:
    static void capture(
        MeasurementRegistry& measurements,
        const Execution& execution,
        const Operation& operation,
        std::uint64_t duration_ns,
        std::size_t concurrent_attempts) {
        const auto* binding = execution.binding();

        if (binding == nullptr) {
            return;
        }

        std::vector<ResourceRef> observed;

        for (const auto& resource_binding : binding->resources) {
            const auto subject = resource_binding.resource;

            if (contains(observed, subject)) {
                continue;
            }

            observed.push_back(subject);

            (void)measurements.record(
                MeasurementObservation{
                    MeasurementQuantity::DURATION_NS,
                    subject,
                    operation.description().id,
                    duration_ns,
                    MeasurementConditions{
                        concurrent_attempts,
                    },
                });
        }
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