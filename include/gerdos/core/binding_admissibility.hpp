#pragma once

#include <cstddef>
#include <vector>

#include "gerdos/core/operation.hpp"
#include "gerdos/core/physical_binding.hpp"

namespace gerdos {

// Semantic execution admissibility: whether a physical binding is an
// appropriate realization of a specific Operation. The evaluation compares
// the Operation's declared requirements against the binding only; it does not
// consult registries, runtime state, topology, measurements, or planner
// policy.
class BindingAdmissibilityValidator {
public:
    [[nodiscard]] bool admissible(
        const Operation& operation,
        const PhysicalBinding& binding) const noexcept {
        const auto& description = operation.description();

        for (const auto& data_binding : binding.data) {
            const auto data = data_binding.residency.data;

            if (is_consuming(data_binding.role)) {
                if (!contains(description.inputs, data)) {
                    return false;
                }
            } else if (is_producing(data_binding.role)) {
                if (!contains(description.outputs, data)) {
                    return false;
                }
            } else {
                return false;
            }
        }

        for (const auto input : description.inputs) {
            if (!has_data_entry(binding, input, true)) {
                return false;
            }
        }

        for (const auto output : description.outputs) {
            if (!has_data_entry(binding, output, false)) {
                return false;
            }
        }

        for (const auto& requirement :
             description.resource_requirements) {
            if (!requirement.valid()) {
                return false;
            }

            if (count_resource_entries(binding, requirement.role) <
                requirement.minimum) {
                return false;
            }
        }

        return true;
    }

private:
    [[nodiscard]] static bool contains(
        const std::vector<DataId>& references,
        DataId id) noexcept {
        for (const auto reference : references) {
            if (reference == id) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] static bool has_data_entry(
        const PhysicalBinding& binding,
        DataId id,
        bool consuming) noexcept {
        for (const auto& data_binding : binding.data) {
            if (data_binding.residency.data != id) {
                continue;
            }

            const auto role = data_binding.role;

            if (consuming ? is_consuming(role) : is_producing(role)) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] static std::size_t count_resource_entries(
        const PhysicalBinding& binding,
        ResourceBindingRole role) noexcept {
        std::size_t count = 0;

        for (const auto& resource_binding : binding.resources) {
            if (resource_binding.role == role) {
                ++count;
            }
        }

        return count;
    }
};

} // namespace gerdos