#pragma once

#include "gerdos/core/physical_binding.hpp"

namespace gerdos {

class PhysicalBindingValidator {
public:
    [[nodiscard]] bool validate(
        const PhysicalBinding& binding) const noexcept {
        for (const auto& data_binding : binding.data) {
            if (!data_binding.residency.valid()) {
                return false;
            }

            switch (data_binding.role) {
            case DataBindingRole::INPUT:
            case DataBindingRole::OUTPUT:
            case DataBindingRole::SOURCE:
            case DataBindingRole::DESTINATION:
                break;
            default:
                return false;
            }
        }

        for (const auto& resource_binding : binding.resources) {
            if (!resource_binding.resource.valid()) {
                return false;
            }

            switch (resource_binding.role) {
            case ResourceBindingRole::COMPUTE:
            case ResourceBindingRole::TRANSFER:
                break;
            default:
                return false;
            }
        }

        return true;
    }
};

} // namespace gerdos
