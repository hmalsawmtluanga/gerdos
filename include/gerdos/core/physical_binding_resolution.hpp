#pragma once

#include <vector>

#include "gerdos/core/data.hpp"
#include "gerdos/core/physical_binding.hpp"
#include "gerdos/core/resource.hpp"

namespace gerdos {

struct ResolvedDataBinding {
    DataBindingRole role;
    DataResidencyRef residency;
    const DataResidency* resolved;
};

struct ResolvedResourceBinding {
    ResourceBindingRole role;
    ResourceRef resource;
    const Resource* resolved;
};

struct BindingResolution {
    std::vector<ResolvedDataBinding> data;
    std::vector<ResolvedResourceBinding> resources;

    [[nodiscard]] bool fully_resolved() const noexcept {
        for (const auto& binding : data) {
            if (binding.resolved == nullptr) {
                return false;
            }
        }

        for (const auto& binding : resources) {
            if (binding.resolved == nullptr) {
                return false;
            }
        }

        return true;
    }
};

} // namespace gerdos
