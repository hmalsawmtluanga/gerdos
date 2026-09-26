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

// Resolution results contain non-owning pointers into the registries.
// A resolved pointer is valid only while the referenced registry-owned
// object remains alive and is not removed. BindingResolution does not
// retain ownership or track registry mutations. A previously obtained
// resolution does not change when registries change; future calls to
// resolve() observe the current registry state.
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
