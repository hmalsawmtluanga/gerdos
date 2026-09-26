#pragma once

#include <vector>

#include "gerdos/core/ids.hpp"

namespace gerdos {

enum class DataBindingRole {
    INPUT,
    OUTPUT,
    SOURCE,
    DESTINATION,
};

enum class ResourceBindingRole {
    COMPUTE,
    TRANSFER,
};

// Data binding role classification. A consuming role references a
// representation the attempt reads; a producing role references a
// representation the attempt creates or rewrites.
[[nodiscard]] constexpr bool is_consuming(DataBindingRole role) noexcept {
    return role == DataBindingRole::INPUT ||
           role == DataBindingRole::SOURCE;
}

[[nodiscard]] constexpr bool is_producing(DataBindingRole role) noexcept {
    return role == DataBindingRole::OUTPUT ||
           role == DataBindingRole::DESTINATION;
}

struct DataBinding {
    DataBindingRole role;
    DataResidencyRef residency;
};

struct ResourceBinding {
    ResourceBindingRole role;
    ResourceRef resource;
};

struct PhysicalBinding {
    std::vector<DataBinding> data;
    std::vector<ResourceBinding> resources;
};

} // namespace gerdos
