#pragma once

#include <vector>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/resource.hpp"

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

// Binding roles name mechanisms: a COMPUTE binding references a compute
// resource and a TRANSFER binding references a transfer resource. Memory
// and storage resources are places, reached through residencies.
[[nodiscard]] constexpr bool is_mechanism(
    ResourceBindingRole role,
    ResourceKind kind) noexcept {
    switch (role) {
    case ResourceBindingRole::COMPUTE:
        return kind == ResourceKind::COMPUTE;
    case ResourceBindingRole::TRANSFER:
        return kind == ResourceKind::TRANSFER;
    default:
        return false;
    }
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

// Which consuming entry supplies the source of one producing entry: the
// consuming record of the same Data — or, for compute shapes, the first
// consuming entry. Returns the binding index, or the binding's data size
// when the producing entry has no source.
[[nodiscard]] constexpr std::size_t work_source_index(
    const PhysicalBinding& binding,
    std::size_t producing_index) noexcept {
    const auto& producing = binding.data[producing_index];

    for (std::size_t in = 0; in < binding.data.size(); ++in) {
        const auto& consuming = binding.data[in];

        if (is_consuming(consuming.role) &&
            consuming.residency.data == producing.residency.data) {
            return in;
        }
    }

    for (std::size_t in = 0; in < binding.data.size(); ++in) {
        if (is_consuming(binding.data[in].role)) {
            return in;
        }
    }

    return binding.data.size();
}

} // namespace gerdos
