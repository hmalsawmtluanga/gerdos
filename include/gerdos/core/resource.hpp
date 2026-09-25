#pragma once

#include <string>
#include <string_view>

#include "gerdos/core/ids.hpp"

namespace gerdos {

enum class ResourceKind {
    COMPUTE,
    MEMORY,
    STORAGE,
    TRANSFER,
    SYNCHRONIZATION,
};

[[nodiscard]] constexpr std::string_view to_string(
    ResourceKind kind) noexcept {
    switch (kind) {
    case ResourceKind::COMPUTE:
        return "compute";
    case ResourceKind::MEMORY:
        return "memory";
    case ResourceKind::STORAGE:
        return "storage";
    case ResourceKind::TRANSFER:
        return "transfer";
    case ResourceKind::SYNCHRONIZATION:
        return "synchronization";
    }

    return "unknown";
}

struct ResourceDescription {
    ResourceId id;
    DeviceId owner;
    ResourceKind kind;
    std::string name;
};

} // namespace gerdos
