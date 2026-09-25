#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/resource_state.hpp"

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

class Resource {
public:
    explicit Resource(ResourceDescription description)
        : description_(std::move(description)) {}

    [[nodiscard]] const ResourceDescription& description() const noexcept {
        return description_;
    }

    [[nodiscard]] ResourceAvailability availability() const noexcept {
        return availability_;
    }

    [[nodiscard]] bool available() const noexcept {
        return is_available(availability_);
    }

    void set_availability(ResourceAvailability state) noexcept {
        availability_ = state;
    }

private:
    ResourceDescription description_;
    ResourceAvailability availability_{ResourceAvailability::INITIALIZING};
};

} // namespace gerdos
