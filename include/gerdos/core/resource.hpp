#pragma once

#include <cstddef>
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
    // Declared total byte capacity for memory/storage resources. Zero
    // means unbounded or unknown: never a refusal. Not identity — a
    // resource keeps its capacity across availability transitions, and
    // references stay valid whatever it declares.
    std::size_t capacity_bytes{0};
};

class Resource {
public:
    explicit Resource(ResourceDescription description)
        : description_(std::move(description)) {}

    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&&) noexcept = default;
    Resource& operator=(Resource&&) noexcept = delete;

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

    // Declared total byte capacity. Runtime state like availability:
    // adjustable without touching identity, references stay valid.
    void set_capacity(std::size_t bytes) noexcept {
        description_.capacity_bytes = bytes;
    }

private:
    ResourceDescription description_;
    ResourceAvailability availability_{ResourceAvailability::INITIALIZING};
};

} // namespace gerdos
