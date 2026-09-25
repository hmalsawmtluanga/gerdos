#pragma once

namespace gerdos {

enum class ResourceAvailability {
    AVAILABLE,
    UNAVAILABLE,
    INITIALIZING,
    DRAINING,
    FAILED,
};

[[nodiscard]] constexpr bool is_available(
    ResourceAvailability state) noexcept {
    return state == ResourceAvailability::AVAILABLE;
}

} // namespace gerdos
