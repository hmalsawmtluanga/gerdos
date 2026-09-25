#pragma once

namespace gerdos {

enum class DataResidencyState {
    VALID,
    STALE,
    TRANSFERRING,
    UNAVAILABLE,
};

[[nodiscard]] constexpr bool is_usable(
    DataResidencyState state) noexcept {
    return state == DataResidencyState::VALID;
}

} // namespace gerdos
