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

[[nodiscard]] constexpr bool can_transition(
    DataResidencyState from,
    DataResidencyState to) noexcept {
    if (from == to) {
        return true;
    }

    switch (from) {
    case DataResidencyState::UNAVAILABLE:
        return to == DataResidencyState::VALID ||
               to == DataResidencyState::TRANSFERRING;

    case DataResidencyState::VALID:
        return to == DataResidencyState::STALE ||
               to == DataResidencyState::TRANSFERRING;

    case DataResidencyState::STALE:
        return to == DataResidencyState::TRANSFERRING ||
               to == DataResidencyState::UNAVAILABLE;

    case DataResidencyState::TRANSFERRING:
        return to == DataResidencyState::VALID ||
               to == DataResidencyState::UNAVAILABLE;
    }

    return false;
}

} // namespace gerdos
