#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gerdos/core/data_state.hpp"
#include "gerdos/core/ids.hpp"

namespace gerdos {

struct DataResidencyDescription {
    DataResidencyId id;
    DataId data;
    ResourceRef resource;
    std::string representation;
};

class DataResidency {
public:
    explicit DataResidency(DataResidencyDescription description)
        : description_(std::move(description)) {}

    DataResidency(const DataResidency&) = delete;
    DataResidency& operator=(const DataResidency&) = delete;
    DataResidency(DataResidency&&) noexcept = default;
    DataResidency& operator=(DataResidency&&) noexcept = delete;

    [[nodiscard]] const DataResidencyDescription&
    description() const noexcept {
        return description_;
    }

    [[nodiscard]] DataResidencyState state() const noexcept {
        return state_;
    }

    [[nodiscard]] bool usable() const noexcept {
        return is_usable(state_);
    }

    [[nodiscard]] bool set_state(DataResidencyState state) noexcept {
        if (!can_transition(state_, state)) {
            return false;
        }

        state_ = state;
        return true;
    }

private:
    DataResidencyDescription description_;
    DataResidencyState state_{DataResidencyState::UNAVAILABLE};
};

struct DataDescription {
    DataId id;
    std::string name;
};

class Data {
public:
    explicit Data(DataDescription description)
        : description_(std::move(description)) {}

    Data(const Data&) = delete;
    Data& operator=(const Data&) = delete;
    Data(Data&&) = delete;
    Data& operator=(Data&&) = delete;

    [[nodiscard]] const DataDescription& description() const noexcept {
        return description_;
    }

    [[nodiscard]] DataResidency* find_residency(
        DataResidencyId id) noexcept {
        const auto it = residencies_.find(id);
        if (it == residencies_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    [[nodiscard]] const DataResidency* find_residency(
        DataResidencyId id) const noexcept {
        const auto it = residencies_.find(id);
        if (it == residencies_.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    [[nodiscard]] bool add_residency(DataResidency&& residency) {
        const auto& description = residency.description();
        const auto id = description.id;

        if (!id.valid()) {
            return false;
        }

        if (!description.data.valid() ||
            description.data != description_.id) {
            return false;
        }

        if (!description.resource.valid()) {
            return false;
        }

        for (const auto& [existing_id, existing_residency] : residencies_) {
            (void)existing_id;
            const auto& existing_description =
                existing_residency->description();

            if (existing_description.resource == description.resource &&
                existing_description.representation ==
                    description.representation) {
                return false;
            }
        }

        if (residencies_.contains(id) ||
            retired_residency_ids_.contains(id)) {
            return false;
        }

        auto owned_residency =
            std::make_unique<DataResidency>(std::move(residency));

        residencies_.emplace(id, std::move(owned_residency));
        return true;
    }

    // Removal is preconditioned on lifecycle state: a residency whose
    // update is in progress must be resolved first, so that the attempt
    // updating it can still apply its finishing effects.
    [[nodiscard]] bool remove_residency(DataResidencyId id) {
        const auto it = residencies_.find(id);
        if (it == residencies_.end()) {
            return false;
        }

        if (it->second->state() == DataResidencyState::TRANSFERRING) {
            return false;
        }

        retired_residency_ids_.insert(id);
        residencies_.erase(it);
        return true;
    }

    [[nodiscard]] bool has_updating_residency() const noexcept {
        for (const auto& [id, residency] : residencies_) {
            (void)id;

            if (residency->state() == DataResidencyState::TRANSFERRING) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] std::size_t residency_count() const noexcept {
        return residencies_.size();
    }

    template <typename Fn>
    void for_each_residency(Fn&& fn) {
        for (auto& [id, residency] : residencies_) {
            (void)id;
            fn(residency.get());
        }
    }

    template <typename Fn>
    void for_each_residency(Fn&& fn) const {
        for (const auto& [id, residency] : residencies_) {
            (void)id;
            fn(residency.get());
        }
    }

private:
    DataDescription description_;
    std::unordered_map<
        DataResidencyId,
        std::unique_ptr<DataResidency>> residencies_;
    std::unordered_set<DataResidencyId> retired_residency_ids_;
};

} // namespace gerdos
