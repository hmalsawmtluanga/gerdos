#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gerdos/core/data.hpp"
#include "gerdos/core/ids.hpp"

namespace gerdos {

class DataRegistry {
public:
    DataRegistry() = default;

    DataRegistry(const DataRegistry&) = delete;
    DataRegistry& operator=(const DataRegistry&) = delete;
    DataRegistry(DataRegistry&&) = delete;
    DataRegistry& operator=(DataRegistry&&) = delete;

    [[nodiscard]] Data* create_data(DataDescription description) {
        const auto id = description.id;

        if (!id.valid()) {
            return nullptr;
        }

        if (data_.contains(id) || retired_ids_.contains(id)) {
            return nullptr;
        }

        auto data = std::make_unique<Data>(std::move(description));
        auto* data_ptr = data.get();

        data_.emplace(id, std::move(data));
        return data_ptr;
    }

    [[nodiscard]] Data* find_data(DataId id) noexcept {
        const auto it = data_.find(id);
        if (it == data_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] const Data* find_data(DataId id) const noexcept {
        const auto it = data_.find(id);
        if (it == data_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    // Removal is preconditioned on lifecycle state: a Data object cannot be
    // removed while one of its residencies holds a claimed update.
    [[nodiscard]] bool remove_data(DataId id) {
        const auto it = data_.find(id);
        if (it == data_.end()) {
            return false;
        }

        if (it->second->has_claimed_update()) {
            return false;
        }

        retired_ids_.insert(id);
        data_.erase(it);
        return true;
    }

    [[nodiscard]] std::size_t data_count() const noexcept {
        return data_.size();
    }

    template <typename Fn>
    void for_each_data(Fn&& fn) {
        for (auto& [id, data] : data_) {
            (void)id;
            fn(data.get());
        }
    }

    template <typename Fn>
    void for_each_data(Fn&& fn) const {
        for (const auto& [id, data] : data_) {
            (void)id;
            fn(static_cast<const Data*>(data.get()));
        }
    }

private:
    std::unordered_map<DataId, std::unique_ptr<Data>> data_;
    std::unordered_set<DataId> retired_ids_;
};

} // namespace gerdos
