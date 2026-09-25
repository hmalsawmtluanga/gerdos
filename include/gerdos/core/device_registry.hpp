#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gerdos/core/device.hpp"
#include "gerdos/core/ids.hpp"

namespace gerdos {

class DeviceRegistry {
public:
    DeviceRegistry() = default;

    DeviceRegistry(const DeviceRegistry&) = delete;
    DeviceRegistry& operator=(const DeviceRegistry&) = delete;
    DeviceRegistry(DeviceRegistry&&) = delete;
    DeviceRegistry& operator=(DeviceRegistry&&) = delete;

    [[nodiscard]] Device* create_device(DeviceDescription description) {
        const auto id = description.id;

        if (!id.valid()) {
            return nullptr;
        }

        if (devices_.contains(id) || retired_ids_.contains(id)) {
            return nullptr;
        }

        auto device = std::make_unique<Device>(std::move(description));
        auto* device_ptr = device.get();

        devices_.emplace(id, std::move(device));
        return device_ptr;
    }

    [[nodiscard]] Device* find_device(DeviceId id) noexcept {
        const auto it = devices_.find(id);

        if (it == devices_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] const Device* find_device(DeviceId id) const noexcept {
        const auto it = devices_.find(id);

        if (it == devices_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] bool remove_device(DeviceId id) {
        const auto it = devices_.find(id);

        if (it == devices_.end()) {
            return false;
        }

        retired_ids_.insert(id);
        devices_.erase(it);
        return true;
    }

    [[nodiscard]] std::size_t device_count() const noexcept {
        return devices_.size();
    }

    template <typename Fn>
    void for_each_device(Fn&& fn) {
        for (auto& [id, device] : devices_) {
            (void)id;
            fn(device.get());
        }
    }

    template <typename Fn>
    void for_each_device(Fn&& fn) const {
        for (const auto& [id, device] : devices_) {
            (void)id;
            fn(device.get());
        }
    }

private:
    std::unordered_map<DeviceId, std::unique_ptr<Device>> devices_;
    std::unordered_set<DeviceId> retired_ids_;
};

} // namespace gerdos
