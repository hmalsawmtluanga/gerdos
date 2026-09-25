#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/resource.hpp"

namespace gerdos {

struct DeviceDescription {
    DeviceId id;
    std::string name;
};

class Device {
public:
    explicit Device(DeviceDescription description)
        : description_(std::move(description)) {}

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    [[nodiscard]] const DeviceDescription& description() const noexcept {
        return description_;
    }

    [[nodiscard]] Resource* find_resource(ResourceId id) noexcept {
        const auto it = resources_.find(id);

        if (it == resources_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] const Resource* find_resource(
        ResourceId id) const noexcept {
        const auto it = resources_.find(id);

        if (it == resources_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] bool add_resource(Resource&& resource) {
        const auto id = resource.description().id;

        if (!id.valid()) {
            return false;
        }

        if (resource.description().owner != description_.id) {
            return false;
        }

        if (resources_.contains(id) || retired_resource_ids_.contains(id)) {
            return false;
        }

        auto owned_resource = std::make_unique<Resource>(
            std::move(resource));

        resources_.emplace(id, std::move(owned_resource));
        return true;
    }

    [[nodiscard]] bool remove_resource(ResourceId id) {
        const auto it = resources_.find(id);

        if (it == resources_.end()) {
            return false;
        }

        retired_resource_ids_.insert(id);
        resources_.erase(it);
        return true;
    }

    [[nodiscard]] std::size_t resource_count() const noexcept {
        return resources_.size();
    }

    template <typename Fn>
    void for_each_resource(Fn&& fn) {
        for (auto& [id, resource] : resources_) {
            (void)id;
            fn(resource.get());
        }
    }

    template <typename Fn>
    void for_each_resource(Fn&& fn) const {
        for (const auto& [id, resource] : resources_) {
            (void)id;
            fn(resource.get());
        }
    }

private:
    DeviceDescription description_;
    std::unordered_map<ResourceId, std::unique_ptr<Resource>> resources_;
    std::unordered_set<ResourceId> retired_resource_ids_;
};

} // namespace gerdos
