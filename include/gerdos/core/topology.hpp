#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gerdos/core/ids.hpp"

namespace gerdos {

enum class TopologyEndpointKind {
    DEVICE,
    RESOURCE,
};

struct TopologyEndpoint {
    TopologyEndpointKind kind;
    DeviceId device;
    ResourceId resource;

    [[nodiscard]] constexpr bool valid() const noexcept {
        switch (kind) {
        case TopologyEndpointKind::DEVICE:
            return device.valid() && !resource.valid();
        case TopologyEndpointKind::RESOURCE:
            return device.valid() && resource.valid();
        }

        return false;
    }

    [[nodiscard]] static constexpr TopologyEndpoint device_endpoint(
        DeviceId id) noexcept {
        return TopologyEndpoint{
            TopologyEndpointKind::DEVICE,
            id,
            ResourceId{},
        };
    }

    [[nodiscard]] static constexpr TopologyEndpoint resource_endpoint(
        DeviceId device_id,
        ResourceId resource_id) noexcept {
        return TopologyEndpoint{
            TopologyEndpointKind::RESOURCE,
            device_id,
            resource_id,
        };
    }
};

enum class TopologyLinkDirection {
    DIRECTED,
    BIDIRECTIONAL,
};

// The static declared performance of a link. Zero means uncharacterized;
// these are declared expectations, not measurements. Measured link behavior
// belongs to Measurement evidence.
struct TopologyLinkAttributes {
    std::uint64_t bandwidth_bytes_per_second{0};
    std::uint64_t latency_ns{0};
};

struct TopologyLinkDescription {
    TopologyLinkId id;
    TopologyEndpoint source;
    TopologyEndpoint destination;
    TopologyLinkDirection direction;
    TopologyLinkAttributes attributes{};
};

class TopologyLink {
public:
    explicit TopologyLink(TopologyLinkDescription description)
        : description_(std::move(description)) {}

    TopologyLink(const TopologyLink&) = delete;
    TopologyLink& operator=(const TopologyLink&) = delete;
    TopologyLink(TopologyLink&&) noexcept = default;
    TopologyLink& operator=(TopologyLink&&) noexcept = delete;

    [[nodiscard]] const TopologyLinkDescription& description() const noexcept {
        return description_;
    }

private:
    TopologyLinkDescription description_;
};

class Topology {
public:
    Topology() = default;

    Topology(const Topology&) = delete;
    Topology& operator=(const Topology&) = delete;
    Topology(Topology&&) = delete;
    Topology& operator=(Topology&&) = delete;

    [[nodiscard]] bool add_link(TopologyLink&& link) {
        const auto id = link.description().id;

        if (!id.valid()) {
            return false;
        }

        if (!link.description().source.valid() ||
            !link.description().destination.valid()) {
            return false;
        }

        const auto direction = link.description().direction;

        if (direction != TopologyLinkDirection::DIRECTED &&
            direction != TopologyLinkDirection::BIDIRECTIONAL) {
            return false;
        }

        if (links_.contains(id) || retired_link_ids_.contains(id)) {
            return false;
        }

        auto owned_link = std::make_unique<TopologyLink>(
            std::move(link));

        links_.emplace(id, std::move(owned_link));
        return true;
    }

    [[nodiscard]] TopologyLink* find_link(TopologyLinkId id) noexcept {
        const auto it = links_.find(id);

        if (it == links_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] const TopologyLink* find_link(
        TopologyLinkId id) const noexcept {
        const auto it = links_.find(id);

        if (it == links_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] bool remove_link(TopologyLinkId id) {
        const auto it = links_.find(id);

        if (it == links_.end()) {
            return false;
        }

        retired_link_ids_.insert(id);
        links_.erase(it);
        return true;
    }

    [[nodiscard]] std::size_t link_count() const noexcept {
        return links_.size();
    }

    template <typename Fn>
    void for_each_link(Fn&& fn) {
        for (auto& [id, link] : links_) {
            (void)id;
            fn(link.get());
        }
    }

    template <typename Fn>
    void for_each_link(Fn&& fn) const {
        for (const auto& [id, link] : links_) {
            (void)id;
            fn(static_cast<const TopologyLink*>(link.get()));
        }
    }

private:
    std::unordered_map<
        TopologyLinkId,
        std::unique_ptr<TopologyLink>> links_;

    std::unordered_set<TopologyLinkId> retired_link_ids_;
};

} // namespace gerdos
