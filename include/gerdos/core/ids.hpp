#pragma once

#include <cstdint>
#include <functional>

namespace gerdos {

template <typename Tag>
class StrongId {
public:
    using value_type = std::uint64_t;

    constexpr StrongId() noexcept = default;
    explicit constexpr StrongId(value_type value) noexcept : value_(value) {}

    [[nodiscard]] constexpr value_type value() const noexcept {
        return value_;
    }

    [[nodiscard]] constexpr bool valid() const noexcept {
        return value_ != 0;
    }

    friend constexpr bool operator==(StrongId, StrongId) = default;

private:
    value_type value_{0};
};

struct DeviceIdTag;
struct ResourceIdTag;
struct TopologyLinkIdTag;
struct DataIdTag;
struct DataResidencyIdTag;
struct OperationIdTag;
struct ExecutionIdTag;
struct MeasurementIdTag;

using DeviceId = StrongId<DeviceIdTag>;
using ResourceId = StrongId<ResourceIdTag>;
using TopologyLinkId = StrongId<TopologyLinkIdTag>;
using DataId = StrongId<DataIdTag>;
using DataResidencyId = StrongId<DataResidencyIdTag>;
using OperationId = StrongId<OperationIdTag>;
using ExecutionId = StrongId<ExecutionIdTag>;
using MeasurementId = StrongId<MeasurementIdTag>;

struct ResourceRef {
    DeviceId device;
    ResourceId resource;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return device.valid() && resource.valid();
    }

    friend constexpr bool operator==(ResourceRef, ResourceRef) = default;
};

struct DataResidencyRef {
    DataId data;
    DataResidencyId residency;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return data.valid() && residency.valid();
    }

    friend constexpr bool operator==(
        DataResidencyRef,
        DataResidencyRef) = default;
};

} // namespace gerdos

namespace std {

template <typename Tag>
struct hash<gerdos::StrongId<Tag>> {
    std::size_t operator()(gerdos::StrongId<Tag> id) const noexcept {
        return std::hash<typename gerdos::StrongId<Tag>::value_type>{}(
            id.value());
    }
};

template <>
struct hash<gerdos::ResourceRef> {
    std::size_t operator()(
        const gerdos::ResourceRef& ref) const noexcept {
        const auto first = std::hash<gerdos::DeviceId>{}(ref.device);
        const auto second = std::hash<gerdos::ResourceId>{}(ref.resource);

        return first ^
               (second + std::size_t{0x9e3779b97f4a7c15ULL} +
                    (first << 6) + (first >> 2));
    }
};

template <>
struct hash<gerdos::DataResidencyRef> {
    std::size_t operator()(
        const gerdos::DataResidencyRef& ref) const noexcept {
        const auto first = std::hash<gerdos::DataId>{}(ref.data);
        const auto second =
            std::hash<gerdos::DataResidencyId>{}(ref.residency);

        return first ^
               (second + std::size_t{0x9e3779b97f4a7c15ULL} +
                    (first << 6) + (first >> 2));
    }
};

} // namespace std
