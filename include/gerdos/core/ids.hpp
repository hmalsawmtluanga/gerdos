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

using DeviceId = StrongId<DeviceIdTag>;
using ResourceId = StrongId<ResourceIdTag>;
using TopologyLinkId = StrongId<TopologyLinkIdTag>;
using DataId = StrongId<DataIdTag>;
using DataResidencyId = StrongId<DataResidencyIdTag>;
using OperationId = StrongId<OperationIdTag>;
using ExecutionId = StrongId<ExecutionIdTag>;

} // namespace gerdos

namespace std {

template <typename Tag>
struct hash<gerdos::StrongId<Tag>> {
    std::size_t operator()(gerdos::StrongId<Tag> id) const noexcept {
        return std::hash<typename gerdos::StrongId<Tag>::value_type>{}(
            id.value());
    }
};

} // namespace std
