#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/physical_binding.hpp"

namespace gerdos {

// A resource requirement declares required runtime participation as a
// resource binding role and a minimum number of resource entries of that
// role. Requirements do not select a concrete Resource or Device and do not
// classify the Operation.
struct ResourceRequirement {
    ResourceBindingRole role;
    std::size_t minimum;

    [[nodiscard]] constexpr bool valid() const noexcept {
        if (minimum == 0) {
            return false;
        }

        switch (role) {
        case ResourceBindingRole::COMPUTE:
        case ResourceBindingRole::TRANSFER:
            return true;
        default:
            return false;
        }
    }
};

struct OperationDescription {
    OperationId id;
    std::vector<DataId> inputs;
    std::vector<DataId> outputs;
    std::vector<OperationId> dependencies;
    std::vector<ResourceRequirement> resource_requirements{};
};

class Operation {
public:
    explicit Operation(OperationDescription description)
        : description_(std::move(description)) {}

    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
    Operation(Operation&&) noexcept = default;
    Operation& operator=(Operation&&) noexcept = delete;

    [[nodiscard]] const OperationDescription&
    description() const noexcept {
        return description_;
    }

private:
    OperationDescription description_;
};

} // namespace gerdos