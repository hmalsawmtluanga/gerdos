#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/physical_binding.hpp"

namespace gerdos {

// A resource requirement declares required runtime participation as a
// resource binding role and a minimum number of distinct resources in that
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

// The declared computation of an attempt, expressed as generic data-parallel
// semantics over the bound representations: each producing representation is
// transformed toward dst * destination_scale + source_scale * src + constant,
// repeated passes times over elements float elements. Exact copying is the
// destination_scale 0, source_scale 1, constant 0, passes 1 case. The
// consuming record of the same Data supplies src — or, for compute shapes,
// the first consuming entry. Gates never inspect the work; backends
// interpret it at the seam.
struct WorkDescription {
    std::size_t elements{0};
    std::size_t passes{0};
    float destination_scale{1.0f};
    float source_scale{0.0f};
    float constant{0.0f};
};

// An Operation is one unit of declarative executable work: it references
// data and dependencies by identity, declares required runtime participation
// as resource requirements, and declares its computation as a work
// description. An Operation does not select a concrete Resource or Device,
// and no gate classifies it.
struct OperationDescription {
    OperationId id;
    std::vector<DataId> inputs;
    std::vector<DataId> outputs;
    std::vector<OperationId> dependencies;
    std::vector<ResourceRequirement> resource_requirements{};
    WorkDescription work{};
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