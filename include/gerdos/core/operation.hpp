#pragma once

#include <utility>
#include <vector>

#include "gerdos/core/ids.hpp"

namespace gerdos {

struct OperationDescription {
    OperationId id;
    std::vector<DataId> inputs;
    std::vector<DataId> outputs;
    std::vector<OperationId> dependencies;
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
