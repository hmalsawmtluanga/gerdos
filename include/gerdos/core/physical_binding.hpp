#pragma once

#include <vector>

#include "gerdos/core/ids.hpp"

namespace gerdos {

enum class DataBindingRole {
    INPUT,
    OUTPUT,
    SOURCE,
    DESTINATION,
};

enum class ResourceBindingRole {
    COMPUTE,
    TRANSFER,
};

struct DataBinding {
    DataBindingRole role;
    DataResidencyRef residency;
};

struct ResourceBinding {
    ResourceBindingRole role;
    ResourceRef resource;
};

struct PhysicalBinding {
    std::vector<DataBinding> data;
    std::vector<ResourceBinding> resources;
};

} // namespace gerdos
