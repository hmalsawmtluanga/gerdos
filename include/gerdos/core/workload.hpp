#pragma once

#include <string>
#include <vector>

#include "gerdos/core/operation.hpp"

namespace gerdos {

// A workload: the computation submitted to the runtime as a named stream of
// work units. Units declare their data references and resource requirements
// in the generic core vocabulary; model adapters produce workloads and the
// core consumes them without knowing their origin. Ordering of the stream is
// the submitting layer's until the planning contract owns dependency
// ordering.
struct Workload {
    std::string name;
    std::vector<OperationDescription> operations;
};

} // namespace gerdos