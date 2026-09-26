#include <cassert>
#include <utility>
#include <vector>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/operation.hpp"

int main() {
    using namespace gerdos;

    OperationDescription description{
        OperationId{100},
        {DataId{10}, DataId{11}},
        {DataId{20}},
        {OperationId{90}, OperationId{91}},
    };

    Operation operation(std::move(description));

    assert(operation.description().id == OperationId{100});

    assert(operation.description().inputs.size() == 2);
    assert(operation.description().inputs[0] == DataId{10});
    assert(operation.description().inputs[1] == DataId{11});

    assert(operation.description().outputs.size() == 1);
    assert(operation.description().outputs[0] == DataId{20});

    assert(operation.description().dependencies.size() == 2);
    assert(operation.description().dependencies[0] == OperationId{90});
    assert(operation.description().dependencies[1] == OperationId{91});

    // Declaration order is preserved; the minimal Operation contract does
    // not silently deduplicate or reinterpret references.
    OperationDescription duplicate_references{
        OperationId{101},
        {DataId{30}, DataId{30}},
        {DataId{40}, DataId{40}},
        {OperationId{90}, OperationId{90}},
    };

    Operation duplicate_operation(std::move(duplicate_references));

    assert(duplicate_operation.description().inputs.size() == 2);
    assert(duplicate_operation.description().outputs.size() == 2);
    assert(duplicate_operation.description().dependencies.size() == 2);

    return 0;
}
