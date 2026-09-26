#include "test_check.hpp"
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

    GERDOS_CHECK(operation.description().id == OperationId{100});

    GERDOS_CHECK(operation.description().inputs.size() == 2);
    GERDOS_CHECK(operation.description().inputs[0] == DataId{10});
    GERDOS_CHECK(operation.description().inputs[1] == DataId{11});

    GERDOS_CHECK(operation.description().outputs.size() == 1);
    GERDOS_CHECK(operation.description().outputs[0] == DataId{20});

    GERDOS_CHECK(operation.description().dependencies.size() == 2);
    GERDOS_CHECK(operation.description().dependencies[0] == OperationId{90});
    GERDOS_CHECK(operation.description().dependencies[1] == OperationId{91});

    // Declaration order is preserved; the minimal Operation contract does
    // not silently deduplicate or reinterpret references.
    OperationDescription duplicate_references{
        OperationId{101},
        {DataId{30}, DataId{30}},
        {DataId{40}, DataId{40}},
        {OperationId{90}, OperationId{90}},
    };

    Operation duplicate_operation(std::move(duplicate_references));

    GERDOS_CHECK(duplicate_operation.description().inputs.size() == 2);
    GERDOS_CHECK(duplicate_operation.description().outputs.size() == 2);
    GERDOS_CHECK(duplicate_operation.description().dependencies.size() == 2);

    return 0;
}
