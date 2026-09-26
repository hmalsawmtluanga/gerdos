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
    GERDOS_CHECK(
        duplicate_operation.description().resource_requirements.empty());

    // Resource requirements are declarative role/minimum pairs and are
    // preserved verbatim.
    OperationDescription required_participation{
        OperationId{102},
        {DataId{50}},
        {DataId{60}},
        {},
        {
            ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
            ResourceRequirement{ResourceBindingRole::TRANSFER, 2},
        },
    };

    Operation required_operation(std::move(required_participation));

    GERDOS_CHECK(
        required_operation.description().resource_requirements.size() == 2);

    GERDOS_CHECK(
        required_operation.description().resource_requirements[0].role ==
        ResourceBindingRole::COMPUTE);

    GERDOS_CHECK(
        required_operation.description().resource_requirements[0].minimum ==
        1);

    GERDOS_CHECK(
        required_operation.description().resource_requirements[1].role ==
        ResourceBindingRole::TRANSFER);

    GERDOS_CHECK(
        required_operation.description().resource_requirements[1].minimum ==
        2);

    // Requirement structural validity: a requirement must require at least
    // one entry of a role in the binding role domain.
    const ResourceRequirement compute_requirement{
        ResourceBindingRole::COMPUTE,
        1,
    };

    const ResourceRequirement transfer_requirement{
        ResourceBindingRole::TRANSFER,
        4,
    };

    const ResourceRequirement vacuous_requirement{
        ResourceBindingRole::COMPUTE,
        0,
    };

    const ResourceRequirement unknown_requirement{
        static_cast<ResourceBindingRole>(255),
        1,
    };

    GERDOS_CHECK(compute_requirement.valid());
    GERDOS_CHECK(transfer_requirement.valid());
    GERDOS_CHECK(!vacuous_requirement.valid());
    GERDOS_CHECK(!unknown_requirement.valid());

    return 0;
}
