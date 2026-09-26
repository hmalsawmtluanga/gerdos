#include "test_check.hpp"
#include <cstddef>

#include "gerdos/core/operation_registry.hpp"

int main() {
    using namespace gerdos;

    OperationRegistry registry;

    // Invalid identity is rejected.
    GERDOS_CHECK(registry.create_operation(OperationDescription{
               OperationId{},
               {DataId{10}},
               {DataId{20}},
               {}}) == nullptr);

    GERDOS_CHECK(registry.operation_count() == 0);

    // Invalid referenced identities are rejected, but existence is not required.
    GERDOS_CHECK(registry.create_operation(OperationDescription{
               OperationId{1},
               {DataId{}},
               {DataId{20}},
               {}}) == nullptr);

    GERDOS_CHECK(registry.create_operation(OperationDescription{
               OperationId{2},
               {DataId{10}},
               {DataId{}},
               {}}) == nullptr);

    GERDOS_CHECK(registry.create_operation(OperationDescription{
               OperationId{3},
               {DataId{10}},
               {DataId{20}},
               {OperationId{}}}) == nullptr);

    // Registration does not require referenced DataIds or dependency
    // OperationIds to already exist.
    Operation* operation_a = registry.create_operation(
        OperationDescription{
            OperationId{100},
            {DataId{10}, DataId{11}},
            {DataId{20}},
            {OperationId{90}},
        });

    GERDOS_CHECK(operation_a != nullptr);
    GERDOS_CHECK(operation_a->description().id == OperationId{100});
    GERDOS_CHECK(operation_a->description().inputs.size() == 2);
    GERDOS_CHECK(operation_a->description().outputs.size() == 1);
    GERDOS_CHECK(operation_a->description().dependencies.size() == 1);
    GERDOS_CHECK(operation_a->description().dependencies[0] ==
           OperationId{90});

    GERDOS_CHECK(registry.find_operation(OperationId{100}) == operation_a);
    GERDOS_CHECK(registry.operation_count() == 1);

    // Duplicate live identity is rejected.
    GERDOS_CHECK(registry.create_operation(OperationDescription{
               OperationId{100},
               {DataId{99}},
               {DataId{98}},
               {}}) == nullptr);

    GERDOS_CHECK(registry.operation_count() == 1);
    GERDOS_CHECK(registry.find_operation(OperationId{100}) == operation_a);

    // A second operation may use the same DataIds and dependency references.
    Operation* operation_b = registry.create_operation(
        OperationDescription{
            OperationId{101},
            {DataId{10}},
            {DataId{20}, DataId{21}},
            {OperationId{100}},
        });

    GERDOS_CHECK(operation_b != nullptr);
    GERDOS_CHECK(operation_b != operation_a);
    GERDOS_CHECK(registry.operation_count() == 2);

    // Mutable lookup.
    GERDOS_CHECK(registry.find_operation(OperationId{101}) == operation_b);

    // Const lookup.
    const auto& const_registry = registry;
    const Operation* const_operation =
        const_registry.find_operation(OperationId{100});

    GERDOS_CHECK(const_operation == operation_a);
    GERDOS_CHECK(const_operation->description().id == OperationId{100});

    // Enumeration sees both registered operations.
    std::size_t mutable_count = 0;
    registry.for_each_operation([&](Operation* operation) {
        GERDOS_CHECK(operation != nullptr);
        ++mutable_count;
    });

    GERDOS_CHECK(mutable_count == 2);

    std::size_t const_count = 0;
    const_registry.for_each_operation(
        [&](const Operation* operation) {
            GERDOS_CHECK(operation != nullptr);
            ++const_count;
        });

    GERDOS_CHECK(const_count == 2);

    // Unknown removal has no effect.
    GERDOS_CHECK(!registry.remove_operation(OperationId{999}));
    GERDOS_CHECK(registry.operation_count() == 2);

    // Removal retires the identity.
    GERDOS_CHECK(registry.remove_operation(OperationId{100}));
    GERDOS_CHECK(registry.operation_count() == 1);
    GERDOS_CHECK(registry.find_operation(OperationId{100}) == nullptr);

    // Retired identities cannot be reused.
    GERDOS_CHECK(registry.create_operation(OperationDescription{
               OperationId{100},
               {DataId{30}},
               {DataId{40}},
               {}}) == nullptr);

    GERDOS_CHECK(registry.operation_count() == 1);

    // The surviving operation remains intact.
    GERDOS_CHECK(registry.find_operation(OperationId{101}) == operation_b);
    GERDOS_CHECK(operation_b->description().id == OperationId{101});

    // Declared resource requirements are accepted when every requirement is
    // structurally valid and roles do not repeat.
    GERDOS_CHECK(
        registry.create_operation(
            OperationDescription{
                OperationId{110},
                {DataId{70}},
                {DataId{71}},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        1,
                    },
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            }) != nullptr);

    GERDOS_CHECK(registry.operation_count() == 2);

    // A requirement must require at least one entry.
    GERDOS_CHECK(
        registry.create_operation(
            OperationDescription{
                OperationId{111},
                {},
                {},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        0,
                    },
                },
            }) == nullptr);

    // Requirement roles must belong to the resource binding role domain.
    GERDOS_CHECK(
        registry.create_operation(
            OperationDescription{
                OperationId{111},
                {},
                {},
                {},
                {
                    ResourceRequirement{
                        static_cast<ResourceBindingRole>(255),
                        1,
                    },
                },
            }) == nullptr);

    // At most one requirement exists per resource binding role.
    GERDOS_CHECK(
        registry.create_operation(
            OperationDescription{
                OperationId{111},
                {},
                {},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        1,
                    },
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        2,
                    },
                },
            }) == nullptr);

    // Rejected descriptions do not register or consume identities.
    GERDOS_CHECK(registry.operation_count() == 2);
    GERDOS_CHECK(registry.find_operation(OperationId{110}) != nullptr);

    return 0;
}
