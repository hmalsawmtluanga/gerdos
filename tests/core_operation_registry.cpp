#include <cassert>
#include <cstddef>

#include "gerdos/core/operation_registry.hpp"

int main() {
    using namespace gerdos;

    OperationRegistry registry;

    // Invalid identity is rejected.
    assert(registry.create_operation(OperationDescription{
               OperationId{},
               {DataId{10}},
               {DataId{20}},
               {}}) == nullptr);

    assert(registry.operation_count() == 0);

    // Registration does not require referenced DataIds or dependency
    // OperationIds to already exist.
    Operation* operation_a = registry.create_operation(
        OperationDescription{
            OperationId{100},
            {DataId{10}, DataId{11}},
            {DataId{20}},
            {OperationId{90}},
        });

    assert(operation_a != nullptr);
    assert(operation_a->description().id == OperationId{100});
    assert(operation_a->description().inputs.size() == 2);
    assert(operation_a->description().outputs.size() == 1);
    assert(operation_a->description().dependencies.size() == 1);
    assert(operation_a->description().dependencies[0] ==
           OperationId{90});

    assert(registry.find_operation(OperationId{100}) == operation_a);
    assert(registry.operation_count() == 1);

    // Duplicate live identity is rejected.
    assert(registry.create_operation(OperationDescription{
               OperationId{100},
               {DataId{99}},
               {DataId{98}},
               {}}) == nullptr);

    assert(registry.operation_count() == 1);
    assert(registry.find_operation(OperationId{100}) == operation_a);

    // A second operation may use the same DataIds and dependency references.
    Operation* operation_b = registry.create_operation(
        OperationDescription{
            OperationId{101},
            {DataId{10}},
            {DataId{20}, DataId{21}},
            {OperationId{100}},
        });

    assert(operation_b != nullptr);
    assert(operation_b != operation_a);
    assert(registry.operation_count() == 2);

    // Mutable lookup.
    assert(registry.find_operation(OperationId{101}) == operation_b);

    // Const lookup.
    const auto& const_registry = registry;
    const Operation* const_operation =
        const_registry.find_operation(OperationId{100});

    assert(const_operation == operation_a);
    assert(const_operation->description().id == OperationId{100});

    // Enumeration sees both registered operations.
    std::size_t mutable_count = 0;
    registry.for_each_operation([&](Operation* operation) {
        assert(operation != nullptr);
        ++mutable_count;
    });

    assert(mutable_count == 2);

    std::size_t const_count = 0;
    const_registry.for_each_operation(
        [&](const Operation* operation) {
            assert(operation != nullptr);
            ++const_count;
        });

    assert(const_count == 2);

    // Unknown removal has no effect.
    assert(!registry.remove_operation(OperationId{999}));
    assert(registry.operation_count() == 2);

    // Removal retires the identity.
    assert(registry.remove_operation(OperationId{100}));
    assert(registry.operation_count() == 1);
    assert(registry.find_operation(OperationId{100}) == nullptr);

    // Retired identities cannot be reused.
    assert(registry.create_operation(OperationDescription{
               OperationId{100},
               {DataId{30}},
               {DataId{40}},
               {}}) == nullptr);

    assert(registry.operation_count() == 1);

    // The surviving operation remains intact.
    assert(registry.find_operation(OperationId{101}) == operation_b);
    assert(operation_b->description().id == OperationId{101});

    return 0;
}
