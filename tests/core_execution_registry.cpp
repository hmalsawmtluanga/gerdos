#include <cassert>

#include "gerdos/core/execution_registry.hpp"

int main() {
    using namespace gerdos;

    ExecutionRegistry registry;

    assert(registry.execution_count() == 0);

    assert(registry.create_execution(
        ExecutionDescription{
            ExecutionId{100},
            OperationId{42},
        }) != nullptr);

    assert(registry.execution_count() == 1);

    // Multiple execution attempts may belong to the same Operation.
    assert(registry.create_execution(
        ExecutionDescription{
            ExecutionId{101},
            OperationId{42},
        }) != nullptr);

    assert(registry.execution_count() == 2);

    // Referenced OperationId need not already exist in this ownership layer.
    assert(registry.create_execution(
        ExecutionDescription{
            ExecutionId{102},
            OperationId{999},
        }) != nullptr);

    assert(registry.execution_count() == 3);

    // Invalid identity is rejected.
    assert(registry.create_execution(
        ExecutionDescription{
            ExecutionId{},
            OperationId{42},
        }) == nullptr);

    assert(registry.execution_count() == 3);

    // Live duplicate identity is rejected.
    assert(registry.create_execution(
        ExecutionDescription{
            ExecutionId{100},
            OperationId{77},
        }) == nullptr);

    assert(registry.execution_count() == 3);

    // Mutable lookup.
    auto* execution = registry.find_execution(ExecutionId{100});
    assert(execution != nullptr);
    assert(execution->description().operation == OperationId{42});

    assert(execution->set_state(ExecutionState::RUNNING));
    assert(execution->state() == ExecutionState::RUNNING);

    // Const lookup.
    const auto& const_registry = registry;
    const auto* const_execution =
        const_registry.find_execution(ExecutionId{101});

    assert(const_execution != nullptr);
    assert(const_execution->description().operation == OperationId{42});
    assert(const_execution->state() == ExecutionState::PENDING);

    // Unknown removal has no effect.
    assert(!registry.remove_execution(ExecutionId{500}));
    assert(registry.execution_count() == 3);

    // Remove and permanently retire an identity.
    assert(registry.remove_execution(ExecutionId{100}));
    assert(registry.execution_count() == 2);
    assert(registry.find_execution(ExecutionId{100}) == nullptr);

    assert(registry.create_execution(
        ExecutionDescription{
            ExecutionId{100},
            OperationId{123},
        }) == nullptr);

    assert(registry.execution_count() == 2);

    // Surviving executions remain intact.
    assert(registry.find_execution(ExecutionId{101}) != nullptr);
    assert(registry.find_execution(ExecutionId{102}) != nullptr);

    std::size_t mutable_count = 0;
    registry.for_each_execution(
        [&](Execution* current) {
            assert(current != nullptr);
            ++mutable_count;
        });

    assert(mutable_count == 2);

    std::size_t const_count = 0;
    const_registry.for_each_execution(
        [&](const Execution* current) {
            assert(current != nullptr);
            ++const_count;
        });

    assert(const_count == 2);

    return 0;
}
