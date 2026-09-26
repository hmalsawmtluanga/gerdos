#include "test_check.hpp"

#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/physical_binding.hpp"

int main() {
    using namespace gerdos;

    ExecutionRegistry registry;

    GERDOS_CHECK(registry.execution_count() == 0);

    // Invalid referenced OperationId is rejected.
    GERDOS_CHECK(registry.create_execution(
        ExecutionDescription{
            ExecutionId{1},
            OperationId{},
        }) == nullptr);

    GERDOS_CHECK(registry.create_execution(
        ExecutionDescription{
            ExecutionId{100},
            OperationId{42},
        }) != nullptr);

    GERDOS_CHECK(registry.execution_count() == 1);

    // Multiple execution attempts may belong to the same Operation.
    GERDOS_CHECK(registry.create_execution(
        ExecutionDescription{
            ExecutionId{101},
            OperationId{42},
        }) != nullptr);

    GERDOS_CHECK(registry.execution_count() == 2);

    // Referenced OperationId need not already exist in this ownership layer.
    GERDOS_CHECK(registry.create_execution(
        ExecutionDescription{
            ExecutionId{102},
            OperationId{999},
        }) != nullptr);

    GERDOS_CHECK(registry.execution_count() == 3);

    // Invalid identity is rejected.
    GERDOS_CHECK(registry.create_execution(
        ExecutionDescription{
            ExecutionId{},
            OperationId{42},
        }) == nullptr);

    GERDOS_CHECK(registry.execution_count() == 3);

    // Live duplicate identity is rejected.
    GERDOS_CHECK(registry.create_execution(
        ExecutionDescription{
            ExecutionId{100},
            OperationId{77},
        }) == nullptr);

    GERDOS_CHECK(registry.execution_count() == 3);

    // Mutable lookup.
    auto* execution = registry.find_execution(ExecutionId{100});
    GERDOS_CHECK(execution != nullptr);
    GERDOS_CHECK(execution->description().operation == OperationId{42});

    GERDOS_CHECK(execution->bind(PhysicalBinding{}));
    GERDOS_CHECK(execution->set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(execution->state() == ExecutionState::RUNNING);

    // Const lookup.
    const auto& const_registry = registry;
    const auto* const_execution =
        const_registry.find_execution(ExecutionId{101});

    GERDOS_CHECK(const_execution != nullptr);
    GERDOS_CHECK(const_execution->description().operation == OperationId{42});
    GERDOS_CHECK(const_execution->state() == ExecutionState::PENDING);

    // Unknown removal has no effect.
    GERDOS_CHECK(!registry.remove_execution(ExecutionId{500}));
    GERDOS_CHECK(registry.execution_count() == 3);

    // Removal is preconditioned on lifecycle state: an in-flight attempt
    // must be cancelled first.
    GERDOS_CHECK(!registry.remove_execution(ExecutionId{100}));
    GERDOS_CHECK(registry.execution_count() == 3);

    GERDOS_CHECK(execution->set_state(ExecutionState::CANCELLED));

    // Remove and permanently retire an identity.
    GERDOS_CHECK(registry.remove_execution(ExecutionId{100}));
    GERDOS_CHECK(registry.execution_count() == 2);
    GERDOS_CHECK(registry.find_execution(ExecutionId{100}) == nullptr);

    GERDOS_CHECK(registry.create_execution(
        ExecutionDescription{
            ExecutionId{100},
            OperationId{123},
        }) == nullptr);

    GERDOS_CHECK(registry.execution_count() == 2);

    // Surviving executions remain intact.
    GERDOS_CHECK(registry.find_execution(ExecutionId{101}) != nullptr);
    GERDOS_CHECK(registry.find_execution(ExecutionId{102}) != nullptr);

    std::size_t mutable_count = 0;
    registry.for_each_execution(
        [&](Execution* current) {
            GERDOS_CHECK(current != nullptr);
            ++mutable_count;
        });

    GERDOS_CHECK(mutable_count == 2);

    std::size_t const_count = 0;
    const_registry.for_each_execution(
        [&](const Execution* current) {
            GERDOS_CHECK(current != nullptr);
            ++const_count;
        });

    GERDOS_CHECK(const_count == 2);

    return 0;
}
