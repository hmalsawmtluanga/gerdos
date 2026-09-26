#include "test_check.hpp"

#include "gerdos/core/execution.hpp"
#include "gerdos/core/physical_binding.hpp"

int main() {
    using namespace gerdos;

    const ExecutionDescription description{
        ExecutionId{100},
        OperationId{42},
    };

    Execution execution(description);

    GERDOS_CHECK(execution.description().id == ExecutionId{100});
    GERDOS_CHECK(execution.description().operation == OperationId{42});
    GERDOS_CHECK(execution.state() == ExecutionState::PENDING);

    GERDOS_CHECK(execution.bind(PhysicalBinding{}));
    GERDOS_CHECK(execution.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(execution.state() == ExecutionState::RUNNING);

    GERDOS_CHECK(execution.set_state(ExecutionState::COMPLETED));
    GERDOS_CHECK(execution.state() == ExecutionState::COMPLETED);

    GERDOS_CHECK(!execution.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(execution.state() == ExecutionState::COMPLETED);

    return 0;
}
