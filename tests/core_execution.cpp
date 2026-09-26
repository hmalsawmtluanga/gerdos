#include <cassert>

#include "gerdos/core/execution.hpp"

int main() {
    using namespace gerdos;

    const ExecutionDescription description{
        ExecutionId{100},
        OperationId{42},
    };

    Execution execution(description);

    assert(execution.description().id == ExecutionId{100});
    assert(execution.description().operation == OperationId{42});
    assert(execution.state() == ExecutionState::PENDING);

    assert(execution.set_state(ExecutionState::RUNNING));
    assert(execution.state() == ExecutionState::RUNNING);

    assert(execution.set_state(ExecutionState::COMPLETED));
    assert(execution.state() == ExecutionState::COMPLETED);

    assert(!execution.set_state(ExecutionState::RUNNING));
    assert(execution.state() == ExecutionState::COMPLETED);

    return 0;
}
