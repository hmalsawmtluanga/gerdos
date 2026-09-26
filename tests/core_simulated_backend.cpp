#include "test_check.hpp"

#include <vector>

#include "gerdos/core/operation.hpp"
#include "gerdos/sim/simulated_backend.hpp"

int main() {
    using namespace gerdos;

    const Operation operation{OperationDescription{
        OperationId{700},
        {},
        {},
        {},
        {},
    }};

    SimulatedBackend backend{2};

    // ---------------------------------------------------------------------
    // 1. Work steps and completion order
    // ---------------------------------------------------------------------

    Execution long_attempt{
        ExecutionDescription{
            ExecutionId{800},
            OperationId{700},
        }};

    Execution short_attempt{
        ExecutionDescription{
            ExecutionId{801},
            OperationId{700},
        }};

    backend.set_work(ExecutionId{800}, 4);
    backend.set_failure(ExecutionId{801});

    GERDOS_CHECK(backend.submit(operation, long_attempt));
    GERDOS_CHECK(backend.submit(operation, short_attempt));
    GERDOS_CHECK(backend.in_flight_count() == 2);

    // Attempts are single-shot.
    GERDOS_CHECK(!backend.submit(operation, long_attempt));

    std::vector<BackendCompletion> completed;

    // Both attempts progress concurrently: one poll is one virtual step.
    backend.poll(completed);
    GERDOS_CHECK(completed.empty());
    GERDOS_CHECK(backend.elapsed_steps() == 1);
    GERDOS_CHECK(backend.in_flight_count() == 2);

    // The default-work attempt completes first; failures are reported.
    backend.poll(completed);
    GERDOS_CHECK(completed.size() == 1);
    GERDOS_CHECK(completed.front().execution == ExecutionId{801});
    GERDOS_CHECK(!completed.front().succeeded);
    GERDOS_CHECK(
        completed.front().duration_ns ==
        2 * SimulatedBackend::ns_per_step);
    GERDOS_CHECK(backend.in_flight_count() == 1);

    backend.poll(completed);
    GERDOS_CHECK(completed.size() == 1);

    backend.poll(completed);
    GERDOS_CHECK(completed.size() == 2);
    GERDOS_CHECK(completed.back().execution == ExecutionId{800});
    GERDOS_CHECK(completed.back().succeeded);
    GERDOS_CHECK(
        completed.back().duration_ns ==
        4 * SimulatedBackend::ns_per_step);
    GERDOS_CHECK(backend.in_flight_count() == 0);

    // Completions are reported once.
    backend.poll(completed);
    GERDOS_CHECK(completed.size() == 2);
    GERDOS_CHECK(backend.elapsed_steps() == 5);

    // Completed attempts cannot be resubmitted.
    GERDOS_CHECK(!backend.submit(operation, long_attempt));

    // ---------------------------------------------------------------------
    // 2. Fresh attempts use the default work
    // ---------------------------------------------------------------------

    Execution default_attempt{
        ExecutionDescription{
            ExecutionId{802},
            OperationId{700},
        }};

    GERDOS_CHECK(backend.submit(operation, default_attempt));

    completed.clear();
    backend.poll(completed);
    GERDOS_CHECK(completed.empty());

    backend.poll(completed);
    GERDOS_CHECK(completed.size() == 1);
    GERDOS_CHECK(completed.front().execution == ExecutionId{802});
    GERDOS_CHECK(completed.front().succeeded);

    return 0;
}