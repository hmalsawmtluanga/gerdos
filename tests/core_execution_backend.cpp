#include "test_check.hpp"

#include <type_traits>
#include <vector>

#include "gerdos/core/execution_backend.hpp"

namespace {

using namespace gerdos;

// The seam shape is part of the contract: the interface is abstract,
// non-copyable, and immovable so implementations live behind it.
static_assert(std::is_abstract_v<ExecutionBackend>);
static_assert(!std::is_copy_constructible_v<ExecutionBackend>);
static_assert(!std::is_copy_assignable_v<ExecutionBackend>);
static_assert(!std::is_move_constructible_v<ExecutionBackend>);
static_assert(!std::is_move_assignable_v<ExecutionBackend>);

// A completion carries identity and outcome only.
static_assert(
    std::is_same_v<
        decltype(BackendCompletion::execution),
        ExecutionId>);

class RecordingBackend final : public ExecutionBackend {
public:
    [[nodiscard]] bool submit(
        const Operation&,
        const Execution& execution) override {
        submitted_ = execution.description().id;
        return true;
    }

    void poll(std::vector<BackendCompletion>& completed) override {
        if (submitted_.valid()) {
            completed.push_back(
                BackendCompletion{
                    submitted_,
                    true,
                    1'000'000,
                });

            submitted_ = ExecutionId{};
        }
    }

private:
    ExecutionId submitted_{};
};

} // namespace

int main() {
    using namespace gerdos;

    RecordingBackend backend;

    const Operation operation{OperationDescription{
        OperationId{1},
        {},
        {},
        {},
        {},
    }};

    const Execution execution{ExecutionDescription{
        ExecutionId{2},
        OperationId{1},
    }};

    GERDOS_CHECK(backend.submit(operation, execution));

    std::vector<BackendCompletion> completed;
    backend.poll(completed);

    GERDOS_CHECK(completed.size() == 1);
    GERDOS_CHECK(completed.front().execution == ExecutionId{2});
    GERDOS_CHECK(completed.front().succeeded);

    // Completions are reported once.
    backend.poll(completed);
    GERDOS_CHECK(completed.size() == 1);

    return 0;
}