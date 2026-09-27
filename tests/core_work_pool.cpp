#include "test_check.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "gerdos/core/work_pool.hpp"
#include "gerdos/cpu/cpu_backend.hpp"

int main() {
    using namespace gerdos;

    // ---------------------------------------------------------------------
    // 1. Saturation: the bounded queue refuses loudly past its bound
    // ---------------------------------------------------------------------

    {
        WorkPool pool;
        std::atomic<std::size_t> started{0};
        std::promise<void> gate;
        std::shared_future<void> open_gate = gate.get_future().share();

        auto gated = [&] {
            ++started;
            open_gate.wait();
            return 1;
        };

        // Occupy all four workers with gated tasks.
        std::vector<std::future<int>> held;

        for (std::size_t i = 0; i < WorkPool::kWorkers; ++i) {
            auto launched = pool.try_submit(gated);
            GERDOS_CHECK(launched.has_value());
            held.push_back(std::move(*launched));
        }

        // Handshake: every worker is inside its gate (queue drains
        // nothing while they wait).
        while (started.load() < WorkPool::kWorkers) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Fill the bounded queue exactly.
        std::vector<std::future<int>> queued;

        for (std::size_t i = 0; i < WorkPool::kQueueBound; ++i) {
            auto launched = pool.try_submit([] { return 2; });
            GERDOS_CHECK(launched.has_value());
            queued.push_back(std::move(*launched));
        }

        // One past the bound refuses loudly.
        GERDOS_CHECK(!pool.try_submit([] { return 3; }).has_value());

        // Open the gate: all 68 tasks drain coherently.
        gate.set_value();

        for (auto& done : held) {
            GERDOS_CHECK(done.get() == 1);
        }

        for (auto& done : queued) {
            GERDOS_CHECK(done.get() == 2);
        }
    }

    return 0;
}

