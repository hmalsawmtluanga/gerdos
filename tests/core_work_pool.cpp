#include "test_check.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
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

    // ---------------------------------------------------------------------
    // 2. Subset hygiene: headers stay file- and console-free
    // ---------------------------------------------------------------------

    {
        // Headers speak in containers and views, never in files or
        // console output: filesystem/fstream/iostream appear in test
        // and demo programs only. Enforced by scanning every header.
        // (This test is the enforcer — it includes filesystem itself
        // to walk the tree, which is exactly the allowed use. Paths
        // resolve against the GERDOS_SOURCE_DIR compile definition.)
        const char* roots[] = {
            GERDOS_SOURCE_DIR "/include/gerdos/core",
            GERDOS_SOURCE_DIR "/include/gerdos/cpu",
            GERDOS_SOURCE_DIR "/include/gerdos/adapters",
        };

        for (const char* root : roots) {
            for (const auto& entry :
                 std::filesystem::recursive_directory_iterator(root)) {
                if (!entry.is_regular_file() ||
                    entry.path().extension() != ".hpp") {
                    continue;
                }

                std::ifstream input(entry.path());
                GERDOS_CHECK(input.good());
                std::string line;

                while (std::getline(input, line)) {
                    GERDOS_CHECK(
                        line.find("filesystem") == std::string::npos);
                    GERDOS_CHECK(
                        line.find("fstream") == std::string::npos);
                    GERDOS_CHECK(
                        line.find("iostream") == std::string::npos);
                    GERDOS_CHECK(
                        line.find("std::cout") == std::string::npos);
                    GERDOS_CHECK(
                        line.find("std::cerr") == std::string::npos);
                    GERDOS_CHECK(
                        line.find("printf") == std::string::npos);
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 3. Revival bound: the whole runtime fits small hardware
    // ---------------------------------------------------------------------

    {
        // Thread budget: three backend pools (CPU, OpenCL, Vulkan) of
        // fixed workers each — bounded, never hardware-derived. The
        // total is small and constant; sharing one pool across backends
        // is declared future work, so this pins the current bound.
        constexpr std::size_t kBackendPools = 3;
        GERDOS_CHECK(WorkPool::kWorkers == 4);
        GERDOS_CHECK(WorkPool::kQueueBound == 64);
        GERDOS_CHECK(kBackendPools * WorkPool::kWorkers <= 12);
        GERDOS_CHECK(kBackendPools * WorkPool::kQueueBound <= 192);
    }

    return 0;
}

