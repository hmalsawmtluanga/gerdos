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

int threaded_main();

int main() {
    using namespace gerdos;

#ifdef GERDOS_NO_THREADS
    // Inline-order proof (replaces saturation under the flag): tasks
    // execute before try_submit returns, in submission order. Running
    // the gated saturation shape here would deadlock by construction
    // (inline tasks block on a gate nobody opens), so it never runs
    // under this flag — the contract says so explicitly.
    {
        WorkPool pool;
        GERDOS_CHECK(pool.pending() == 0);
        int order = 0;
        int first = -1;
        int second = -1;
        auto a = pool.try_submit([&] { return order++; });
        auto b = pool.try_submit([&] { return order++; });
        GERDOS_CHECK(a.has_value() && b.has_value());
        first = a->get();
        second = b->get();
        GERDOS_CHECK(first == 0);
        GERDOS_CHECK(second == 1);
        GERDOS_CHECK(pool.pending() == 0);
    }

    return 0;
#else
    return threaded_main();
#endif
}

int threaded_main() {
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
    // 3. Choke-point guard: no direct producing-target assignment
    // ---------------------------------------------------------------------

    {
        // Every producing record passes through claim_target: scan the
        // planner for direct assignments that bypass it. Allowed shapes
        // are claim_target(...), choose_target(...) (which filters
        // internally), free_sole_residency(...) (which filters
        // internally), and nullptr initialization. Anything else fails
        // loudly here instead of reaching review.
        std::ifstream input(
            std::string(GERDOS_SOURCE_DIR) +
            "/include/gerdos/core/binding_planner.hpp");
        GERDOS_CHECK(input.good());
        std::string line;
        std::size_t line_number = 0;

        while (std::getline(input, line)) {
            ++line_number;

            // Normalize whitespace first: tabs and runs collapse, so
            // `target\t=\t...` cannot evade the assignment match.
            std::string flat;

            for (const char c : line) {
                if (c == ' ' || c == '\t') {
                    if (!flat.empty() && flat.back() != ' ') {
                        flat.push_back(' ');
                    }
                } else {
                    flat.push_back(c);
                }
            }

            const bool assigns =
                flat.find("target =") != std::string::npos;

            if (!assigns) {
                continue;
            }

            const bool allowed =
                flat.find("claim_target") != std::string::npos ||
                flat.find("choose_target") != std::string::npos ||
                flat.find("free_sole_residency") != std::string::npos ||
                flat.find("= nullptr") != std::string::npos;
            GERDOS_CHECK(allowed);

            // sole_residency is identity-only: placement reads through
            // free_sole_residency. A producing assignment naming the
            // raw helper fails even beside an allowed callee.
            GERDOS_CHECK(
                flat.find("sole_residency") == std::string::npos ||
                flat.find("free_sole_residency") != std::string::npos);
        }
    }

    // ---------------------------------------------------------------------
    // 4. Revival bound: the whole runtime fits small hardware
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

