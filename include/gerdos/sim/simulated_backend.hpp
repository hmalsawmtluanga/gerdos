#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gerdos/core/execution_backend.hpp"

namespace gerdos {

// A deterministic simulated backend: submitted attempts require an
// configured amount of virtual work, every poll advances all in-flight
// attempts by one step, and completions are reported in submission order.
// Overlapping attempts progress concurrently in virtual time, so overlap
// behaviour is reproducible without threads or hardware.
class SimulatedBackend final : public ExecutionBackend {
public:
    // The simulated clock ticks nanoseconds: one virtual step is one
    // simulated millisecond of work.
    static constexpr std::uint64_t ns_per_step = 1'000'000;

    explicit SimulatedBackend(std::size_t default_work = 1) noexcept
        : default_work_(default_work) {}

    // Scenario configuration: the virtual work one attempt requires.
    void set_work(ExecutionId id, std::size_t steps) {
        work_[id] = steps;
    }

    // Scenario configuration: the speed of one mechanism. An attempt takes
    // the slowest configured step count among the mechanisms it binds, so
    // the planner's choice of mechanism determines realized duration.
    void set_mechanism_steps(ResourceRef ref, std::size_t steps) {
        mechanism_steps_[ref] = steps;
    }

    // Scenario configuration: the attempt completes unsuccessfully.
    void set_failure(ExecutionId id) {
        failures_.insert(id);
    }

    [[nodiscard]] bool submit(
        const Operation&,
        const Execution& execution) override {
        const auto id = execution.description().id;

        if (submitted_.contains(id)) {
            return false;
        }

        const auto work_it = work_.find(id);
        auto work =
            work_it == work_.end() ? default_work_ : work_it->second;

        // Mechanism speed model: an attempt completes when its slowest
        // bound mechanism does.
        const auto* binding = execution.binding();

        if (binding != nullptr) {
            bool configured = false;

            for (const auto& entry : binding->resources) {
                const auto steps_it =
                    mechanism_steps_.find(entry.resource);

                if (steps_it == mechanism_steps_.end()) {
                    continue;
                }

                if (!configured || steps_it->second > work) {
                    work = steps_it->second;
                }

                configured = true;
            }
        }

        in_flight_.push_back(
            InFlight{
                id,
                work,
                !failures_.contains(id),
                work * ns_per_step,
            });

        submitted_.insert(id);
        return true;
    }

    void poll(std::vector<BackendCompletion>& completed) override {
        ++elapsed_steps_;

        std::vector<InFlight> remaining;

        for (auto& entry : in_flight_) {
            if (entry.remaining <= 1) {
                completed.push_back(
                    BackendCompletion{
                        entry.id,
                        entry.succeeded,
                        entry.duration_ns,
                    });
            } else {
                --entry.remaining;
                remaining.push_back(entry);
            }
        }

        in_flight_ = std::move(remaining);
    }

    [[nodiscard]] std::size_t elapsed_steps() const noexcept {
        return elapsed_steps_;
    }

    [[nodiscard]] std::size_t in_flight_count() const noexcept {
        return in_flight_.size();
    }

private:
    struct InFlight {
        ExecutionId id;
        std::size_t remaining;
        bool succeeded;
        std::uint64_t duration_ns;
    };

    std::size_t default_work_;
    std::unordered_map<ExecutionId, std::size_t> work_;
    std::unordered_map<ResourceRef, std::size_t> mechanism_steps_;
    std::unordered_set<ExecutionId> failures_;
    std::vector<InFlight> in_flight_;
    std::unordered_set<ExecutionId> submitted_;
    std::size_t elapsed_steps_{0};
};

} // namespace gerdos