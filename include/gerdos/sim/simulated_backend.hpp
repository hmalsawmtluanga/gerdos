#pragma once

#include <cstddef>
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
    explicit SimulatedBackend(std::size_t default_work = 1) noexcept
        : default_work_(default_work) {}

    // Scenario configuration: the virtual work one attempt requires.
    void set_work(ExecutionId id, std::size_t steps) {
        work_[id] = steps;
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
        const auto work =
            work_it == work_.end() ? default_work_ : work_it->second;

        in_flight_.push_back(
            InFlight{
                id,
                work,
                !failures_.contains(id),
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
    };

    std::size_t default_work_;
    std::unordered_map<ExecutionId, std::size_t> work_;
    std::unordered_set<ExecutionId> failures_;
    std::vector<InFlight> in_flight_;
    std::unordered_set<ExecutionId> submitted_;
    std::size_t elapsed_steps_{0};
};

} // namespace gerdos