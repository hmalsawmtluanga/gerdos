#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gerdos/core/execution_backend.hpp"
#include "gerdos/core/physical_binding.hpp"

namespace gerdos {

// The CPU backend: real work on real hardware. It owns real allocations for
// the representations it touches — the core's data plane lives behind the
// seam — and executes the operation's declared work over them with real CPU
// kernels. Observed durations are real wall-clock nanoseconds measured
// where the work happens.
//
// Work runs on real background threads; poll never blocks. Allocations are
// resolved at submission and worker threads touch only the buffers they are
// given.
class CpuBackend final : public ExecutionBackend {
public:
    // Scenario configuration: the attempt completes unsuccessfully.
    void set_failure(ExecutionId id) {
        failures_.insert(id);
    }

    [[nodiscard]] bool submit(
        const Operation& operation,
        const Execution& execution) override {
        const auto id = execution.description().id;

        if (submitted_.contains(id)) {
            return false;
        }

        const auto* binding = execution.binding();

        // The seam rejects work that is not well-formed: hostile sizing
        // must never reach an allocation or a kernel.
        const auto work = operation.description().work;

        if (binding == nullptr || !work.valid()) {
            return false;
        }

        const bool succeeded = !failures_.contains(id);
        const auto bytes = work.elements * sizeof(float);

        // Rejection-atomic submission: allocations created here are rolled
        // back if the work cannot be launched, so a refused submit leaves
        // nothing behind.
        std::vector<DataResidencyRef> created;
        jobs_.reserve(jobs_.size() + 1);

        try {
            // Real allocations exist before the work begins, and the
            // worker receives buffer pointers only.
            std::vector<float*> buffers;

            for (const auto& entry : binding->data) {
                if (!allocations_.contains(entry.residency)) {
                    created.push_back(entry.residency);
                }

                buffers.push_back(
                    allocation(entry.residency, bytes).data());
            }

            jobs_.push_back(
                Job{
                    id,
                    std::async(
                        std::launch::async,
                        [binding_copy = *binding,
                         buffers = std::move(buffers),
                         work,
                         succeeded]() {
                            return run(
                                binding_copy, buffers, work, succeeded);
                        }),
                });
        } catch (...) {
            // The launch failed: nothing was enqueued, and allocations
            // created for it are released.
            for (const auto& ref : created) {
                (void)allocations_.erase(ref);
            }

            return false;
        }

        submitted_.insert(id);
        return true;
    }

    void poll(std::vector<BackendCompletion>& completed) override {
        std::vector<Job> remaining;

        for (auto& job : jobs_) {
            if (job.result.wait_for(std::chrono::seconds(0)) !=
                std::future_status::ready) {
                remaining.push_back(std::move(job));
                continue;
            }

            const auto outcome = job.result.get();

            completed.push_back(
                BackendCompletion{
                    job.id,
                    outcome.succeeded,
                    outcome.duration_ns,
                });
        }

        jobs_ = std::move(remaining);
    }

    // Inspection surface for tests: real allocations live behind the seam.
    [[nodiscard]] std::size_t allocation_count() const noexcept {
        return allocations_.size();
    }

    [[nodiscard]] std::size_t allocation_bytes(
        DataResidencyRef ref) const noexcept {
        const auto it = allocations_.find(ref);

        return it == allocations_.end() ? 0 : it->second.size();
    }

    [[nodiscard]] float sample(DataResidencyRef ref, std::size_t index) {
        const auto it = allocations_.find(ref);

        if (it == allocations_.end() ||
            it->second.size() <= index) {
            return 0.0f;
        }

        return it->second[index];
    }

    [[nodiscard]] bool allocations_equal(
        DataResidencyRef left,
        DataResidencyRef right) const noexcept {
        const auto left_it = allocations_.find(left);
        const auto right_it = allocations_.find(right);

        return left_it != allocations_.end() &&
               right_it != allocations_.end() &&
               left_it->second == right_it->second;
    }

private:
    struct Outcome {
        bool succeeded;
        std::uint64_t duration_ns;
    };

    struct Job {
        ExecutionId id;
        std::future<Outcome> result;
    };

    [[nodiscard]] std::vector<float>& allocation(
        DataResidencyRef ref,
        std::size_t bytes) {
        auto& buffer = allocations_[ref];
        const auto elements = bytes / sizeof(float);

        if (buffer.size() != elements) {
            buffer.assign(elements, 1.0f);
        }

        return buffer;
    }

    // The declared work over the bound representations: exact copying when
    // the description says so, otherwise the affine elementwise transform.
    static Outcome run(
        const PhysicalBinding& binding,
        const std::vector<float*>& buffers,
        const WorkDescription& work,
        bool configured) {
        const auto begin = std::chrono::steady_clock::now();

        const bool exact_copy =
            work.passes == 1 && work.destination_scale == 0.0f &&
            work.source_scale == 1.0f && work.constant == 0.0f;

        for (std::size_t out = 0; out < binding.data.size(); ++out) {
            if (!is_producing(binding.data[out].role)) {
                continue;
            }

            float* destination = buffers[out];
            const auto source_index =
                work_source_index(binding, out);
            const bool has_source = source_index < binding.data.size();

            if (!has_source) {
                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    for (std::size_t i = 0; i < work.elements; ++i) {
                        destination[i] =
                            destination[i] * work.destination_scale +
                            work.constant;
                    }
                }

                continue;
            }

            const float* origin = buffers[source_index];

            if (exact_copy && source_index != out) {
                std::memcpy(
                    destination,
                    origin,
                    work.elements * sizeof(float));
                continue;
            }

            for (std::size_t pass = 0; pass < work.passes; ++pass) {
                for (std::size_t i = 0; i < work.elements; ++i) {
                    destination[i] =
                        destination[i] * work.destination_scale +
                        origin[i] * work.source_scale + work.constant;
                }
            }
        }

        const auto end = std::chrono::steady_clock::now();

        return Outcome{
            configured,
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - begin)
                    .count()),
        };
    }

    std::unordered_set<ExecutionId> failures_;
    std::unordered_map<DataResidencyRef, std::vector<float>> allocations_;
    std::vector<Job> jobs_;
    std::unordered_set<ExecutionId> submitted_;
};

} // namespace gerdos