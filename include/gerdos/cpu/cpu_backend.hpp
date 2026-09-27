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

// The CPU backend: the first backend performing real work on real hardware.
// It owns real allocations for the representations it touches — the core's
// data plane lives behind the seam — and executes real kernels over them:
// movement pairs are copied between representations, compute pairs run
// arithmetic between them. Observed durations are real wall-clock
// nanoseconds measured where the work happens.
//
// Work runs on real background threads; poll never blocks. Allocations are
// resolved at submission and worker threads touch only the buffers they are
// given. The amount of work is backend configuration until Operation work
// descriptions exist.
class CpuBackend final : public ExecutionBackend {
public:
    // Scenario configuration: the real size of one attempt's work.
    void set_operation_work(
        OperationId id,
        std::size_t bytes,
        std::size_t passes) {
        work_[id] = Work{bytes, passes};
    }

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

        if (binding == nullptr) {
            return false;
        }

        const auto work = work_for(operation.description().id);
        const bool succeeded = !failures_.contains(id);

        // Real allocations exist before the work begins, and the worker
        // receives buffer pointers only — it never touches shared maps.
        std::vector<float*> buffers;

        for (const auto& entry : binding->data) {
            buffers.push_back(
                allocation(entry.residency, work.bytes).data());
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
                        return run(binding_copy, buffers, work, succeeded);
                    }),
            });

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
    struct Work {
        std::size_t bytes;
        std::size_t passes;
    };

    struct Outcome {
        bool succeeded;
        std::uint64_t duration_ns;
    };

    struct Job {
        ExecutionId id;
        std::future<Outcome> result;
    };

    [[nodiscard]] Work work_for(OperationId id) const noexcept {
        const auto it = work_.find(id);

        return it == work_.end() ? Work{4096, 1} : it->second;
    }

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

    // Real work over the bound representations: movement pairs are copied
    // between representations; compute pairs run arithmetic between them;
    // unpaired producing entries are written in place.
    static Outcome run(
        const PhysicalBinding& binding,
        const std::vector<float*>& buffers,
        Work work,
        bool configured) {
        const auto begin = std::chrono::steady_clock::now();

        for (std::size_t out = 0; out < binding.data.size(); ++out) {
            const auto& producing = binding.data[out];

            if (!is_producing(producing.role)) {
                continue;
            }

            float* destination = buffers[out];
            const float* origin = nullptr;
            std::size_t origin_index = 0;

            for (std::size_t in = 0; in < binding.data.size(); ++in) {
                const auto& consuming = binding.data[in];

                if (is_consuming(consuming.role) &&
                    consuming.residency.data == producing.residency.data) {
                    origin = buffers[in];
                    origin_index = in;
                    break;
                }
            }

            const auto elements = work.bytes / sizeof(float);

            if (origin == nullptr) {
                for (std::size_t i = 0; i < elements; ++i) {
                    destination[i] = destination[i] * 1.0001f + 0.0001f;
                }

                continue;
            }

            if (binding.data[origin_index].residency !=
                    producing.residency &&
                producing.role == DataBindingRole::DESTINATION) {
                std::memcpy(
                    destination,
                    origin,
                    elements * sizeof(float));
            } else {
                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    for (std::size_t i = 0; i < elements; ++i) {
                        destination[i] =
                            destination[i] * 0.5f + origin[i] * 1.5f;
                    }
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

    std::unordered_map<OperationId, Work> work_;
    std::unordered_set<ExecutionId> failures_;
    std::unordered_map<DataResidencyRef, std::vector<float>> allocations_;
    std::vector<Job> jobs_;
    std::unordered_set<ExecutionId> submitted_;
};

} // namespace gerdos