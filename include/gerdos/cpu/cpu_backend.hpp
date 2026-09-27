#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
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
    // Destruction waits for outstanding work: the futures own it.
    ~CpuBackend() {
        for (auto& job : jobs_) {
            (void)job.result.get();
        }
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

        // The seam rejects work that is not well-formed: hostile sizing
        // must never reach an allocation or a kernel.
        const auto work = operation.description().work;

        if (binding == nullptr || !work.valid()) {
            return false;
        }

        const bool succeeded = !failures_.contains(id);

        // Rejection-atomic submission: workers hold shared ownership of
        // their storage, so replacements cannot dangle under a live job,
        // and a failed launch restores exactly what it changed.
        std::vector<std::pair<DataResidencyRef, Buffer>> undo;
        jobs_.reserve(jobs_.size() + 1);

        try {
            std::vector<Buffer> keeps;

            for (const auto& entry : binding->data) {
                const auto& ref = entry.residency;
                const auto it = allocations_.find(ref);

                if (it == allocations_.end() ||
                    it->second->size() != work.storage_elements()) {
                    undo.emplace_back(
                        ref,
                        it == allocations_.end() ? Buffer{}
                                                 : it->second);
                    allocations_[ref] = std::make_shared<
                        std::vector<float>>(
                        work.storage_elements(),
                        1.0f);
                }

                keeps.push_back(allocations_[ref]);
            }

            jobs_.push_back(
                Job{
                    id,
                    std::async(
                        std::launch::async,
                        [binding_copy = *binding,
                         keeps = std::move(keeps),
                         work,
                         succeeded]() {
                            std::vector<float*> buffers;

                            for (const auto& keep : keeps) {
                                buffers.push_back(keep->data());
                            }

                            return run(
                                binding_copy, buffers, work, succeeded);
                        }),
                });
        } catch (...) {
            // The launch failed: nothing was enqueued, and every change
            // made for it is undone.
            for (const auto& [ref, previous] : undo) {
                if (previous) {
                    allocations_[ref] = previous;
                } else {
                    (void)allocations_.erase(ref);
                }
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

        return it == allocations_.end() ? 0 : it->second->size();
    }

    [[nodiscard]] float sample(DataResidencyRef ref, std::size_t index) {
        const auto it = allocations_.find(ref);

        if (it == allocations_.end() ||
            it->second->size() <= index) {
            return 0.0f;
        }

        return (*it->second)[index];
    }

    [[nodiscard]] bool allocations_equal(
        DataResidencyRef left,
        DataResidencyRef right) const noexcept {
        const auto left_it = allocations_.find(left);
        const auto right_it = allocations_.find(right);

        return left_it != allocations_.end() &&
               right_it != allocations_.end() &&
               *left_it->second == *right_it->second;
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

    using Buffer = std::shared_ptr<std::vector<float>>;

    // The declared work over the bound representations: exact copying when
    // the description says so, otherwise the affine elementwise transform.
    static Outcome run(
        const PhysicalBinding& binding,
        const std::vector<float*>& buffers,
        const WorkDescription& work,
        bool configured) {
        const auto begin = std::chrono::steady_clock::now();

        const bool exact_copy =
            work.form == WorkForm::ELEMENTWISE_AFFINE &&
            work.passes == 1 && work.destination_scale == 0.0f &&
            work.source_scale == 1.0f && work.constant == 0.0f;

        // Matrix operands are the first two consuming entries, in binding
        // order.
        std::vector<std::size_t> consuming;

        for (std::size_t in = 0; in < binding.data.size(); ++in) {
            if (is_consuming(binding.data[in].role)) {
                consuming.push_back(in);
            }
        }

        for (std::size_t out = 0; out < binding.data.size(); ++out) {
            if (!is_producing(binding.data[out].role)) {
                continue;
            }

            float* destination = buffers[out];
            const auto source_index =
                work_source_index(binding, out);
            const bool has_source = source_index < binding.data.size();

            if (work.form == WorkForm::MATRIX_PRODUCT) {
                if (consuming.size() < 2) {
                    continue;
                }

                const float* left = buffers[consuming[0]];
                const float* right = buffers[consuming[1]];

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    for (std::size_t i = 0; i < work.rows; ++i) {
                        for (std::size_t j = 0;
                             j < work.columns;
                             ++j) {
                            float accumulated = 0.0f;

                            for (std::size_t k = 0; k < work.inner; ++k) {
                                accumulated +=
                                    left[i * work.inner + k] *
                                    right[k * work.columns + j];
                            }

                            const auto index =
                                i * work.columns + j;
                            destination[index] =
                                destination[index] *
                                    work.destination_scale +
                                accumulated * work.source_scale +
                                work.constant;
                        }
                    }
                }

                continue;
            }

            if (work.form == WorkForm::REDUCE_SUM) {
                if (!has_source) {
                    continue;
                }

                const float* origin = buffers[source_index];

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    float sum = 0.0f;

                    for (std::size_t i = 0; i < work.elements; ++i) {
                        sum += origin[i];
                    }

                    destination[0] =
                        destination[0] * work.destination_scale +
                        sum * work.source_scale + work.constant;
                }

                continue;
            }

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
    std::unordered_map<DataResidencyRef, Buffer> allocations_;
    std::vector<Job> jobs_;
    std::unordered_set<ExecutionId> submitted_;
};

} // namespace gerdos