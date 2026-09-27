#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gerdos/core/dtype.hpp"
#include "gerdos/core/work_pool.hpp"
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
        // Allocations are byte buffers in the work's dtype; a record
        // retargeted to another dtype reinitializes, never reinterprets.
        std::vector<std::pair<DataResidencyRef, std::optional<Allocation>>> undo;
        jobs_.reserve(jobs_.size() + 1);

        try {
            std::vector<Buffer> keeps;
            std::vector<WorkDtype> dtypes;

            for (const auto& entry : binding->data) {
                const auto& ref = entry.residency;
                auto it = allocations_.find(ref);
                const auto bytes =
                    work.storage_elements() * work.storage_bytes();

                // Allocations keep their own dtype across attempts: a
                // record touched by F32 work then I8 work keeps its
                // storage and converts per operation in run(). Only a
                // missing record or an element-count mismatch
                // reallocates (fresh storage takes the work's dtype).
                // Retargeting on dtype alone would wipe the operand.
                const bool count_matches =
                    it != allocations_.end() &&
                    it->second.bytes / dtype_bytes(it->second.dtype) ==
                        work.storage_elements();

                if (!count_matches) {
                    undo.emplace_back(
                        ref,
                        it == allocations_.end()
                            ? std::optional<Allocation>{}
                            : std::optional<Allocation>{it->second});
                    Allocation fresh;
                    fresh.dtype = work.dtype;
                    fresh.bytes = bytes;
                    fresh.storage =
                        std::make_shared<std::vector<unsigned char>>(
                            bytes, 0);
                    // Fresh storage decodes to F32 1.0 per element — the
                    // same initial value the F32 path always had.
                    std::vector<float> ones(work.storage_elements(), 1.0f);
                    encode_all(
                        work.dtype,
                        ones.data(),
                        fresh.storage->data(),
                        ones.size());
                    it = allocations_.insert_or_assign(ref, std::move(fresh))
                             .first;
                }

                keeps.push_back(it->second.storage);
                dtypes.push_back(it->second.dtype);
            }

            auto launched = pool_.try_submit(
                [binding_copy = *binding,
                 keeps = std::move(keeps),
                 dtypes = std::move(dtypes),
                 work,
                 succeeded]() {
                    std::vector<unsigned char*> buffers;

                    for (const auto& keep : keeps) {
                        buffers.push_back(keep->data());
                    }

                    return run(
                        binding_copy, buffers, dtypes, work, succeeded);
                });

            if (!launched.has_value()) {
                throw std::bad_alloc();
            }

            jobs_.push_back(Job{id, std::move(*launched)});
        } catch (...) {
            // The launch failed: nothing was enqueued, and every change
            // made for it is undone.
            for (const auto& [ref, previous] : undo) {
                if (previous.has_value()) {
                    allocations_[ref] = *previous;
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

    // Inspection surface: element count, not bytes — the name predates
    // dtypes and stays for compatibility; byte size is elements times
    // the record's dtype width.
    [[nodiscard]] std::size_t allocation_bytes(
        DataResidencyRef ref) const noexcept {
        const auto it = allocations_.find(ref);

        if (it == allocations_.end()) {
            return 0;
        }

        return it->second.bytes / dtype_bytes(it->second.dtype);
    }

    [[nodiscard]] float sample(DataResidencyRef ref, std::size_t index) {
        const auto it = allocations_.find(ref);

        if (it == allocations_.end()) {
            return 0.0f;
        }

        const auto width = dtype_bytes(it->second.dtype);

        if ((index + 1) * width > it->second.bytes) {
            return 0.0f;
        }

        return decode_element(
            it->second.dtype, it->second.storage->data(), index);
    }

    [[nodiscard]] bool allocations_equal(
        DataResidencyRef left,
        DataResidencyRef right) const noexcept {
        const auto left_it = allocations_.find(left);
        const auto right_it = allocations_.find(right);

        return left_it != allocations_.end() &&
               right_it != allocations_.end() &&
               left_it->second.dtype == right_it->second.dtype &&
                left_it->second.bytes == right_it->second.bytes &&
                *left_it->second.storage == *right_it->second.storage;
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

    using Buffer = std::shared_ptr<std::vector<unsigned char>>;

    struct Allocation {
        WorkDtype dtype{WorkDtype::F32};
        std::size_t bytes{0};
        Buffer storage;
    };

    // The declared work over the bound representations: exact copying when
    // the description says so, otherwise the affine elementwise transform.
    static Outcome run(
        const PhysicalBinding& binding,
        const std::vector<unsigned char*>& buffers,
        const std::vector<WorkDtype>& dtypes,
        const WorkDescription& work,
        bool configured) {
        const auto begin = std::chrono::steady_clock::now();

        // Structural conversion: decode every bound representation to
        // F32 scratch once, compute entirely in F32, encode back once.
        // Mixed-dtype attempts convert explicitly through the shared
        // helpers; engine semantics stay identical across dtypes.
        // Entries sharing one residency share one scratch: aliased
        // in-place sources must observe each pass's write, exactly as
        // the F32 path always did.
        std::vector<std::vector<float>> owned;
        std::vector<std::size_t> scratch_for(buffers.size(), 0);
        for (std::size_t b = 0; b < buffers.size(); ++b) {
            bool shared = false;
            for (std::size_t earlier = 0; earlier < b; ++earlier) {
                if (buffers[earlier] == buffers[b]) {
                    scratch_for[b] = scratch_for[earlier];
                    shared = true;
                    break;
                }
            }
            if (shared) {
                continue;
            }
            scratch_for[b] = owned.size();
            owned.emplace_back(work.storage_elements(), 0.0f);
            decode_all(dtypes[b], buffers[b], owned.back().data(), work.storage_elements());
        }
        std::vector<float*> f32;
        f32.reserve(buffers.size());
        for (std::size_t b = 0; b < buffers.size(); ++b) {
            f32.push_back(owned[scratch_for[b]].data());
        }
        const auto encode_back = [&]() {
            for (std::size_t b = 0; b < buffers.size(); ++b) {
                if (is_producing(binding.data[b].role)) {
                    encode_all(dtypes[b], owned[scratch_for[b]].data(), buffers[b], work.storage_elements());
                }
            }
        };

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

            float* destination = f32[out];
            const auto source_index =
                work_source_index(binding, out);
            const bool has_source = source_index < binding.data.size();

            if (work.form == WorkForm::MATRIX_PRODUCT) {
                if (consuming.size() < 2) {
                    continue;
                }

                const float* left = f32[consuming[0]];
                const float* right = f32[consuming[1]];

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

                const float* origin = f32[source_index];

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

            if (work.form == WorkForm::REDUCE_MAX) {
                if (!has_source) {
                    continue;
                }

                const float* origin = f32[source_index];

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    float peak = origin[0];

                    for (std::size_t i = 1; i < work.elements; ++i) {
                        if (origin[i] > peak) {
                            peak = origin[i];
                        }
                    }

                    destination[0] =
                        destination[0] * work.destination_scale +
                        peak * work.source_scale + work.constant;
                }

                continue;
            }

            if (work.form == WorkForm::REDUCE_MIN) {
                if (!has_source) {
                    continue;
                }

                const float* origin = f32[source_index];

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    float floor = origin[0];

                    for (std::size_t i = 1; i < work.elements; ++i) {
                        if (origin[i] < floor) {
                            floor = origin[i];
                        }
                    }

                    destination[0] =
                        destination[0] * work.destination_scale +
                        floor * work.source_scale + work.constant;
                }

                continue;
            }

            if (work.form == WorkForm::ELEMENTWISE_MIN ||
                work.form == WorkForm::ELEMENTWISE_MAX) {
                if (consuming.size() < 2 ||
                    binding.data[consuming[0]].residency.data ==
                        binding.data[out].residency.data ||
                    binding.data[consuming[1]].residency.data ==
                        binding.data[out].residency.data) {
                    continue;
                }

                const float* left = f32[consuming[0]];
                const float* right = f32[consuming[1]];
                const bool take_min =
                    work.form == WorkForm::ELEMENTWISE_MIN;

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    for (std::size_t i = 0; i < work.elements; ++i) {
                        const float chosen = take_min
                            ? (left[i] < right[i] ? left[i] : right[i])
                            : (left[i] > right[i] ? left[i] : right[i]);
                        destination[i] =
                            destination[i] * work.destination_scale +
                            chosen * work.source_scale + work.constant;
                    }
                }

                continue;
            }

            if (work.form == WorkForm::MASK_SELECT) {
                if (consuming.size() < 3 ||
                    binding.data[consuming[0]].residency.data ==
                        binding.data[out].residency.data ||
                    binding.data[consuming[1]].residency.data ==
                        binding.data[out].residency.data ||
                    binding.data[consuming[2]].residency.data ==
                        binding.data[out].residency.data) {
                    continue;
                }

                const float* predicate = f32[consuming[0]];
                const float* first = f32[consuming[1]];
                const float* second = f32[consuming[2]];

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    for (std::size_t i = 0; i < work.elements; ++i) {
                        const float chosen =
                            predicate[i] != 0.0f ? first[i] : second[i];
                        destination[i] =
                            destination[i] * work.destination_scale +
                            chosen * work.source_scale + work.constant;
                    }
                }

                continue;
            }

            if (work.form == WorkForm::GATHER) {
                if (consuming.size() < 2 ||
                    binding.data[consuming[0]].residency.data ==
                        binding.data[out].residency.data ||
                    binding.data[consuming[1]].residency.data ==
                        binding.data[out].residency.data) {
                    continue;
                }

                const float* table = f32[consuming[0]];
                const float* indices = f32[consuming[1]];
                const auto last =
                    static_cast<std::int64_t>(work.elements - 1);
                const float last_float = static_cast<float>(
                    work.elements - 1);

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    std::vector<float> staged(
                        table, table + work.elements);

                    for (std::size_t i = 0; i < work.elements; ++i) {
                        // Hostile index content can never read out of
                        // bounds: range-check in float first (which also
                        // folds NaN to zero), cast only inside the range,
                        // then clamp the integer again against rounding.
                        const float raw = indices[i];
                        std::int64_t position = 0;

                        if (raw >= 0.0f && raw <= last_float) {
                            position = static_cast<std::int64_t>(raw);

                            if (position > last) {
                                position = last;
                            }
                        } else if (raw > last_float) {
                            position = last;
                        }

                        destination[i] =
                            destination[i] * work.destination_scale +
                            staged[static_cast<std::size_t>(position)] *
                                work.source_scale +
                            work.constant;
                    }
                }

                continue;
            }

            if (work.form == WorkForm::EXPONENTIAL) {
                if (!has_source) {
                    continue;
                }

                const float* origin = f32[source_index];

                for (std::size_t pass = 0; pass < work.passes; ++pass) {
                    for (std::size_t i = 0; i < work.elements; ++i) {
                        destination[i] =
                            destination[i] * work.destination_scale +
                            std::exp(origin[i]) * work.source_scale +
                            work.constant;
                    }
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

            const float* origin = f32[source_index];

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

        encode_back();
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
    std::unordered_map<DataResidencyRef, Allocation> allocations_;
    WorkPool pool_;
    std::vector<Job> jobs_;
    std::unordered_set<ExecutionId> submitted_;
};

} // namespace gerdos