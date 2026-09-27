#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <CL/opencl.hpp>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/work_pool.hpp"
#include "gerdos/core/dtype.hpp"
#include "gerdos/core/execution_backend.hpp"

namespace gerdos {

// The heterogeneous backend: one backend seam, real work executed by the
// engine that owns an attempt's bound mechanisms — CPU loops for host
// mechanisms, real OpenCL kernels on the accelerator for accelerator
// mechanisms. Which devices are accelerators is configuration, never
// vocabulary.
//
// The data plane is home-respecting: every representation has exactly one
// real allocation, living where its residency record says it lives — host
// memory for host-homed records, device memory for accelerator-homed ones.
// Work crossing homes performs real staging, exactly as a transfer engine
// mediates between host and device memory. Durations are real wall-clock
// nanoseconds measured where the work happens.
//
// Both engines execute the operation's declared work — the same elementwise
// affine semantics — and exact copying is the fast path.
//
// Threading contract: work runs on real background threads and poll never
// blocks. Workers hold copied OpenCL handles and their own slots — never
// this object's maps. The inspection surface is valid only while no jobs
// are in flight, and destruction waits for outstanding work.
class HeterogeneousBackend final : public ExecutionBackend {
public:
    // Destruction waits for outstanding work: the futures own it.
    ~HeterogeneousBackend() {
        for (auto& job : jobs_) {
            (void)job.result.get();
        }
    }

    HeterogeneousBackend(
        const DataRegistry& data,
        DeviceId accelerator)
        : data_(data),
          accelerator_(accelerator) {
        std::vector<cl::Platform> platforms;

        if (cl::Platform::get(&platforms) != CL_SUCCESS ||
            platforms.empty()) {
            return;
        }

        std::vector<cl::Device> devices;

        if (platforms.front().getDevices(
                CL_DEVICE_TYPE_GPU, &devices) != CL_SUCCESS ||
            devices.empty()) {
            return;
        }

        device_ = devices.front();
        context_ = cl::Context(device_);
        program_ = cl::Program(
            context_,
            "__kernel void transform(__global float* d,"
            " const __global float* o, float dscale, float sscale,"
            " float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    d[i] = d[i] * dscale + o[i] * sscale + bias;"
            "  }"
            "}"
            "__kernel void fill(__global float* d, float dscale,"
            " float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    d[i] = d[i] * dscale + bias;"
            "  }"
            "}"
            "__kernel void matrix_product(__global float* d,"
            " const __global float* a, const __global float* b,"
            " ulong rows, ulong inner, ulong columns, float dscale,"
            " float sscale, float bias, ulong passes) {"
            "  size_t index = get_global_id(0);"
            "  ulong i = index / columns;"
            "  ulong j = index % columns;"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    float acc = 0.0f;"
            "    for (ulong k = 0; k < inner; ++k) {"
            "      acc += a[i * inner + k] * b[k * columns + j];"
            "    }"
            "    d[index] = d[index] * dscale + acc * sscale + bias;"
            "  }"
            "}"
            "__kernel void matmul_tile(__global float* d,"
            " const __global float* a, const __global float* b,"
            " ulong rows, ulong inner, ulong columns, float dscale,"
            " float sscale, float bias, ulong passes) {"
            "  __local float at[16][16];"
            "  __local float bt[16][16];"
            "  ulong row = get_group_id(1) * 16 + get_local_id(1);"
            "  ulong col = get_group_id(0) * 16 + get_local_id(0);"
            "  ulong lid1 = get_local_id(1);"
            "  ulong lid0 = get_local_id(0);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    float acc = 0.0f;"
            "    for (ulong t = 0; t < (inner + 15) / 16; ++t) {"
            "      ulong arow = row;"
            "      ulong acol = t * 16 + lid0;"
            "      at[lid1][lid0] = (arow < rows && acol < inner) ?"
            "        a[arow * inner + acol] : 0.0f;"
            "      ulong brow = t * 16 + lid1;"
            "      ulong bcol = col;"
            "      bt[lid1][lid0] = (brow < inner && bcol < columns) ?"
            "        b[brow * columns + bcol] : 0.0f;"
            "      barrier(CLK_LOCAL_MEM_FENCE);"
            "      for (ulong k = 0; k < 16; ++k) {"
            "        acc += at[lid1][k] * bt[k][lid0];"
            "      }"
            "      barrier(CLK_LOCAL_MEM_FENCE);"
            "    }"
            "    ulong index = row * columns + col;"
            "    if (row < rows && col < columns) {"
            "      d[index] = d[index] * dscale + acc * sscale + bias;"
            "    }"
            "  }"
            "}"
            "__kernel void transform_vec(__global float* d,"
            " const __global float* o, float dscale, float sscale,"
            " float bias, ulong passes, ulong n) {"
            "  size_t i = get_global_id(0) * 4;"
            "  ulong remaining = (n > i) ? (n - i) : 0;"
            "  if (remaining == 0) { return; }"
            "  for (ulong p = 0; p < passes; ++p) {"
            "    if (remaining >= 4) {"
            "      float4 dd = vload4(0, d + i);"
            "      float4 oo = vload4(0, o + i);"
            "      vstore4(dd * dscale + oo * sscale + bias, 0, d + i);"
            "    } else {"
            "      for (ulong k = 0; k < remaining; ++k) {"
            "        d[i + k] = d[i + k] * dscale + o[i + k] * sscale + bias;"
            "      }"
            "    }"
            "  }"
            "}"
            "__kernel void exponential_vec(__global float* d,"
            " const __global float* o, float dscale, float sscale,"
            " float bias, ulong passes, ulong n) {"
            "  size_t i = get_global_id(0) * 4;"
            "  ulong remaining = (n > i) ? (n - i) : 0;"
            "  if (remaining == 0) { return; }"
            "  for (ulong p = 0; p < passes; ++p) {"
            "    if (remaining >= 4) {"
            "      float4 dd = vload4(0, d + i);"
            "      float4 oo = vload4(0, o + i);"
            "      vstore4(dd * dscale + exp(oo) * sscale + bias, 0, d + i);"
            "    } else {"
            "      for (ulong k = 0; k < remaining; ++k) {"
            "        d[i + k] = d[i + k] * dscale + exp(o[i + k]) * sscale + bias;"
            "      }"
            "    }"
            "  }"
            "}"
            "__kernel void fill_vec(__global float* d,"
            " float dscale, float bias, ulong passes, ulong n) {"
            "  size_t i = get_global_id(0) * 4;"
            "  ulong remaining = (n > i) ? (n - i) : 0;"
            "  if (remaining == 0) { return; }"
            "  for (ulong p = 0; p < passes; ++p) {"
            "    if (remaining >= 4) {"
            "      float4 dd = vload4(0, d + i);"
            "      vstore4(dd * dscale + bias, 0, d + i);"
            "    } else {"
            "      for (ulong k = 0; k < remaining; ++k) {"
            "        d[i + k] = d[i + k] * dscale + bias;"
            "      }"
            "    }"
            "  }"
            "}"
            "__kernel void reduce_sum(__global float* d,"
            " const __global float* o, ulong n, float dscale,"
            " float sscale, float bias, ulong passes) {"
            "  if (get_global_id(0) != 0) { return; }"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    float sum = 0.0f;"
            "    for (ulong k = 0; k < n; ++k) { sum += o[k]; }"
            "    d[0] = d[0] * dscale + sum * sscale + bias;"
            "  }"
            "}"
            "__kernel void exponential(__global float* d,"
            " const __global float* o, float dscale, float sscale,"
            " float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    d[i] = d[i] * dscale + exp(o[i]) * sscale + bias;"
            "  }"
            "}"
            "__kernel void reduce_max(__global float* d,"
            " const __global float* o, ulong n, float dscale,"
            " float sscale, float bias, ulong passes) {"
            "  if (get_global_id(0) != 0) { return; }"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    float peak = o[0];"
            "    for (ulong k = 1; k < n; ++k) {"
            "      peak = fmax(peak, o[k]);"
            "    }"
            "    d[0] = d[0] * dscale + peak * sscale + bias;"
            "  }"
            "}"
            "__kernel void reduce_min(__global float* d,"
            " const __global float* o, ulong n, float dscale,"
            " float sscale, float bias, ulong passes) {"
            "  if (get_global_id(0) != 0) { return; }"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    float floor_value = o[0];"
            "    for (ulong k = 1; k < n; ++k) {"
            "      floor_value = fmin(floor_value, o[k]);"
            "    }"
            "    d[0] = d[0] * dscale + floor_value * sscale + bias;"
            "  }"
            "}"
            "__kernel void elementwise_min(__global float* d,"
            " const __global float* a, const __global float* b,"
            " float dscale, float sscale, float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    d[i] = d[i] * dscale + fmin(a[i], b[i]) * sscale + bias;"
            "  }"
            "}"
            "__kernel void elementwise_max(__global float* d,"
            " const __global float* a, const __global float* b,"
            " float dscale, float sscale, float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    d[i] = d[i] * dscale + fmax(a[i], b[i]) * sscale + bias;"
            "  }"
            "}"
            "__kernel void mask_select(__global float* d,"
            " const __global float* m, const __global float* a,"
            " const __global float* b,"
            " float dscale, float sscale, float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    float chosen = (m[i] != 0.0f) ? a[i] : b[i];"
            "    d[i] = d[i] * dscale + chosen * sscale + bias;"
            "  }"
            "}"
            "__kernel void gather(__global float* d,"
            " const __global float* v, const __global float* idx,"
            " ulong n, float dscale, float sscale, float bias,"
            " ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  float last_value = (float) (n - 1);"
            "  long last = (long) (n - 1);"
            "  for (ulong pass = 0; pass < passes; ++pass) {"
            "    float raw = idx[i];"
            "    long position = 0;"
            "    if (raw >= 0.0f && raw <= last_value) {"
            "      position = (long) raw;"
            "      position = position > last ? last : position;"
            "    } else if (raw > last_value) {"
            "      position = last;"
            "    }"
            "    d[i] = d[i] * dscale + v[position] * sscale + bias;"
            "  }"
            "}");

        if (program_.build(std::vector<cl::Device>{device_}) !=
            CL_SUCCESS) {
            context_ = cl::Context();
            return;
        }

        // Warmup: one-time driver costs must not appear in measured work.
        {
            cl::CommandQueue warmup(context_, device_);
            cl::Buffer scratch(
                context_, CL_MEM_READ_WRITE, sizeof(float), nullptr);
            cl::Kernel kernel(program_, "fill");
            kernel.setArg(0, scratch);
            kernel.setArg(1, 1.0f);
            kernel.setArg(2, 0.0f);
            kernel.setArg(3, static_cast<cl_ulong>(1));
            warmup.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(1));
            warmup.finish();
        }

        // Tiled matmul needs 16x16 work-groups: query capability once
        // and fall back to the naive kernel fail-closed where the
        // device cannot serve it.
        std::size_t max_group = 0;
        device_.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_group);
        tiled_ok_ = max_group >= 256;

        available_ = true;
    }

    [[nodiscard]] bool gpu_available() const noexcept {
        return available_;
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

        const bool gpu = on_accelerator(*binding);

        if (gpu && !available_) {
            return false;
        }

        const bool succeeded = !failures_.contains(id);
        const auto bytes = work.storage_elements() * work.storage_bytes();

        // Rejection-atomic submission: workers hold shared ownership of
        // their storage and copied OpenCL handles — never this object's
        // maps — and a failed launch restores exactly what it changed.
        std::vector<std::pair<DataResidencyRef, std::optional<Allocation>>>
            undo;
        jobs_.reserve(jobs_.size() + 1);

        try {
            std::vector<Slot> slots;

            for (const auto& entry : binding->data) {
                const auto& ref = entry.residency;
                const auto it = allocations_.find(ref);

                // Allocations keep their own dtype across attempts (see
                // the CPU backend): only a missing record or an
                // element-count mismatch reallocates.
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
                }

                slots.push_back(slot_for(ref, bytes, work.dtype));
            }

            auto launched = pool_.try_submit(
                [context = context_,
                 device = device_,
                 program = program_,
                 binding_copy = *binding,
                 slots = std::move(slots),
                 work,
                 succeeded,
                 gpu,
                 tiled = tiled_ok_]() {
                    return run(
                        context,
                        device,
                        program,
                        binding_copy,
                        slots,
                        work,
                        succeeded,
                        gpu,
                        tiled);
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

    // Inspection surface for tests: real allocations live behind the seam,
    // one per representation, at the home its record declares.
    [[nodiscard]] std::size_t allocation_count() const noexcept {
        return allocations_.size();
    }

    [[nodiscard]] bool allocations_equal(
        DataResidencyRef left,
        DataResidencyRef right) {
        const auto left_bytes = allocation_bytes(left);
        const auto right_bytes = allocation_bytes(right);

        if (left_bytes == 0 || left_bytes != right_bytes) {
            return false;
        }

        std::vector<unsigned char> left_data;
        std::vector<unsigned char> right_data;
        left_data.resize(left_bytes);
        right_data.resize(right_bytes);

        read(left, left_data.data());
        read(right, right_data.data());

        return left_data == right_data;
    }

    [[nodiscard]] float sample(DataResidencyRef ref, std::size_t index) {
        const auto it = allocations_.find(ref);

        if (it == allocations_.end()) {
            return 0.0f;
        }

        const auto width = dtype_bytes(it->second.dtype);
        const auto bytes = it->second.bytes;

        if ((index + 1) * width > bytes) {
            return 0.0f;
        }

        // read() copies the whole allocation; give it the whole buffer.
        // Device homes stage F32 working copies: convert back through
        // the shared helpers so the inspection path matches the
        // compute path.
        if (it->second.on_device) {
            std::vector<float> working(bytes / width, 0.0f);
            read_device(ref, reinterpret_cast<unsigned char*>(working.data()));
            std::vector<unsigned char> raw(bytes, 0);
            encode_all(it->second.dtype, working.data(), raw.data(), working.size());
            return decode_element(it->second.dtype, raw.data(), index);
        }

        std::vector<unsigned char> raw;
        raw.resize(bytes);
        read(ref, raw.data());
        return decode_element(it->second.dtype, raw.data(), index);
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

    // One real allocation per representation, at its declared home. Host
    // storage is shared-owning: workers hold their buffers alive for as
    // long as they run, so a replacement can never dangle under them.
    struct Allocation {
        bool on_device;
        WorkDtype dtype{WorkDtype::F32};
        std::shared_ptr<std::vector<unsigned char>> host;
        cl::Buffer device;
        std::size_t bytes;
    };

    // What a worker needs, resolved before it starts.
    struct Slot {
        bool on_device;
        WorkDtype dtype{WorkDtype::F32};
        std::shared_ptr<std::vector<unsigned char>> host;
        cl::Buffer device;
        std::size_t bytes;
    };

    [[nodiscard]] bool on_accelerator(const PhysicalBinding& binding) const
        noexcept {
        for (const auto& entry : binding.resources) {
            if (entry.resource.device == accelerator_) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] bool home_on_accelerator(DataResidencyRef ref) const {
        const auto* datum = data_.find_data(ref.data);

        if (datum == nullptr) {
            return false;
        }

        const auto* record = datum->find_residency(ref.residency);

        return record != nullptr &&
               record->description().resource.device == accelerator_;
    }

    [[nodiscard]] Slot slot_for(
        DataResidencyRef ref,
        std::size_t bytes,
        WorkDtype dtype) {
        auto it = allocations_.find(ref);

        const bool count_matches =
            it != allocations_.end() &&
            it->second.bytes / dtype_bytes(it->second.dtype) ==
                bytes / dtype_bytes(dtype);

        if (!count_matches) {
            const bool on_device =
                available_ && home_on_accelerator(ref);

            Allocation fresh{};
            fresh.on_device = on_device;
            fresh.dtype = dtype;
            fresh.bytes = bytes;

            fresh.host =
                std::make_shared<std::vector<unsigned char>>(bytes, 0);
            std::vector<float> ones(bytes / dtype_bytes(dtype), 1.0f);
            encode_all(dtype, ones.data(), fresh.host->data(), ones.size());

            if (on_device) {
                // Device buffers always stage F32 working copies: decode
                // host storage to F32 scratch, then upload the scratch.
                // Conversion is structural at the boundary, identical on
                // both engines.
                std::vector<float> working(ones.size(), 0.0f);
                decode_all(dtype, fresh.host->data(), working.data(), ones.size());
                fresh.device = cl::Buffer(
                    context_,
                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    working.size() * sizeof(float),
                    working.data());
            }

            it = allocations_.insert_or_assign(ref, std::move(fresh)).first;
        }

        auto& allocation = it->second;

        return Slot{
            allocation.on_device,
            allocation.dtype,
            allocation.host,
            allocation.on_device ? allocation.device : cl::Buffer(),
            allocation.bytes,
        };
    }

    [[nodiscard]] std::size_t allocation_bytes(DataResidencyRef ref) {
        const auto it = allocations_.find(ref);

        return it == allocations_.end() ? 0 : it->second.bytes;
    }

    void read_device(DataResidencyRef ref, unsigned char* out) {
        const auto it = allocations_.find(ref);

        if (it == allocations_.end() || !it->second.on_device) {
            return;
        }

        cl::CommandQueue queue(context_, device_);
        queue.enqueueReadBuffer(
            it->second.device,
            CL_TRUE,
            0,
            it->second.bytes / dtype_bytes(it->second.dtype) * sizeof(float),
            out);
    }

    void read(DataResidencyRef ref, unsigned char* out) {
        const auto it = allocations_.find(ref);

        if (it == allocations_.end()) {
            return;
        }

        if (!it->second.on_device) {
            std::memcpy(out, it->second.host->data(), it->second.bytes);
            return;
        }

        cl::CommandQueue queue(context_, device_);
        queue.enqueueReadBuffer(
            it->second.device, CL_TRUE, 0, it->second.bytes, out);
    }

    // The declared work. The engine is chosen by the bound mechanisms; data
    // moves between declared homes with real staging. Static by design: the
    // worker touches only copied OpenCL handles and the slots it was given.
    [[nodiscard]] static Outcome run(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const PhysicalBinding& binding,
        const std::vector<Slot>& slots,
        const WorkDescription& work,
        bool configured,
        bool gpu,
        bool tiled) {
        const auto begin = std::chrono::steady_clock::now();

        // Staging across homes needs a real queue regardless of engine.
        cl::CommandQueue queue(context, device);

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

            const auto& destination = slots[out];
            const auto source_index = work_source_index(binding, out);
            const bool has_source = source_index < binding.data.size();

            if (work.form == WorkForm::MATRIX_PRODUCT) {
                if (consuming.size() >= 2) {
                    matrix_product(
                        context,
                        device,
                        program,
                        slots[consuming[0]],
                        slots[consuming[1]],
                        destination,
                        queue,
                        gpu,
                        tiled,
                        work);
                }

                continue;
            }

            if (work.form == WorkForm::REDUCE_SUM && has_source) {
                reduce_sum(
                    context,
                    device,
                    program,
                    slots[source_index],
                    destination,
                    queue,
                    gpu,
                    work);
                continue;
            }

            if (work.form == WorkForm::REDUCE_MAX && has_source) {
                reduce_max(
                    context,
                    device,
                    program,
                    slots[source_index],
                    destination,
                    queue,
                    gpu,
                    work);
                continue;
            }

            if (work.form == WorkForm::REDUCE_MIN && has_source) {
                reduce_min(
                    context,
                    device,
                    program,
                    slots[source_index],
                    destination,
                    queue,
                    gpu,
                    work);
                continue;
            }

            if ((work.form == WorkForm::ELEMENTWISE_MIN ||
                 work.form == WorkForm::ELEMENTWISE_MAX) &&
                consuming.size() >= 2 &&
                binding.data[consuming[0]].residency.data !=
                    binding.data[out].residency.data &&
                binding.data[consuming[1]].residency.data !=
                    binding.data[out].residency.data) {
                elementwise_select(
                    context,
                    device,
                    program,
                    slots[consuming[0]],
                    slots[consuming[1]],
                    destination,
                    queue,
                    gpu,
                    work);
                continue;
            }

            if (work.form == WorkForm::MASK_SELECT &&
                consuming.size() >= 3 &&
                binding.data[consuming[0]].residency.data !=
                    binding.data[out].residency.data &&
                binding.data[consuming[1]].residency.data !=
                    binding.data[out].residency.data &&
                binding.data[consuming[2]].residency.data !=
                    binding.data[out].residency.data) {
                mask_select(
                    context,
                    device,
                    program,
                    slots[consuming[0]],
                    slots[consuming[1]],
                    slots[consuming[2]],
                    destination,
                    queue,
                    gpu,
                    work);
                continue;
            }

            if (work.form == WorkForm::GATHER &&
                consuming.size() >= 2 &&
                binding.data[consuming[0]].residency.data !=
                    binding.data[out].residency.data &&
                binding.data[consuming[1]].residency.data !=
                    binding.data[out].residency.data) {
                gather(
                    context,
                    device,
                    program,
                    slots[consuming[0]],
                    slots[consuming[1]],
                    destination,
                    queue,
                    gpu,
                    work);
                continue;
            }

            if (work.form == WorkForm::EXPONENTIAL && has_source) {
                exponential(
                    context,
                    device,
                    program,
                    slots[source_index],
                    destination,
                    queue,
                    gpu,
                    work);
                continue;
            }

            if (!has_source) {
                fill(context, device, program, destination, queue, gpu, work);
                continue;
            }

            const auto& origin = slots[source_index];

            if (exact_copy && source_index != out) {
                transfer(origin, destination, work, queue);
                continue;
            }

            transform(
                context,
                device,
                program,
                origin,
                destination,
                queue,
                gpu,
                work);
        }

        if (gpu) {
            queue.finish();
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

    // Exact copying: real staging across homes, engine-independent — like a
    // transfer engine, the path follows where the data lives. Same-dtype
    // copies move stored bytes; mixed-dtype copies convert explicitly
    // through F32 with the shared helpers. Device buffers hold F32
    // working copies sized by element count, so device paths convert at
    // the boundary in both directions.
    static void transfer(
        const Slot& origin,
        const Slot& destination,
        const WorkDescription& work,
        cl::CommandQueue& queue) {
        const auto elements = work.storage_elements();
        const auto bytes =
            origin.bytes < destination.bytes ? origin.bytes
                                             : destination.bytes;

        if (origin.dtype != destination.dtype) {
            // Mixed-dtype move: decode the source to F32, encode into
            // the destination dtype, then stage to the destination home.
            // Device destinations receive F32 working copies.
            std::vector<float> working(elements, 0.0f);
            if (origin.on_device) {
                queue.enqueueReadBuffer(
                    origin.device,
                    CL_TRUE,
                    0,
                    elements * sizeof(float),
                    working.data());
            } else {
                decode_all(
                    origin.dtype, origin.host->data(), working.data(), elements);
            }
            if (destination.on_device) {
                queue.enqueueWriteBuffer(
                    destination.device,
                    CL_TRUE,
                    0,
                    elements * sizeof(float),
                    working.data());
            } else {
                encode_all(
                    destination.dtype,
                    working.data(),
                    destination.host->data(),
                    elements);
            }
            return;
        }

        if (origin.dtype == destination.dtype) {
            if (!origin.on_device && !destination.on_device) {
                std::memcpy(
                    destination.host->data(), origin.host->data(), bytes);
                return;
            }

            if (origin.on_device && destination.on_device) {
                queue.enqueueCopyBuffer(
                    origin.device,
                    destination.device,
                    0,
                    0,
                    elements * sizeof(float));
                return;
            }

            if (destination.on_device) {
                std::vector<float> working(elements, 0.0f);
                decode_all(
                    origin.dtype, origin.host->data(), working.data(), elements);
                queue.enqueueWriteBuffer(
                    destination.device,
                    CL_TRUE,
                    0,
                    elements * sizeof(float),
                    working.data());
            } else {
                std::vector<float> working(elements, 0.0f);
                queue.enqueueReadBuffer(
                    origin.device,
                    CL_TRUE,
                    0,
                    elements * sizeof(float),
                    working.data());
                encode_all(
                    destination.dtype,
                    working.data(),
                    destination.host->data(),
                    elements);
            }
        }
    }

    // The affine elementwise transform on either engine, with real staging
    // across homes.
    static void transform(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& origin,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto elements = work.elements;
        const auto floats = elements * sizeof(float);

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;

            if (!origin.on_device) {
                std::vector<float> decoded(elements, 0.0f);
                decode_all(
                    origin.dtype, origin.host->data(), decoded.data(), elements);
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, floats, nullptr);
            }

            // Vectorized float4 streaming with scalar tails: identical
            // values to the scalar kernel (pinned by test).
            cl::Kernel kernel(program, "transform_vec");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_origin);
            kernel.setArg(2, work.destination_scale);
            kernel.setArg(3, work.source_scale);
            kernel.setArg(4, work.constant);
            kernel.setArg(5, static_cast<cl_ulong>(work.passes));
            kernel.setArg(6, static_cast<cl_ulong>(elements));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange((elements + 3) / 4));

            if (!destination.on_device) {
                std::vector<float> working(elements, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    floats,
                    working.data());
                encode_all(
                    destination.dtype,
                    working.data(),
                    destination.host->data(),
                    elements);
            }

            return;
        }

        // CPU engine over host storage, staging device homes as needed.
        // Device homes hold F32 working copies; host homes decode from
        // stored dtype through the shared helpers.
        std::vector<float> staged_origin(elements, 1.0f);
        std::vector<float> staged_destination(elements, 1.0f);

        if (origin.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                origin.device,
                CL_TRUE,
                0,
                floats,
                staged_origin.data());
        } else {
            decode_all(
                origin.dtype, origin.host->data(), staged_origin.data(), elements);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                floats,
                staged_destination.data());
        } else {
            decode_all(
                destination.dtype,
                destination.host->data(),
                staged_destination.data(),
                elements);
        }

        // Aliasing is defined as the iterated form: when the source is the
        // destination record, each pass composes over the previous result.
        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < elements; ++i) {
                staged_destination[i] =
                    staged_destination[i] * work.destination_scale +
                    staged_origin[i] * work.source_scale + work.constant;
            }
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                floats,
                staged_destination.data());
        } else {
            encode_all(
                destination.dtype,
                staged_destination.data(),
                destination.host->data(),
                elements);
        }
    }

    // The matrix product form: dst = dst * destination_scale +
    // source_scale * (A x B) + constant, row-major.
    static void matrix_product(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& left,
        const Slot& right,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        bool tiled,
        const WorkDescription& work) {
        const auto rows = work.rows;
        const auto inner = work.inner;
        const auto columns = work.columns;
        const auto floats_out = rows * columns * sizeof(float);

        if (gpu) {
            cl::Buffer staged_left = left.device;
            cl::Buffer staged_right = right.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_left(rows * inner, 0.0f);
            std::vector<float> decoded_right(inner * columns, 0.0f);

            if (!left.on_device) {
                decode_all(
                    left.dtype,
                    left.host->data(),
                    decoded_left.data(),
                    decoded_left.size());
                staged_left = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    decoded_left.size() * sizeof(float),
                    decoded_left.data());
            }

            if (!right.on_device) {
                decode_all(
                    right.dtype,
                    right.host->data(),
                    decoded_right.data(),
                    decoded_right.size());
                staged_right = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    decoded_right.size() * sizeof(float),
                    decoded_right.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, floats_out, nullptr);
            }

            // Tiled 16x16 work-groups over local-memory tiles when the
            // device serves them; the naive kernel stays the fail-closed
            // fallback. Both compute identical values (pinned by test).
            const bool use_tile =
                tiled && rows <= 4096 && columns <= 4096 &&
                rows * columns <= (std::size_t{1} << 30);
            cl::Kernel kernel(
                program, use_tile ? "matmul_tile" : "matrix_product");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_left);
            kernel.setArg(2, staged_right);
            kernel.setArg(3, static_cast<cl_ulong>(rows));
            kernel.setArg(4, static_cast<cl_ulong>(inner));
            kernel.setArg(5, static_cast<cl_ulong>(columns));
            kernel.setArg(6, work.destination_scale);
            kernel.setArg(7, work.source_scale);
            kernel.setArg(8, work.constant);
            kernel.setArg(9, static_cast<cl_ulong>(work.passes));

            if (use_tile) {
                const auto groups_x = (columns + 15) / 16;
                const auto groups_y = (rows + 15) / 16;
                queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(groups_x * 16, groups_y * 16),
                    cl::NDRange(16, 16));
            } else {
                queue.enqueueNDRangeKernel(
                    kernel,
                    cl::NullRange,
                    cl::NDRange(rows * columns));
            }

            if (!destination.on_device) {
                std::vector<float> working(rows * columns, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    floats_out,
                    working.data());
                encode_all(
                    destination.dtype,
                    working.data(),
                    destination.host->data(),
                    working.size());
            }

            return;
        }

        // CPU engine, staging device homes as needed. Device homes hold
        // F32 working copies; host homes decode through shared helpers.
        std::vector<float> staged_left(rows * inner, 1.0f);
        std::vector<float> staged_right(inner * columns, 1.0f);
        std::vector<float> staged_destination(rows * columns, 1.0f);

        const auto stage_in = [&](const Slot& slot,
                                  std::vector<float>& target) {
            if (slot.on_device) {
                cl::CommandQueue staging(context, device);
                staging.enqueueReadBuffer(
                    slot.device,
                    CL_TRUE,
                    0,
                    target.size() * sizeof(float),
                    target.data());
            } else {
                decode_all(
                    slot.dtype, slot.host->data(), target.data(), target.size());
            }
        };

        stage_in(left, staged_left);
        stage_in(right, staged_right);
        stage_in(destination, staged_destination);

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    float accumulated = 0.0f;

                    for (std::size_t k = 0; k < inner; ++k) {
                        accumulated +=
                            staged_left[i * inner + k] *
                            staged_right[k * columns + j];
                    }

                    const auto index = i * columns + j;
                    staged_destination[index] =
                        staged_destination[index] *
                            work.destination_scale +
                        accumulated * work.source_scale + work.constant;
                }
            }
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                staged_destination.size() * sizeof(float),
                staged_destination.data());
        } else {
            encode_all(
                destination.dtype,
                staged_destination.data(),
                destination.host->data(),
                staged_destination.size());
        }
    }

    // The reduction form: dst[0] = dst[0] * destination_scale +
    // source_scale * sum(src) + constant.
    static void reduce_sum(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& origin,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto count = work.elements;

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_origin(count, 0.0f);
            decode_all(origin.dtype, origin.host->data(), decoded_origin.data(), count);

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    count * sizeof(float),
                    decoded_origin.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, count * sizeof(float), nullptr);
            }

            cl::Kernel kernel(program, "reduce_sum");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_origin);
            kernel.setArg(2, static_cast<cl_ulong>(count));
            kernel.setArg(3, work.destination_scale);
            kernel.setArg(4, work.source_scale);
            kernel.setArg(5, work.constant);
            kernel.setArg(6, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(1));

            if (!destination.on_device) {
                std::vector<float> working(count, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    count * sizeof(float),
                    working.data());
                encode_element(destination.dtype, destination.host->data(), 0, working[0]);
            }

            return;
        }

        std::vector<float> staged_origin(count, 1.0f);
        std::vector<float> staged_destination(1, 1.0f);

        if (origin.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                origin.device,
                CL_TRUE,
                0,
                count * sizeof(float),
                staged_origin.data());
        } else {
            decode_all(origin.dtype, origin.host->data(), staged_origin.data(), count);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                sizeof(float),
                staged_destination.data());
        } else {
            staged_destination[0] = decode_element(destination.dtype, destination.host->data(), 0);
        }

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            float sum = 0.0f;

            for (std::size_t i = 0; i < count; ++i) {
                sum += staged_origin[i];
            }

            staged_destination[0] =
                staged_destination[0] * work.destination_scale +
                sum * work.source_scale + work.constant;
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                sizeof(float),
                staged_destination.data());
        } else {
            encode_element(destination.dtype, destination.host->data(), 0, staged_destination[0]);
        }
    }

    // The max-reduction form: dst[0] = dst[0] * destination_scale +
    // source_scale * max(src) + constant.
    static void reduce_max(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& origin,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto count = work.elements;

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_origin(count, 0.0f);
            decode_all(origin.dtype, origin.host->data(), decoded_origin.data(), count);

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    count * sizeof(float),
                    decoded_origin.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, count * sizeof(float), nullptr);
            }

            cl::Kernel kernel(program, "reduce_max");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_origin);
            kernel.setArg(2, static_cast<cl_ulong>(count));
            kernel.setArg(3, work.destination_scale);
            kernel.setArg(4, work.source_scale);
            kernel.setArg(5, work.constant);
            kernel.setArg(6, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(1));

            if (!destination.on_device) {
                std::vector<float> working(count, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    count * sizeof(float),
                    working.data());
                encode_element(destination.dtype, destination.host->data(), 0, working[0]);
            }

            return;
        }

        std::vector<float> staged_origin(count, 1.0f);
        std::vector<float> staged_destination(1, 1.0f);

        if (origin.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                origin.device,
                CL_TRUE,
                0,
                count * sizeof(float),
                staged_origin.data());
        } else {
            decode_all(origin.dtype, origin.host->data(), staged_origin.data(), count);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                sizeof(float),
                staged_destination.data());
        } else {
            staged_destination[0] = decode_element(destination.dtype, destination.host->data(), 0);
        }

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            float peak = staged_origin[0];

            for (std::size_t i = 1; i < count; ++i) {
                if (staged_origin[i] > peak) {
                    peak = staged_origin[i];
                }
            }

            staged_destination[0] =
                staged_destination[0] * work.destination_scale +
                peak * work.source_scale + work.constant;
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                sizeof(float),
                staged_destination.data());
        } else {
            encode_element(destination.dtype, destination.host->data(), 0, staged_destination[0]);
        }
    }

    // The exponential form: dst = dst * destination_scale +
    // source_scale * exp(src) + constant, elementwise.
    static void exponential(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& origin,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto elements = work.elements;
        const auto floats = elements * sizeof(float);

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_origin(elements, 0.0f);
            decode_all(origin.dtype, origin.host->data(), decoded_origin.data(), elements);

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_origin.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, floats, nullptr);
            }

            cl::Kernel kernel(program, "exponential_vec");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_origin);
            kernel.setArg(2, work.destination_scale);
            kernel.setArg(3, work.source_scale);
            kernel.setArg(4, work.constant);
            kernel.setArg(5, static_cast<cl_ulong>(work.passes));
            kernel.setArg(6, static_cast<cl_ulong>(elements));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange((elements + 3) / 4));

            if (!destination.on_device) {
                std::vector<float> working(elements, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    floats,
                    working.data());
                encode_all(destination.dtype, working.data(), destination.host->data(), elements);
            }

            return;
        }

        // CPU engine, staging device homes as needed.
        std::vector<float> staged_origin(elements, 1.0f);
        std::vector<float> staged_destination(elements, 1.0f);

        if (origin.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                origin.device,
                CL_TRUE,
                0,
                floats,
                staged_origin.data());
        } else {
            decode_all(origin.dtype, origin.host->data(), staged_origin.data(), elements);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                floats,
                staged_destination.data());
        } else {
            decode_all(destination.dtype, destination.host->data(), staged_destination.data(), elements);
        }

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < elements; ++i) {
                staged_destination[i] =
                    staged_destination[i] * work.destination_scale +
                    std::exp(staged_origin[i]) * work.source_scale +
                    work.constant;
            }
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                floats,
                staged_destination.data());
        } else {
            encode_all(destination.dtype, staged_destination.data(), destination.host->data(), elements);
        }
    }

    // The min-reduction form: dst[0] = dst[0] * destination_scale +
    // source_scale * min(src) + constant.
    static void reduce_min(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& origin,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto count = work.elements;

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_origin(count, 0.0f);
            decode_all(origin.dtype, origin.host->data(), decoded_origin.data(), count);

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    count * sizeof(float),
                    decoded_origin.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, count * sizeof(float), nullptr);
            }

            cl::Kernel kernel(program, "reduce_min");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_origin);
            kernel.setArg(2, static_cast<cl_ulong>(count));
            kernel.setArg(3, work.destination_scale);
            kernel.setArg(4, work.source_scale);
            kernel.setArg(5, work.constant);
            kernel.setArg(6, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(1));

            if (!destination.on_device) {
                std::vector<float> working(count, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    count * sizeof(float),
                    working.data());
                encode_element(destination.dtype, destination.host->data(), 0, working[0]);
            }

            return;
        }

        std::vector<float> staged_origin(count, 1.0f);
        std::vector<float> staged_destination(1, 1.0f);

        if (origin.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                origin.device,
                CL_TRUE,
                0,
                count * sizeof(float),
                staged_origin.data());
        } else {
            decode_all(origin.dtype, origin.host->data(), staged_origin.data(), count);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                sizeof(float),
                staged_destination.data());
        } else {
            staged_destination[0] = decode_element(destination.dtype, destination.host->data(), 0);
        }

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            float floor_value = staged_origin[0];

            for (std::size_t i = 1; i < count; ++i) {
                if (staged_origin[i] < floor_value) {
                    floor_value = staged_origin[i];
                }
            }

            staged_destination[0] =
                staged_destination[0] * work.destination_scale +
                floor_value * work.source_scale + work.constant;
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                sizeof(float),
                staged_destination.data());
        } else {
            encode_element(destination.dtype, destination.host->data(), 0, staged_destination[0]);
        }
    }

    // The two-operand elementwise selection: min or max of A and B under
    // the affine wrapper.
    static void elementwise_select(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& left,
        const Slot& right,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto elements = work.elements;
        const char* name = work.form == WorkForm::ELEMENTWISE_MIN
            ? "elementwise_min"
            : "elementwise_max";

        const auto floats = elements * sizeof(float);

        if (gpu) {
            cl::Buffer staged_left = left.device;
            cl::Buffer staged_right = right.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_left(elements, 0.0f);
            std::vector<float> decoded_right(elements, 0.0f);
            decode_all(left.dtype, left.host->data(), decoded_left.data(), elements);
            decode_all(right.dtype, right.host->data(), decoded_right.data(), elements);

            if (!left.on_device) {
                staged_left = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_left.data());
            }

            if (!right.on_device) {
                staged_right = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_right.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, floats, nullptr);
            }

            cl::Kernel kernel(program, name);
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_left);
            kernel.setArg(2, staged_right);
            kernel.setArg(3, work.destination_scale);
            kernel.setArg(4, work.source_scale);
            kernel.setArg(5, work.constant);
            kernel.setArg(6, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(elements));

            if (!destination.on_device) {
                std::vector<float> working(elements, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    floats,
                    working.data());
                encode_all(destination.dtype, working.data(), destination.host->data(), elements);
            }

            return;
        }

        std::vector<float> staged_left(elements, 1.0f);
        std::vector<float> staged_right(elements, 1.0f);
        std::vector<float> staged_destination(elements, 1.0f);
        stage_pair(
            context,
            device,
            left,
            right,
            destination,
            staged_left,
            staged_right,
            staged_destination);

        const bool take_min = work.form == WorkForm::ELEMENTWISE_MIN;

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < elements; ++i) {
                const float chosen = take_min
                    ? (staged_left[i] < staged_right[i] ? staged_left[i]
                                                       : staged_right[i])
                    : (staged_left[i] > staged_right[i] ? staged_left[i]
                                                       : staged_right[i]);
                staged_destination[i] =
                    staged_destination[i] * work.destination_scale +
                    chosen * work.source_scale + work.constant;
            }
        }

        write_back(context, device, destination, staged_destination);
    }

    // The three-operand predicate selection: P != 0 ? A : B under the
    // affine wrapper.
    static void mask_select(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& predicate,
        const Slot& first,
        const Slot& second,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto elements = work.elements;

        const auto floats = elements * sizeof(float);

        if (gpu) {
            cl::Buffer staged_predicate = predicate.device;
            cl::Buffer staged_first = first.device;
            cl::Buffer staged_second = second.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_predicate(elements, 0.0f);
            std::vector<float> decoded_first(elements, 0.0f);
            std::vector<float> decoded_second(elements, 0.0f);
            decode_all(predicate.dtype, predicate.host->data(), decoded_predicate.data(), elements);
            decode_all(first.dtype, first.host->data(), decoded_first.data(), elements);
            decode_all(second.dtype, second.host->data(), decoded_second.data(), elements);

            if (!predicate.on_device) {
                staged_predicate = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_predicate.data());
            }

            if (!first.on_device) {
                staged_first = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_first.data());
            }

            if (!second.on_device) {
                staged_second = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_second.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, floats, nullptr);
            }

            cl::Kernel kernel(program, "mask_select");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_predicate);
            kernel.setArg(2, staged_first);
            kernel.setArg(3, staged_second);
            kernel.setArg(4, work.destination_scale);
            kernel.setArg(5, work.source_scale);
            kernel.setArg(6, work.constant);
            kernel.setArg(7, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(elements));

            if (!destination.on_device) {
                std::vector<float> working(elements, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    floats,
                    working.data());
                encode_all(destination.dtype, working.data(), destination.host->data(), elements);
            }

            return;
        }

        std::vector<float> staged_predicate(elements, 1.0f);
        std::vector<float> staged_first(elements, 1.0f);
        std::vector<float> staged_second(elements, 1.0f);
        std::vector<float> staged_destination(elements, 1.0f);
        stage_triple(
            context,
            device,
            predicate,
            first,
            second,
            destination,
            staged_predicate,
            staged_first,
            staged_second,
            staged_destination);

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < elements; ++i) {
                const float chosen = staged_predicate[i] != 0.0f
                    ? staged_first[i]
                    : staged_second[i];
                staged_destination[i] =
                    staged_destination[i] * work.destination_scale +
                    chosen * work.source_scale + work.constant;
            }
        }

        write_back(context, device, destination, staged_destination);
    }

    // The gather form: the value table indexed by the truncated and
    // clamped index table, under the affine wrapper.
    static void gather(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& table,
        const Slot& indices,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto elements = work.elements;
        const auto floats = elements * sizeof(float);

        if (gpu) {
            cl::Buffer staged_table = table.device;
            cl::Buffer staged_indices = indices.device;
            cl::Buffer staged_destination = destination.device;
            std::vector<float> decoded_table(elements, 0.0f);
            std::vector<float> decoded_indices(elements, 0.0f);
            decode_all(table.dtype, table.host->data(), decoded_table.data(), elements);
            decode_all(indices.dtype, indices.host->data(), decoded_indices.data(), elements);

            if (!table.on_device) {
                staged_table = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_table.data());
            }

            if (!indices.on_device) {
                staged_indices = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floats,
                    decoded_indices.data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, floats, nullptr);
            }

            cl::Kernel kernel(program, "gather");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_table);
            kernel.setArg(2, staged_indices);
            kernel.setArg(3, static_cast<cl_ulong>(elements));
            kernel.setArg(4, work.destination_scale);
            kernel.setArg(5, work.source_scale);
            kernel.setArg(6, work.constant);
            kernel.setArg(7, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(elements));

            if (!destination.on_device) {
                std::vector<float> working(elements, 0.0f);
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    floats,
                    working.data());
                encode_all(destination.dtype, working.data(), destination.host->data(), elements);
            }

            return;
        }

        std::vector<float> staged_table(elements, 1.0f);
        std::vector<float> staged_indices(elements, 1.0f);
        std::vector<float> staged_destination(elements, 1.0f);
        stage_pair(
            context,
            device,
            table,
            indices,
            destination,
            staged_table,
            staged_indices,
            staged_destination);

        const auto last = static_cast<std::int64_t>(elements - 1);
        const float last_float = static_cast<float>(elements - 1);

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            const std::vector<float> values = staged_table;

            for (std::size_t i = 0; i < elements; ++i) {
                const float raw = staged_indices[i];
                std::int64_t position = 0;

                if (raw >= 0.0f && raw <= last_float) {
                    position = static_cast<std::int64_t>(raw);

                    if (position > last) {
                        position = last;
                    }
                } else if (raw > last_float) {
                    position = last;
                }

                staged_destination[i] =
                    staged_destination[i] * work.destination_scale +
                    values[static_cast<std::size_t>(position)] *
                        work.source_scale +
                    work.constant;
            }
        }

        write_back(context, device, destination, staged_destination);
    }

    // Stage one operand pair plus the destination into host scratch.
    static void stage_pair(
        const cl::Context& context,
        const cl::Device& device,
        const Slot& left,
        const Slot& right,
        const Slot& destination,
        std::vector<float>& staged_left,
        std::vector<float>& staged_right,
        std::vector<float>& staged_destination) {
        stage_one(context, device, left, staged_left);
        stage_one(context, device, right, staged_right);
        stage_one(context, device, destination, staged_destination);
    }

    // Stage a predicate triple plus the destination into host scratch.
    static void stage_triple(
        const cl::Context& context,
        const cl::Device& device,
        const Slot& predicate,
        const Slot& first,
        const Slot& second,
        const Slot& destination,
        std::vector<float>& staged_predicate,
        std::vector<float>& staged_first,
        std::vector<float>& staged_second,
        std::vector<float>& staged_destination) {
        stage_one(context, device, predicate, staged_predicate);
        stage_one(context, device, first, staged_first);
        stage_one(context, device, second, staged_second);
        stage_one(context, device, destination, staged_destination);
    }

    // Stage one slot into host scratch.
    static void stage_one(
        const cl::Context& context,
        const cl::Device& device,
        const Slot& slot,
        std::vector<float>& staged) {
        if (slot.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                slot.device,
                CL_TRUE,
                0,
                staged.size() * sizeof(float),
                staged.data());
        } else {
            decode_all(slot.dtype, slot.host->data(), staged.data(), staged.size());
        }
    }

    // Write host scratch back to the destination home.
    static void write_back(
        const cl::Context& context,
        const cl::Device& device,
        const Slot& destination,
        const std::vector<float>& staged) {
        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                staged.size() * sizeof(float),
                staged.data());
        } else {
            encode_all(destination.dtype, staged.data(), destination.host->data(), staged.size());
        }
    }

    // The transform without a source: dst = dst * destination_scale +
    // constant.
    static void fill(
        const cl::Context& context,
        const cl::Device& device,
        const cl::Program& program,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) {
        const auto elements = work.elements;
        const auto floats = elements * sizeof(float);

        if (gpu) {
            cl::Buffer staged = destination.device;

            if (!destination.on_device) {
                // Sourceless fill over a non-F32 home: decode the current
                // values to F32 first so the kernel's dst term is right,
                // then upload the working copy.
                std::vector<float> decoded(elements, 0.0f);
                decode_all(destination.dtype, destination.host->data(), decoded.data(), elements);
                staged = cl::Buffer(
                    context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, floats, decoded.data());
            }

            cl::Kernel kernel(program, "fill_vec");
            kernel.setArg(0, staged);
            kernel.setArg(1, work.destination_scale);
            kernel.setArg(2, work.constant);
            kernel.setArg(3, static_cast<cl_ulong>(work.passes));
            kernel.setArg(4, static_cast<cl_ulong>(elements));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange((elements + 3) / 4));

            if (!destination.on_device) {
                std::vector<float> working(elements, 0.0f);
                queue.enqueueReadBuffer(
                    staged, CL_TRUE, 0, floats, working.data());
                encode_all(destination.dtype, working.data(), destination.host->data(), elements);
            }

            return;
        }

        if (destination.on_device) {
            std::vector<float> staged(elements, 1.0f);

            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                floats,
                staged.data());

            for (std::size_t pass = 0; pass < work.passes; ++pass) {
                for (std::size_t i = 0; i < elements; ++i) {
                    staged[i] =
                        staged[i] * work.destination_scale + work.constant;
                }
            }

            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                floats,
                staged.data());

            return;
        }

        std::vector<float> decoded(elements, 0.0f);
        decode_all(destination.dtype, destination.host->data(), decoded.data(), elements);
        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < elements; ++i) {
                decoded[i] =
                    decoded[i] * work.destination_scale +
                    work.constant;
            }
        }
        encode_all(destination.dtype, decoded.data(), destination.host->data(), elements);
    }

    const DataRegistry& data_;
    DeviceId accelerator_;
    bool available_{false};
    bool tiled_ok_{false};
    cl::Device device_;
    cl::Context context_;
    cl::Program program_;

    std::unordered_set<ExecutionId> failures_;
    std::unordered_map<DataResidencyRef, Allocation> allocations_;
    WorkPool pool_;
    std::vector<Job> jobs_;
    std::unordered_set<ExecutionId> submitted_;
};

} // namespace gerdos