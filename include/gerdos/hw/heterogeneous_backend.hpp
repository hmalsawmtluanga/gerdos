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
            "  for (ulong p = 0; p < passes; ++p) {"
            "    d[i] = d[i] * dscale + o[i] * sscale + bias;"
            "  }"
            "}"
            "__kernel void fill(__global float* d, float dscale,"
            " float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong p = 0; p < passes; ++p) {"
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
            "  for (ulong p = 0; p < passes; ++p) {"
            "    float acc = 0.0f;"
            "    for (ulong k = 0; k < inner; ++k) {"
            "      acc += a[i * inner + k] * b[k * columns + j];"
            "    }"
            "    d[index] = d[index] * dscale + acc * sscale + bias;"
            "  }"
            "}"
            "__kernel void reduce_sum(__global float* d,"
            " const __global float* o, ulong n, float dscale,"
            " float sscale, float bias, ulong passes) {"
            "  if (get_global_id(0) != 0) { return; }"
            "  for (ulong p = 0; p < passes; ++p) {"
            "    float sum = 0.0f;"
            "    for (ulong k = 0; k < n; ++k) { sum += o[k]; }"
            "    d[0] = d[0] * dscale + sum * sscale + bias;"
            "  }"
            "}"
            "__kernel void exponential(__global float* d,"
            " const __global float* o, float dscale, float sscale,"
            " float bias, ulong passes) {"
            "  size_t i = get_global_id(0);"
            "  for (ulong p = 0; p < passes; ++p) {"
            "    d[i] = d[i] * dscale + exp(o[i]) * sscale + bias;"
            "  }"
            "}"
            "__kernel void reduce_max(__global float* d,"
            " const __global float* o, ulong n, float dscale,"
            " float sscale, float bias, ulong passes) {"
            "  if (get_global_id(0) != 0) { return; }"
            "  for (ulong p = 0; p < passes; ++p) {"
            "    float peak = o[0];"
            "    for (ulong k = 1; k < n; ++k) {"
            "      peak = fmax(peak, o[k]);"
            "    }"
            "    d[0] = d[0] * dscale + peak * sscale + bias;"
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
        const auto bytes = work.storage_elements() * sizeof(float);

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

                if (it == allocations_.end() ||
                    it->second.bytes != bytes) {
                    undo.emplace_back(
                        ref,
                        it == allocations_.end()
                            ? std::optional<Allocation>{}
                            : std::optional<Allocation>{it->second});
                }

                slots.push_back(slot_for(ref, bytes));
            }

            jobs_.push_back(
                Job{
                    id,
                    std::async(
                        std::launch::async,
                        [context = context_,
                         device = device_,
                         program = program_,
                         binding_copy = *binding,
                         slots = std::move(slots),
                         work,
                         succeeded,
                         gpu]() {
                            return run(
                                context,
                                device,
                                program,
                                binding_copy,
                                slots,
                                work,
                                succeeded,
                                gpu);
                        }),
                });
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
        float value = 0.0f;

        const auto bytes = allocation_bytes(ref);

        if (bytes < (index + 1) * sizeof(float)) {
            return value;
        }

        // read() copies the whole allocation; give it the whole buffer.
        std::vector<unsigned char> raw;
        raw.resize(bytes);
        read(ref, raw.data());
        std::memcpy(&value, raw.data() + index * sizeof(float), sizeof(float));

        return value;
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
        std::shared_ptr<std::vector<float>> host;
        cl::Buffer device;
        std::size_t bytes;
    };

    // What a worker needs, resolved before it starts.
    struct Slot {
        bool on_device;
        std::shared_ptr<std::vector<float>> host;
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

    [[nodiscard]] Slot slot_for(DataResidencyRef ref, std::size_t bytes) {
        auto it = allocations_.find(ref);

        if (it == allocations_.end() || it->second.bytes != bytes) {
            const bool on_device =
                available_ && home_on_accelerator(ref);

            Allocation fresh{};
            fresh.on_device = on_device;
            fresh.bytes = bytes;

            fresh.host =
                std::make_shared<std::vector<float>>(
                    bytes / sizeof(float),
                    1.0f);

            if (on_device) {
                fresh.device = cl::Buffer(
                    context_,
                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    bytes,
                    fresh.host->data());
            }

            it = allocations_.insert_or_assign(ref, std::move(fresh)).first;
        }

        auto& allocation = it->second;

        return Slot{
            allocation.on_device,
            allocation.host,
            allocation.on_device ? allocation.device : cl::Buffer(),
            allocation.bytes,
        };
    }

    [[nodiscard]] std::size_t allocation_bytes(DataResidencyRef ref) {
        const auto it = allocations_.find(ref);

        return it == allocations_.end() ? 0 : it->second.bytes;
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
        bool gpu) {
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
                transfer(origin, destination, queue);
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
    // transfer engine, the path follows where the data lives.
    static void transfer(
        const Slot& origin,
        const Slot& destination,
        cl::CommandQueue& queue) {
        const auto bytes =
            origin.bytes < destination.bytes ? origin.bytes
                                             : destination.bytes;

        if (!origin.on_device && !destination.on_device) {
            std::memcpy(
                destination.host->data(), origin.host->data(), bytes);
            return;
        }

        if (origin.on_device && destination.on_device) {
            queue.enqueueCopyBuffer(
                origin.device, destination.device, 0, 0, bytes);
            return;
        }

        if (destination.on_device) {
            queue.enqueueWriteBuffer(
                destination.device, CL_TRUE, 0, bytes, origin.host->data());
        } else {
            queue.enqueueReadBuffer(
                origin.device, CL_TRUE, 0, bytes, destination.host->data());
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

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    origin.bytes,
                    origin.host->data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, destination.bytes, nullptr);
            }

            cl::Kernel kernel(program, "transform");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_origin);
            kernel.setArg(2, work.destination_scale);
            kernel.setArg(3, work.source_scale);
            kernel.setArg(4, work.constant);
            kernel.setArg(5, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(elements));

            if (!destination.on_device) {
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    destination.bytes,
                    destination.host->data());
            }

            return;
        }

        // CPU engine over host storage, staging device homes as needed.
        std::vector<float> staged_origin(elements, 1.0f);
        std::vector<float> staged_destination(elements, 1.0f);

        if (origin.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                origin.device,
                CL_TRUE,
                0,
                origin.bytes,
                staged_origin.data());
        } else {
            std::memcpy(
                staged_origin.data(), origin.host->data(), origin.bytes);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                destination.bytes,
                staged_destination.data());
        } else {
            std::memcpy(
                staged_destination.data(),
                destination.host->data(),
                destination.bytes);
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
                destination.bytes,
                staged_destination.data());
        } else {
            std::memcpy(
                destination.host->data(),
                staged_destination.data(),
                destination.bytes);
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
        const WorkDescription& work) {
        const auto rows = work.rows;
        const auto inner = work.inner;
        const auto columns = work.columns;

        if (gpu) {
            cl::Buffer staged_left = left.device;
            cl::Buffer staged_right = right.device;
            cl::Buffer staged_destination = destination.device;

            if (!left.on_device) {
                staged_left = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    left.bytes,
                    left.host->data());
            }

            if (!right.on_device) {
                staged_right = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    right.bytes,
                    right.host->data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, destination.bytes, nullptr);
            }

            cl::Kernel kernel(program, "matrix_product");
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
            queue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange(rows * columns));

            if (!destination.on_device) {
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    rows * columns * sizeof(float),
                    destination.host->data());
            }

            return;
        }

        // CPU engine, staging device homes as needed.
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
                std::memcpy(
                    target.data(),
                    slot.host->data(),
                    target.size() * sizeof(float));
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
            std::memcpy(
                destination.host->data(),
                staged_destination.data(),
                staged_destination.size() * sizeof(float));
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

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    origin.bytes,
                    origin.host->data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, destination.bytes, nullptr);
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
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    destination.bytes,
                    destination.host->data());
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
            std::memcpy(
                staged_origin.data(),
                origin.host->data(),
                count * sizeof(float));
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
            staged_destination[0] = (*destination.host)[0];
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
            (*destination.host)[0] = staged_destination[0];
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

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    origin.bytes,
                    origin.host->data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, destination.bytes, nullptr);
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
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    destination.bytes,
                    destination.host->data());
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
            std::memcpy(
                staged_origin.data(),
                origin.host->data(),
                count * sizeof(float));
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
            staged_destination[0] = (*destination.host)[0];
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
            (*destination.host)[0] = staged_destination[0];
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

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    origin.bytes,
                    origin.host->data());
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context, CL_MEM_READ_WRITE, destination.bytes, nullptr);
            }

            cl::Kernel kernel(program, "exponential");
            kernel.setArg(0, staged_destination);
            kernel.setArg(1, staged_origin);
            kernel.setArg(2, work.destination_scale);
            kernel.setArg(3, work.source_scale);
            kernel.setArg(4, work.constant);
            kernel.setArg(5, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(elements));

            if (!destination.on_device) {
                queue.enqueueReadBuffer(
                    staged_destination,
                    CL_TRUE,
                    0,
                    destination.bytes,
                    destination.host->data());
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
                origin.bytes,
                staged_origin.data());
        } else {
            std::memcpy(
                staged_origin.data(), origin.host->data(), origin.bytes);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context, device);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                destination.bytes,
                staged_destination.data());
        } else {
            std::memcpy(
                staged_destination.data(),
                destination.host->data(),
                destination.bytes);
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
                destination.bytes,
                staged_destination.data());
        } else {
            std::memcpy(
                destination.host->data(),
                staged_destination.data(),
                destination.bytes);
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

        if (gpu) {
            cl::Buffer staged = destination.device;

            if (!destination.on_device) {
                staged = cl::Buffer(
                    context, CL_MEM_READ_WRITE, destination.bytes, nullptr);
            }

            cl::Kernel kernel(program, "fill");
            kernel.setArg(0, staged);
            kernel.setArg(1, work.destination_scale);
            kernel.setArg(2, work.constant);
            kernel.setArg(3, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(elements));

            if (!destination.on_device) {
                queue.enqueueReadBuffer(
                    staged, CL_TRUE, 0, destination.bytes, destination.host->data());
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
                destination.bytes,
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
                destination.bytes,
                staged.data());

            return;
        }

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < elements; ++i) {
                (*destination.host)[i] =
                    (*destination.host)[i] * work.destination_scale +
                    work.constant;
            }
        }
    }

    const DataRegistry& data_;
    DeviceId accelerator_;
    bool available_{false};
    cl::Device device_;
    cl::Context context_;
    cl::Program program_;

    std::unordered_set<ExecutionId> failures_;
    std::unordered_map<DataResidencyRef, Allocation> allocations_;
    std::vector<Job> jobs_;
    std::unordered_set<ExecutionId> submitted_;
};

} // namespace gerdos