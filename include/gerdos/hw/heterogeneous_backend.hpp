#pragma once

#include <chrono>
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
// affine semantics — and exact copying is the fast path. Work runs on real
// background threads and poll never blocks.
class HeterogeneousBackend final : public ExecutionBackend {
public:
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

        if (binding == nullptr) {
            return false;
        }

        const auto work = operation.description().work;
        const bool succeeded = !failures_.contains(id);
        const auto bytes = work.elements * sizeof(float);

        // Real allocations exist before the work begins, and the worker
        // receives storage handles only — it never touches shared maps.
        std::vector<Slot> slots;

        for (const auto& entry : binding->data) {
            slots.push_back(slot_for(entry.residency, bytes));
        }

        const bool gpu = on_accelerator(*binding);

        if (gpu && !available_) {
            return false;
        }

        jobs_.push_back(
            Job{
                id,
                std::async(
                    std::launch::async,
                    [this,
                     binding_copy = *binding,
                     slots = std::move(slots),
                     work,
                     succeeded,
                     gpu]() {
                        return run(
                            binding_copy, slots, work, succeeded, gpu);
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

    // One real allocation per representation, at its declared home.
    struct Allocation {
        bool on_device;
        std::vector<float> host;
        cl::Buffer device;
        std::size_t bytes;
    };

    // What a worker needs, resolved before it starts.
    struct Slot {
        bool on_device;
        float* host;
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

            std::vector<float> initial(bytes / sizeof(float), 1.0f);

            if (on_device) {
                fresh.device = cl::Buffer(
                    context_,
                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    bytes,
                    initial.data());
            } else {
                fresh.host = std::move(initial);
            }

            it = allocations_.insert_or_assign(ref, std::move(fresh)).first;
        }

        auto& allocation = it->second;

        return Slot{
            allocation.on_device,
            allocation.host.empty() ? nullptr : allocation.host.data(),
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
            std::memcpy(out, it->second.host.data(), it->second.bytes);
            return;
        }

        cl::CommandQueue queue(context_, device_);
        queue.enqueueReadBuffer(
            it->second.device, CL_TRUE, 0, it->second.bytes, out);
    }

    // The declared work. The engine is chosen by the bound mechanisms; data
    // moves between declared homes with real staging. Workers touch only
    // the slots they were given.
    [[nodiscard]] Outcome run(
        const PhysicalBinding& binding,
        const std::vector<Slot>& slots,
        const WorkDescription& work,
        bool configured,
        bool gpu) {
        const auto begin = std::chrono::steady_clock::now();

        // Staging across homes needs a real queue regardless of engine.
        cl::CommandQueue queue;

        if (available_) {
            queue = cl::CommandQueue(context_, device_);
        }

        const bool exact_copy =
            work.passes == 1 && work.destination_scale == 0.0f &&
            work.source_scale == 1.0f && work.constant == 0.0f;

        for (std::size_t out = 0; out < binding.data.size(); ++out) {
            if (!is_producing(binding.data[out].role)) {
                continue;
            }

            const auto& destination = slots[out];
            const auto source_index = work_source_index(binding, out);
            const bool has_source = source_index < binding.data.size();

            if (!has_source) {
                fill(destination, queue, gpu, work);
                continue;
            }

            const auto& origin = slots[source_index];

            if (exact_copy && source_index != out) {
                transfer(origin, destination, queue);
                continue;
            }

            transform(origin, destination, queue, gpu, work);
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
            std::memcpy(destination.host, origin.host, bytes);
            return;
        }

        if (origin.on_device && destination.on_device) {
            queue.enqueueCopyBuffer(
                origin.device, destination.device, 0, 0, bytes);
            return;
        }

        if (destination.on_device) {
            queue.enqueueWriteBuffer(
                destination.device, CL_TRUE, 0, bytes, origin.host);
        } else {
            queue.enqueueReadBuffer(
                origin.device, CL_TRUE, 0, bytes, destination.host);
        }
    }

    // The affine elementwise transform on either engine, with real staging
    // across homes.
    void transform(
        const Slot& origin,
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) const {
        const auto elements = work.elements;

        if (gpu) {
            cl::Buffer staged_origin = origin.device;
            cl::Buffer staged_destination = destination.device;

            if (!origin.on_device) {
                staged_origin = cl::Buffer(
                    context_,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    origin.bytes,
                    origin.host);
            }

            if (!destination.on_device) {
                staged_destination = cl::Buffer(
                    context_, CL_MEM_READ_WRITE, destination.bytes, nullptr);
            }

            cl::Kernel kernel(program_, "transform");
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
                    destination.host);
            }

            return;
        }

        // CPU engine over host storage, staging device homes as needed.
        std::vector<float> staged_origin(elements, 1.0f);
        std::vector<float> staged_destination(elements, 1.0f);

        if (origin.on_device) {
            cl::CommandQueue staging(context_, device_);
            staging.enqueueReadBuffer(
                origin.device,
                CL_TRUE,
                0,
                origin.bytes,
                staged_origin.data());
        } else {
            std::memcpy(
                staged_origin.data(), origin.host, origin.bytes);
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context_, device_);
            staging.enqueueReadBuffer(
                destination.device,
                CL_TRUE,
                0,
                destination.bytes,
                staged_destination.data());
        } else {
            std::memcpy(
                staged_destination.data(),
                destination.host,
                destination.bytes);
        }

        for (std::size_t pass = 0; pass < work.passes; ++pass) {
            for (std::size_t i = 0; i < elements; ++i) {
                staged_destination[i] =
                    staged_destination[i] * work.destination_scale +
                    staged_origin[i] * work.source_scale + work.constant;
            }
        }

        if (destination.on_device) {
            cl::CommandQueue staging(context_, device_);
            staging.enqueueWriteBuffer(
                destination.device,
                CL_TRUE,
                0,
                destination.bytes,
                staged_destination.data());
        } else {
            std::memcpy(
                destination.host,
                staged_destination.data(),
                destination.bytes);
        }
    }

    // The transform without a source: dst = dst * destination_scale +
    // constant.
    void fill(
        const Slot& destination,
        cl::CommandQueue& queue,
        bool gpu,
        const WorkDescription& work) const {
        const auto elements = work.elements;

        if (gpu) {
            cl::Buffer staged = destination.device;

            if (!destination.on_device) {
                staged = cl::Buffer(
                    context_, CL_MEM_READ_WRITE, destination.bytes, nullptr);
            }

            cl::Kernel kernel(program_, "fill");
            kernel.setArg(0, staged);
            kernel.setArg(1, work.destination_scale);
            kernel.setArg(2, work.constant);
            kernel.setArg(3, static_cast<cl_ulong>(work.passes));
            queue.enqueueNDRangeKernel(
                kernel, cl::NullRange, cl::NDRange(elements));

            if (!destination.on_device) {
                queue.enqueueReadBuffer(
                    staged, CL_TRUE, 0, destination.bytes, destination.host);
            }

            return;
        }

        if (destination.on_device) {
            std::vector<float> staged(elements, 1.0f);

            cl::CommandQueue staging(context_, device_);
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
                destination.host[i] =
                    destination.host[i] * work.destination_scale +
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