#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <vulkan/vulkan.h>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/dtype.hpp"
#include "gerdos/core/work_pool.hpp"
#include "gerdos/core/execution_backend.hpp"
#include "gerdos/vk/vk_shaders.hpp"

namespace gerdos {

// The Vulkan compute backend: the second GPU family behind the same seam.
// It owns accelerator-homed records and executes the full work algebra
// with identical semantics to the OpenCL path — both engines compute in
// F32 with once-per-direction boundary conversion through the shared
// helpers. Attempts bound to no accelerator mechanism are refused.
// Durations are real wall-clock nanoseconds around queue submission and
// fence wait, on real background threads; poll never blocks.
//
// Device selection is configuration: the first real GPU (integrated or
// discrete) with a compute queue. Software rasterizers are excluded — a
// CPU masquerading as a GPU would fabricate the hardware claim. Compute
// shaders are compiled offline to SPIR-V and checked in; there is no
// runtime shader toolchain dependency.
//
// Queue submission is mutex-serialized across concurrent attempts (one
// queue, many workers); command buffers, descriptor pools, and fences are
// per-run worker-owned transient state. Workers hold shared ownership of
// their storage; destruction joins all jobs before tearing down the
// device. The inspection surface is valid only while no jobs are in
// flight.
class VkComputeBackend final : public ExecutionBackend {
public:
    ~VkComputeBackend() {
        for (auto& job : jobs_) {
            try {
                (void)job.result.get();
            } catch (...) {
                // Destruction joins; throwing here would terminate.
            }
        }

        teardown();
    }

    VkComputeBackend(const DataRegistry& data, DeviceId accelerator)
        : data_(data),
          accelerator_(accelerator) {
        if (!init_instance()) {
            return;
        }

        if (!init_device()) {
            teardown_instance();
            return;
        }

        if (!init_pipelines()) {
            teardown_device();
            teardown_instance();
            return;
        }

        warmup();
        available_ = true;
    }

    [[nodiscard]] bool vk_available() const noexcept {
        return available_;
    }

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

        const auto work = operation.description().work;

        if (binding == nullptr || !work.valid()) {
            return false;
        }

        if (!on_accelerator(*binding)) {
            return false;
        }

        if (!available_) {
            return false;
        }

        const bool succeeded = !failures_.contains(id);
        const auto bytes =
            work.storage_elements() * work.storage_bytes();

        std::vector<std::pair<DataResidencyRef, std::optional<Allocation>>>
            undo;
        jobs_.reserve(jobs_.size() + 1);

        try {
            std::vector<Slot> slots;

            for (const auto& entry : binding->data) {
                const auto& ref = entry.residency;
                auto it = allocations_.find(ref);

                const bool count_sufficient =
                    it != allocations_.end() &&
                    it->second.bytes / dtype_bytes(it->second.dtype) >=
                        work.storage_elements();

                if (!count_sufficient) {
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
                 binding_copy = *binding,
                 slots = std::move(slots),
                 work,
                 succeeded]() {
                    return run(
                        context,
                        binding_copy,
                        slots,
                        work,
                        succeeded);
                });

            if (!launched.has_value()) {
                throw std::bad_alloc();
            }

            jobs_.push_back(Job{id, std::move(*launched)});
        } catch (...) {
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

            // Seam containment: a worker that throws instead of
            // returning (allocation failure under pressure) reports a
            // failed zero-duration completion with bookkeeping intact —
            // exceptions never escape poll.
            try {
                const auto outcome = job.result.get();

                completed.push_back(
                    BackendCompletion{
                        job.id,
                        outcome.succeeded,
                        outcome.duration_ns,
                    });
            } catch (...) {
                completed.push_back(
                    BackendCompletion{job.id, false, 0});
            }
        }

        jobs_ = std::move(remaining);
    }

    [[nodiscard]] std::size_t allocation_count() const noexcept {
        return allocations_.size();
    }

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

        if (it->second.on_device) {
            std::vector<float> working(it->second.bytes / width, 0.0f);
            download(it->second.device, working.data(), working.size());
            std::vector<unsigned char> raw(it->second.bytes, static_cast<unsigned char>(0));
            encode_all(it->second.dtype, working.data(), raw.data(), working.size());
            return decode_element(it->second.dtype, raw.data(), index);
        }

        return decode_element(it->second.dtype, it->second.host->data(), index);
    }

    [[nodiscard]] bool allocations_equal(
        DataResidencyRef left,
        DataResidencyRef right) {
        const auto left_it = allocations_.find(left);
        const auto right_it = allocations_.find(right);

        if (left_it == allocations_.end() ||
            right_it == allocations_.end() ||
            left_it->second.dtype != right_it->second.dtype ||
            left_it->second.bytes != right_it->second.bytes) {
            return false;
        }

        std::vector<unsigned char> left_raw(left_it->second.bytes, static_cast<unsigned char>(0));
        std::vector<unsigned char> right_raw(right_it->second.bytes, static_cast<unsigned char>(0));
        read(left, left_raw.data());
        read(right, right_raw.data());

        return left_raw == right_raw;
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

    struct DeviceBuffer {
        VkBuffer buffer{VK_NULL_HANDLE};
        VkDeviceMemory memory{VK_NULL_HANDLE};
        std::size_t floats{0};
    };

    using Buffer = std::shared_ptr<std::vector<unsigned char>>;

    struct Allocation {
        bool on_device{false};
        WorkDtype dtype{WorkDtype::F32};
        Buffer host;
        std::shared_ptr<DeviceBuffer> device;
        std::size_t bytes{0};
    };

    struct Slot {
        bool on_device{false};
        WorkDtype dtype{WorkDtype::F32};
        Buffer host;
        std::shared_ptr<DeviceBuffer> device;
        std::size_t bytes{0};
    };

    struct Context {
        VkDevice device{VK_NULL_HANDLE};
        std::shared_ptr<std::mutex> submission;
        VkQueue queue{VK_NULL_HANDLE};
        std::uint32_t queue_family{0};
        VkPhysicalDevice physical{VK_NULL_HANDLE};
        std::uint32_t memory_type{0};
        VkPipeline transform{VK_NULL_HANDLE};
        VkPipelineLayout transform_layout{VK_NULL_HANDLE};
        VkPipeline fill{VK_NULL_HANDLE};
        VkPipelineLayout fill_layout{VK_NULL_HANDLE};
        VkPipeline matmul{VK_NULL_HANDLE};
        VkPipelineLayout matmul_layout{VK_NULL_HANDLE};
        VkPipeline matmul_tiled{VK_NULL_HANDLE};
        VkPipelineLayout matmul_tiled_layout{VK_NULL_HANDLE};
        VkPipeline transform_vec{VK_NULL_HANDLE};
        VkPipelineLayout transform_vec_layout{VK_NULL_HANDLE};
        VkPipeline exponential_vec{VK_NULL_HANDLE};
        VkPipelineLayout exponential_vec_layout{VK_NULL_HANDLE};
        VkPipeline fill_vec{VK_NULL_HANDLE};
        VkPipelineLayout fill_vec_layout{VK_NULL_HANDLE};
        std::uint32_t max_workgroup{0};
        VkPipeline reduce_sum{VK_NULL_HANDLE};
        VkPipelineLayout reduce_sum_layout{VK_NULL_HANDLE};
        VkPipeline reduce_minmax{VK_NULL_HANDLE};
        VkPipelineLayout reduce_minmax_layout{VK_NULL_HANDLE};
        VkPipeline select{VK_NULL_HANDLE};
        VkPipelineLayout select_layout{VK_NULL_HANDLE};
        VkPipeline mask{VK_NULL_HANDLE};
        VkPipelineLayout mask_layout{VK_NULL_HANDLE};
        VkPipeline gather{VK_NULL_HANDLE};
        VkPipelineLayout gather_layout{VK_NULL_HANDLE};
        VkPipeline divide{VK_NULL_HANDLE};
        VkPipelineLayout divide_layout{VK_NULL_HANDLE};
        VkPipeline exponential{VK_NULL_HANDLE};
        VkPipelineLayout exponential_layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout layout1{VK_NULL_HANDLE};
        VkDescriptorSetLayout layout2{VK_NULL_HANDLE};
        VkDescriptorSetLayout layout3{VK_NULL_HANDLE};
        VkDescriptorSetLayout layout4{VK_NULL_HANDLE};
    };

    struct PcAffine {
        float dscale;
        float sscale;
        float bias;
        std::uint32_t passes;
        std::uint32_t n;
    };

    struct PcMatmul {
        std::uint32_t rows;
        std::uint32_t inner;
        std::uint32_t columns;
        float dscale;
        float sscale;
        float bias;
        std::uint32_t passes;
    };

    struct PcReduce {
        std::uint32_t n;
        float dscale;
        float sscale;
        float bias;
        std::uint32_t passes;
    };

    struct PcReduceMm {
        std::uint32_t n;
        float dscale;
        float sscale;
        float bias;
        std::uint32_t passes;
        std::uint32_t take_max;
    };

    struct PcSelect {
        float dscale;
        float sscale;
        float bias;
        std::uint32_t passes;
        std::uint32_t n;
        std::uint32_t take_max;
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

    [[nodiscard]] bool init_instance() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &app;

        return vkCreateInstance(&info, nullptr, &instance_) == VK_SUCCESS;
    }

    void teardown_instance() {
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    [[nodiscard]] std::uint32_t find_memory_type() const {
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &properties);

        for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
            constexpr VkMemoryPropertyFlags wanted =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            if ((properties.memoryTypes[i].propertyFlags & wanted) ==
                wanted) {
                return i;
            }
        }

        return properties.memoryTypeCount;
    }

    [[nodiscard]] bool init_device() {
        std::uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);

        if (count == 0) {
            return false;
        }

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        for (const auto candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);

            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
                continue;
            }

            std::uint32_t families = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(
                candidate, &families, nullptr);
            std::vector<VkQueueFamilyProperties> queues(families);
            vkGetPhysicalDeviceQueueFamilyProperties(
                candidate, &families, queues.data());

            for (std::uint32_t family = 0; family < families; ++family) {
                if ((queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT) !=
                    0u) {
                    physical_ = candidate;
                    queue_family_ = family;
                    break;
                }
            }

            if (physical_ != VK_NULL_HANDLE) {
                break;
            }
        }

        if (physical_ == VK_NULL_HANDLE) {
            return false;
        }

        VkPhysicalDeviceProperties device_properties{};
        vkGetPhysicalDeviceProperties(physical_, &device_properties);
        context_.max_workgroup =
            device_properties.limits.maxComputeWorkGroupInvocations;

        memory_type_ = find_memory_type();

        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_, &properties);

        if (memory_type_ >= properties.memoryTypeCount) {
            return false;
        }

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;

        VkDevice device{VK_NULL_HANDLE};

        if (vkCreateDevice(physical_, &device_info, nullptr, &device) !=
            VK_SUCCESS) {
            return false;
        }

        device_ = device;
        context_.device = device;
        context_.queue_family = queue_family_;
        context_.physical = physical_;
        context_.memory_type = memory_type_;
        context_.submission = std::make_shared<std::mutex>();
        vkGetDeviceQueue(device, queue_family_, 0, &context_.queue);

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_;

        return vkCreateCommandPool(
                   device, &pool_info, nullptr, &command_pool_) == VK_SUCCESS;
    }

    [[nodiscard]] VkShaderModule make_module(const std::uint32_t* code,
                                             std::size_t words) const {
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = words * sizeof(std::uint32_t);
        info.pCode = code;

        VkShaderModule module{VK_NULL_HANDLE};

        if (vkCreateShaderModule(device_, &info, nullptr, &module) !=
            VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        return module;
    }

    [[nodiscard]] bool make_layout(std::uint32_t bindings,
                                   VkDescriptorSetLayout* layout) const {
        std::vector<VkDescriptorSetLayoutBinding> entries(bindings);

        for (std::uint32_t i = 0; i < bindings; ++i) {
            entries[i].binding = i;
            entries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            entries[i].descriptorCount = 1;
            entries[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = bindings;
        info.pBindings = entries.data();

        return vkCreateDescriptorSetLayout(
                   device_, &info, nullptr, layout) == VK_SUCCESS;
    }

    [[nodiscard]] bool make_pipeline(const std::uint32_t* code,
                                     std::size_t words,
                                     VkDescriptorSetLayout layout,
                                     std::uint32_t push_size,
                                     VkPipeline* pipeline,
                                     VkPipelineLayout* pipeline_layout) {
        VkShaderModule module = make_module(code, words);

        if (module == VK_NULL_HANDLE) {
            return false;
        }

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = push_size;

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;

        bool ok = vkCreatePipelineLayout(
                      device_, &layout_info, nullptr, pipeline_layout) ==
                  VK_SUCCESS;

        if (ok) {
            VkComputePipelineCreateInfo pipeline_info{};
            pipeline_info.sType =
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipeline_info.stage.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipeline_info.stage.module = module;
            pipeline_info.stage.pName = "main";
            pipeline_info.layout = *pipeline_layout;

            ok = vkCreateComputePipelines(
                     device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                     pipeline) == VK_SUCCESS;
        }

        vkDestroyShaderModule(device_, module, nullptr);
        return ok;
    }

    [[nodiscard]] bool init_pipelines() {
        using namespace vk_detail;

        if (!make_layout(1, &context_.layout1) ||
            !make_layout(2, &context_.layout2) ||
            !make_layout(3, &context_.layout3) ||
            !make_layout(4, &context_.layout4)) {
            return false;
        }

        return make_pipeline(
                   kSpvTransform.data(), kSpvTransform.size(),
                   context_.layout2, sizeof(PcAffine), &context_.transform,
                   &context_.transform_layout) &&
               make_pipeline(
                   kSpvFill.data(), kSpvFill.size(), context_.layout1,
                   sizeof(PcAffine), &context_.fill,
                   &context_.fill_layout) &&
               make_pipeline(
                   kSpvMatmul.data(), kSpvMatmul.size(),
                   context_.layout3, sizeof(PcMatmul), &context_.matmul,
                   &context_.matmul_layout) &&
               make_pipeline(
                   kSpvMatmulTile.data(), kSpvMatmulTile.size(),
                   context_.layout3, sizeof(PcMatmul), &context_.matmul_tiled,
                   &context_.matmul_tiled_layout) &&
               make_pipeline(
                   kSpvTransformVec.data(), kSpvTransformVec.size(),
                   context_.layout2, sizeof(PcAffine),
                   &context_.transform_vec,
                   &context_.transform_vec_layout) &&
               make_pipeline(
                   kSpvExponentialVec.data(), kSpvExponentialVec.size(),
                   context_.layout2, sizeof(PcAffine),
                   &context_.exponential_vec,
                   &context_.exponential_vec_layout) &&
               make_pipeline(
                   kSpvFillVec.data(), kSpvFillVec.size(),
                   context_.layout1, sizeof(PcAffine), &context_.fill_vec,
                   &context_.fill_vec_layout) &&
               make_pipeline(
                   kSpvReduceSum.data(), kSpvReduceSum.size(),
                   context_.layout2, sizeof(PcReduce), &context_.reduce_sum,
                   &context_.reduce_sum_layout) &&
               make_pipeline(
                   kSpvReduceMinmax.data(), kSpvReduceMinmax.size(),
                   context_.layout2, sizeof(PcReduceMm),
                   &context_.reduce_minmax,
                   &context_.reduce_minmax_layout) &&
               make_pipeline(
                   kSpvElementwiseMinmax.data(),
                   kSpvElementwiseMinmax.size(), context_.layout3,
                   sizeof(PcSelect), &context_.select,
                   &context_.select_layout) &&
               make_pipeline(
                   kSpvMaskSelect.data(), kSpvMaskSelect.size(),
                   context_.layout4, sizeof(PcAffine), &context_.mask,
                   &context_.mask_layout) &&
               make_pipeline(
                   kSpvGather.data(), kSpvGather.size(),
                   context_.layout3, sizeof(PcAffine), &context_.gather,
                   &context_.gather_layout) &&
               make_pipeline(
                   kSpvDivide.data(), kSpvDivide.size(),
                   context_.layout3, sizeof(PcAffine), &context_.divide,
                   &context_.divide_layout) &&
               make_pipeline(
                   kSpvExponential.data(), kSpvExponential.size(),
                   context_.layout2, sizeof(PcAffine),
                   &context_.exponential, &context_.exponential_layout);
    }

    void warmup() {
        auto scratch = make_device_buffer(sizeof(float));

        if (scratch == nullptr) {
            return;
        }

        VkCommandPool pool{VK_NULL_HANDLE};
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_;

        if (vkCreateCommandPool(device_, &pool_info, nullptr, &pool) !=
            VK_SUCCESS) {
            return;
        }

        // Warmup dispatches one fill through the same record path as
        // every worker: transient pool, push constants, one group.
        VkCommandBuffer commands{VK_NULL_HANDLE};
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device_, &alloc_info, &commands) ==
            VK_SUCCESS) {
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            if (vkBeginCommandBuffer(commands, &begin_info) == VK_SUCCESS) {
                PcAffine push{1.0f, 0.0f, 0.0f, 1, 1};
                VkDescriptorPoolSize pool_size{};
                pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                pool_size.descriptorCount = 1;
                VkDescriptorPoolCreateInfo descriptor_info{};
                descriptor_info.sType =
                    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                descriptor_info.maxSets = 1;
                descriptor_info.poolSizeCount = 1;
                descriptor_info.pPoolSizes = &pool_size;
                VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};

                if (vkCreateDescriptorPool(
                        device_, &descriptor_info, nullptr,
                        &descriptor_pool) == VK_SUCCESS) {
                    Dispatch dispatch{
                        context_.fill,
                        context_.fill_layout,
                        context_.layout1,
                        {scratch->buffer},
                        to_bytes(push),
                        1,
                    };
                    record_dispatch(
                        device_, commands, descriptor_pool, dispatch);
                    vkDestroyDescriptorPool(
                        device_, descriptor_pool, nullptr);
                }
            }
        }

        vkDestroyCommandPool(device_, pool, nullptr);
    }

    void teardown() {
        // Release device buffers through the still-live device first:
        // allocation deleters capture the device handle, which dies below.
        allocations_.clear();

        if (device_ == VK_NULL_HANDLE) {
            teardown_instance();
            return;
        }

        const VkPipeline pipelines[] = {context_.transform, context_.fill,
            context_.matmul, context_.matmul_tiled, context_.transform_vec,
            context_.exponential_vec, context_.fill_vec,
            context_.reduce_sum, context_.reduce_minmax,
            context_.select, context_.mask, context_.gather,
            context_.divide, context_.exponential};

        for (const auto pipeline : pipelines) {
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline, nullptr);
            }
        }

        const VkPipelineLayout layouts[] = {context_.transform_layout,
            context_.fill_layout, context_.matmul_layout,
            context_.matmul_tiled_layout, context_.transform_vec_layout,
            context_.exponential_vec_layout, context_.fill_vec_layout,
            context_.reduce_sum_layout, context_.reduce_minmax_layout,
            context_.select_layout, context_.mask_layout,
            context_.gather_layout, context_.divide_layout,
            context_.exponential_layout};

        for (const auto layout : layouts) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, layout, nullptr);
            }
        }

        const VkDescriptorSetLayout sets[] = {context_.layout1,
            context_.layout2, context_.layout3, context_.layout4};

        for (const auto layout : sets) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, layout, nullptr);
            }
        }

        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }

        teardown_device();
        teardown_instance();
        context_ = Context{};
    }

    void teardown_device() {
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
    }

    [[nodiscard]] std::shared_ptr<DeviceBuffer> make_device_buffer(
        std::size_t floats) const {
        const VkDeviceSize bytes = floats * sizeof(float);

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = bytes == 0 ? 4 : bytes;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer{VK_NULL_HANDLE};

        if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer) !=
            VK_SUCCESS) {
            return nullptr;
        }

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer, &requirements);

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = requirements.size;
        alloc_info.memoryTypeIndex = memory_type_;

        VkDeviceMemory memory{VK_NULL_HANDLE};

        if (vkAllocateMemory(device_, &alloc_info, nullptr, &memory) !=
            VK_SUCCESS) {
            vkDestroyBuffer(device_, buffer, nullptr);
            return nullptr;
        }

        if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
            vkFreeMemory(device_, memory, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            return nullptr;
        }

        auto held = std::shared_ptr<DeviceBuffer>(
            new DeviceBuffer{buffer, memory, floats},
            [device = device_](DeviceBuffer* done) {
                vkFreeMemory(device, done->memory, nullptr);
                vkDestroyBuffer(device, done->buffer, nullptr);
                delete done;
            });

        return held;
    }

    void upload(const std::shared_ptr<DeviceBuffer>& target,
                const float* values,
                std::size_t floats) const {
        void* mapped = nullptr;
        vkMapMemory(
            device_, target->memory, 0, floats * sizeof(float), 0, &mapped);
        std::memcpy(mapped, values, floats * sizeof(float));
        vkUnmapMemory(device_, target->memory);
    }

    void download(const std::shared_ptr<DeviceBuffer>& source,
                  float* values,
                  std::size_t floats) const {
        void* mapped = nullptr;
        vkMapMemory(
            device_, source->memory, 0, floats * sizeof(float), 0, &mapped);
        std::memcpy(values, mapped, floats * sizeof(float));
        vkUnmapMemory(device_, source->memory);
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

        std::vector<float> working(
            it->second.bytes / dtype_bytes(it->second.dtype), 0.0f);
        download(it->second.device, working.data(), working.size());
        encode_all(
            it->second.dtype, working.data(), out, working.size());
    }

    [[nodiscard]] Slot slot_for(DataResidencyRef ref,
                                std::size_t bytes,
                                WorkDtype dtype) {
        auto it = allocations_.find(ref);

        const bool count_sufficient =
            it != allocations_.end() &&
            it->second.bytes / dtype_bytes(it->second.dtype) >=
                bytes / dtype_bytes(dtype);

        if (!count_sufficient) {
            const bool on_device = available_ && home_on_accelerator(ref);

            Allocation fresh{};
            fresh.on_device = on_device;
            fresh.dtype = dtype;
            fresh.bytes = bytes;
            fresh.host = std::make_shared<std::vector<unsigned char>>(
                bytes, static_cast<unsigned char>(0));
            std::vector<float> ones(bytes / dtype_bytes(dtype), 1.0f);
            encode_all(dtype, ones.data(), fresh.host->data(), ones.size());

            if (on_device) {
                std::vector<float> working(ones.size(), 0.0f);
                decode_all(
                    dtype, fresh.host->data(), working.data(), ones.size());
                fresh.device = make_device_buffer(working.size());

                if (fresh.device == nullptr) {
                    throw std::bad_alloc();
                }

                upload(fresh.device, working.data(), working.size());
            }

            it = allocations_.insert_or_assign(ref, std::move(fresh)).first;
        }

        auto& allocation = it->second;

        return Slot{
            allocation.on_device,
            allocation.dtype,
            allocation.host,
            allocation.device,
            allocation.bytes,
        };
    }

    [[nodiscard]] std::size_t allocation_bytes(DataResidencyRef ref) {
        const auto it = allocations_.find(ref);

        return it == allocations_.end() ? 0 : it->second.bytes;
    }

    struct Dispatch {
        VkPipeline pipeline{VK_NULL_HANDLE};
        VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
        VkDescriptorSetLayout set_layout{VK_NULL_HANDLE};
        std::vector<VkBuffer> buffers;
        std::vector<std::uint8_t> push;
        std::uint32_t groups{1};
        std::uint32_t groups_y{1};
    };

    struct Scratch {
        std::vector<std::shared_ptr<DeviceBuffer>> buffers;
    };

    static void record_dispatch(
        VkDevice device,
        VkCommandBuffer commands,
        VkDescriptorPool pool,
        const Dispatch& dispatch) {
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &dispatch.set_layout;

        VkDescriptorSet set{VK_NULL_HANDLE};
        vkAllocateDescriptorSets(device, &alloc_info, &set);

        std::vector<VkDescriptorBufferInfo> infos(dispatch.buffers.size());
        std::vector<VkWriteDescriptorSet> writes(dispatch.buffers.size());

        for (std::size_t i = 0; i < dispatch.buffers.size(); ++i) {
            infos[i].buffer = dispatch.buffers[i];
            infos[i].offset = 0;
            infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }

        vkUpdateDescriptorSets(
            device, static_cast<std::uint32_t>(writes.size()), writes.data(),
            0, nullptr);
        vkCmdBindPipeline(
            commands, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.pipeline);
        vkCmdBindDescriptorSets(
            commands, VK_PIPELINE_BIND_POINT_COMPUTE,
            dispatch.pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(
            commands, dispatch.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, static_cast<std::uint32_t>(dispatch.push.size()),
            dispatch.push.data());
        vkCmdDispatch(commands, dispatch.groups, dispatch.groups_y, 1);

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(
            commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    [[nodiscard]] static bool execute_run(
        const Context& context,
        VkCommandPool /*pool*/,
        VkCommandBuffer commands) {
        if (vkEndCommandBuffer(commands) != VK_SUCCESS) {
            return false;
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        VkFence fence{VK_NULL_HANDLE};

        if (vkCreateFence(context.device, &fence_info, nullptr, &fence) !=
            VK_SUCCESS) {
            return false;
        }

        bool ok = false;
        {
            std::lock_guard<std::mutex> guard(*context.submission);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &commands;
            ok = vkQueueSubmit(context.queue, 1, &submit_info, fence) ==
                 VK_SUCCESS;
        }

        if (ok) {
            ok = vkWaitForFences(
                     context.device, 1, &fence, VK_TRUE,
                     10000000000ull) == VK_SUCCESS;
        }

        vkDestroyFence(context.device, fence, nullptr);
        return ok;
    }

    [[nodiscard]] static bool begin_run(
        const Context& context,
        VkCommandPool* pool,
        VkCommandBuffer* commands,
        VkDescriptorPool* descriptor_pool,
        std::uint32_t descriptors) {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = context.queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        if (vkCreateCommandPool(
                context.device, &pool_info, nullptr, pool) != VK_SUCCESS) {
            return false;
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = *pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(
                context.device, &alloc_info, commands) != VK_SUCCESS) {
            vkDestroyCommandPool(context.device, *pool, nullptr);
            return false;
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(*commands, &begin_info) != VK_SUCCESS) {
            vkDestroyCommandPool(context.device, *pool, nullptr);
            return false;
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = descriptors;
        VkDescriptorPoolCreateInfo descriptor_info{};
        descriptor_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_info.maxSets = descriptors;
        descriptor_info.poolSizeCount = 1;
        descriptor_info.pPoolSizes = &pool_size;

        if (vkCreateDescriptorPool(
                context.device, &descriptor_info, nullptr,
                descriptor_pool) != VK_SUCCESS) {
            return false;
        }

        return true;
    }

    template <typename Push>
    [[nodiscard]] static std::vector<std::uint8_t> to_bytes(
        const Push& push) {
        std::vector<std::uint8_t> bytes(sizeof(Push), 0);
        std::memcpy(bytes.data(), &push, sizeof(Push));
        return bytes;
    }

    [[nodiscard]] static std::shared_ptr<DeviceBuffer> make_transient(
        const Context& context,
        const float* values,
        std::size_t floats,
        Scratch& scratch) {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = floats == 0 ? 4 : floats * sizeof(float);
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer{VK_NULL_HANDLE};

        if (vkCreateBuffer(context.device, &buffer_info, nullptr, &buffer) !=
            VK_SUCCESS) {
            return nullptr;
        }

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context.device, buffer, &requirements);

        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(context.physical, &properties);

        std::uint32_t memory_type = properties.memoryTypeCount;

        for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
            constexpr VkMemoryPropertyFlags wanted =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            if (((requirements.memoryTypeBits & (1u << i)) != 0u) &&
                ((properties.memoryTypes[i].propertyFlags & wanted) ==
                 wanted)) {
                memory_type = i;
                break;
            }
        }

        if (memory_type >= properties.memoryTypeCount) {
            vkDestroyBuffer(context.device, buffer, nullptr);
            return nullptr;
        }

        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = requirements.size;
        alloc_info.memoryTypeIndex = memory_type;

        VkDeviceMemory memory{VK_NULL_HANDLE};

        if (vkAllocateMemory(context.device, &alloc_info, nullptr, &memory) !=
            VK_SUCCESS) {
            vkDestroyBuffer(context.device, buffer, nullptr);
            return nullptr;
        }

        if (vkBindBufferMemory(context.device, buffer, memory, 0) !=
            VK_SUCCESS) {
            vkFreeMemory(context.device, memory, nullptr);
            vkDestroyBuffer(context.device, buffer, nullptr);
            return nullptr;
        }

        void* mapped_ptr = nullptr;
        vkMapMemory(
            context.device, memory, 0, floats * sizeof(float), 0, &mapped_ptr);
        std::memcpy(mapped_ptr, values, floats * sizeof(float));
        vkUnmapMemory(context.device, memory);

        auto held = std::shared_ptr<DeviceBuffer>(
            new DeviceBuffer{buffer, memory, floats},
            [device = context.device](DeviceBuffer* done) {
                vkFreeMemory(device, done->memory, nullptr);
                vkDestroyBuffer(device, done->buffer, nullptr);
                delete done;
            });

        scratch.buffers.push_back(held);
        return held;
    }

    static void download_into(const Context& context,
                              const std::shared_ptr<DeviceBuffer>& buffer,
                              float* values,
                              std::size_t floats) {
        void* mapped_ptr = nullptr;
        vkMapMemory(
            context.device, buffer->memory, 0, floats * sizeof(float), 0,
            &mapped_ptr);
        std::memcpy(values, mapped_ptr, floats * sizeof(float));
        vkUnmapMemory(context.device, buffer->memory);
    }

    [[nodiscard]] static bool dispatch_form(
        const Context& context,
        VkCommandBuffer commands,
        VkDescriptorPool pool,
        Scratch& scratch,
        const PhysicalBinding& binding,
        const std::vector<Slot>& slots,
        std::vector<std::vector<float>>& owned,
        const std::vector<std::size_t>& staging_for,
        const std::vector<std::size_t>& consuming,
        const WorkDescription& work,
        std::size_t out,
        std::size_t source_index,
        bool has_source) {
        const auto elements = work.storage_elements();
        const std::uint32_t groups =
            static_cast<std::uint32_t>((elements + 63) / 64);

        auto transient_of = [&](std::size_t staging) {
            return make_transient(
                context, owned[staging].data(), elements, scratch);
        };

        auto producing_buffer = [&](std::size_t target,
                                    std::size_t staging) {
            if (slots[target].on_device &&
                slots[target].device != nullptr) {
                return slots[target].device;
            }

            return transient_of(staging);
        };

        auto download_transient = [&](std::size_t target,
                                      std::size_t staging,
                                      const std::shared_ptr<DeviceBuffer>& buffer) {
            if (slots[target].on_device) {
                return;
            }

            download_into(context, buffer, owned[staging].data(), elements);
        };

        const bool exact_copy =
            work.form == WorkForm::ELEMENTWISE_AFFINE &&
            work.passes == 1 && work.destination_scale == 0.0f &&
            work.source_scale == 1.0f && work.constant == 0.0f;

        if (work.form == WorkForm::MATRIX_PRODUCT) {
            if (consuming.size() < 2) {
                return true;
            }

            auto left = transient_of(staging_for[consuming[0]]);
            auto right = transient_of(staging_for[consuming[1]]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (left == nullptr || right == nullptr ||
                destination == nullptr) {
                return false;
            }

            const PcMatmul push{
                static_cast<std::uint32_t>(work.rows),
                static_cast<std::uint32_t>(work.inner),
                static_cast<std::uint32_t>(work.columns),
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
            };
            const std::uint32_t total = static_cast<std::uint32_t>(
                work.rows * work.columns);
            const bool use_tile =
                context.max_workgroup >= 256 && work.rows <= 4096 &&
                work.columns <= 4096 &&
                work.rows * work.columns <= (std::size_t{1} << 30);

            if (use_tile) {
                const std::uint32_t groups_x = static_cast<std::uint32_t>(
                    (work.columns + 15) / 16);
                const std::uint32_t groups_y = static_cast<std::uint32_t>(
                    (work.rows + 15) / 16);
                record_dispatch(
                    context.device, commands, pool,
                    Dispatch{
                        context.matmul_tiled,
                        context.matmul_layout,
                        context.layout3,
                        {destination->buffer, left->buffer, right->buffer},
                        to_bytes(push),
                        groups_x,
                        groups_y,
                    });
            } else {
                record_dispatch(
                    context.device, commands, pool,
                    Dispatch{
                        context.matmul,
                        context.matmul_layout,
                        context.layout3,
                        {destination->buffer, left->buffer, right->buffer},
                        to_bytes(push),
                        (total + 63) / 64,
                    });
            }
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (work.form == WorkForm::REDUCE_SUM) {
            if (!has_source) {
                return true;
            }

            auto origin = transient_of(staging_for[source_index]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (origin == nullptr || destination == nullptr) {
                return false;
            }

            const PcReduce push{
                static_cast<std::uint32_t>(work.elements),
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.reduce_sum,
                    context.reduce_sum_layout,
                    context.layout2,
                    {destination->buffer, origin->buffer},
                    to_bytes(push),
                    1,
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (work.form == WorkForm::REDUCE_MAX ||
            work.form == WorkForm::REDUCE_MIN) {
            if (!has_source) {
                return true;
            }

            auto origin = transient_of(staging_for[source_index]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (origin == nullptr || destination == nullptr) {
                return false;
            }

            const PcReduceMm push{
                static_cast<std::uint32_t>(work.elements),
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
                work.form == WorkForm::REDUCE_MAX ? 1u : 0u,
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.reduce_minmax,
                    context.reduce_minmax_layout,
                    context.layout2,
                    {destination->buffer, origin->buffer},
                    to_bytes(push),
                    1,
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (work.form == WorkForm::ELEMENTWISE_MIN ||
            work.form == WorkForm::ELEMENTWISE_MAX) {
            if (consuming.size() < 2 ||
                binding.data[consuming[0]].residency.data ==
                    binding.data[out].residency.data ||
                binding.data[consuming[1]].residency.data ==
                    binding.data[out].residency.data) {
                return true;
            }

            auto left = transient_of(staging_for[consuming[0]]);
            auto right = transient_of(staging_for[consuming[1]]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (left == nullptr || right == nullptr ||
                destination == nullptr) {
                return false;
            }

            const PcSelect push{
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
                static_cast<std::uint32_t>(work.elements),
                work.form == WorkForm::ELEMENTWISE_MAX ? 1u : 0u,
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.select,
                    context.select_layout,
                    context.layout3,
                    {destination->buffer, left->buffer, right->buffer},
                    to_bytes(push),
                    groups,
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (work.form == WorkForm::MASK_SELECT) {
            if (consuming.size() < 3 ||
                binding.data[consuming[0]].residency.data ==
                    binding.data[out].residency.data ||
                binding.data[consuming[1]].residency.data ==
                    binding.data[out].residency.data ||
                binding.data[consuming[2]].residency.data ==
                    binding.data[out].residency.data) {
                return true;
            }

            auto predicate = transient_of(staging_for[consuming[0]]);
            auto first = transient_of(staging_for[consuming[1]]);
            auto second = transient_of(staging_for[consuming[2]]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (predicate == nullptr || first == nullptr ||
                second == nullptr || destination == nullptr) {
                return false;
            }

            const PcAffine push{
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
                static_cast<std::uint32_t>(work.elements),
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.mask,
                    context.mask_layout,
                    context.layout4,
                    {destination->buffer, predicate->buffer, first->buffer,
                     second->buffer},
                    to_bytes(push),
                    groups,
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (work.form == WorkForm::GATHER) {
            if (consuming.size() < 2 ||
                binding.data[consuming[0]].residency.data ==
                    binding.data[out].residency.data ||
                binding.data[consuming[1]].residency.data ==
                    binding.data[out].residency.data) {
                return true;
            }

            auto table = transient_of(staging_for[consuming[0]]);
            auto indices = transient_of(staging_for[consuming[1]]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (table == nullptr || indices == nullptr ||
                destination == nullptr) {
                return false;
            }

            const PcAffine push{
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
                static_cast<std::uint32_t>(work.elements),
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.gather,
                    context.gather_layout,
                    context.layout3,
                    {destination->buffer, table->buffer, indices->buffer},
                    to_bytes(push),
                    groups,
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (work.form == WorkForm::ELEMENTWISE_DIVIDE) {
            if (consuming.size() < 2 ||
                binding.data[consuming[0]].residency.data ==
                    binding.data[out].residency.data ||
                binding.data[consuming[1]].residency.data ==
                    binding.data[out].residency.data) {
                return true;
            }

            auto dividend = transient_of(staging_for[consuming[0]]);
            auto divisor = transient_of(staging_for[consuming[1]]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (dividend == nullptr || divisor == nullptr ||
                destination == nullptr) {
                return false;
            }

            const PcAffine push{
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
                static_cast<std::uint32_t>(work.elements),
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.divide,
                    context.divide_layout,
                    context.layout3,
                    {destination->buffer, dividend->buffer, divisor->buffer},
                    to_bytes(push),
                    groups,
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (work.form == WorkForm::EXPONENTIAL) {
            if (!has_source) {
                return true;
            }

            auto origin = transient_of(staging_for[source_index]);
            auto destination = producing_buffer(out, staging_for[out]);

            if (origin == nullptr || destination == nullptr) {
                return false;
            }

            const PcAffine push{
                work.destination_scale,
                work.source_scale,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
                static_cast<std::uint32_t>(work.elements),
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.exponential_vec,
                    context.exponential_vec_layout,
                    context.layout2,
                    {destination->buffer, origin->buffer},
                    to_bytes(push),
                    static_cast<std::uint32_t>(
                        (work.elements + 3) / 4),
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        if (!has_source) {
            auto destination = producing_buffer(out, staging_for[out]);

            if (destination == nullptr) {
                return false;
            }

            const PcAffine push{
                work.destination_scale,
                0.0f,
                work.constant,
                static_cast<std::uint32_t>(work.passes),
                static_cast<std::uint32_t>(work.elements),
            };
            record_dispatch(
                context.device, commands, pool,
                Dispatch{
                    context.fill_vec,
                    context.fill_vec_layout,
                    context.layout1,
                    {destination->buffer},
                    to_bytes(push),
                    static_cast<std::uint32_t>(
                        (work.elements + 3) / 4),
                });
            download_transient(out, staging_for[out], destination);
            return true;
        }

        auto origin = transient_of(staging_for[source_index]);
        auto destination = producing_buffer(out, staging_for[out]);

        if (origin == nullptr || destination == nullptr) {
            return false;
        }

        if (exact_copy && source_index != out) {
            if (!copy_buffers(context, commands, pool, origin, destination, elements)) {
                return false;
            }

            download_transient(out, staging_for[out], destination);
            return true;
        }

        const PcAffine push{
            work.destination_scale,
            work.source_scale,
            work.constant,
            static_cast<std::uint32_t>(work.passes),
            static_cast<std::uint32_t>(work.elements),
        };
        record_dispatch(
            context.device, commands, pool,
            Dispatch{
                context.transform_vec,
                context.transform_vec_layout,
                context.layout2,
                {destination->buffer, origin->buffer},
                to_bytes(push),
                static_cast<std::uint32_t>(
                    (work.elements + 3) / 4),
            });
        download_transient(out, staging_for[out], destination);
        return true;
    }

    [[nodiscard]] static bool copy_buffers(
        const Context& /*context*/,
        VkCommandBuffer commands,
        VkDescriptorPool pool,
        const std::shared_ptr<DeviceBuffer>& origin,
        const std::shared_ptr<DeviceBuffer>& destination,
        std::size_t floats) {
        (void)pool;
        VkBufferCopy region{};
        region.size = floats * sizeof(float);
        vkCmdCopyBuffer(commands, origin->buffer, destination->buffer, 1, &region);

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(
            commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
        return true;
    }

    [[nodiscard]] static Outcome run(const Context& context,
                                     const PhysicalBinding& binding,
                                     const std::vector<Slot>& slots,
                                     const WorkDescription& work,
                                     bool configured) {
        const auto begin = std::chrono::steady_clock::now();
        const auto done = [&](bool vulkan_ok) {
            const auto end = std::chrono::steady_clock::now();
            return Outcome{
                configured && vulkan_ok,
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - begin)
                        .count()),
            };
        };

        std::vector<std::size_t> consuming;

        for (std::size_t in = 0; in < binding.data.size(); ++in) {
            if (is_consuming(binding.data[in].role)) {
                consuming.push_back(in);
            }
        }

        const auto elements = work.storage_elements();
        std::vector<std::vector<float>> owned;
        std::vector<std::size_t> staging_for(slots.size(), 0);

        for (std::size_t s = 0; s < slots.size(); ++s) {
            bool shared = false;

            for (std::size_t earlier = 0; earlier < s; ++earlier) {
                if (slots[earlier].host == slots[s].host &&
                    slots[earlier].host != nullptr) {
                    staging_for[s] = staging_for[earlier];
                    shared = true;
                    break;
                }
            }

            if (shared) {
                continue;
            }

            staging_for[s] = owned.size();
            owned.emplace_back(elements, 0.0f);

            if (slots[s].on_device && slots[s].device != nullptr) {
                download_into(
                    context, slots[s].device, owned.back().data(), elements);
            } else if (slots[s].host != nullptr) {
                decode_all(
                    slots[s].dtype, slots[s].host->data(), owned.back().data(),
                    elements);
            }
        }

        std::uint32_t descriptor_budget = 0;

        for (std::size_t out = 0; out < binding.data.size(); ++out) {
            if (is_producing(binding.data[out].role)) {
                descriptor_budget += 4;
            }
        }

        VkCommandPool pool{VK_NULL_HANDLE};
        VkCommandBuffer commands{VK_NULL_HANDLE};
        VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};

        if (!begin_run(
                context, &pool, &commands, &descriptor_pool,
                descriptor_budget == 0 ? 1 : descriptor_budget)) {
            return done(false);
        }

        Scratch scratch;
        bool ok = true;

        for (std::size_t out = 0; out < binding.data.size() && ok; ++out) {
            if (!is_producing(binding.data[out].role)) {
                continue;
            }

            const auto source_index = work_source_index(binding, out);
            const bool has_source = source_index < binding.data.size();
            ok = dispatch_form(
                context, commands, descriptor_pool, scratch, binding, slots,
                owned, staging_for, consuming, work, out, source_index,
                has_source);
        }

        if (ok) {
            ok = execute_run(context, pool, commands);
        }

        vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
        vkDestroyCommandPool(context.device, pool, nullptr);

        if (!ok) {
            return done(false);
        }

        for (std::size_t out = 0; out < binding.data.size(); ++out) {
            if (!is_producing(binding.data[out].role)) {
                continue;
            }

            if (slots[out].on_device && slots[out].device != nullptr) {
                download_into(
                    context, slots[out].device, owned[staging_for[out]].data(),
                    elements);
            }

            encode_all(
                slots[out].dtype, owned[staging_for[out]].data(),
                slots[out].host->data(), elements);
        }

        return done(true);
    }

    const DataRegistry& data_;
    DeviceId accelerator_;
    bool available_{false};
    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_{VK_NULL_HANDLE};
    std::uint32_t queue_family_{0};
    std::uint32_t memory_type_{0};
    VkDevice device_{VK_NULL_HANDLE};
    VkCommandPool command_pool_{VK_NULL_HANDLE};
    Context context_;

    std::unordered_set<ExecutionId> failures_;
    std::unordered_map<DataResidencyRef, Allocation> allocations_;
    WorkPool pool_;
    std::vector<Job> jobs_;
    std::unordered_set<ExecutionId> submitted_;
};

} // namespace gerdos

