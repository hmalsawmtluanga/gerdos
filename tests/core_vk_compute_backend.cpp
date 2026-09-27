#include "test_check.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <thread>
#include <vector>

#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/vk/vk_compute_backend.hpp"

namespace {

using namespace gerdos;

struct Machine {
    DeviceRegistry devices;
    DataRegistry data;
    OperationRegistry operations;
    ExecutionRegistry executions;
    Topology topology;
    MeasurementRegistry measurements;

    Machine() {
        auto* host = devices.create_device(
            DeviceDescription{DeviceId{100}, "host"});

        auto* accelerator = devices.create_device(
            DeviceDescription{DeviceId{200}, "accelerator"});

        add(*host, ResourceId{101}, ResourceKind::MEMORY, "ram");
        add(*host, ResourceId{102}, ResourceKind::TRANSFER, "dma");
        add(*accelerator, ResourceId{201}, ResourceKind::MEMORY,
            "device-memory");
        add(*accelerator, ResourceId{202}, ResourceKind::COMPUTE, "compute");
        add(*accelerator, ResourceId{203}, ResourceKind::TRANSFER,
            "copy-engine");

        (void)topology.add_link(
            TopologyLink{
                TopologyLinkDescription{
                    TopologyLinkId{1},
                    TopologyEndpoint::resource_endpoint(
                        DeviceId{100}, ResourceId{101}),
                    TopologyEndpoint::resource_endpoint(
                        DeviceId{200}, ResourceId{201}),
                    TopologyLinkDirection::BIDIRECTIONAL,
                    TopologyLinkAttributes{48000000000, 500},
                },
            });
    }

    static void add(
        Device& device,
        ResourceId id,
        ResourceKind kind,
        const char* name) {
        (void)device.add_resource(
            Resource{
                ResourceDescription{
                    id,
                    device.description().id,
                    kind,
                    name,
                },
            });

        device.find_resource(id)->set_availability(
            ResourceAvailability::AVAILABLE);
    }

    static void record(
        Data& datum,
        DataResidencyId id,
        ResourceRef home,
        const char* representation) {
        (void)datum.add_residency(
            DataResidency{
                DataResidencyDescription{
                    id,
                    datum.description().id,
                    home,
                    representation,
                },
            });
    }

    static void usable(Data* datum, DataResidencyId id) {
        (void)datum->find_residency(id)->set_state(
            DataResidencyState::VALID);
    }
};

void run_to_completion(VkComputeBackend& backend, int patience_ms = 30000) {
    auto waited = 0;

    while (waited < patience_ms) {
        std::vector<BackendCompletion> completed;
        backend.poll(completed);

        if (!completed.empty()) {
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        waited += 10;
    }
}

} // namespace

int main() {
    using namespace gerdos;

    Machine machine;

    VkComputeBackend backend(machine.data, DeviceId{200});
    GERDOS_CHECK(backend.vk_available());

    BindingPlanner planner(
        machine.devices,
        machine.data,
        machine.topology,
        machine.measurements);

    const ResourceRef gpu_compute{DeviceId{200}, ResourceId{202}};

    auto entry = [](DataBindingRole role,
                     DataId data,
                     DataResidencyId residency) {
        return DataBinding{role, DataResidencyRef{data, residency}};
    };

    auto compute_on = [&](ResourceRef mechanism) {
        PhysicalBinding binding;
        binding.resources.push_back(
            ResourceBinding{ResourceBindingRole::COMPUTE, mechanism});
        return binding;
    };

    auto run = [&](const OperationDescription& description,
                   PhysicalBinding binding,
                   ExecutionId id) {
        const Operation operation{description};
        Execution attempt{ExecutionDescription{id, description.id}};
        GERDOS_CHECK(attempt.bind(binding));
        GERDOS_CHECK(backend.submit(operation, attempt));
        run_to_completion(backend);
    };

    // ---------------------------------------------------------------------
    // 1. Affine elementwise on the Vulkan engine
    // ---------------------------------------------------------------------

    {
        auto* source = machine.data.create_data(
            DataDescription{DataId{700}, "source"});
        Machine::record(
            *source, DataResidencyId{7001},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto* result = machine.data.create_data(
            DataDescription{DataId{701}, "result"});
        Machine::record(
            *result, DataResidencyId{7002},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        (void)machine.operations.create_operation(OperationDescription{
            OperationId{800},
            {DataId{700}},
            {DataId{701}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{64, 1, 0.5f, 1.5f, 0.0f},
        });

        // Seed the source through the backend itself (fill without a
        // source writes dst * 1 + 0 over fresh 1.0 storage: stays 1.0).
        // The Vulkan path starts from device-homed 1.0 storage, exactly
        // like the OpenCL path.
        const OperationDescription fill_desc{
            OperationId{801},
            {},
            {DataId{700}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{64, 1, 1.0f, 0.0f, 0.0f},
        };

        auto fill_binding = compute_on(gpu_compute);
        fill_binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{700}, DataResidencyId{7001}));
        run(fill_desc, fill_binding, ExecutionId{901});

        const OperationDescription work_desc{
            OperationId{800},
            {DataId{700}},
            {DataId{701}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{64, 1, 0.5f, 1.5f, 0.0f},
        };

        auto binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{700}, DataResidencyId{7001}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{701}, DataResidencyId{7002}));
        run(work_desc, binding, ExecutionId{900});

        const float value = backend.sample(
            DataResidencyRef{DataId{701}, DataResidencyId{7002}}, 0);
        std::printf("vk affine: %f (expect 2.0)\n", value);
        std::fflush(stdout);
        GERDOS_CHECK(value == 2.0f);
    }

    // ---------------------------------------------------------------------
    // 2. Chained reductions + exponential, matching the OpenCL engine
    // ---------------------------------------------------------------------

    {
        auto* table = machine.data.create_data(
            DataDescription{DataId{710}, "table"});
        Machine::record(
            *table, DataResidencyId{7101},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        Machine::usable(table, DataResidencyId{7101});

        auto* peak = machine.data.create_data(
            DataDescription{DataId{711}, "peak"});
        Machine::record(
            *peak, DataResidencyId{7102},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto* raised = machine.data.create_data(
            DataDescription{DataId{712}, "raised"});
        Machine::record(
            *raised, DataResidencyId{7103},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        // Reduce six 1.0 elements to 6 (element 0), leaving [6,1,1,1,1,1].
        run(
            OperationDescription{
                OperationId{810},
                {DataId{710}},
                {DataId{710}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(
                    entry(DataBindingRole::INPUT, DataId{710}, DataResidencyId{7101}));
                binding.data.push_back(
                    entry(DataBindingRole::OUTPUT, DataId{710}, DataResidencyId{7101}));
                return binding;
            }(),
            ExecutionId{910});

        const float folded = backend.sample(
            DataResidencyRef{DataId{710}, DataResidencyId{7101}}, 0);
        std::printf("vk reduce_sum: %f (expect 6.0)\n", folded);
        std::fflush(stdout);
        GERDOS_CHECK(folded == 6.0f);

        // Max over [6,1,1,1,1,1] is exactly 6.
        run(
            OperationDescription{
                OperationId{811},
                {DataId{710}},
                {DataId{711}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MAX},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(
                    entry(DataBindingRole::INPUT, DataId{710}, DataResidencyId{7101}));
                binding.data.push_back(
                    entry(DataBindingRole::OUTPUT, DataId{711}, DataResidencyId{7102}));
                return binding;
            }(),
            ExecutionId{911});

        const float peak_value = backend.sample(
            DataResidencyRef{DataId{711}, DataResidencyId{7102}}, 0);
        std::printf("vk reduce_max: %f (expect 6.0)\n", peak_value);
        std::fflush(stdout);
        GERDOS_CHECK(peak_value == 6.0f);

        // Exponential off the peak record: 1 + exp(6), 1 + exp(1).
        // Vulkan exp may differ from host libm in the last ulp: 1e-4.
        run(
            OperationDescription{
                OperationId{812},
                {DataId{710}},
                {DataId{712}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 1.0f, 1.0f, 0.0f, WorkForm::EXPONENTIAL},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(
                    entry(DataBindingRole::INPUT, DataId{710}, DataResidencyId{7101}));
                binding.data.push_back(
                    entry(DataBindingRole::OUTPUT, DataId{712}, DataResidencyId{7103}));
                return binding;
            }(),
            ExecutionId{912});

        const float raised_zero = backend.sample(
            DataResidencyRef{DataId{712}, DataResidencyId{7103}}, 0);
        const float raised_one = backend.sample(
            DataResidencyRef{DataId{712}, DataResidencyId{7103}}, 1);
        const float expected_zero = 1.0f + std::exp(6.0f);
        const float expected_one = 1.0f + std::exp(1.0f);
        std::printf(
            "vk exponential: %f %f (expected %f %f)\n",
            raised_zero, raised_one, expected_zero, expected_one);
        std::fflush(stdout);
        GERDOS_CHECK(std::fabs(raised_zero - expected_zero) <= 1e-4f * expected_zero);
        GERDOS_CHECK(std::fabs(raised_one - expected_one) <= 1e-4f * expected_one);
    }

    // ---------------------------------------------------------------------
    // 3. Matrix product on the Vulkan engine
    // ---------------------------------------------------------------------

    {
        // A = [[6,1,1],[1,1,1]] seeded by reducing a uniform record,
        // B = ones(3,2) from fill; C = [[8,8],[3,3]] exactly.
        auto* uniform = machine.data.create_data(
            DataDescription{DataId{720}, "uniform"});
        Machine::record(
            *uniform, DataResidencyId{7201},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        Machine::usable(uniform, DataResidencyId{7201});

        auto* left = machine.data.create_data(
            DataDescription{DataId{721}, "left"});
        Machine::record(
            *left, DataResidencyId{7202},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto* right = machine.data.create_data(
            DataDescription{DataId{722}, "right"});
        Machine::record(
            *right, DataResidencyId{7203},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto* product = machine.data.create_data(
            DataDescription{DataId{723}, "product"});
        Machine::record(
            *product, DataResidencyId{7204},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto bind2 = [&](DataId a, DataResidencyId ra, DataId b, DataResidencyId rb) {
            auto binding = compute_on(gpu_compute);
            binding.data.push_back(entry(DataBindingRole::INPUT, a, ra));
            binding.data.push_back(entry(DataBindingRole::INPUT, b, rb));
            return binding;
        };

        // B = ones via fill-exact? Fresh storage is already 1.0, so no
        // seeding needed: B reads 1.0 everywhere.
        run(
            OperationDescription{
                OperationId{820},
                {DataId{720}},
                {DataId{721}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{720}, DataResidencyId{7201}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{721}, DataResidencyId{7202}));
                return binding;
            }(),
            ExecutionId{920});

        run(
            OperationDescription{
                OperationId{821},
                {DataId{721}, DataId{722}},
                {DataId{723}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{0, 1, 0.0f, 1.0f, 0.0f, WorkForm::MATRIX_PRODUCT, 2, 3, 2},
            },
            [&] {
                auto binding = bind2(DataId{721}, DataResidencyId{7202}, DataId{722}, DataResidencyId{7203});
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{723}, DataResidencyId{7204}));
                return binding;
            }(),
            ExecutionId{921});

        std::printf(
            "vk matmul: %f %f %f %f (expect 8 8 3 3)\n",
            backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 0),
            backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 1),
            backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 2),
            backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 3));
        std::fflush(stdout);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 0) == 8.0f);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 1) == 8.0f);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 2) == 3.0f);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{723}, DataResidencyId{7204}}, 3) == 3.0f);
    }

    // ---------------------------------------------------------------------
    // 4. I8 exact + mixed conversion on the Vulkan engine
    // ---------------------------------------------------------------------

    {
        auto* table = machine.data.create_data(
            DataDescription{DataId{730}, "table"});
        Machine::record(
            *table, DataResidencyId{7301},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        Machine::usable(table, DataResidencyId{7301});

        auto* picked = machine.data.create_data(
            DataDescription{DataId{731}, "picked"});
        Machine::record(
            *picked, DataResidencyId{7302},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        run(
            OperationDescription{
                OperationId{830},
                {DataId{730}},
                {DataId{731}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM, 0, 0, 0, WorkDtype::I8},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{730}, DataResidencyId{7301}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{731}, DataResidencyId{7302}));
                return binding;
            }(),
            ExecutionId{930});

        const float folded = backend.sample(
            DataResidencyRef{DataId{731}, DataResidencyId{7302}}, 0);
        std::printf("vk i8 reduce_sum: %f (expect 6.0)\n", folded);
        std::fflush(stdout);
        GERDOS_CHECK(folded == 6.0f);

        // Mixed-dtype exact copy through F32.
        auto* half = machine.data.create_data(
            DataDescription{DataId{732}, "half"});
        Machine::record(
            *half, DataResidencyId{7303},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        run(
            OperationDescription{
                OperationId{831},
                {DataId{730}},
                {DataId{732}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{730}, DataResidencyId{7301}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{732}, DataResidencyId{7303}));
                return binding;
            }(),
            ExecutionId{931});

        const float copied = backend.sample(
            DataResidencyRef{DataId{732}, DataResidencyId{7303}}, 0);
        std::printf("vk mixed copy: %f (expect 1.0)\n", copied);
        std::fflush(stdout);
        GERDOS_CHECK(copied == 1.0f);
    }

    // ---------------------------------------------------------------------
    // 5. Selection forms on the Vulkan engine
    // ---------------------------------------------------------------------

    {
        // Table [6,1,1,1,1,1] via reduce; partner uniform 2; mask and
        // gather mirror the OpenCL chains with the same expectations.
        auto* table = machine.data.create_data(
            DataDescription{DataId{740}, "table"});
        Machine::record(
            *table, DataResidencyId{7401},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        Machine::usable(table, DataResidencyId{7401});

        auto* partner = machine.data.create_data(
            DataDescription{DataId{741}, "partner"});
        Machine::record(
            *partner, DataResidencyId{7402},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto* picked = machine.data.create_data(
            DataDescription{DataId{742}, "picked"});
        Machine::record(
            *picked, DataResidencyId{7403},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto* predicate = machine.data.create_data(
            DataDescription{DataId{743}, "predicate"});
        Machine::record(
            *predicate, DataResidencyId{7404},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto* indices = machine.data.create_data(
            DataDescription{DataId{744}, "indices"});
        Machine::record(
            *indices, DataResidencyId{7405},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        auto single = [&](OperationId op, DataId src, DataResidencyId rsrc, DataId dst, DataResidencyId rdst, WorkDescription work, ExecutionId eid) {
            run(
                OperationDescription{op, {src}, {dst}, {}, {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}}, work},
                [&] {
                    auto binding = compute_on(gpu_compute);
                    binding.data.push_back(entry(DataBindingRole::INPUT, src, rsrc));
                    binding.data.push_back(entry(DataBindingRole::OUTPUT, dst, rdst));
                    return binding;
                }(),
                eid);
        };

        // Table folds to [6,1,1,1,1,1]; partner fills uniform 2.
        single(OperationId{840}, DataId{740}, DataResidencyId{7401}, DataId{740}, DataResidencyId{7401}, WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM}, ExecutionId{940});
        single(OperationId{841}, DataId{741}, DataResidencyId{7402}, DataId{741}, DataResidencyId{7402}, WorkDescription{6, 1, 0.0f, 0.0f, 2.0f}, ExecutionId{941});

        // REDUCE_MIN over [6,1,...] is exactly 1.
        single(OperationId{842}, DataId{740}, DataResidencyId{7401}, DataId{742}, DataResidencyId{7403}, WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MIN}, ExecutionId{942});
        const float floor_value = backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 0);
        std::printf("vk reduce_min: %f (expect 1.0)\n", floor_value);
        std::fflush(stdout);
        GERDOS_CHECK(floor_value == 1.0f);

        // max(T,2) = [6,2,2,2,2,2]; min(T,2) = [2,1,1,1,1,1].
        run(
            OperationDescription{
                OperationId{843},
                {DataId{740}, DataId{741}},
                {DataId{742}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::ELEMENTWISE_MAX},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{740}, DataResidencyId{7401}));
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{741}, DataResidencyId{7402}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{742}, DataResidencyId{7403}));
                return binding;
            }(),
            ExecutionId{943});
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 0) == 6.0f);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 1) == 2.0f);

        run(
            OperationDescription{
                OperationId{844},
                {DataId{740}, DataId{741}},
                {DataId{742}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::ELEMENTWISE_MIN},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{740}, DataResidencyId{7401}));
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{741}, DataResidencyId{7402}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{742}, DataResidencyId{7403}));
                return binding;
            }(),
            ExecutionId{944});
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 0) == 2.0f);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 1) == 1.0f);

        // Predicate P = T - 1 = [5,0,0,0,0,0]; mask picks T at element 0.
        single(OperationId{845}, DataId{740}, DataResidencyId{7401}, DataId{743}, DataResidencyId{7404}, WorkDescription{6, 1, -1.0f, 1.0f, 0.0f}, ExecutionId{945});
        run(
            OperationDescription{
                OperationId{846},
                {DataId{743}, DataId{740}, DataId{741}},
                {DataId{742}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::MASK_SELECT},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{743}, DataResidencyId{7404}));
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{740}, DataResidencyId{7401}));
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{741}, DataResidencyId{7402}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{742}, DataResidencyId{7403}));
                return binding;
            }(),
            ExecutionId{946});
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 0) == 6.0f);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 1) == 2.0f);

        // Gather: I = T - 1 = [5,0,0,0,0,0] reads T[5]=1, T[0]=6.
        single(OperationId{847}, DataId{740}, DataResidencyId{7401}, DataId{744}, DataResidencyId{7405}, WorkDescription{6, 1, -1.0f, 1.0f, 0.0f}, ExecutionId{947});
        run(
            OperationDescription{
                OperationId{848},
                {DataId{740}, DataId{744}},
                {DataId{742}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::GATHER},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{740}, DataResidencyId{7401}));
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{744}, DataResidencyId{7405}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{742}, DataResidencyId{7403}));
                return binding;
            }(),
            ExecutionId{948});
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 0) == 1.0f);
        GERDOS_CHECK(backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 1) == 6.0f);
    }

    // ---------------------------------------------------------------------
    // 6. The seam refuses non-accelerator attempts and hostile sizing
    // ---------------------------------------------------------------------

    {
        // Host-bound compute belongs to another engine.
        const OperationDescription host_work{
            OperationId{860},
            {DataId{750}},
            {DataId{751}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{8, 1, 0.0f, 1.0f, 0.0f},
        };
        const Operation host_op{host_work};
        Execution host_attempt{ExecutionDescription{ExecutionId{960}, OperationId{860}}};
        PhysicalBinding host_binding;
        host_binding.data.push_back(entry(DataBindingRole::INPUT, DataId{750}, DataResidencyId{7501}));
        host_binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{751}, DataResidencyId{7502}));
        host_binding.resources.push_back(ResourceBinding{ResourceBindingRole::COMPUTE, ResourceRef{DeviceId{100}, ResourceId{102}}});
        GERDOS_CHECK(host_attempt.bind(host_binding));
        GERDOS_CHECK(!backend.submit(host_op, host_attempt));

        // Hostile sizing never reaches an allocation.
        auto* hostile_data = machine.data.create_data(DataDescription{DataId{752}, "hostile"});
        Machine::record(*hostile_data, DataResidencyId{7521}, ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        const OperationDescription hostile_work{
            OperationId{861},
            {DataId{752}},
            {DataId{752}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{(std::size_t{1} << 61) + 1, 1, 0.0f, 1.0f, 0.0f},
        };
        const Operation hostile_op{hostile_work};
        Execution hostile_attempt{ExecutionDescription{ExecutionId{961}, OperationId{861}}};
        PhysicalBinding hostile_binding;
        hostile_binding.data.push_back(entry(DataBindingRole::INPUT, DataId{752}, DataResidencyId{7521}));
        hostile_binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{752}, DataResidencyId{7521}));
        hostile_binding.resources.push_back(ResourceBinding{ResourceBindingRole::COMPUTE, gpu_compute});
        GERDOS_CHECK(hostile_attempt.bind(hostile_binding));
        const auto before = backend.allocation_count();
        GERDOS_CHECK(!backend.submit(hostile_op, hostile_attempt));
        GERDOS_CHECK(backend.allocation_count() == before);
    }

    return 0;
}

