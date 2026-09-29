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
    if (!backend.vk_available()) {
        std::printf("SKIP: no Vulkan GPU device\n");
        return 0;
    }

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

        // Tiled path: 17x17x17 with A = 2.0, B = 3.0 gives 102
        // everywhere — tile guards cross bounds in every dimension.
        auto* tile_a = machine.data.create_data(
            DataDescription{DataId{724}, "tile_a"});
        Machine::record(
            *tile_a, DataResidencyId{7241},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        auto* tile_b = machine.data.create_data(
            DataDescription{DataId{725}, "tile_b"});
        Machine::record(
            *tile_b, DataResidencyId{7242},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        auto* tile_c = machine.data.create_data(
            DataDescription{DataId{726}, "tile_c"});
        Machine::record(
            *tile_c, DataResidencyId{7243},
            ResourceRef{DeviceId{200}, ResourceId{201}}, "device");
        auto fill_n = [&](OperationId op, DataId data, DataResidencyId res, float constant, ExecutionId eid) {
            run(
                OperationDescription{op, {}, {data}, {}, {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}}, WorkDescription{17 * 17, 1, 0.0f, 0.0f, constant}},
                [&] {
                    auto binding = compute_on(gpu_compute);
                    binding.data.push_back(entry(DataBindingRole::OUTPUT, data, res));
                    return binding;
                }(),
                eid);
        };
        fill_n(OperationId{822}, DataId{724}, DataResidencyId{7241}, 2.0f, ExecutionId{922});
        fill_n(OperationId{823}, DataId{725}, DataResidencyId{7242}, 3.0f, ExecutionId{923});
        run(
            OperationDescription{
                OperationId{824},
                {DataId{724}, DataId{725}},
                {DataId{726}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{0, 1, 0.0f, 1.0f, 0.0f, WorkForm::MATRIX_PRODUCT, 17, 17, 17},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{724}, DataResidencyId{7241}));
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{725}, DataResidencyId{7242}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{726}, DataResidencyId{7243}));
                return binding;
            }(),
            ExecutionId{924});
        const float t00 = backend.sample(DataResidencyRef{DataId{726}, DataResidencyId{7243}}, 0);
        const float t016 = backend.sample(DataResidencyRef{DataId{726}, DataResidencyId{7243}}, 16);
        const float t158 = backend.sample(DataResidencyRef{DataId{726}, DataResidencyId{7243}}, 16 * 17 + 16);
        std::printf("vk tiled 17x17x17: %f %f %f (expect 102)\n", t00, t016, t158);
        std::fflush(stdout);
        GERDOS_CHECK(t00 == 102.0f);
        GERDOS_CHECK(t016 == 102.0f);
        GERDOS_CHECK(t158 == 102.0f);
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

        // Divide: T / 2 reads the scalar divisor at B[0]: [3,0.5,...].
        run(
            OperationDescription{
                OperationId{849},
                {DataId{740}, DataId{741}},
                {DataId{742}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::ELEMENTWISE_DIVIDE},
            },
            [&] {
                auto binding = compute_on(gpu_compute);
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{740}, DataResidencyId{7401}));
                binding.data.push_back(entry(DataBindingRole::INPUT, DataId{741}, DataResidencyId{7402}));
                binding.data.push_back(entry(DataBindingRole::OUTPUT, DataId{742}, DataResidencyId{7403}));
                return binding;
            }(),
            ExecutionId{949});
        const float div_head = backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 0);
        const float div_tail = backend.sample(DataResidencyRef{DataId{742}, DataResidencyId{7403}}, 1);
        std::printf("vk divide: %f %f (expect 3.0 0.5)\n", div_head, div_tail);
        std::fflush(stdout);
        GERDOS_CHECK(div_head == 3.0f);
        GERDOS_CHECK(div_tail == 0.5f);
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

    // ---------------------------------------------------------------------
    // Tiny classifier on the Vulkan engine: exact values, surplus kept
    // ---------------------------------------------------------------------

    {
        // The tiny-model shapes through the ENGINE backend: mixed
        // footprints (matmul-sized-6 record read 2-wide) must keep
        // data here exactly as on the CPU engine. Hand-bound to the
        // device compute mechanism; skipped cleanly without a device.
        const ResourceRef home{DeviceId{200}, ResourceId{201}};

        auto* weights = machine.data.create_data(
            DataDescription{DataId{780}, "weights"});
        Machine::record(*weights, DataResidencyId{7801}, home, "weights");
        Machine::usable(weights, DataResidencyId{7801});

        auto* features = machine.data.create_data(
            DataDescription{DataId{781}, "features"});
        Machine::record(*features, DataResidencyId{7802}, home, "features");
        Machine::usable(features, DataResidencyId{7802});

        auto* scores = machine.data.create_data(
            DataDescription{DataId{782}, "scores"});
        Machine::record(*scores, DataResidencyId{7803}, home, "scores");
        Machine::usable(scores, DataResidencyId{7803});

        auto* biased = machine.data.create_data(
            DataDescription{DataId{783}, "biased"});
        Machine::record(*biased, DataResidencyId{7804}, home, "biased");
        Machine::usable(biased, DataResidencyId{7804});

        auto* second = machine.data.create_data(
            DataDescription{DataId{784}, "second"});
        Machine::record(*second, DataResidencyId{7805}, home, "second");
        Machine::usable(second, DataResidencyId{7805});

        auto* best = machine.data.create_data(
            DataDescription{DataId{785}, "best"});
        Machine::record(*best, DataResidencyId{7806}, home, "best");
        Machine::usable(best, DataResidencyId{7806});

        auto* peak = machine.data.create_data(
            DataDescription{DataId{786}, "peak"});
        Machine::record(*peak, DataResidencyId{7807}, home, "peak");
        Machine::usable(peak, DataResidencyId{7807});

        auto* average = machine.data.create_data(
            DataDescription{DataId{787}, "average"});
        Machine::record(*average, DataResidencyId{7808}, home, "average");
        Machine::usable(average, DataResidencyId{7808});

        auto binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{780}, DataResidencyId{7801}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{780}, DataResidencyId{7801}));
        run(OperationDescription{
                OperationId{920},
                {DataId{780}},
                {DataId{780}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{6, 1, 1.0f, 1.0f, 0.0f, WorkForm::ELEMENTWISE_AFFINE},
            },
            binding, ExecutionId{990});

        binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{780}, DataResidencyId{7801}));
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{781}, DataResidencyId{7802}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{782}, DataResidencyId{7803}));
        run(OperationDescription{
                OperationId{921},
                {DataId{780}, DataId{781}},
                {DataId{782}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{0, 1, 0.0f, 1.0f, 0.0f, WorkForm::MATRIX_PRODUCT, 2, 3, 1},
            },
            binding, ExecutionId{991});

        binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{782}, DataResidencyId{7803}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{783}, DataResidencyId{7804}));
        run(OperationDescription{
                OperationId{922},
                {DataId{782}},
                {DataId{783}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{2, 1, 0.0f, 1.0f, 0.5f},
            },
            binding, ExecutionId{992});

        binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{783}, DataResidencyId{7804}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{784}, DataResidencyId{7805}));
        run(OperationDescription{
                OperationId{923},
                {DataId{783}},
                {DataId{784}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{2, 1, 0.0f, 2.0f, 1.0f},
            },
            binding, ExecutionId{993});

        binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{784}, DataResidencyId{7805}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{785}, DataResidencyId{7806}));
        run(OperationDescription{
                OperationId{924},
                {DataId{784}},
                {DataId{785}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{2, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MIN},
            },
            binding, ExecutionId{994});

        binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{784}, DataResidencyId{7805}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{786}, DataResidencyId{7807}));
        run(OperationDescription{
                OperationId{925},
                {DataId{784}},
                {DataId{786}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{2, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MAX},
            },
            binding, ExecutionId{995});

        binding = compute_on(gpu_compute);
        binding.data.push_back(
            entry(DataBindingRole::INPUT, DataId{784}, DataResidencyId{7805}));
        binding.data.push_back(
            entry(DataBindingRole::OUTPUT, DataId{787}, DataResidencyId{7808}));
        run(OperationDescription{
                OperationId{926},
                {DataId{784}},
                {DataId{787}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{2, 1, 0.0f, 0.5f, 0.0f, WorkForm::REDUCE_SUM},
            },
            binding, ExecutionId{996});

        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{780}, DataResidencyId{7801}}, 0) == 2.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{782}, DataResidencyId{7803}}, 0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{782}, DataResidencyId{7803}}, 1) == 6.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{783}, DataResidencyId{7804}}, 0) == 6.5f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{784}, DataResidencyId{7805}}, 0) == 14.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{785}, DataResidencyId{7806}}, 0) == 14.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{786}, DataResidencyId{7807}}, 0) == 14.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{787}, DataResidencyId{7808}}, 0) == 14.0f);
        std::printf("tiny Vulkan: W=2 scores=6 biased=6.5 layer2=14 stats=14\n");
        std::fflush(stdout);
    }

    // ---------------------------------------------------------------------
    // Gated selector on the Vulkan engine: the same chain as the CPU proof
    // ---------------------------------------------------------------------

    {
        // The second tiny model through the ENGINE backend: fill a
        // uniform partner, fold the table in place, bound it both ways,
        // gate by predicate, gather by computed index, reduce the pick.
        // Same closed forms as the CPU proof; skipped cleanly with it
        // when no device is present (whole binary).
        const ResourceRef home{DeviceId{200}, ResourceId{201}};

        auto* table = machine.data.create_data(
            DataDescription{DataId{790}, "table"});
        Machine::record(*table, DataResidencyId{7901}, home, "table");
        Machine::usable(table, DataResidencyId{7901});

        auto* partner = machine.data.create_data(
            DataDescription{DataId{791}, "partner"});
        Machine::record(*partner, DataResidencyId{7902}, home, "partner");
        Machine::usable(partner, DataResidencyId{7902});

        auto* hi = machine.data.create_data(
            DataDescription{DataId{792}, "hi"});
        Machine::record(*hi, DataResidencyId{7903}, home, "hi");
        Machine::usable(hi, DataResidencyId{7903});

        auto* lo = machine.data.create_data(
            DataDescription{DataId{793}, "lo"});
        Machine::record(*lo, DataResidencyId{7904}, home, "lo");
        Machine::usable(lo, DataResidencyId{7904});

        auto* gate = machine.data.create_data(
            DataDescription{DataId{794}, "gate"});
        Machine::record(*gate, DataResidencyId{7905}, home, "gate");
        Machine::usable(gate, DataResidencyId{7905});

        auto* pred = machine.data.create_data(
            DataDescription{DataId{795}, "pred"});
        Machine::record(*pred, DataResidencyId{7906}, home, "pred");
        Machine::usable(pred, DataResidencyId{7906});

        auto* idx = machine.data.create_data(
            DataDescription{DataId{796}, "idx"});
        Machine::record(*idx, DataResidencyId{7907}, home, "idx");
        Machine::usable(idx, DataResidencyId{7907});

        auto* picked = machine.data.create_data(
            DataDescription{DataId{797}, "picked"});
        Machine::record(*picked, DataResidencyId{7908}, home, "picked");
        Machine::usable(picked, DataResidencyId{7908});

        auto* floor = machine.data.create_data(
            DataDescription{DataId{798}, "floor"});
        Machine::record(*floor, DataResidencyId{7909}, home, "floor");
        Machine::usable(floor, DataResidencyId{7909});

        auto* peak = machine.data.create_data(
            DataDescription{DataId{799}, "peak"});
        Machine::record(*peak, DataResidencyId{7910}, home, "peak");
        Machine::usable(peak, DataResidencyId{7910});

        auto* floor2 = machine.data.create_data(
            DataDescription{DataId{800}, "floor2"});
        Machine::record(*floor2, DataResidencyId{7911}, home, "floor2");
        Machine::usable(floor2, DataResidencyId{7911});

        auto* total = machine.data.create_data(
            DataDescription{DataId{801}, "total"});
        Machine::record(*total, DataResidencyId{7912}, home, "total");
        Machine::usable(total, DataResidencyId{7912});

        auto single = [&](OperationId op, DataId src, DataResidencyId rsrc,
                          DataId dst, DataResidencyId rdst,
                          WorkDescription work, ExecutionId eid) {
            auto binding = compute_on(gpu_compute);
            binding.data.push_back(entry(DataBindingRole::INPUT, src, rsrc));
            binding.data.push_back(entry(DataBindingRole::OUTPUT, dst, rdst));
            run(OperationDescription{
                    op,
                    {src},
                    {dst},
                    {},
                    {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                    work,
                },
                binding, eid);
        };

        // P = 0*P + 2; T folds in place to [6,1,1,1,1,1]; floor = 1.
        single(OperationId{930}, DataId{791}, DataResidencyId{7902},
               DataId{791}, DataResidencyId{7902},
               WorkDescription{6, 1, 0.0f, 0.0f, 2.0f},
               ExecutionId{965});
        single(OperationId{931}, DataId{790}, DataResidencyId{7901},
               DataId{790}, DataResidencyId{7901},
               WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
               ExecutionId{966});
        single(OperationId{932}, DataId{790}, DataResidencyId{7901},
               DataId{798}, DataResidencyId{7909},
               WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MIN},
               ExecutionId{967});

        // hi = max(T,P); lo = min(T,P).
        auto pair = [&](OperationId op, DataId dst, DataResidencyId rdst,
                        WorkForm form, ExecutionId eid) {
            auto binding = compute_on(gpu_compute);
            binding.data.push_back(
                entry(DataBindingRole::INPUT, DataId{790}, DataResidencyId{7901}));
            binding.data.push_back(
                entry(DataBindingRole::INPUT, DataId{791}, DataResidencyId{7902}));
            binding.data.push_back(
                entry(DataBindingRole::OUTPUT, dst, rdst));
            run(OperationDescription{
                    op,
                    {DataId{790}, DataId{791}},
                    {dst},
                    {},
                    {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                    WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, form},
                },
                binding, eid);
        };
        pair(OperationId{933}, DataId{792}, DataResidencyId{7903},
             WorkForm::ELEMENTWISE_MAX, ExecutionId{968});
        pair(OperationId{934}, DataId{793}, DataResidencyId{7904},
             WorkForm::ELEMENTWISE_MIN, ExecutionId{969});

        // pred = T - 1; gate = select(pred, T, P).
        single(OperationId{935}, DataId{790}, DataResidencyId{7901},
               DataId{795}, DataResidencyId{7906},
               WorkDescription{6, 1, 0.0f, 1.0f, -1.0f},
               ExecutionId{970});
        {
            auto binding = compute_on(gpu_compute);
            binding.data.push_back(
                entry(DataBindingRole::INPUT, DataId{795}, DataResidencyId{7906}));
            binding.data.push_back(
                entry(DataBindingRole::INPUT, DataId{790}, DataResidencyId{7901}));
            binding.data.push_back(
                entry(DataBindingRole::INPUT, DataId{791}, DataResidencyId{7902}));
            binding.data.push_back(
                entry(DataBindingRole::OUTPUT, DataId{794}, DataResidencyId{7905}));
            run(OperationDescription{
                    OperationId{936},
                    {DataId{795}, DataId{790}, DataId{791}},
                    {DataId{794}},
                    {},
                    {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                    WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::MASK_SELECT},
                },
                binding, ExecutionId{971});
        }

        // idx = T - 1; picked = gather(T, idx); peak/floor/total.
        single(OperationId{937}, DataId{790}, DataResidencyId{7901},
               DataId{796}, DataResidencyId{7907},
               WorkDescription{6, 1, 0.0f, 1.0f, -1.0f},
               ExecutionId{972});
        {
            auto binding = compute_on(gpu_compute);
            binding.data.push_back(
                entry(DataBindingRole::INPUT, DataId{790}, DataResidencyId{7901}));
            binding.data.push_back(
                entry(DataBindingRole::INPUT, DataId{796}, DataResidencyId{7907}));
            binding.data.push_back(
                entry(DataBindingRole::OUTPUT, DataId{797}, DataResidencyId{7908}));
            run(OperationDescription{
                    OperationId{938},
                    {DataId{790}, DataId{796}},
                    {DataId{797}},
                    {},
                    {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                    WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::GATHER},
                },
                binding, ExecutionId{973});
        }
        single(OperationId{939}, DataId{797}, DataResidencyId{7908},
               DataId{799}, DataResidencyId{7910},
               WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MAX},
               ExecutionId{974});
        single(OperationId{940}, DataId{797}, DataResidencyId{7908},
               DataId{800}, DataResidencyId{7911},
               WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MIN},
               ExecutionId{975});
        single(OperationId{941}, DataId{797}, DataResidencyId{7908},
               DataId{801}, DataResidencyId{7912},
               WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
               ExecutionId{976});

        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{791}, DataResidencyId{7902}}, 0) == 2.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{790}, DataResidencyId{7901}}, 0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{790}, DataResidencyId{7901}}, 1) == 1.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{798}, DataResidencyId{7909}}, 0) == 1.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{792}, DataResidencyId{7903}}, 0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{792}, DataResidencyId{7903}}, 1) == 2.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{793}, DataResidencyId{7904}}, 0) == 2.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{793}, DataResidencyId{7904}}, 1) == 1.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{794}, DataResidencyId{7905}}, 0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{794}, DataResidencyId{7905}}, 1) == 2.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{797}, DataResidencyId{7908}}, 0) == 1.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{797}, DataResidencyId{7908}}, 1) == 6.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{799}, DataResidencyId{7910}}, 0) == 6.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{800}, DataResidencyId{7911}}, 0) == 1.0f);
        GERDOS_CHECK(
            backend.sample(DataResidencyRef{DataId{801}, DataResidencyId{7912}}, 0) == 31.0f);
        std::printf("tiny-gated Vulkan: T=[6,1..] P=2 hi=6/2 lo=2/1 gate=6/2 picked=1/6 peak=6 floor=1 total=31\n");
        std::fflush(stdout);
    }

    return 0;
}

