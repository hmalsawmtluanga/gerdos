#include "test_check.hpp"

#include <cmath>
#include <cstdio>
#include <optional>
#include <vector>

#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/hw/heterogeneous_backend.hpp"

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
                    TopologyLinkAttributes{48'000'000'000, 500},
                },
            });

        // Payload: usable on ram, fresh on the device.
        auto* payload = data.create_data(
            DataDescription{DataId{500}, "payload"});

        record(*payload, DataResidencyId{5001},
               ResourceRef{DeviceId{100}, ResourceId{101}}, "staged");
        record(*payload, DataResidencyId{5002},
               ResourceRef{DeviceId{200}, ResourceId{201}}, "resident");
        usable(payload, DataResidencyId{5001});

        // Result: fresh on the device.
        auto* result = data.create_data(
            DataDescription{DataId{501}, "result"});

        record(*result, DataResidencyId{5101},
               ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

        (void)operations.create_operation(
            OperationDescription{
                OperationId{800},
                {DataId{500}},
                {DataId{500}},
                {},
                {
                    ResourceRequirement{ResourceBindingRole::TRANSFER, 1},
                },
                WorkDescription{1 << 18, 1, 0.0f, 1.0f, 0.0f},
            });

        (void)operations.create_operation(
            OperationDescription{
                OperationId{801},
                {DataId{500}},
                {DataId{501}},
                {},
                {
                    ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
                },
                WorkDescription{1 << 12, 1, 0.5f, 1.5f, 0.0f},
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
        ResourceRef host,
        const char* representation) {
        (void)datum.add_residency(
            DataResidency{
                DataResidencyDescription{
                    id,
                    datum.description().id,
                    host,
                    representation,
                },
            });
    }

    static void usable(Data* datum, DataResidencyId id) {
        (void)datum->find_residency(id)->set_state(
            DataResidencyState::VALID);
    }
};

void run_to_completion(Executor& executor, ExecutionId id) {
    std::vector<AttemptStatus> outcomes;
    executor.advance(outcomes);

    while (outcomes.empty()) {
        executor.advance(outcomes);
    }

    (void)id;
}

} // namespace

int main() {
    using namespace gerdos;

    Machine machine;

    HeterogeneousBackend backend(machine.data, DeviceId{200});
    GERDOS_CHECK(backend.gpu_available());

    BindingPlanner planner(
        machine.devices,
        machine.data,
        machine.topology,
        machine.measurements);

    Executor executor(
        machine.executions,
        machine.operations,
        machine.devices,
        machine.data,
        backend,
        &machine.measurements);

    const DataResidencyRef staged{DataId{500}, DataResidencyId{5001}};
    const DataResidencyRef resident{DataId{500}, DataResidencyId{5002}};
    const DataResidencyRef result{DataId{501}, DataResidencyId{5101}};

    // ---------------------------------------------------------------------
    // 1. A host mechanism runs the CPU engine over the real data plane
    // ---------------------------------------------------------------------

    {
        const auto* operation =
            machine.operations.find_operation(OperationId{800});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(
            binding->resources[0].resource ==
            (ResourceRef{DeviceId{100}, ResourceId{102}}));

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{900}, OperationId{800}});

        GERDOS_CHECK(execution->bind(*binding));
        GERDOS_CHECK(executor.start(ExecutionId{900}));
        run_to_completion(executor, ExecutionId{900});

        // Real movement across homes: the ram record and the device record
        // now hold identical bytes.
        GERDOS_CHECK(backend.allocations_equal(staged, resident));
    }

    // ---------------------------------------------------------------------
    // 2. An accelerator mechanism runs the real OpenCL engine
    // ---------------------------------------------------------------------

    {
        const auto* operation =
            machine.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(
            binding->resources[0].resource ==
            (ResourceRef{DeviceId{200}, ResourceId{202}}));

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{901}, OperationId{801}});

        GERDOS_CHECK(execution->bind(*binding));
        GERDOS_CHECK(executor.start(ExecutionId{901}));
        run_to_completion(executor, ExecutionId{901});

        // The real iGPU kernel computed dst = dst * 0.5 + src * 1.5 over
        // one pass: 1.0 * 0.5 + 1.0 * 1.5 = 2.0, exactly.
        GERDOS_CHECK(backend.sample(result, 0) == 2.0f);
    }

    // ---------------------------------------------------------------------
    // 3. When the host engine is unavailable, the accelerator's mechanism
    //    performs the movement
    // ---------------------------------------------------------------------

    {
        machine.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{102})
            ->set_availability(ResourceAvailability::FAILED);

        const auto* operation =
            machine.operations.find_operation(OperationId{800});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(
            binding->resources[0].resource ==
            (ResourceRef{DeviceId{200}, ResourceId{203}}));

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{902}, OperationId{800}});

        GERDOS_CHECK(execution->bind(*binding));
        GERDOS_CHECK(executor.start(ExecutionId{902}));
        run_to_completion(executor, ExecutionId{902});

        GERDOS_CHECK(backend.allocations_equal(staged, resident));
    }

    // ---------------------------------------------------------------------
    // 4. One evidence log holds real durations from both physical engines
    // ---------------------------------------------------------------------

    {
        const auto dma_summary = machine.measurements.summarize(
            ResourceRef{DeviceId{100}, ResourceId{102}},
            MeasurementQuantity::DURATION_NS);

        const auto copy_summary = machine.measurements.summarize(
            ResourceRef{DeviceId{200}, ResourceId{203}},
            MeasurementQuantity::DURATION_NS);

        const auto compute_summary = machine.measurements.summarize(
            ResourceRef{DeviceId{200}, ResourceId{202}},
            MeasurementQuantity::DURATION_NS);

        // Host engine, accelerator transfer engine, accelerator compute
        // engine: all measured in real nanoseconds.
        GERDOS_CHECK(dma_summary.succeeded_observations == 1);
        GERDOS_CHECK(copy_summary.succeeded_observations == 1);
        GERDOS_CHECK(compute_summary.succeeded_observations == 1);
        GERDOS_CHECK(dma_summary.latest_value > 0);
        GERDOS_CHECK(copy_summary.latest_value > 0);
        GERDOS_CHECK(compute_summary.latest_value > 0);
    }

    // ---------------------------------------------------------------------
    // 5. The new work forms execute on the real accelerator engine
    // ---------------------------------------------------------------------

    {
        auto* source = machine.data.create_data(
            DataDescription{DataId{700}, "source"});
        (void)source->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7001},
                    DataId{700},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "source",
                },
            });

        auto* partial = machine.data.create_data(
            DataDescription{DataId{701}, "partial"});
        (void)partial->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7002},
                    DataId{701},
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "partial",
                },
            });

        auto* product = machine.data.create_data(
            DataDescription{DataId{702}, "product"});
        (void)product->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7003},
                    DataId{702},
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "product",
                },
            });

        // Reduction on the GPU engine: sum of six 1.0 elements = 6.
        OperationDescription reduce{
            OperationId{810},
            {DataId{700}},
            {DataId{701}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
        };

        const Operation reduce_operation{reduce};
        Execution reduce_attempt{
            ExecutionDescription{ExecutionId{910}, OperationId{810}}};

        PhysicalBinding reduce_binding;
        reduce_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
            });
        reduce_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
            });
        reduce_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{200}, ResourceId{202}},
            });

        GERDOS_CHECK(reduce_attempt.bind(reduce_binding));
        GERDOS_CHECK(backend.submit(reduce_operation, reduce_attempt));

        std::vector<BackendCompletion> completed;

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
                0) == 6.0f);

        // The matrix form on the GPU engine: the same exact result as the
        // CPU engine — A = [[6,1,1],[1,1,1]], B = ones(3,2).
        OperationDescription matmul{
            OperationId{811},
            {DataId{701}, DataId{700}},
            {DataId{702}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{
                0, 1, 0.0f, 1.0f, 0.0f, WorkForm::MATRIX_PRODUCT, 2, 3, 2},
        };

        const Operation matmul_operation{matmul};
        Execution matmul_attempt{
            ExecutionDescription{ExecutionId{911}, OperationId{811}}};

        PhysicalBinding matmul_binding;
        matmul_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{701}, DataResidencyId{7002}},
            });
        matmul_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{700}, DataResidencyId{7001}},
            });
        matmul_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
            });
        matmul_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{200}, ResourceId{202}},
            });

        GERDOS_CHECK(matmul_attempt.bind(matmul_binding));
        GERDOS_CHECK(backend.submit(matmul_operation, matmul_attempt));
        completed.clear();

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                0) == 8.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                1) == 8.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                2) == 3.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{702}, DataResidencyId{7003}},
                3) == 3.0f);
    }

    // ---------------------------------------------------------------------
    // 6. The new normalization forms execute on the accelerator engine
    // ---------------------------------------------------------------------

    {
        auto* source = machine.data.create_data(
            DataDescription{DataId{710}, "source"});
        (void)source->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7101},
                    DataId{710},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "source",
                },
            });

        auto* peak = machine.data.create_data(
            DataDescription{DataId{711}, "peak"});
        (void)peak->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7102},
                    DataId{711},
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "peak",
                },
            });

        auto* raised = machine.data.create_data(
            DataDescription{DataId{712}, "raised"});
        (void)raised->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{7103},
                    DataId{712},
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "raised",
                },
            });

        // Chain the same discriminating operand as the CPU engine test:
        // the uniform source folds to 6, the maximum of [6,1,1,1,1,1]
        // is 6, and exp(6) differs from exp(1) by two orders of
        // magnitude.
        OperationDescription reduce{
            OperationId{820},
            {DataId{710}},
            {DataId{711}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_SUM},
        };

        const Operation reduce_operation{reduce};
        Execution reduce_attempt{
            ExecutionDescription{ExecutionId{920}, OperationId{820}}};

        PhysicalBinding reduce_binding;
        reduce_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{710}, DataResidencyId{7101}},
            });
        reduce_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{711}, DataResidencyId{7102}},
            });
        reduce_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{200}, ResourceId{202}},
            });

        GERDOS_CHECK(reduce_attempt.bind(reduce_binding));
        GERDOS_CHECK(backend.submit(reduce_operation, reduce_attempt));

        std::vector<BackendCompletion> completed;

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{711}, DataResidencyId{7102}},
                0) == 6.0f);

        OperationDescription maximum{
            OperationId{821},
            {DataId{711}},
            {DataId{711}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 1, 0.0f, 1.0f, 0.0f, WorkForm::REDUCE_MAX},
        };

        const Operation maximum_operation{maximum};
        Execution maximum_attempt{
            ExecutionDescription{ExecutionId{921}, OperationId{821}}};

        PhysicalBinding maximum_binding;
        maximum_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{711}, DataResidencyId{7102}},
            });
        maximum_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{711}, DataResidencyId{7102}},
            });
        maximum_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{200}, ResourceId{202}},
            });

        GERDOS_CHECK(maximum_attempt.bind(maximum_binding));
        GERDOS_CHECK(backend.submit(maximum_operation, maximum_attempt));
        completed.clear();

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);

        const float peak_value = backend.sample(
            DataResidencyRef{DataId{711}, DataResidencyId{7102}}, 0);
        std::printf("gpu reduce_max peak: %f\n", peak_value);
        GERDOS_CHECK(peak_value == 6.0f);
        GERDOS_CHECK(
            backend.sample(
                DataResidencyRef{DataId{711}, DataResidencyId{7102}},
                1) == 1.0f);

        // Six elements wide, like the CPU chain: a narrower work
        // would reallocate the peak record and wipe the operand.
        OperationDescription raised_work{
            OperationId{822},
            {DataId{711}},
            {DataId{712}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{6, 1, 1.0f, 1.0f, 0.0f, WorkForm::EXPONENTIAL},
        };

        const Operation raised_operation{raised_work};
        Execution raised_attempt{
            ExecutionDescription{ExecutionId{922}, OperationId{822}}};

        PhysicalBinding raised_binding;
        raised_binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{DataId{711}, DataResidencyId{7102}},
            });
        raised_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{712}, DataResidencyId{7103}},
            });
        raised_binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{DeviceId{200}, ResourceId{202}},
            });

        GERDOS_CHECK(raised_attempt.bind(raised_binding));
        GERDOS_CHECK(backend.submit(raised_operation, raised_attempt));
        completed.clear();

        while (completed.empty()) {
            backend.poll(completed);
        }

        GERDOS_CHECK(completed.front().succeeded);

        // GPU floating point may differ in the last ulp from the host
        // libm: compare against a 1e-4 relative tolerance instead of
        // exact equality, with the expected values printed first.
        const float raised_zero = backend.sample(
            DataResidencyRef{DataId{712}, DataResidencyId{7103}}, 0);
        const float raised_one = backend.sample(
            DataResidencyRef{DataId{712}, DataResidencyId{7103}}, 1);
        const float expected_zero = 1.0f + std::exp(6.0f);
        const float expected_one = 1.0f + std::exp(1.0f);
        std::printf(
            "gpu exponential raised: %f %f (expected %f %f)\n",
            raised_zero,
            raised_one,
            expected_zero,
            expected_one);
        GERDOS_CHECK(
            std::fabs(raised_zero - expected_zero) <=
            1e-4f * expected_zero);
        GERDOS_CHECK(
            std::fabs(raised_one - expected_one) <= 1e-4f * expected_one);
    }

    return 0;
}