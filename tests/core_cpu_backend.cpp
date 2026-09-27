#include "test_check.hpp"

#include <optional>
#include <vector>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/cpu/cpu_backend.hpp"

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

        record(*payload, DataResidencyId{5001}, ResourceRef{DeviceId{100}, ResourceId{101}}, "staged");
        record(*payload, DataResidencyId{5002}, ResourceRef{DeviceId{200}, ResourceId{201}}, "resident");
        usable(payload, DataResidencyId{5001});

        // Result: fresh on the device.
        auto* result = data.create_data(
            DataDescription{DataId{501}, "result"});

        record(*result, DataResidencyId{5101}, ResourceRef{DeviceId{200}, ResourceId{201}}, "device");

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
                WorkDescription{1 << 14, 4, 0.5f, 1.5f, 0.0f},
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

} // namespace

int main() {
    using namespace gerdos;

    Machine machine;
    CpuBackend backend;

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

    // ---------------------------------------------------------------------
    // 1. Real work moves between real representations
    // ---------------------------------------------------------------------

    {
        const auto* operation =
            machine.operations.find_operation(OperationId{800});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{900}, OperationId{800}});

        GERDOS_CHECK(execution->bind(*binding));
        GERDOS_CHECK(executor.start(ExecutionId{900}));

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        while (outcomes.empty()) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(
            outcomes.front().integrity == AttemptIntegrity::COHERENT);

        // The movement kernel copied real bytes between representations.
        GERDOS_CHECK(
            backend.allocations_equal(
                DataResidencyRef{DataId{500}, DataResidencyId{5001}},
                DataResidencyRef{DataId{500}, DataResidencyId{5002}}));
    }

    // ---------------------------------------------------------------------
    // 2. Real durations flow through the seam into evidence
    // ---------------------------------------------------------------------

    {
        const auto summary = machine.measurements.summarize(
            ResourceRef{DeviceId{100}, ResourceId{102}},
            MeasurementQuantity::DURATION_NS);

        GERDOS_CHECK(summary.observations >= 1);
        GERDOS_CHECK(summary.succeeded_observations >= 1);
        GERDOS_CHECK(summary.latest_value > 0);

        machine.measurements.for_each(
            ResourceRef{DeviceId{100}, ResourceId{102}},
            MeasurementQuantity::DURATION_NS,
            [](const MeasurementRecord& record) {
                // Physical time, not virtual steps.
                GERDOS_CHECK(record.observation.value > 0);
            });
    }

    // ---------------------------------------------------------------------
    // 3. Claim arbitration serializes work on one representation
    // ---------------------------------------------------------------------

    {
        auto* first = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{901}, OperationId{801}});

        auto* second = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{902}, OperationId{801}});

        const auto* operation =
            machine.operations.find_operation(OperationId{801});

        const auto first_binding = planner.plan(*operation);
        const auto second_binding = planner.plan(*operation);
        GERDOS_CHECK(first_binding.has_value());
        GERDOS_CHECK(second_binding.has_value());

        GERDOS_CHECK(first->bind(*first_binding));
        GERDOS_CHECK(second->bind(*second_binding));

        GERDOS_CHECK(executor.start(ExecutionId{901}));

        // The second attempt binds a producing record the first is
        // updating: planning refuses it until the first finishes, so the
        // two compute attempts run back to back rather than racing on one
        // representation.
        const auto refused = executor.start(ExecutionId{902});
        GERDOS_CHECK(!refused);
        GERDOS_CHECK(second->state() == ExecutionState::PENDING);

        std::vector<AttemptStatus> outcomes;
        executor.advance(outcomes);

        while (outcomes.empty()) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(executor.start(ExecutionId{902}));

        outcomes.clear();
        executor.advance(outcomes);

        while (outcomes.empty()) {
            executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(first->state() == ExecutionState::COMPLETED);
        GERDOS_CHECK(second->state() == ExecutionState::COMPLETED);
    }

    // ---------------------------------------------------------------------
    // 4. Failure is real and reported
    // ---------------------------------------------------------------------

    {
        CpuBackend failing_backend;
        failing_backend.set_failure(ExecutionId{903});

        Executor failing_executor(
            machine.executions,
            machine.operations,
            machine.devices,
            machine.data,
            failing_backend,
            &machine.measurements);

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{ExecutionId{903}, OperationId{801}});

        const auto* operation =
            machine.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(execution->bind(*binding));
        GERDOS_CHECK(failing_executor.start(ExecutionId{903}));

        std::vector<AttemptStatus> outcomes;
        failing_executor.advance(outcomes);

        while (outcomes.empty()) {
            failing_executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(execution->state() == ExecutionState::FAILED);
        GERDOS_CHECK(
            execution->result()->outcome == ExecutionState::FAILED);

        // A backend failure is not an incoherence: the attempt's effects
        // applied and its completion is fully reported.
        GERDOS_CHECK(
            execution->result()->integrity ==
            AttemptIntegrity::COHERENT);
    }

    // ---------------------------------------------------------------------
    // 5. The data plane stays behind the seam
    // ---------------------------------------------------------------------

    {
        // The core's registries still hold identities only; real buffers
        // live inside the backend. The device-homed record's allocation
        // reflects the last writer: §1's movement sized it at 2^18, then
        // §3's compute attempt — whose planner prefers the engine-local
        // device copy (§2d locality) — resized it to op 801's storage.
        // The size is derived from the operation, not hardcoded.
        const auto* compute =
            machine.operations.find_operation(OperationId{801});
        GERDOS_CHECK(compute != nullptr);
        GERDOS_CHECK(backend.allocation_count() > 0);
        GERDOS_CHECK(
            backend.allocation_bytes(
                DataResidencyRef{DataId{500}, DataResidencyId{5002}}) ==
            compute->description().work.storage_elements());
    }

    return 0;
}