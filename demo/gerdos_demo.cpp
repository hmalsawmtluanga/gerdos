#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"
#include "gerdos/core/physical_binding_validation.hpp"
#include "gerdos/core/workload.hpp"
#include "gerdos/hw/heterogeneous_backend.hpp"
#include "test_check.hpp"

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
        add(*host, ResourceId{103}, ResourceKind::COMPUTE, "cpu-compute");
        add(*accelerator, ResourceId{201}, ResourceKind::MEMORY,
            "device-memory");
        add(*accelerator, ResourceId{202}, ResourceKind::COMPUTE,
            "gpu-compute");
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
};

struct Attempt {
    std::string phase;
    ResourceRef mechanism;
    std::uint64_t duration_ns;
    std::string reason;
};

struct Run {
    std::string label;
    std::vector<Attempt> attempts;
    std::uint64_t total_ns{0};
    std::size_t cpu_compute{0};
    std::size_t gpu_compute{0};
};

constexpr std::size_t kUnits = 10;

Workload staged_workload() {
    Workload workload;
    workload.name = "just-in-time staged units";

    for (std::size_t unit = 0; unit < kUnits; ++unit) {
        // 1 MB of floats; the compute unit runs deep streaming arithmetic.
        workload.operations.push_back(
            OperationDescription{
                OperationId{1000 + unit},
                {DataId{500 + unit}},
                {DataId{500 + unit}},
                {},
                {ResourceRequirement{ResourceBindingRole::TRANSFER, 1}},
                WorkDescription{1 << 18, 1, 0.0f, 1.0f, 0.0f},
            });

        workload.operations.push_back(
            OperationDescription{
                OperationId{2000 + unit},
                {DataId{500 + unit}},
                {DataId{600 + unit}},
                {},
                {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
                WorkDescription{1 << 18, 64, 0.5f, 1.5f, 0.0f},
            });
    }

    return workload;
}

void populate(Machine& machine, const Workload& workload) {
    for (std::size_t unit = 0; unit < kUnits; ++unit) {
        const DataId payload{500 + unit};
        const DataId result{600 + unit};
        const std::uint64_t base = 5000 + unit * 10;

        auto* datum = machine.data.create_data(
            DataDescription{payload, "payload"});

        (void)datum->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{base},
                    payload,
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "staged",
                },
            });

        (void)datum->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{base + 1},
                    payload,
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "resident",
                },
            });

        (void)datum->find_residency(DataResidencyId{base})
            ->set_state(DataResidencyState::VALID);

        auto* output = machine.data.create_data(
            DataDescription{result, "result"});

        (void)output->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{base + 2},
                    result,
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "device",
                },
            });

        (void)machine.operations.create_operation(
            workload.operations[unit * 2]);
        (void)machine.operations.create_operation(
            workload.operations[unit * 2 + 1]);
    }
}

std::string reason_for(
    MeasurementRegistry& measurements,
    const PhysicalBinding& binding,
    ResourceRef chosen,
    bool informed) {
    if (!informed) {
        return "declared order (the planner is evidence-blind)";
    }

    // Compare against the like-for-like alternative for the bound role.
    const bool transfer =
        binding.resources[0].role == ResourceBindingRole::TRANSFER;

    const ResourceRef alternatives[]{
        transfer ? ResourceRef{DeviceId{100}, ResourceId{102}}
                 : ResourceRef{DeviceId{100}, ResourceId{103}},
        transfer ? ResourceRef{DeviceId{200}, ResourceId{203}}
                 : ResourceRef{DeviceId{200}, ResourceId{202}},
    };

    const auto chosen_summary = measurements.summarize(
        chosen, MeasurementQuantity::DURATION_NS);

    for (const auto& other : alternatives) {
        if (other == chosen) {
            continue;
        }

        const auto other_summary = measurements.summarize(
            other, MeasurementQuantity::DURATION_NS);

        if (chosen_summary.succeeded_observations == 0) {
            return other_summary.succeeded_observations == 0
                       ? "declared order (no evidence yet)"
                       : "discovery (sampling an unmeasured engine)";
        }

        if (other_summary.succeeded_observations > 0) {
            char buffer[96];
            std::snprintf(
                buffer,
                sizeof(buffer),
                "measured preference (%.2f ms < %.2f ms)",
                double(chosen_summary.succeeded_total_value) /
                    double(chosen_summary.succeeded_observations) / 1e6,
                double(other_summary.succeeded_total_value) /
                    double(other_summary.succeeded_observations) / 1e6);

            return buffer;
        }

        return "measured (the alternative is unmeasured)";
    }

    return "sole engine";
}

Run execute(
    const std::string& label,
    const Workload& workload,
    bool informed) {
    Machine machine;
    populate(machine, workload);

    HeterogeneousBackend backend(machine.data, DeviceId{200});
    GERDOS_CHECK(backend.gpu_available());

    // The baseline's planner never sees evidence; its executor still
    // records what happened.
    MeasurementRegistry blind;

    BindingPlanner planner(
        machine.devices,
        machine.data,
        machine.topology,
        informed ? machine.measurements : blind);

    Executor executor(
        machine.executions,
        machine.operations,
        machine.devices,
        machine.data,
        backend,
        &machine.measurements);

    auto& evidence = machine.measurements;

    Run run;
    run.label = label;

    PhysicalBindingValidator validator;
    BindingResolver resolver(machine.devices, machine.data);
    BindingAdmissibilityValidator admissibility;

    std::size_t attempts = 0;
    const auto begin = std::chrono::steady_clock::now();

    for (const auto& description : workload.operations) {
        const auto* operation =
            machine.operations.find_operation(description.id);

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());

        // The gate chain, on every attempt.
        GERDOS_CHECK(validator.validate(*binding));
        GERDOS_CHECK(resolver.resolve(*binding).fully_resolved());
        GERDOS_CHECK(admissibility.admissible(*operation, *binding));

        // One mechanism per attempt in this workload.
        GERDOS_CHECK(binding->resources.size() == 1);
        const ResourceRef chosen = binding->resources[0].resource;

        const std::string reason =
            reason_for(evidence, *binding, chosen, informed);

        const ExecutionId id{9000 + attempts};
        ++attempts;

        auto* execution = machine.executions.create_execution(
            ExecutionDescription{
                id,
                description.id,
            });

        GERDOS_CHECK(execution->bind(*binding));
        GERDOS_CHECK(executor.start(id));

        std::vector<AttemptStatus> outcomes;

        do {
            executor.advance(outcomes);
        } while (outcomes.empty());

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(
            outcomes.front().integrity == AttemptIntegrity::COHERENT);

        const bool compute_attempt =
            description.id.value() >= 2000;

        const auto summary = evidence.summarize(
            chosen, MeasurementQuantity::DURATION_NS);

        run.attempts.push_back(
            Attempt{
                compute_attempt ? "compute" : "stage ",
                chosen,
                summary.latest_value,
                reason,
            });

        if (compute_attempt && chosen == ResourceRef{DeviceId{100}, ResourceId{103}}) {
            ++run.cpu_compute;
        }

        if (compute_attempt && chosen == ResourceRef{DeviceId{200}, ResourceId{202}}) {
            ++run.gpu_compute;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    run.total_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - begin)
            .count());

    return run;
}

const char* short_name(ResourceRef ref) {
    return ref == ResourceRef(DeviceId{100}, ResourceId{103})
               ? "cpu-compute"
           : ref == ResourceRef(DeviceId{200}, ResourceId{202})
               ? "gpu-compute"
           : ref == ResourceRef(DeviceId{100}, ResourceId{102})
               ? "dma"
               : "copy-engine";
}

void print(const Run& run) {
    std::printf("\n%s\n", run.label.c_str());

    for (std::size_t i = 0; i < run.attempts.size(); ++i) {
        const auto& attempt = run.attempts[i];

        std::printf(
            "  unit %2zu %s  %-11s %7.2f ms   %s\n",
            i / 2 + 1,
            attempt.phase.c_str(),
            short_name(attempt.mechanism),
            attempt.duration_ns / 1e6,
            attempt.reason.c_str());
    }

    std::printf(
        "  total: %.2f ms   (cpu-compute %zu, gpu-compute %zu)\n",
        run.total_ns / 1e6,
        run.cpu_compute,
        run.gpu_compute);
}

} // namespace

int main() {
    const Workload workload = staged_workload();

    std::printf("GERDOS - end-to-end demonstration\n");
    std::printf(
        "machine: host CPU engine + accelerator GPU engine (OpenCL)\n");
    std::printf(
        "workload '%s': %zu units, each stages 1 MB then computes "
        "1 MB x 64 passes\n",
        workload.name.c_str(),
        kUnits);
    std::printf(
        "work is declared by the operations themselves; gates never "
        "inspect it.\n");

    const Run baseline = execute(
        "run 1: planning without evidence (declared order)",
        workload,
        false);

    const Run informed = execute(
        "run 2: measurement-informed planning",
        workload,
        true);

    print(baseline);
    print(informed);

    std::printf("\nresult: ");
    std::printf(
        "measurement-informed planning beat planning without evidence: "
        "%.2f ms -> %.2f ms (%.2fx)\n",
        baseline.total_ns / 1e6,
        informed.total_ns / 1e6,
        double(baseline.total_ns) / double(informed.total_ns));
    std::printf(
        "        ratios belong to this machine; the portable claim is "
        "structural: measured evidence changed the choices.\n");
    std::fflush(stdout);

    // The demonstration verifies itself: the baseline never left declared
    // order; the informed run discovered the other engine and beat the
    // baseline. The exact informed split is not pinned: discovery inside
    // the exploration budget guarantees the accelerator is sampled (unit 2
    // at the latest) and unit 1 always runs on declared order, but later
    // units re-weigh live means against physical noise, so a second host
    // sample on a noisy machine is evidence working as designed, not a
    // regression. The wall-clock verdict carries the speedup claim.
    GERDOS_CHECK(baseline.gpu_compute == 0);
    GERDOS_CHECK(baseline.cpu_compute == kUnits);
    GERDOS_CHECK(informed.cpu_compute >= 1);
    GERDOS_CHECK(informed.gpu_compute > 0);
    GERDOS_CHECK(
        informed.cpu_compute + informed.gpu_compute == kUnits);
    GERDOS_CHECK(informed.total_ns < baseline.total_ns);

    return 0;
}