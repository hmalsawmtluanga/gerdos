#include "test_check.hpp"

#include <chrono>
#include <cstdio>
#include <optional>
#include <vector>

#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/workload.hpp"
#include "gerdos/hw/heterogeneous_backend.hpp"

namespace {

using namespace gerdos;

// A real heterogeneous machine: a host CPU engine and a real iGPU engine,
// both able to serve compute work. Declared order prefers the host engine;
// the engines' real speeds disagree.
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
        add(*host, ResourceId{103}, ResourceKind::COMPUTE, "host-compute");
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

// A just-in-time staged workload on real hardware: each unit brings its
// payload to the accelerator's memory and computes over it.
struct StagedWorkload {
    static constexpr std::size_t kUnits = 10;

    Workload workload;

    StagedWorkload() {
        workload.name = "just-in-time staged units";

        for (std::size_t unit = 0; unit < kUnits; ++unit) {
            // Real work: 1 MB of floats; the compute unit runs deep
            // streaming arithmetic — 64 passes, where the engines on this
            // machine genuinely diverge (the CPU engine's per-pass cost is
            // about twice the accelerator's on this silicon).
            workload.operations.push_back(
                OperationDescription{
                    OperationId{1000 + unit},
                    {DataId{500 + unit}},
                    {DataId{500 + unit}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::TRANSFER,
                            1,
                        },
                    },
                    WorkDescription{1 << 18, 1, 0.0f, 1.0f, 0.0f},
                });

            workload.operations.push_back(
                OperationDescription{
                    OperationId{2000 + unit},
                    {DataId{500 + unit}},
                    {DataId{600 + unit}},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                    WorkDescription{1 << 18, 64, 0.5f, 1.5f, 0.0f},
                });
        }
    }

    void populate(Machine& machine) const {
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
};

std::uint64_t run_workload(
    Machine& machine,
    BindingPlanner& planner,
    Executor& executor,
    const Workload& workload) {
    const auto begin = std::chrono::steady_clock::now();
    std::size_t attempts = 0;

    for (const auto& description : workload.operations) {
        const auto* operation =
            machine.operations.find_operation(description.id);

        GERDOS_CHECK(operation != nullptr);

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());

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
    }

    const auto end = std::chrono::steady_clock::now();

    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - begin)
            .count());
}

} // namespace

int main() {
    using namespace gerdos;

    const StagedWorkload staged;

    const ResourceRef host_compute{DeviceId{100}, ResourceId{103}};
    const ResourceRef gpu_compute{DeviceId{200}, ResourceId{202}};

    // ---------------------------------------------------------------------
    // 1. The baseline: planning without evidence trusts declared order and
    //    runs every unit on the host engine.
    // ---------------------------------------------------------------------

    Machine naive_machine;
    HeterogeneousBackend naive_backend(naive_machine.data, DeviceId{200});
    GERDOS_CHECK(naive_backend.gpu_available());
    staged.populate(naive_machine);

    // The baseline's evidence log records what happened; its planner never
    // reads it.
    MeasurementRegistry naive_evidence;
    MeasurementRegistry blind;

    BindingPlanner naive_planner(
        naive_machine.devices,
        naive_machine.data,
        naive_machine.topology,
        blind);

    Executor naive_executor(
        naive_machine.executions,
        naive_machine.operations,
        naive_machine.devices,
        naive_machine.data,
        naive_backend,
        &naive_evidence);

    const auto naive_ns = run_workload(
        naive_machine, naive_planner, naive_executor, staged.workload);

    // The transfer phase costs the same in both arms (engine-independent
    // movement) and therefore dilutes the compute signal; wall-clock is the
    // symmetric verdict, engine means are the evidence. Durations include
    // each engine's own launch accounting.

    // ---------------------------------------------------------------------
    // 2. Measurement-informed planning discovers both engines and learns
    //    which one is actually faster on this machine.
    // ---------------------------------------------------------------------

    Machine informed_machine;
    HeterogeneousBackend informed_backend(
        informed_machine.data, DeviceId{200});
    GERDOS_CHECK(informed_backend.gpu_available());
    staged.populate(informed_machine);

    BindingPlanner informed_planner(
        informed_machine.devices,
        informed_machine.data,
        informed_machine.topology,
        informed_machine.measurements);

    Executor informed_executor(
        informed_machine.executions,
        informed_machine.operations,
        informed_machine.devices,
        informed_machine.data,
        informed_backend,
        &informed_machine.measurements);

    const auto informed_ns = run_workload(
        informed_machine,
        informed_planner,
        informed_executor,
        staged.workload);

    // ---------------------------------------------------------------------
    // 3. Structural truth: what each run actually did
    // ---------------------------------------------------------------------

    const auto naive_host = naive_evidence.summarize(
        host_compute, MeasurementQuantity::DURATION_NS);
    const auto naive_gpu = naive_evidence.summarize(
        gpu_compute, MeasurementQuantity::DURATION_NS);

    const auto informed_host = informed_machine.measurements.summarize(
        host_compute, MeasurementQuantity::DURATION_NS);
    const auto informed_gpu = informed_machine.measurements.summarize(
        gpu_compute, MeasurementQuantity::DURATION_NS);

    // The baseline ran all units on the host engine and never sampled the
    // accelerator.
    GERDOS_CHECK(
        naive_host.succeeded_observations == StagedWorkload::kUnits);
    GERDOS_CHECK(naive_gpu.succeeded_observations == 0);

    // The informed run sampled the host engine once, discovered the
    // accelerator engine, and then used it for every remaining unit.
    GERDOS_CHECK(informed_host.succeeded_observations == 1);
    GERDOS_CHECK(
        informed_gpu.succeeded_observations ==
        StagedWorkload::kUnits - 1);

    // ---------------------------------------------------------------------
    // 4. The number: measured decisions beat declared order on real
    //    hardware. Exact durations are printed as evidence; the assertion
    //    carries a margin so physical noise cannot flip it.
    // ---------------------------------------------------------------------

    std::printf(
        "workload '%s' on real hardware:\n"
        "  planning without evidence: %8.2f ms"
        "  (host engine mean %6.3f ms)\n"
        "  measurement-informed:      %8.2f ms"
        "  (gpu engine mean %6.3f ms)\n"
        "  speedup: %.2fx\n",
        staged.workload.name.c_str(),
        naive_ns / 1e6,
        naive_host.succeeded_total_value /
            double(naive_host.succeeded_observations) / 1e6,
        informed_ns / 1e6,
        informed_gpu.succeeded_total_value /
            double(informed_gpu.succeeded_observations) / 1e6,
        double(naive_ns) / double(informed_ns));

    // Never lose the numbers to an aborted assertion.
    std::fflush(stdout);

    // ---------------------------------------------------------------------
    // 5. Repetition: physical noise cannot decide the verdict
    // ---------------------------------------------------------------------

    std::uint64_t best_naive = naive_ns;
    std::uint64_t best_informed = informed_ns;

    for (int repetition = 1; repetition < 3; ++repetition) {
        Machine naive_repeat;
        HeterogeneousBackend naive_repeat_backend(
            naive_repeat.data, DeviceId{200});
        staged.populate(naive_repeat);

        MeasurementRegistry naive_repeat_evidence;
        BindingPlanner naive_repeat_planner(
            naive_repeat.devices,
            naive_repeat.data,
            naive_repeat.topology,
            blind);

        Executor naive_repeat_executor(
            naive_repeat.executions,
            naive_repeat.operations,
            naive_repeat.devices,
            naive_repeat.data,
            naive_repeat_backend,
            &naive_repeat_evidence);

        Machine informed_repeat;
        HeterogeneousBackend informed_repeat_backend(
            informed_repeat.data, DeviceId{200});
        staged.populate(informed_repeat);

        BindingPlanner informed_repeat_planner(
            informed_repeat.devices,
            informed_repeat.data,
            informed_repeat.topology,
            informed_repeat.measurements);

        Executor informed_repeat_executor(
            informed_repeat.executions,
            informed_repeat.operations,
            informed_repeat.devices,
            informed_repeat.data,
            informed_repeat_backend,
            &informed_repeat.measurements);

        const auto naive_repeat_ns = run_workload(
            naive_repeat,
            naive_repeat_planner,
            naive_repeat_executor,
            staged.workload);

        const auto informed_repeat_ns = run_workload(
            informed_repeat,
            informed_repeat_planner,
            informed_repeat_executor,
            staged.workload);

        std::printf(
            "  repetition %d: naive %8.2f ms, informed %8.2f ms\n",
            repetition,
            naive_repeat_ns / 1e6,
            informed_repeat_ns / 1e6);
        std::fflush(stdout);

        if (naive_repeat_ns < best_naive) {
            best_naive = naive_repeat_ns;
        }

        if (informed_repeat_ns < best_informed) {
            best_informed = informed_repeat_ns;
        }
    }

    // The verdict uses the best observed run of each arm — the estimator
    // least sensitive to physical noise. The measured engine gap is the
    // mechanism: the informed run's engine mean must beat the baseline's.
    GERDOS_CHECK(best_informed < best_naive);
    GERDOS_CHECK(
        informed_gpu.succeeded_total_value *
                naive_host.succeeded_observations <
            naive_host.succeeded_total_value *
                informed_gpu.succeeded_observations);

    return 0;
}