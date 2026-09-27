#include "test_check.hpp"

#include <cstdio>
#include <vector>

#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/workload.hpp"
#include "gerdos/sim/simulated_backend.hpp"

namespace {

using namespace gerdos;

// A synthetic heterogeneous machine with one misleading detail: the
// declared attributes look decisive, but the engines have their own proven
// speeds. The dma is slow; the copy-engine is fast; the compute engine is
// in between.
struct Machine {
    DeviceRegistry devices;
    DataRegistry data;
    OperationRegistry operations;
    ExecutionRegistry executions;
    Topology topology;
    MeasurementRegistry measurements;
    SimulatedBackend backend{1};

    Machine() {
        auto* host = devices.create_device(
            DeviceDescription{DeviceId{100}, "host"});

        auto* accelerator = devices.create_device(
            DeviceDescription{DeviceId{200}, "accelerator"});

        add(*host, ResourceId{101}, ResourceKind::STORAGE, "nvme");
        add(*host, ResourceId{102}, ResourceKind::MEMORY, "ram");
        add(*host, ResourceId{103}, ResourceKind::TRANSFER, "dma");
        add(*accelerator, ResourceId{201}, ResourceKind::MEMORY,
            "device-memory");
        add(*accelerator, ResourceId{202}, ResourceKind::COMPUTE, "compute");
        add(*accelerator, ResourceId{203}, ResourceKind::TRANSFER,
            "copy-engine");

        // One promising-looking link: nvme <-> device-memory.
        (void)topology.add_link(
            TopologyLink{
                TopologyLinkDescription{
                    TopologyLinkId{1},
                    TopologyEndpoint::resource_endpoint(
                        DeviceId{100}, ResourceId{101}),
                    TopologyEndpoint::resource_endpoint(
                        DeviceId{200}, ResourceId{201}),
                    TopologyLinkDirection::BIDIRECTIONAL,
                    TopologyLinkAttributes{
                        48'000'000'000,
                        500,
                    },
                },
            });

        // Proven speeds, in virtual steps per attempt.
        backend.set_mechanism_steps(
            ResourceRef{DeviceId{100}, ResourceId{103}}, 60);
        backend.set_mechanism_steps(
            ResourceRef{DeviceId{200}, ResourceId{203}}, 10);
        backend.set_mechanism_steps(
            ResourceRef{DeviceId{200}, ResourceId{202}}, 20);
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

// A just-in-time staged workload: each unit of work brings its payload to
// the accelerator and then computes over it. Units are generic: no model
// vocabulary appears anywhere in the shape.
struct StagedWorkload {
    Workload workload;

    explicit StagedWorkload(std::size_t units) {
        workload.name = "just-in-time staged units";

        for (std::size_t unit = 0; unit < units; ++unit) {
            const DataId payload{500 + unit};
            const DataId result{600 + unit};
            const std::uint64_t base = 5000 + unit * 10;

            staging_records.push_back({payload, base, unit});
            result_records.push_back({result, base + 2, unit});

            workload.operations.push_back(
                OperationDescription{
                    OperationId{1000 + unit},
                    {payload},
                    {payload},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::TRANSFER,
                            1,
                        },
                    },
                });

            workload.operations.push_back(
                OperationDescription{
                    OperationId{2000 + unit},
                    {payload},
                    {result},
                    {},
                    {
                        ResourceRequirement{
                            ResourceBindingRole::COMPUTE,
                            1,
                        },
                    },
                });
        }
    }

    struct RecordPlan {
        DataId data;
        std::uint64_t base;
        std::size_t unit;
    };

    std::vector<RecordPlan> staging_records;
    std::vector<RecordPlan> result_records;

    // Populates the registries for the workload: each payload has a usable
    // disk record and a fresh device record; each result has a fresh device
    // record.
    void populate(Machine& machine) const {
        for (const auto& plan : staging_records) {
            auto* datum = machine.data.create_data(
                DataDescription{plan.data, "payload"});

            (void)datum->add_residency(
                DataResidency{
                    DataResidencyDescription{
                        DataResidencyId{plan.base},
                        plan.data,
                        ResourceRef{DeviceId{100}, ResourceId{101}},
                        "disk",
                    },
                });

            (void)datum->add_residency(
                DataResidency{
                    DataResidencyDescription{
                        DataResidencyId{plan.base + 1},
                        plan.data,
                        ResourceRef{DeviceId{200}, ResourceId{201}},
                        "device",
                    },
                });

            (void)datum->find_residency(DataResidencyId{plan.base})
                ->set_state(DataResidencyState::VALID);
        }

        for (const auto& plan : result_records) {
            auto* datum = machine.data.create_data(
                DataDescription{plan.data, "result"});

            (void)datum->add_residency(
                DataResidency{
                    DataResidencyDescription{
                        DataResidencyId{plan.base},
                        plan.data,
                        ResourceRef{DeviceId{200}, ResourceId{201}},
                        "device",
                    },
                });
        }

        for (const auto& operation : workload.operations) {
            (void)machine.operations.create_operation(operation);
        }
    }
};

// Runs the workload sequentially: plan each operation, attempt it, and wait
// for its completion. Returns the realized virtual steps.
std::size_t run(
    Machine& machine,
    BindingPlanner& planner,
    Executor& executor,
    const Workload& workload) {
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

        GERDOS_CHECK(execution != nullptr);
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

    return machine.backend.elapsed_steps();
}

} // namespace

int main() {
    using namespace gerdos;

    const StagedWorkload staged{8};

    // ---------------------------------------------------------------------
    // 1. The naive baseline: planning without evidence trusts the declared
    //    order and the slow engine does every staging move.
    // ---------------------------------------------------------------------

    Machine naive_machine;
    staged.populate(naive_machine);

    MeasurementRegistry naive_evidence;
    BindingPlanner naive_planner(
        naive_machine.devices,
        naive_machine.data,
        naive_machine.topology,
        naive_evidence);

    Executor naive_executor(
        naive_machine.executions,
        naive_machine.operations,
        naive_machine.devices,
        naive_machine.data,
        naive_machine.backend);

    const auto naive_steps =
        run(naive_machine, naive_planner, naive_executor, staged.workload);

    // 8 staging moves on the 60-step engine + 8 compute attempts at 20.
    GERDOS_CHECK(naive_steps == 8 * 60 + 8 * 20);

    // ---------------------------------------------------------------------
    // 2. Measurement-informed planning: the engines prove their speeds and
    //    the workload gets faster.
    // ---------------------------------------------------------------------

    Machine informed_machine;
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
        informed_machine.backend,
        &informed_machine.measurements);

    const auto informed_steps = run(
        informed_machine,
        informed_planner,
        informed_executor,
        staged.workload);

    // The first staging move discovers the slow engine; the second samples
    // the unmeasured copy-engine; the remaining six run on measured truth.
    // 60 + 7 * 10 staging + 8 * 20 compute.
    GERDOS_CHECK(informed_steps == 60 + 7 * 10 + 8 * 20);

    // The number: measurement-informed planning beats planning without
    // evidence on the same machine and the same workload.
    GERDOS_CHECK(informed_steps < naive_steps);

    // The evidence trail explains the difference.
    const auto dma_summary = informed_machine.measurements.summarize(
        ResourceRef{DeviceId{100}, ResourceId{103}},
        MeasurementQuantity::DURATION_NS);

    const auto copy_summary = informed_machine.measurements.summarize(
        ResourceRef{DeviceId{200}, ResourceId{203}},
        MeasurementQuantity::DURATION_NS);

    GERDOS_CHECK(dma_summary.succeeded_observations == 1);
    GERDOS_CHECK(copy_summary.succeeded_observations == 7);

    std::printf(
        "workload '%s': naive %zu steps, measurement-informed %zu steps\n",
        staged.workload.name.c_str(),
        naive_steps,
        informed_steps);

    return 0;
}