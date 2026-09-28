#include "test_check.hpp"

#include <optional>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/sim/simulated_backend.hpp"
#include "gerdos/core/execution_effects.hpp"
#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"
#include "gerdos/core/physical_binding_validation.hpp"

namespace {

using namespace gerdos;

// A synthetic machine: host (device 100) with ram 101, scratch 103 and
// dma 102; accelerator (device 200) with device-memory 201, compute 202 and
// copy-engine 203. A fast link covers ram<->device-memory; a slow link
// covers ram<->scratch.
struct Machine {
    DeviceRegistry devices;
    DataRegistry data;
    OperationRegistry operations;
    Topology topology;
    MeasurementRegistry measurements;

    Machine() {
        auto* host = devices.create_device(
            DeviceDescription{DeviceId{100}, "host"});

        auto* accelerator = devices.create_device(
            DeviceDescription{DeviceId{200}, "accelerator"});

        add(*host, ResourceId{101}, ResourceKind::MEMORY, "ram");
        add(*host, ResourceId{102}, ResourceKind::TRANSFER, "dma");
        add(*host, ResourceId{103}, ResourceKind::MEMORY, "scratch");
        add(*accelerator, ResourceId{201}, ResourceKind::MEMORY,
            "device-memory");
        add(*accelerator, ResourceId{202}, ResourceKind::COMPUTE, "compute");
        add(*accelerator, ResourceId{203}, ResourceKind::TRANSFER,
            "copy-engine");

        // Weights: usable on ram, with a low-id copy on the slow path and a
        // high-id copy on the fast path. Link quality must beat id order.
        auto* weights = data.create_data(
            DataDescription{DataId{500}, "weights"});

        record(*weights, DataResidencyId{5001}, host_ram(), "staged");
        record(*weights, DataResidencyId{5002}, scratch(), "host-copy");
        record(*weights, DataResidencyId{5003}, device_memory(), "resident");

        usable(weights, DataResidencyId{5001});

        // Activation on the device; output records on both sides.
        auto* activation = data.create_data(
            DataDescription{DataId{501}, "activation"});

        record(*activation, DataResidencyId{5101}, device_memory(), "device");
        usable(activation, DataResidencyId{5101});

        auto* output = data.create_data(
            DataDescription{DataId{502}, "output"});

        record(*output, DataResidencyId{5201}, device_memory(), "device");
        record(*output, DataResidencyId{5202}, host_ram(), "host-copy");

        // The fast link: ram <-> device-memory.
        add_link(
            TopologyLinkId{1},
            host_ram(),
            device_memory(),
            TopologyLinkAttributes{48'000'000'000, 500});

        // The slow link: ram <-> scratch.
        add_link(
            TopologyLinkId{2},
            host_ram(),
            scratch(),
            TopologyLinkAttributes{1'000'000'000, 9'000});

        // Scratch state with a sole representation: in+out realizes in
        // place.
        auto* state = data.create_data(
            DataDescription{DataId{503}, "state"});

        record(*state, DataResidencyId{5301}, host_ram(), "state");
        usable(state, DataResidencyId{5301});

        (void)operations.create_operation(
            OperationDescription{
                OperationId{802},
                {DataId{503}},
                {DataId{503}},
                {},
                {
                    ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
                },
            });

        // Compute realization: activation -> output.
        (void)operations.create_operation(
            OperationDescription{
                OperationId{800},
                {DataId{501}},
                {DataId{502}},
                {},
                {
                    ResourceRequirement{ResourceBindingRole::COMPUTE, 1},
                },
            });

        // Movement realization: weights in -> weights out.
        (void)operations.create_operation(
            OperationDescription{
                OperationId{801},
                {DataId{500}},
                {DataId{500}},
                {},
                {
                    ResourceRequirement{ResourceBindingRole::TRANSFER, 1},
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

    void add_link(
        TopologyLinkId id,
        ResourceRef from,
        ResourceRef to,
        TopologyLinkAttributes attributes) {
        (void)topology.add_link(
            TopologyLink{
                TopologyLinkDescription{
                    id,
                    TopologyEndpoint::resource_endpoint(
                        from.device, from.resource),
                    TopologyEndpoint::resource_endpoint(
                        to.device, to.resource),
                    TopologyLinkDirection::BIDIRECTIONAL,
                    attributes,
                },
            });
    }

    static ResourceRef host_ram() {
        return ResourceRef{DeviceId{100}, ResourceId{101}};
    }

    static ResourceRef scratch() {
        return ResourceRef{DeviceId{100}, ResourceId{103}};
    }

    static ResourceRef device_memory() {
        return ResourceRef{DeviceId{200}, ResourceId{201}};
    }
};

} // namespace

int main() {
    using namespace gerdos;

    Machine machine;
    BindingPlanner planner(
        machine.devices,
        machine.data,
        machine.topology,
        machine.measurements);
    PhysicalBindingValidator validator;
    BindingAdmissibilityValidator admissibility;
    BindingResolver resolver(machine.devices, machine.data);

    // ---------------------------------------------------------------------
    // 1. A compute realization is chosen and passes every gate
    // ---------------------------------------------------------------------

    {
        const auto* operation =
            machine.operations.find_operation(OperationId{800});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(validator.validate(*binding));
        GERDOS_CHECK(admissibility.admissible(*operation, *binding));
        GERDOS_CHECK(resolver.resolve(*binding).fully_resolved());

        GERDOS_CHECK(binding->data.size() == 2);
        GERDOS_CHECK(binding->data[0].role == DataBindingRole::INPUT);
        GERDOS_CHECK(
            binding->data[0].residency ==
            (DataResidencyRef{DataId{501}, DataResidencyId{5101}}));

        GERDOS_CHECK(binding->data[1].role == DataBindingRole::OUTPUT);
        GERDOS_CHECK(
            binding->data[1].residency ==
            (DataResidencyRef{DataId{502}, DataResidencyId{5201}}));

        GERDOS_CHECK(binding->resources.size() == 1);
        GERDOS_CHECK(
            binding->resources[0].resource ==
            (ResourceRef{DeviceId{200}, ResourceId{202}}));
    }

    // ---------------------------------------------------------------------
    // 2. A movement realization prefers the record on the best link
    // ---------------------------------------------------------------------

    {
        const auto* operation =
            machine.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(validator.validate(*binding));
        GERDOS_CHECK(admissibility.admissible(*operation, *binding));

        GERDOS_CHECK(binding->data.size() == 2);
        GERDOS_CHECK(binding->data[0].role == DataBindingRole::SOURCE);
        GERDOS_CHECK(
            binding->data[0].residency ==
            (DataResidencyRef{DataId{500}, DataResidencyId{5001}}));

        // Declared bandwidth beats identifier order: 5003 sits on the
        // 48 GB/s link while the lower-id 5002 sits on the 1 GB/s link.
        GERDOS_CHECK(binding->data[1].role == DataBindingRole::DESTINATION);
        GERDOS_CHECK(
            binding->data[1].residency ==
            (DataResidencyRef{DataId{500}, DataResidencyId{5003}}));

        // The movement mechanism sits on a covering link's endpoint and is
        // kind-matched.
        GERDOS_CHECK(binding->resources.size() == 1);
        GERDOS_CHECK(binding->resources[0].role ==
                      ResourceBindingRole::TRANSFER);
    }

    // ---------------------------------------------------------------------
    // 3. Determinism: the same state yields the same plan
    // ---------------------------------------------------------------------

    {
        const auto* operation =
            machine.operations.find_operation(OperationId{801});

        const auto again = planner.plan(*operation);
        GERDOS_CHECK(again.has_value());
        GERDOS_CHECK(again->data.size() == 2);
        GERDOS_CHECK(
            again->data[1].residency ==
            (DataResidencyRef{DataId{500}, DataResidencyId{5003}}));
        GERDOS_CHECK(
            again->resources[0].resource ==
            (ResourceRef{DeviceId{100}, ResourceId{102}}));
    }

    // ---------------------------------------------------------------------
    // 4. Planning fails closed
    // ---------------------------------------------------------------------

    {
        // Movement with unreachable sibling records is refused loudly:
        // in-place realization is only for sole representations.
        Topology barren_links;
        BindingPlanner barren(
            machine.devices,
            machine.data,
            barren_links,
            machine.measurements);

        const auto* movement =
            machine.operations.find_operation(OperationId{801});

        GERDOS_CHECK(!barren.plan(*movement).has_value());

        // An operation with nothing to realize produces no binding.
        const Operation empty{OperationDescription{
            OperationId{900},
            {},
            {},
            {},
            {},
        }};

        GERDOS_CHECK(!barren.plan(empty).has_value());

        // Unusable consuming records cannot be planned from.
        auto* weights = machine.data.find_data(DataId{500});
        GERDOS_CHECK(weights != nullptr);

        (void)weights->find_residency(DataResidencyId{5001})
            ->set_state(DataResidencyState::STALE);

        GERDOS_CHECK(!barren.plan(*movement).has_value());
        GERDOS_CHECK(!planner.plan(*movement).has_value());
    }

    // ---------------------------------------------------------------------
    // 5. A sole representation realizes the update in place
    // ---------------------------------------------------------------------

    {
        const auto* operation =
            machine.operations.find_operation(OperationId{802});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(validator.validate(*binding));
        GERDOS_CHECK(admissibility.admissible(*operation, *binding));

        GERDOS_CHECK(binding->data.size() == 2);
        GERDOS_CHECK(binding->data[0].role == DataBindingRole::INPUT);
        GERDOS_CHECK(binding->data[1].role == DataBindingRole::OUTPUT);
        GERDOS_CHECK(
            binding->data[0].residency ==
            (DataResidencyRef{DataId{503}, DataResidencyId{5301}}));
        GERDOS_CHECK(
            binding->data[1].residency ==
            (DataResidencyRef{DataId{503}, DataResidencyId{5301}}));
    }

    // ---------------------------------------------------------------------
    // 6. Discovery before exploitation: measured behavior decides
    // ---------------------------------------------------------------------

    const ResourceRef dma{DeviceId{100}, ResourceId{102}};
    const ResourceRef copy_engine{DeviceId{200}, ResourceId{203}};

    {
        Machine measured;
        BindingPlanner planner(
            measured.devices,
            measured.data,
            measured.topology,
            measured.measurements);

        const auto* operation =
            measured.operations.find_operation(OperationId{801});

        // Premise: without evidence, declared rules pick the dma.
        const auto baseline = planner.plan(*operation);
        GERDOS_CHECK(baseline.has_value());
        GERDOS_CHECK(baseline->resources[0].resource == dma);

        // One slow observation of the dma is enough to give the unmeasured
        // copy-engine its discovery turn.
        (void)measured.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{2},
                true,
                60'000'000'000,
                {},
            });

        GERDOS_CHECK(
            planner.plan(*operation)->resources[0].resource ==
            copy_engine);

        // Evidence from any operation contributes to the mechanism's
        // proven behavior; the now-measured copy-engine keeps winning.
        (void)measured.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{800},
                ExecutionId{3},
                true,
                1'000'000'000,
                {},
            });

        const auto informed = planner.plan(*operation);
        GERDOS_CHECK(informed.has_value());
        GERDOS_CHECK(informed->resources[0].resource == copy_engine);
        GERDOS_CHECK(informed->data.size() == 2);
    }

    // ---------------------------------------------------------------------
    // 7. Measured comparison works both ways; failure is not speed
    // ---------------------------------------------------------------------

    {
        Machine first;
        BindingPlanner planner(
            first.devices, first.data, first.topology, first.measurements);

        // The dma has proven faster: declared order loses.
        (void)first.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{4},
                true,
                1'000'000'000,
                {},
            });

        (void)first.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{801},
                ExecutionId{5},
                true,
                60'000'000'000,
                {},
            });

        const auto* operation =
            first.operations.find_operation(OperationId{801});

        GERDOS_CHECK(
            planner.plan(*operation)->resources[0].resource == dma);
    }

    {
        Machine second;
        BindingPlanner planner(
            second.devices,
            second.data,
            second.topology,
            second.measurements);

        // The dma is slightly slower on success but has a fast failure;
        // the failure is not speed evidence, so the faster copy-engine
        // still wins.
        (void)second.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{6},
                true,
                60'000'000'000,
                {},
            });

        (void)second.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{7},
                false,
                1,
                {},
            });

        (void)second.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{801},
                ExecutionId{8},
                true,
                50'000'000'000,
                {},
            });

        const auto* operation =
            second.operations.find_operation(OperationId{801});

        GERDOS_CHECK(
            planner.plan(*operation)->resources[0].resource ==
            copy_engine);
    }

    // ---------------------------------------------------------------------
    // 8. Discovery is per-comparison: no log size starves newcomers
    // ---------------------------------------------------------------------

    {
        Machine budgeted;
        BindingPlanner planner(
            budgeted.devices,
            budgeted.data,
            budgeted.topology,
            budgeted.measurements);

        // Flood the log with unrelated evidence: under the old global
        // budget this would exhaust discovery. The per-comparison rule
        // ignores log size — discovery depends only on the compared
        // pair's own observations.
        for (std::size_t i = 0; i < 32; ++i) {
            (void)budgeted.measurements.record(
                MeasurementObservation{
                    MeasurementQuantity::DURATION_NS,
                    Machine::host_ram(),
                    OperationId{777},
                    ExecutionId{100 + i},
                    true,
                    1,
                    {},
                });
        }

        (void)budgeted.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{200},
                true,
                60'000'000'000,
                {},
            });

        // The unmeasured copy-engine is sampled ahead of measured dma:
        // one bounded re-exploration sample, regardless of log size.
        const auto* operation =
            budgeted.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(binding->resources[0].resource == copy_engine);

        // After its one sample, means decide: a slow copy-engine sample
        // (120 s) loses to dma's 60 s mean. Bounded re-exploration does
        // not starve proven engines — it samples once, then submits to
        // evidence.
        (void)budgeted.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{801},
                ExecutionId{201},
                true,
                120'000'000'000,
                {},
            });

        const auto replan = planner.plan(*operation);
        GERDOS_CHECK(replan.has_value());
        GERDOS_CHECK(replan->resources[0].resource == dma);
    }

    // ---------------------------------------------------------------------
    // 9. Saturated totals are not evidence of speed
    // ---------------------------------------------------------------------

    {
        Machine poisoned;
        BindingPlanner planner(
            poisoned.devices,
            poisoned.data,
            poisoned.topology,
            poisoned.measurements);

        const auto maximum = std::numeric_limits<std::uint64_t>::max();

        (void)poisoned.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{300},
                true,
                maximum,
                {},
            });

        (void)poisoned.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{301},
                true,
                maximum,
                {},
            });

        (void)poisoned.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{801},
                ExecutionId{302},
                true,
                50'000'000'000,
                {},
            });

        // The saturated dma falls back to the unmeasured class and gets its
        // discovery turn instead of ranking on a meaningless mean.
        const auto* operation =
            poisoned.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(binding->resources[0].resource == dma);
    }

    // ---------------------------------------------------------------------
    // 10. Means pool work shapes — the documented limitation
    // ---------------------------------------------------------------------

    {
        Machine mixed;
        BindingPlanner planner(
            mixed.devices,
            mixed.data,
            mixed.topology,
            mixed.measurements);

        // The copy-engine is fast at the relevant shape but carries one
        // huge outlier from unrelated work; its pooled mean looks slow.
        (void)mixed.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{600},
                ExecutionId{400},
                true,
                500'000'000'000,
                {},
            });

        (void)mixed.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{601},
                ExecutionId{401},
                true,
                1'000'000'000,
                {},
            });

        (void)mixed.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{402},
                true,
                60'000'000'000,
                {},
            });

        const auto* operation =
            mixed.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());

        // Pooled means rank the uniform dma first; work-shape-scoped
        // comparison is the named future refinement.
        GERDOS_CHECK(binding->resources[0].resource == dma);
    }

    // ---------------------------------------------------------------------
    // 11. Adversarial planning edges
    // ---------------------------------------------------------------------

    {
        Machine edges;
        BindingPlanner planner(
            edges.devices,
            edges.data,
            edges.topology,
            edges.measurements);

        // A garbage requirement is refused.
        const Operation garbage{OperationDescription{
            OperationId{901},
            {},
            {},
            {},
            {
                ResourceRequirement{
                    ResourceBindingRole::COMPUTE,
                    0,
                },
            },
        }};

        GERDOS_CHECK(!planner.plan(garbage).has_value());

        // A failed mechanism is not chosen.
        edges.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{102})
            ->set_availability(ResourceAvailability::FAILED);

        const auto* movement =
            edges.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*movement);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(binding->resources[0].resource == copy_engine);

        // With every movement mechanism failed, planning fails loudly.
        edges.devices.find_device(DeviceId{200})
            ->find_resource(ResourceId{203})
            ->set_availability(ResourceAvailability::FAILED);

        GERDOS_CHECK(!planner.plan(*movement).has_value());

        // A claimed producing record is never planned onto.
        edges.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{102})
            ->set_availability(ResourceAvailability::AVAILABLE);
        edges.devices.find_device(DeviceId{200})
            ->find_resource(ResourceId{203})
            ->set_availability(ResourceAvailability::AVAILABLE);

        // The claim is created through the effects layer, like real
        // attempts do — claims cannot be forged.
        auto* weights = edges.data.find_data(DataId{500});
        auto* candidate = weights->find_residency(DataResidencyId{5002});

        ExecutionRegistry claim_executions;
        ExecutionEffects claim_effects(edges.data, claim_executions);

        auto* claimant = claim_executions.create_execution(
            ExecutionDescription{ExecutionId{999}, OperationId{800}});

        PhysicalBinding claim_binding;
        claim_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{500}, DataResidencyId{5002}},
            });

        GERDOS_CHECK(claimant->bind(claim_binding));
        GERDOS_CHECK(claim_effects.start(*claimant));
        GERDOS_CHECK(candidate->update_owner() == ExecutionId{999});

        const auto moved = planner.plan(*movement);
        GERDOS_CHECK(moved.has_value());
        GERDOS_CHECK(
            moved->data[1].residency !=
            (DataResidencyRef{DataId{500}, DataResidencyId{5002}}));
    }

    // ---------------------------------------------------------------------
    // 12. Minimums count distinct mechanisms
    // ---------------------------------------------------------------------

    {
        Machine twin;
        BindingPlanner planner(
            twin.devices,
            twin.data,
            twin.topology,
            twin.measurements);

        const Operation two_engines{OperationDescription{
            OperationId{902},
            {},
            {},
            {},
            {
                ResourceRequirement{
                    ResourceBindingRole::TRANSFER,
                    2,
                },
            },
        }};

        const auto binding = planner.plan(two_engines);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(binding->resources.size() == 2);
        GERDOS_CHECK(
            binding->resources[0].resource !=
            binding->resources[1].resource);

        // One engine cannot satisfy a minimum of two.
        const Operation three_engines{OperationDescription{
            OperationId{903},
            {},
            {},
            {},
            {
                ResourceRequirement{
                    ResourceBindingRole::TRANSFER,
                    3,
                },
            },
        }};

        GERDOS_CHECK(!planner.plan(three_engines).has_value());
    }

    // ---------------------------------------------------------------------
    // 13. Dependency scheduling: chains and diamonds order correctly
    // ---------------------------------------------------------------------

    {
        Machine chain;
        BindingPlanner planner(
            chain.devices,
            chain.data,
            chain.topology,
            chain.measurements);

        // Chain: 810 -> 811 -> 812, registered out of order. Each
        // carries a satisfiable TRANSFER requirement so plan() succeeds
        // and the graph logic is isolated from data state.
        auto transferable = [](OperationId id,
                               std::vector<OperationId> deps) {
            return OperationDescription{
                id,
                {},
                {},
                deps,
                {
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            };
        };

        (void)chain.operations.create_operation(transferable(OperationId{812}, {OperationId{811}}));
        (void)chain.operations.create_operation(transferable(OperationId{810}, {}));
        (void)chain.operations.create_operation(transferable(OperationId{811}, {OperationId{810}}));

        // The fixture's own operations (800/801/802, no dependencies)
        // schedule too — the chain asserts relative order, not absolute
        // membership: 810 before 811 before 812, all present.
        const auto order = planner.schedule(chain.operations);
        auto position = [&](OperationId id) -> std::size_t {
            for (std::size_t i = 0; i < order.size(); ++i) {
                if (order[i] == id) {
                    return i;
                }
            }
            return order.size();
        };
        GERDOS_CHECK(position(OperationId{810}) < order.size());
        GERDOS_CHECK(position(OperationId{811}) < order.size());
        GERDOS_CHECK(position(OperationId{812}) < order.size());
        GERDOS_CHECK(position(OperationId{810}) < position(OperationId{811}));
        GERDOS_CHECK(position(OperationId{811}) < position(OperationId{812}));

        // Diamond: 820 fans to 821 + 822, joining at 823.
        Machine diamond;
        BindingPlanner diamond_planner(
            diamond.devices,
            diamond.data,
            diamond.topology,
            diamond.measurements);

        (void)diamond.operations.create_operation(transferable(OperationId{823}, {OperationId{821}, OperationId{822}}));
        (void)diamond.operations.create_operation(transferable(OperationId{822}, {OperationId{820}}));
        (void)diamond.operations.create_operation(transferable(OperationId{821}, {OperationId{820}}));
        (void)diamond.operations.create_operation(transferable(OperationId{820}, {}));

        const auto diamond_order = diamond_planner.schedule(diamond.operations);
        auto diamond_position = [&](OperationId id) -> std::size_t {
            for (std::size_t i = 0; i < diamond_order.size(); ++i) {
                if (diamond_order[i] == id) {
                    return i;
                }
            }
            return diamond_order.size();
        };
        GERDOS_CHECK(diamond_position(OperationId{820}) < diamond_order.size());
        GERDOS_CHECK(diamond_position(OperationId{821}) < diamond_order.size());
        GERDOS_CHECK(diamond_position(OperationId{822}) < diamond_order.size());
        GERDOS_CHECK(diamond_position(OperationId{823}) < diamond_order.size());
        GERDOS_CHECK(diamond_position(OperationId{820}) < diamond_position(OperationId{821}));
        GERDOS_CHECK(diamond_position(OperationId{820}) < diamond_position(OperationId{822}));
        GERDOS_CHECK(diamond_position(OperationId{821}) < diamond_position(OperationId{823}));
        GERDOS_CHECK(diamond_position(OperationId{822}) < diamond_position(OperationId{823}));
        // Siblings order deterministically by id.
        GERDOS_CHECK(diamond_position(OperationId{821}) < diamond_position(OperationId{822}));
    }

    // ---------------------------------------------------------------------
    // 14. Unready operations are excluded, never waited on
    // ---------------------------------------------------------------------

    {
        Machine blocked;
        BindingPlanner planner(
            blocked.devices,
            blocked.data,
            blocked.topology,
            blocked.measurements);

        auto transferable = [](OperationId id,
                               std::vector<OperationId> deps) {
            return OperationDescription{
                id,
                {},
                {},
                deps,
                {
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            };
        };

        // 831 depends on 830, which names a dependency the registry
        // does not hold (999): 830 is unready, 831 cascades unready.
        // 832 is independent and ready.
        (void)blocked.operations.create_operation(transferable(OperationId{830}, {OperationId{999}}));
        (void)blocked.operations.create_operation(transferable(OperationId{831}, {OperationId{830}}));
        (void)blocked.operations.create_operation(transferable(OperationId{832}, {}));

        // Fixture ops stay ready; the blocked pair (830, 831) must be
        // absent while independent 832 is present.
        const auto order = planner.schedule(blocked.operations);
        auto blocked_position = [&](OperationId id) -> bool {
            for (const auto got : order) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(!blocked_position(OperationId{830}));
        GERDOS_CHECK(!blocked_position(OperationId{831}));
        GERDOS_CHECK(blocked_position(OperationId{832}));

        // An operation whose own binding cannot plan is excluded even
        // with satisfied dependencies: garbage requirement, no waiting.
        Machine unplannable;
        BindingPlanner unplannable_planner(
            unplannable.devices,
            unplannable.data,
            unplannable.topology,
            unplannable.measurements);

        (void)unplannable.operations.create_operation(OperationDescription{
            OperationId{840},
            {},
            {},
            {},
            {
                ResourceRequirement{
                    ResourceBindingRole::TRANSFER,
                    99,
                },
            },
        });
        (void)unplannable.operations.create_operation(transferable(OperationId{841}, {}));

        const auto ready = unplannable_planner.schedule(unplannable.operations);
        auto ready_has = [&](OperationId id) -> bool {
            for (const auto got : ready) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(!ready_has(OperationId{840}));
        GERDOS_CHECK(ready_has(OperationId{841}));
    }

    // ---------------------------------------------------------------------
    // 15. Cycles fail closed with the empty set
    // ---------------------------------------------------------------------

    {
        Machine cyclic;
        BindingPlanner planner(
            cyclic.devices,
            cyclic.data,
            cyclic.topology,
            cyclic.measurements);

        auto transferable = [](OperationId id,
                               std::vector<OperationId> deps) {
            return OperationDescription{
                id,
                {},
                {},
                deps,
                {
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            };
        };

        // Two-cycle: 850 <-> 851. Neither can ever be ready.
        (void)cyclic.operations.create_operation(transferable(OperationId{850}, {OperationId{851}}));
        (void)cyclic.operations.create_operation(transferable(OperationId{851}, {OperationId{850}}));

        // Fixture ops stay schedulable; the cyclic pair is excluded
        // and — critically — the acyclic fixture ops still schedule.
        // A PURE cycle (fixture removed) yields the empty set.
        const auto order = planner.schedule(cyclic.operations);
        auto cyclic_has = [&](OperationId id) -> bool {
            for (const auto got : order) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(!cyclic_has(OperationId{850}));
        GERDOS_CHECK(!cyclic_has(OperationId{851}));
        GERDOS_CHECK(!order.empty());

        // Self-dependency is a one-cycle: 852 -> 852.
        Machine self;
        BindingPlanner self_planner(
            self.devices, self.data, self.topology, self.measurements);
        (void)self.operations.create_operation(transferable(OperationId{852}, {OperationId{852}}));
        const auto self_order = self_planner.schedule(self.operations);
        auto self_has = [&](OperationId id) -> bool {
            for (const auto got : self_order) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(!self_has(OperationId{852}));

        // The cycle stays visible: dependencies still name each other,
        // so the caller can break it. Retired ids are permanent (removal
        // is forever), so breaking means withdrawing 851 — 850's
        // dependency then dangles and 850 cascades unready — while a
        // fresh operation schedules normally.
        GERDOS_CHECK(cyclic.operations.remove_operation(OperationId{851}));
        (void)cyclic.operations.create_operation(transferable(OperationId{853}, {}));
        const auto repaired = planner.schedule(cyclic.operations);
        auto repaired_has = [&](OperationId id) -> bool {
            for (const auto got : repaired) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(!repaired_has(OperationId{850}));
        GERDOS_CHECK(!repaired_has(OperationId{851}));
        GERDOS_CHECK(repaired_has(OperationId{853}));
    }

    // ---------------------------------------------------------------------
    // 16. Adversarial: failed outputs and claimed records stay unready
    // ---------------------------------------------------------------------

    {
        Machine hostile;
        BindingPlanner planner(
            hostile.devices,
            hostile.data,
            hostile.topology,
            hostile.measurements);

        // A dependency on a removed operation: the dependent is
        // excluded, not guessed at — a failed attempt's output is a
        // contradiction the planner refuses.
        (void)hostile.operations.create_operation(OperationDescription{
            OperationId{860},
            {DataId{501}},
            {DataId{502}},
            {},
            {
                ResourceRequirement{
                    ResourceBindingRole::COMPUTE,
                    1,
                },
            },
        });
        (void)hostile.operations.create_operation(OperationDescription{
            OperationId{861},
            {DataId{502}},
            {DataId{502}},
            {OperationId{860}},
            {
                ResourceRequirement{
                    ResourceBindingRole::COMPUTE,
                    1,
                },
            },
        });
        // The producer's output is usable (as if 860 completed): the
        // consumer's later exclusion comes from the dangling dependency
        // alone, not from data state — the adversarial separation.
        (void)hostile.data.find_data(DataId{502})
            ->find_residency(DataResidencyId{5201})
            ->set_state(DataResidencyState::VALID);
        auto hostile_has = [&](OperationId id) -> bool {
            const auto scheduled = planner.schedule(hostile.operations);
            for (const auto got : scheduled) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(hostile_has(OperationId{860}));
        GERDOS_CHECK(hostile_has(OperationId{861}));

        // The producer is withdrawn (its attempt failed; nothing
        // produces its output): the consumer cascades unready.
        GERDOS_CHECK(hostile.operations.remove_operation(OperationId{860}));
        const auto after = planner.schedule(hostile.operations);
        auto after_has = [&](OperationId id) -> bool {
            for (const auto got : after) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(!after_has(OperationId{861}));

        // Claimed records keep consumers unready: claim the only usable
        // activation record through the effects layer, and an op reading
        // it still binds (consuming reads are fine) — but an op that
        // must PRODUCE onto the claimed record cannot plan, so a chain
        // through it is excluded at the readiness probe. Pinned
        // structurally: the producing op has no binding.
        Machine claimed;
        BindingPlanner claimed_planner(
            claimed.devices,
            claimed.data,
            claimed.topology,
            claimed.measurements);

        (void)claimed.operations.create_operation(OperationDescription{
            OperationId{870},
            {DataId{501}},
            {DataId{501}},
            {},
            {
                ResourceRequirement{
                    ResourceBindingRole::COMPUTE,
                    1,
                },
            },
        });

        ExecutionRegistry claim_executions;
        ExecutionEffects claim_effects(claimed.data, claim_executions);
        auto* claimant = claim_executions.create_execution(
            ExecutionDescription{ExecutionId{997}, OperationId{870}});
        PhysicalBinding claim_binding;
        claim_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{501}, DataResidencyId{5101}},
            });
        GERDOS_CHECK(claimant->bind(claim_binding));
        GERDOS_CHECK(claim_effects.start(*claimant));

        // Producing onto the claimed record cannot plan; the op is
        // excluded from the ready set even with no dependencies, while
        // unrelated fixture ops still schedule.
        const auto* producer =
            claimed.operations.find_operation(OperationId{870});
        GERDOS_CHECK(!claimed_planner.plan(*producer).has_value());
        const auto claimed_ready = claimed_planner.schedule(claimed.operations);
        auto claimed_has = [&](OperationId id) -> bool {
            for (const auto got : claimed_ready) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(!claimed_has(OperationId{870}));
        GERDOS_CHECK(!claimed_ready.empty());
    }

    // ---------------------------------------------------------------------
    // 17. Derived capacity: contention excludes over-budget ops loudly
    // ---------------------------------------------------------------------

    {
        // Two attempts, one memory budget: ram (101) declares room for
        // exactly one 1.0 MB producing representation. Each op writes 1 MB
        // (1 << 18 F32 elements = 1 MiB) to its own device-homed... no —
        // home the outputs on ram so demand lands on the budgeted
        // resource. Op order decides: first fits, second refused, its
        // dependent cascades.
        Machine budgeted;
        BindingPlanner planner(
            budgeted.devices,
            budgeted.data,
            budgeted.topology,
            budgeted.measurements);

        budgeted.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{101})
            ->set_capacity(1u << 20);

        auto* first_out = budgeted.data.create_data(
            DataDescription{DataId{610}, "first_out"});
        (void)first_out->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6101},
                    DataId{610},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "ram-home",
                },
            });

        auto* second_out = budgeted.data.create_data(
            DataDescription{DataId{611}, "second_out"});
        (void)second_out->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6111},
                    DataId{611},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "ram-home",
                },
            });

        auto* third_out = budgeted.data.create_data(
            DataDescription{DataId{612}, "third_out"});
        (void)third_out->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6121},
                    DataId{612},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "ram-home",
                },
            });

        // Sources: usable records feeding each op (activation 5101 is
        // usable in the fixture; reuse it for all three — consuming
        // reads never conflict).
        auto megabyte = [](OperationId id,
                           DataId out,
                           std::vector<OperationId> deps) {
            return OperationDescription{
                id,
                {DataId{501}},
                {out},
                deps,
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        1,
                    },
                },
                WorkDescription{1 << 18, 1, 0.0f, 1.0f, 0.0f},
            };
        };

        (void)budgeted.operations.create_operation(megabyte(OperationId{920}, DataId{610}, {}));
        (void)budgeted.operations.create_operation(megabyte(OperationId{921}, DataId{611}, {}));
        (void)budgeted.operations.create_operation(
            megabyte(OperationId{922}, DataId{612}, {OperationId{921}}));

        const auto ready = planner.schedule(budgeted.operations);
        auto ready_has = [&](OperationId id) -> bool {
            for (const auto got : ready) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };

        // First megabyte fits the 1 MiB budget; the second (same
        // resource, cumulative 2 MiB) is refused loudly by absence;
        // the third depends on the refused second and cascades.
        GERDOS_CHECK(ready_has(OperationId{920}));
        GERDOS_CHECK(!ready_has(OperationId{921}));
        GERDOS_CHECK(!ready_has(OperationId{922}));

        // plan_attempts inherits the exclusion: no pair for refused ops.
        const auto planned = planner.plan_attempts(budgeted.operations);
        auto planned_has = [&](OperationId id) -> bool {
            for (const auto& attempt : planned) {
                if (attempt.id == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(planned_has(OperationId{920}));
        GERDOS_CHECK(!planned_has(OperationId{921}));
        GERDOS_CHECK(!planned_has(OperationId{922}));
    }

    // ---------------------------------------------------------------------
    // 18. Unbounded resources never refuse; exact-fit fits
    // ---------------------------------------------------------------------

    {
        Machine open;
        BindingPlanner planner(
            open.devices, open.data, open.topology, open.measurements);

        // scratch (103) declares zero: unbounded. Two megabyte outputs
        // home there; both schedule regardless of cumulative demand.
        auto* first_out = open.data.create_data(
            DataDescription{DataId{620}, "first_out"});
        (void)first_out->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6201},
                    DataId{620},
                    ResourceRef{DeviceId{100}, ResourceId{103}},
                    "scratch-home",
                },
            });

        auto* second_out = open.data.create_data(
            DataDescription{DataId{621}, "second_out"});
        (void)second_out->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6202},
                    DataId{621},
                    ResourceRef{DeviceId{100}, ResourceId{103}},
                    "scratch-home",
                },
            });

        auto megabyte = [](OperationId id, DataId out) {
            return OperationDescription{
                id,
                {DataId{501}},
                {out},
                {},
                {
                    ResourceRequirement{
                        ResourceBindingRole::COMPUTE,
                        1,
                    },
                },
                WorkDescription{1 << 18, 1, 0.0f, 1.0f, 0.0f},
            };
        };

        (void)open.operations.create_operation(megabyte(OperationId{930}, DataId{620}));
        (void)open.operations.create_operation(megabyte(OperationId{931}, DataId{621}));

        const auto ready = planner.schedule(open.operations);
        auto ready_has = [&](OperationId id) -> bool {
            for (const auto got : ready) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(ready_has(OperationId{930}));
        GERDOS_CHECK(ready_has(OperationId{931}));

        // Exact fit: ram declares exactly 1 MiB; one megabyte fits.
        Machine exact;
        BindingPlanner exact_planner(
            exact.devices, exact.data, exact.topology, exact.measurements);
        exact.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{101})
            ->set_capacity(1u << 20);

        auto* exact_out = exact.data.create_data(
            DataDescription{DataId{630}, "exact_out"});
        (void)exact_out->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6301},
                    DataId{630},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "ram-home",
                },
            });

        (void)exact.operations.create_operation(OperationDescription{
            OperationId{932},
            {DataId{501}},
            {DataId{630}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{1 << 18, 1, 0.0f, 1.0f, 0.0f},
        });

        const auto exact_ready = exact_planner.schedule(exact.operations);
        auto exact_has = [&](OperationId id) -> bool {
            for (const auto got : exact_ready) {
                if (got == id) {
                    return true;
                }
            }
            return false;
        };
        GERDOS_CHECK(exact_has(OperationId{932}));
    }

    // ---------------------------------------------------------------------
    // 19. Multi-attempt plans preserve order with gate-passable pairs
    // ---------------------------------------------------------------------

    {
        Machine ordered;
        BindingPlanner planner(
            ordered.devices,
            ordered.data,
            ordered.topology,
            ordered.measurements);

        auto transferable = [](OperationId id,
                               std::vector<OperationId> deps) {
            return OperationDescription{
                id,
                {},
                {},
                deps,
                {
                    ResourceRequirement{
                        ResourceBindingRole::TRANSFER,
                        1,
                    },
                },
            };
        };

        (void)ordered.operations.create_operation(transferable(OperationId{892}, {OperationId{891}}));
        (void)ordered.operations.create_operation(transferable(OperationId{890}, {}));
        (void)ordered.operations.create_operation(transferable(OperationId{891}, {OperationId{890}}));

        const auto planned = planner.plan_attempts(ordered.operations);

        // The chain's pairs appear in dependency order...
        auto planned_position = [&](OperationId id) -> std::size_t {
            for (std::size_t i = 0; i < planned.size(); ++i) {
                if (planned[i].id == id) {
                    return i;
                }
            }
            return planned.size();
        };
        GERDOS_CHECK(planned_position(OperationId{890}) < planned.size());
        GERDOS_CHECK(planned_position(OperationId{891}) < planned.size());
        GERDOS_CHECK(planned_position(OperationId{892}) < planned.size());
        GERDOS_CHECK(
            planned_position(OperationId{890}) <
            planned_position(OperationId{891}));
        GERDOS_CHECK(
            planned_position(OperationId{891}) <
            planned_position(OperationId{892}));

        // ...and every pair is individually gate-passable through the
        // same chain a single plan faces.
        PhysicalBindingValidator validator;
        BindingResolver resolver(
            ordered.devices, ordered.data);
        BindingAdmissibilityValidator admissibility;
        ExecutionAdmissionValidator admission(
            ordered.devices, ordered.data);
        ExecutionRegistry probe_executions;

        for (const auto& attempt : planned) {
            const auto* operation = ordered.operations.find_operation(attempt.id);
            GERDOS_CHECK(operation != nullptr);
            GERDOS_CHECK(validator.validate(attempt.binding));
            GERDOS_CHECK(
                resolver.resolve(attempt.binding).fully_resolved());
            GERDOS_CHECK(
                admissibility.admissible(*operation, attempt.binding));
            Execution probe{
                ExecutionDescription{ExecutionId{1900 + attempt.id.value()}, attempt.id}};
            GERDOS_CHECK(probe.bind(attempt.binding));
            GERDOS_CHECK(admission.admit(probe).has_value());
        }
    }

    // ---------------------------------------------------------------------
    // 20. The staged workload plans as stage/compute pairs; executed
    //     attempts leave later pairs' bindings valid (atomicity)
    // ---------------------------------------------------------------------

    {
        // Two-unit staged workload in the demo's shape: each unit stages
        // (TRANSFER) then computes (COMPUTE), unit 2 depending on unit 1.
        // The planner — not hand-pairing — produces stage, compute,
        // stage, compute with the right mechanism roles.
        Machine staged;
        BindingPlanner planner(
            staged.devices,
            staged.data,
            staged.topology,
            staged.measurements);

        auto* payload = staged.data.create_data(
            DataDescription{DataId{600}, "payload"});
        (void)payload->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6001},
                    DataId{600},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "staged",
                },
            });
        (void)payload->find_residency(DataResidencyId{6001})
            ->set_state(DataResidencyState::VALID);

        auto* result = staged.data.create_data(
            DataDescription{DataId{601}, "result"});
        (void)result->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6101},
                    DataId{601},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "result",
                },
            });

        auto* payload2 = staged.data.create_data(
            DataDescription{DataId{602}, "payload2"});
        (void)payload2->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6201},
                    DataId{602},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "staged",
                },
            });
        (void)payload2->find_residency(DataResidencyId{6201})
            ->set_state(DataResidencyState::VALID);

        auto* result2 = staged.data.create_data(
            DataDescription{DataId{603}, "result2"});
        (void)result2->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6301},
                    DataId{603},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "result",
                },
            });

        (void)staged.operations.create_operation(OperationDescription{
            OperationId{910},
            {DataId{600}},
            {DataId{600}},
            {},
            {ResourceRequirement{ResourceBindingRole::TRANSFER, 1}},
        });
        (void)staged.operations.create_operation(OperationDescription{
            OperationId{911},
            {DataId{600}},
            {DataId{601}},
            {OperationId{910}},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
        });
        (void)staged.operations.create_operation(OperationDescription{
            OperationId{912},
            {DataId{602}},
            {DataId{602}},
            {OperationId{911}},
            {ResourceRequirement{ResourceBindingRole::TRANSFER, 1}},
        });
        (void)staged.operations.create_operation(OperationDescription{
            OperationId{913},
            {DataId{602}},
            {DataId{603}},
            {OperationId{912}},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
        });

        const auto planned = planner.plan_attempts(staged.operations);

        // The two units' pairs in dependency order...
        auto staged_position = [&](OperationId id) -> std::size_t {
            for (std::size_t i = 0; i < planned.size(); ++i) {
                if (planned[i].id == id) {
                    return i;
                }
            }
            return planned.size();
        };
        GERDOS_CHECK(staged_position(OperationId{910}) < planned.size());
        GERDOS_CHECK(staged_position(OperationId{913}) < planned.size());
        GERDOS_CHECK(
            staged_position(OperationId{910}) <
            staged_position(OperationId{911}));
        GERDOS_CHECK(
            staged_position(OperationId{911}) <
            staged_position(OperationId{912}));
        GERDOS_CHECK(
            staged_position(OperationId{912}) <
            staged_position(OperationId{913}));

        // ...with stage pairs on TRANSFER mechanisms and compute pairs
        // on COMPUTE mechanisms — the pairing structure, planned.
        for (const auto& attempt : planned) {
            const bool is_stage =
                attempt.id == OperationId{910} ||
                attempt.id == OperationId{912};
            const bool is_compute =
                attempt.id == OperationId{911} ||
                attempt.id == OperationId{913};

            if (!is_stage && !is_compute) {
                continue;
            }

            GERDOS_CHECK(attempt.binding.resources.size() == 1);
            GERDOS_CHECK(
                attempt.binding.resources[0].role ==
                (is_stage ? ResourceBindingRole::TRANSFER
                          : ResourceBindingRole::COMPUTE));
        }

        // Per-attempt atomicity: execute the first pair through the
        // deterministic simulator; the later pairs' already-planned
        // bindings still gate-pass afterwards — the first attempt's
        // evidence, state, and results do not invalidate them.
        SimulatedBackend backend;
        ExecutionRegistry executions;
        Executor executor(
            executions,
            staged.operations,
            staged.devices,
            staged.data,
            backend,
            &staged.measurements);

        {
            const auto& first = planned[staged_position(OperationId{910})];
            const auto* operation =
                staged.operations.find_operation(first.id);
            PhysicalBindingValidator validator;
            GERDOS_CHECK(validator.validate(first.binding));
            auto* execution = executions.create_execution(
                ExecutionDescription{ExecutionId{2000}, first.id});
            GERDOS_CHECK(execution->bind(first.binding));
            GERDOS_CHECK(executor.start(ExecutionId{2000}));
            std::vector<AttemptStatus> outcomes;
            executor.advance(outcomes);
            while (outcomes.empty()) {
                executor.advance(outcomes);
            }
            GERDOS_CHECK(outcomes.size() == 1);
            GERDOS_CHECK(
                outcomes.front().integrity == AttemptIntegrity::COHERENT);
            (void)operation;
        }

        PhysicalBindingValidator revalidator;
        BindingResolver reresolver(staged.devices, staged.data);
        BindingAdmissibilityValidator readmissibility;
        for (const auto& attempt : planned) {
            if (attempt.id != OperationId{912} &&
                attempt.id != OperationId{913}) {
                continue;
            }
            const auto* operation =
                staged.operations.find_operation(attempt.id);
            GERDOS_CHECK(revalidator.validate(attempt.binding));
            GERDOS_CHECK(
                reresolver.resolve(attempt.binding).fully_resolved());
            GERDOS_CHECK(
                readmissibility.admissible(*operation, attempt.binding));
        }
    }

    // ---------------------------------------------------------------------
    // 21. Locality: engine-local usable copies beat lower identifiers
    // ---------------------------------------------------------------------

    {
        // Two usable copies of one Data: low-id on host ram, high-id on
        // device memory. The only compute engine is the accelerator's,
        // so the device-homed copy must win despite its higher id —
        // identifier order would pick the host copy.
        Machine local;
        BindingPlanner planner(
            local.devices, local.data, local.topology, local.measurements);

        auto* both = local.data.create_data(
            DataDescription{DataId{640}, "both"});
        (void)both->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6401},
                    DataId{640},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "host-copy",
                },
            });
        (void)both->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6402},
                    DataId{640},
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "device-copy",
                },
            });
        (void)both->find_residency(DataResidencyId{6401})
            ->set_state(DataResidencyState::VALID);
        (void)both->find_residency(DataResidencyId{6402})
            ->set_state(DataResidencyState::VALID);

        auto* out = local.data.create_data(
            DataDescription{DataId{641}, "out"});
        (void)out->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{6403},
                    DataId{641},
                    ResourceRef{DeviceId{200}, ResourceId{201}},
                    "device-out",
                },
            });

        (void)local.operations.create_operation(OperationDescription{
            OperationId{940},
            {DataId{640}},
            {DataId{641}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{64, 1, 0.0f, 1.0f, 0.0f},
        });

        const auto* compute =
            local.operations.find_operation(OperationId{940});
        const auto binding = planner.plan(*compute);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(binding->data.size() == 2);
        GERDOS_CHECK(
            binding->data[0].residency ==
            (DataResidencyRef{DataId{640}, DataResidencyId{6402}}));

        // Without a compute requirement there is no engine: legacy
        // lowest-identifier order picks the host copy.
        (void)local.operations.create_operation(OperationDescription{
            OperationId{941},
            {DataId{640}},
            {DataId{640}},
            {},
            {ResourceRequirement{ResourceBindingRole::TRANSFER, 1}},
        });

        const auto* movement =
            local.operations.find_operation(OperationId{941});
        const auto moved = planner.plan(*movement);
        GERDOS_CHECK(moved.has_value());
        GERDOS_CHECK(
            moved->data[0].residency ==
            (DataResidencyRef{DataId{640}, DataResidencyId{6401}}));
    }

    // ---------------------------------------------------------------------
    // 22. Later arrivals earn one bounded re-exploration sample
    // ---------------------------------------------------------------------

    {
        // A transfer engine arrives mid-run: dma starts UNAVAILABLE
        // (not yet arrived), so copy-engine serves and accrues fast
        // evidence. When dma becomes AVAILABLE with zero observations,
        // it is sampled once ahead of the measured incumbent — then
        // means decide.
        Machine arrival;
        BindingPlanner planner(
            arrival.devices,
            arrival.data,
            arrival.topology,
            arrival.measurements);

        arrival.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{102})
            ->set_availability(ResourceAvailability::UNAVAILABLE);

        const auto* operation =
            arrival.operations.find_operation(OperationId{801});

        const auto first = planner.plan(*operation);
        GERDOS_CHECK(first.has_value());
        GERDOS_CHECK(first->resources[0].resource == copy_engine);

        // Incumbent accrues fast evidence.
        (void)arrival.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                copy_engine,
                OperationId{801},
                ExecutionId{300},
                true,
                10'000'000,
                {},
            });

        // The engine arrives: unmeasured dma dislodges the measured
        // incumbent exactly once.
        arrival.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{102})
            ->set_availability(ResourceAvailability::AVAILABLE);

        const auto sampled = planner.plan(*operation);
        GERDOS_CHECK(sampled.has_value());
        GERDOS_CHECK(sampled->resources[0].resource == dma);

        // Its sample disappoints (1 s vs 10 ms): means reinstall the
        // incumbent. One sample, then evidence rules.
        (void)arrival.measurements.record(
            MeasurementObservation{
                MeasurementQuantity::DURATION_NS,
                dma,
                OperationId{801},
                ExecutionId{301},
                true,
                1'000'000'000,
                {},
            });

        const auto settled = planner.plan(*operation);
        GERDOS_CHECK(settled.has_value());
        GERDOS_CHECK(settled->resources[0].resource == copy_engine);
    }

    // ---------------------------------------------------------------------
    // 23. Proven incumbents hold against newcomer churn (audit A3)
    // ---------------------------------------------------------------------

    {
        // An incumbent with two successes is proven: a stream of
        // never-tried newcomers no longer steals every plan (the M1
        // churn residual, closed per-comparison). With one success the
        // incumbent is unproven and the newcomer still samples once.
        Machine churn;
        BindingPlanner planner(
            churn.devices, churn.data, churn.topology, churn.measurements);

        const auto* operation =
            churn.operations.find_operation(OperationId{801});

        auto record_success = [&](ResourceRef ref, ExecutionId eid,
                                  std::uint64_t ns) {
            (void)churn.measurements.record(
                MeasurementObservation{
                    MeasurementQuantity::DURATION_NS,
                    ref,
                    OperationId{801},
                    eid,
                    true,
                    ns,
                    {},
                });
        };

        // One success: incumbent unproven — newcomer samples once.
        record_success(dma, ExecutionId{400}, 50'000'000);
        const auto sampled = planner.plan(*operation);
        GERDOS_CHECK(sampled.has_value());
        GERDOS_CHECK(sampled->resources[0].resource == copy_engine);

        // Second success: incumbent proven — the next newcomer waits.
        // (The first newcomer was sampled but recorded nothing, so it
        // is still the unmeasured challenger.)
        record_success(dma, ExecutionId{401}, 50'000'000);
        const auto held = planner.plan(*operation);
        GERDOS_CHECK(held.has_value());
        GERDOS_CHECK(held->resources[0].resource == dma);

        // A FURTHER distinct newcomer also waits: churn closed, not
        // merely delayed one round. (No new mechanism exists in this
        // fixture beyond the pair; the same challenger re-planned
        // stands in for the stream — still unmeasured, still refused.)
        const auto held_again = planner.plan(*operation);
        GERDOS_CHECK(held_again.has_value());
        GERDOS_CHECK(held_again->resources[0].resource == dma);
    }

    // ---------------------------------------------------------------------
    // 24. Claimed sole records refuse through every fallback (audit F1)
    // ---------------------------------------------------------------------

    {
        // The sole-record fallbacks (same-Data in-place, pure-output
        // placement) consult claims through free_sole_residency, never
        // around them: a claimed sole record refuses loudly rather than
        // planning work the executor must abandon. Both pins use the
        // sealed effects layer for claims — claims cannot be forged.
        Machine guarded;
        BindingPlanner planner(
            guarded.devices, guarded.data, guarded.topology,
            guarded.measurements);

        auto* datum = guarded.data.create_data(
            DataDescription{DataId{950}, "guarded"});
        (void)datum->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{9501},
                    DataId{950},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "guarded",
                },
            });
        (void)datum->find_residency(DataResidencyId{9501})
            ->set_state(DataResidencyState::VALID);

        (void)guarded.operations.create_operation(OperationDescription{
            OperationId{950},
            {},
            {DataId{950}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{4, 1, 0.0f, 1.0f, 0.0f},
        });

        // Unclaimed: the pure output writes its sole record in place.
        const auto* pure =
            guarded.operations.find_operation(OperationId{950});
        const auto unclaimed = planner.plan(*pure);
        GERDOS_CHECK(unclaimed.has_value());

        // Claimed through the sealed effects layer: refuses.
        ExecutionRegistry claim_executions;
        ExecutionEffects claim_effects(guarded.data, claim_executions);
        auto* claimant = claim_executions.create_execution(
            ExecutionDescription{ExecutionId{995}, OperationId{950}});
        PhysicalBinding claim_binding;
        claim_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{950}, DataResidencyId{9501}},
            });
        GERDOS_CHECK(claimant->bind(claim_binding));
        GERDOS_CHECK(claim_effects.start(*claimant));
        GERDOS_CHECK(!planner.plan(*pure).has_value());

        // Excluded from the ready set while unrelated fixture ops
        // still schedule.
        const auto guarded_ready = planner.schedule(guarded.operations);
        bool guarded_has = false;

        for (const auto got : guarded_ready) {
            guarded_has = guarded_has || got == OperationId{950};
        }

        GERDOS_CHECK(!guarded_has);
    }

    {
        // Same-Data in-place over a claimed sole record refuses too:
        // the source is usable (it passed the claim filter before... no
        // — a claimed source is unusable, so choose_source refuses
        // first. This pin proves the composition, not just the guard.
        Machine same;
        BindingPlanner planner(
            same.devices, same.data, same.topology, same.measurements);

        auto* datum = same.data.create_data(
            DataDescription{DataId{951}, "same"});
        (void)datum->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{9511},
                    DataId{951},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "same",
                },
            });
        (void)datum->find_residency(DataResidencyId{9511})
            ->set_state(DataResidencyState::VALID);

        (void)same.operations.create_operation(OperationDescription{
            OperationId{951},
            {DataId{951}},
            {DataId{951}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{4, 1, 0.0f, 1.0f, 0.0f},
        });

        const auto* inplace =
            same.operations.find_operation(OperationId{951});
        GERDOS_CHECK(planner.plan(*inplace).has_value());

        ExecutionRegistry claim_executions;
        ExecutionEffects claim_effects(same.data, claim_executions);
        auto* claimant = claim_executions.create_execution(
            ExecutionDescription{ExecutionId{996}, OperationId{951}});
        PhysicalBinding claim_binding;
        claim_binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{DataId{951}, DataResidencyId{9511}},
            });
        GERDOS_CHECK(claimant->bind(claim_binding));
        GERDOS_CHECK(claim_effects.start(*claimant));
        GERDOS_CHECK(!planner.plan(*inplace).has_value());
    }

    return 0;
}