#include "test_check.hpp"

#include <optional>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/binding_planner.hpp"
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
    // 8. Discovery is bounded by the exploration budget
    // ---------------------------------------------------------------------

    {
        Machine budgeted;
        BindingPlanner planner(
            budgeted.devices,
            budgeted.data,
            budgeted.topology,
            budgeted.measurements);

        // Exhaust the exploration budget with unrelated evidence.
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

        // With the budget exhausted, the unmeasured copy-engine no longer
        // dislodges the measured dma: newcomers cannot starve proven
        // engines forever.
        const auto* operation =
            budgeted.operations.find_operation(OperationId{801});

        const auto binding = planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());
        GERDOS_CHECK(binding->resources[0].resource == dma);
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

        auto* weights = edges.data.find_data(DataId{500});
        auto* candidate = weights->find_residency(DataResidencyId{5002});
        candidate->set_update_owner(ExecutionId{999});

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

    return 0;
}