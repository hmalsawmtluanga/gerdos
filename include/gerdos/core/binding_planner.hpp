#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/measurement.hpp"
#include "gerdos/core/operation.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/physical_binding.hpp"
#include "gerdos/core/topology.hpp"

namespace gerdos {

// Binding planning: produces one physical binding for an attempt of an
// Operation from the current runtime state. Planning is deterministic and
// fail-closed: it returns no binding rather than one that cannot pass
// structural validation, resolution, semantic admissibility, and admission.
//
// Version-zero selection rules:
//   - a declared input binds a consuming entry; a declared output binds a
//     producing entry; data declared as both consumes from one record and
//     produces to a distinct record where one is reachable (movement:
//     SOURCE/DESTINATION), otherwise in place (INPUT/OUTPUT)
//   - consuming entries use usable records with available hosts, lowest
//     identifiers first
//   - producing entries prefer distinct unclaimed records with available
//     hosts, ranked by the best covering link for movement, then identifier
//     order
//   - resource requirements are satisfied by distinct, available mechanisms
//     whose kind matches the required role; movement transfer mechanisms must
//     sit on the devices at the ends of a covering link — link endpoints are
//     the data locations, and engines drive hops between them — preferred by
//     declared bandwidth, then latency, then identifier order
//   - movement between distinct records requires a covering topology link
//
// Claimed producing records are never planned onto. Measurement-informed
// preference is discovery before exploitation, decided per comparison:
// unmeasured mechanisms are sampled ahead of measured ones (ties by
// declared attributes), and once measured, mean successful duration
// decides. Failure evidence is not speed evidence, and saturated totals
// are not evidence of speed.
class BindingPlanner {
public:
    // Discovery is per-comparison, not budgeted: unmeasured mechanisms
    // are sampled ahead of measured ones, ties break by declared
    // attributes, and later arrivals earn one bounded re-exploration
    // sample before means decide. Kept for compatibility: existing
    // callers and tests may reference the historical budget value.
    // New code must not branch on it.
    static constexpr std::size_t kExplorationBudget = 32;

    // Proven incumbents keep their plans: an unmeasured candidate wins
    // only while the measured incumbent is unproven (fewer successes).
    // Two successes — not one lucky sample. Closes the churn residual
    // where endless newcomers steal every plan; late arrivals past proof
    // wait, disclosed in the contract.
    static constexpr std::size_t kProvenSuccesses = 2;

    BindingPlanner(
        const DeviceRegistry& devices,
        const DataRegistry& data,
        const Topology& topology,
        const MeasurementRegistry& measurements) noexcept
        : devices_(devices),
          data_(data),
          topology_(topology),
          measurements_(measurements) {}

    [[nodiscard]] std::optional<PhysicalBinding> plan(
        const Operation& operation) const {
        const auto& description = operation.description();

        struct Placement {
            DataId id;
            const DataResidency* source;
            const DataResidency* target;
            bool movement;
        };

        std::vector<Placement> placements;
        std::vector<DataId> declared;
        PhysicalBinding binding;

        for (const auto input : description.inputs) {
            if (!contains(declared, input)) {
                declared.push_back(input);
            }
        }

        for (const auto output : description.outputs) {
            if (!contains(declared, output)) {
                declared.push_back(output);
            }
        }

        // Engine-first ordering: compute mechanisms are chosen before
        // data placement (with no hops — movement hops derive from
        // placements below), so consuming-copy choice can prefer
        // engine-local records. Transfer mechanisms follow placement.
        std::vector<const TopologyLink*> no_hops;
        for (const auto& requirement :
             description.resource_requirements) {
            if (!requirement.valid()) {
                return std::nullopt;
            }

            if (requirement.role != ResourceBindingRole::COMPUTE) {
                continue;
            }

            const auto chosen = choose_mechanisms(
                requirement,
                no_hops,
                binding);

            if (chosen < requirement.minimum) {
                return std::nullopt;
            }
        }

        std::vector<DeviceId> engines;
        for (const auto& entry : binding.resources) {
            if (entry.role != ResourceBindingRole::COMPUTE) {
                continue;
            }

            bool known = false;
            for (const auto engine : engines) {
                if (engine == entry.resource.device) {
                    known = true;
                    break;
                }
            }

            if (!known) {
                engines.push_back(entry.resource.device);
            }
        }

        for (const auto id : declared) {
            const bool is_input = contains(description.inputs, id);
            const bool is_output = contains(description.outputs, id);

            const DataResidency* source = nullptr;
            const DataResidency* target = nullptr;

            if (is_input) {
                source = choose_source(id, engines);

                if (source == nullptr) {
                    return std::nullopt;
                }
            }

            if (is_output) {
                // A producing entry for the same Data as the consuming
                // entry writes the same record in place: skip the
                // distinct-target search, which excludes the source
                // record by construction. The source is usable (chosen
                // through the claim filter), so in-place is claim-safe.
                const bool same_data =
                    is_input && source != nullptr &&
                    sole_residency(id) == source;
                target = same_data ? source : choose_target(id, source);

                // In-place realization only when the consuming record is
                // the sole representation: with sibling records that are
                // unreachable or busy, the update was movement and planning
                // fails loudly. A pure output (no consuming source) with
                // exactly one representation writes it in place — only
                // when that record is unclaimed with an available host.
                // Claimed records are never planned onto, by any path.
                if (target == nullptr &&
                    (source != nullptr || record_count(id) == 1)) {
                    if (source != nullptr && record_count(id) != 1) {
                        return std::nullopt;
                    }

                    if (source != nullptr &&
                        (source->update_owner().valid() ||
                         !host_available(
                             source->description().resource))) {
                        return std::nullopt;
                    }

                    target = source;

                    if (target == nullptr) {
                        target = free_sole_residency(id);
                    }
                }

                if (target == nullptr) {
                    return std::nullopt;
                }
            }

            const bool movement =
                is_input && is_output && target != source;

            placements.push_back(
                Placement{
                    id,
                    source,
                    target,
                    movement,
                });

            if (is_input) {
                binding.data.push_back(
                    DataBinding{
                        movement ? DataBindingRole::SOURCE
                                 : DataBindingRole::INPUT,
                        DataResidencyRef{
                            id,
                            source->description().id,
                        },
                    });
            }

            if (is_output) {
                binding.data.push_back(
                    DataBinding{
                        movement ? DataBindingRole::DESTINATION
                                 : DataBindingRole::OUTPUT,
                        DataResidencyRef{
                            id,
                            target->description().id,
                        },
                    });
            }
        }

        if (binding.data.empty() &&
            description.resource_requirements.empty()) {
            return std::nullopt;
        }

        // Movement between distinct records needs a covering topology link;
        // movement transfer mechanisms must sit on those links.
        std::vector<const TopologyLink*> hops;

        for (const auto& placement : placements) {
            if (!placement.movement) {
                continue;
            }

            const auto* hop = best_link(
                placement.source->description().resource,
                placement.target->description().resource);

            if (hop == nullptr) {
                return std::nullopt;
            }

            hops.push_back(hop);
        }

        for (const auto& requirement :
             description.resource_requirements) {
            if (!requirement.valid()) {
                return std::nullopt;
            }

            // Compute mechanisms were chosen before placement; the
            // remaining roles follow with movement hops.
            if (requirement.role == ResourceBindingRole::COMPUTE) {
                continue;
            }

            const auto chosen = choose_mechanisms(
                requirement,
                hops,
                binding);

            if (chosen < requirement.minimum) {
                return std::nullopt;
            }
        }

        return binding;
    }

    // Ready operations in topological order over declared dependencies.
    // Deterministic (lowest OperationId first among ready), fail-closed
    // (cycles yield the empty set), non-blocking (unready operations are
    // excluded, never waited on).
    [[nodiscard]] std::vector<OperationId> schedule(
        const OperationRegistry& operations) const {
        // Index present operations by id.
        std::vector<const Operation*> present;
        operations.for_each_operation(
            [&](const Operation* operation) {
                present.push_back(operation);
            });

        auto present_by_id = [&](OperationId id) -> const Operation* {
            for (const auto* operation : present) {
                if (operation->description().id == id) {
                    return operation;
                }
            }

            return nullptr;
        };

        // Kahn's algorithm over edges dependency -> operation, restricted
        // to dependencies the registry holds. Operations with absent
        // dependencies are excluded upfront (not ready); their dependents
        // cascade as unready through fixpoint iteration: an operation is
        // eligible only when every dependency is itself eligible.
        std::vector<const Operation*> eligible;
        for (const auto* operation : present) {
            bool missing = false;
            for (const auto dependency :
                 operation->description().dependencies) {
                if (present_by_id(dependency) == nullptr) {
                    missing = true;
                    break;
                }
            }

            if (!missing) {
                eligible.push_back(operation);
            }
        }

        auto eligible_contains = [&](OperationId id) -> bool {
            for (const auto* operation : eligible) {
                if (operation->description().id == id) {
                    return true;
                }
            }

            return false;
        };

        bool narrowed = true;
        while (narrowed) {
            narrowed = false;
            std::vector<const Operation*> kept;
            for (const auto* operation : eligible) {
                bool unready = false;
                for (const auto dependency :
                     operation->description().dependencies) {
                    if (!eligible_contains(dependency)) {
                        unready = true;
                        break;
                    }
                }

                if (unready) {
                    narrowed = true;
                } else {
                    kept.push_back(operation);
                }
            }
            eligible = std::move(kept);
        }

        // In-degree over eligible-only edges.
        auto eligible_by_id = [&](OperationId id) -> bool {
            for (const auto* operation : eligible) {
                if (operation->description().id == id) {
                    return true;
                }
            }

            return false;
        };

        std::vector<OperationId> ordered;
        std::vector<const Operation*> remaining = eligible;

        while (!remaining.empty()) {
            // Lowest id first: deterministic ready order.
            const Operation* next = nullptr;
            for (const auto* candidate : remaining) {
                bool blocked = false;
                for (const auto dependency :
                     candidate->description().dependencies) {
                    if (!eligible_by_id(dependency)) {
                        continue;
                    }

                    bool satisfied = false;
                    for (const auto done : ordered) {
                        if (done == dependency) {
                            satisfied = true;
                            break;
                        }
                    }

                    if (!satisfied) {
                        blocked = true;
                        break;
                    }
                }

                if (!blocked &&
                    (next == nullptr ||
                     candidate->description().id.value() <
                         next->description().id.value())) {
                    next = candidate;
                }
            }

            if (next == nullptr) {
                // No unblocked operation remains: the remainder is a
                // dependency cycle. Fail closed for the cyclic nodes —
                // they are excluded and never scheduled — while already
                // ordered ready work is kept. The cycle stays visible in
                // the declared dependencies for the caller to break.
                // A pure cycle therefore yields the empty set.
                break;
            }

            ordered.push_back(next->description().id);
            std::vector<const Operation*> still;
            for (const auto* operation : remaining) {
                if (operation->description().id !=
                    next->description().id) {
                    still.push_back(operation);
                }
            }
            remaining = std::move(still);
        }

        // Readiness probe: keep only operations whose binding plans
        // successfully — usability and claim state evaluated by plan(),
        // the one semantic home — and whose derived demand fits declared
        // capacity. Demand accumulates in scheduled order per home
        // memory/storage resource; over-budget operations are excluded
        // with dependents cascading. Order preserved.
        std::vector<OperationId> ready;
        std::vector<OperationId> dropped;
        std::vector<std::pair<ResourceRef, std::size_t>> committed;
        for (const auto id : ordered) {
            const auto* operation = present_by_id(id);

            if (operation == nullptr) {
                continue;
            }

            // Cascaded capacity exclusion: dependents of dropped
            // operations are themselves dropped.
            bool blocked = false;
            for (const auto dependency :
                 operation->description().dependencies) {
                for (const auto dropped_id : dropped) {
                    if (dropped_id == dependency) {
                        blocked = true;
                        break;
                    }
                }

                if (blocked) {
                    break;
                }
            }

            if (blocked) {
                dropped.push_back(id);
                continue;
            }

            const auto binding = plan(*operation);

            if (!binding.has_value()) {
                continue;
            }

            if (!fits_capacity(
                    *operation, *binding, committed)) {
                dropped.push_back(id);
                continue;
            }

            ready.push_back(id);
        }

        return ready;
    }

    // Derived demand check: accumulate this attempt's producing-entry
    // bytes per home memory/storage resource; refuse (false) when any
    // declared (nonzero) total would be exceeded. Zero total means
    // unbounded. Overflow-safe: compares without wrapping.
    [[nodiscard]] bool fits_capacity(
        const Operation& operation,
        const PhysicalBinding& binding,
        std::vector<std::pair<ResourceRef, std::size_t>>& committed) const {
        const auto& work = operation.description().work;
        const auto bytes =
            work.storage_elements() * work.storage_bytes();
        std::vector<std::pair<ResourceRef, std::size_t>> claims;

        for (const auto& entry : binding.data) {
            if (!is_producing(entry.role)) {
                continue;
            }

            const auto* datum = data_.find_data(entry.residency.data);

            if (datum == nullptr) {
                continue;
            }

            const auto* record =
                datum->find_residency(entry.residency.residency);

            if (record == nullptr) {
                continue;
            }

            const auto home = record->description().resource;
            const auto* device = devices_.find_device(home.device);

            if (device == nullptr) {
                continue;
            }

            const auto* resource = device->find_resource(home.resource);

            if (resource == nullptr) {
                continue;
            }

            const auto kind = resource->description().kind;

            if (kind != ResourceKind::MEMORY &&
                kind != ResourceKind::STORAGE) {
                continue;
            }

            const auto total = resource->description().capacity_bytes;

            if (total == 0) {
                continue;
            }

            std::size_t used = bytes;
            for (const auto& [ref, amount] : committed) {
                if (ref == home) {
                    used += amount;

                    if (used < amount) {
                        return false;
                    }
                }
            }

            if (used > total) {
                return false;
            }

            claims.emplace_back(home, bytes);
        }

        for (const auto& claim : claims) {
            committed.push_back(claim);
        }

        return true;
    }

    // One planned attempt: the operation plus its physical binding,
    // computed against a single runtime state snapshot. Each pair is
    // individually gate-passable; execution still advances attempt by
    // attempt with per-attempt rejection atomicity.
    struct PlannedAttempt {
        OperationId id;
        PhysicalBinding binding;
    };

    // Multi-attempt plans in dependency order: schedule() composed with
    // plan() — one semantic home each, no new binding logic. The list is
    // a plan, not a promise: later attempts may still be refused at
    // execution (claims, contention), leaving earlier evidence intact.
    [[nodiscard]] std::vector<PlannedAttempt> plan_attempts(
        const OperationRegistry& operations) const {
        const auto order = schedule(operations);
        std::vector<PlannedAttempt> planned;

        for (const auto id : order) {
            const Operation* found = nullptr;
            operations.for_each_operation(
                [&](const Operation* operation) {
                    if (operation->description().id == id) {
                        found = operation;
                    }
                });

            if (found == nullptr) {
                continue;
            }

            const auto binding = plan(*found);

            if (!binding.has_value()) {
                continue;
            }

            planned.push_back(PlannedAttempt{id, *binding});
        }

        return planned;
    }

private:
    [[nodiscard]] static bool contains(
        const std::vector<DataId>& references,
        DataId id) noexcept {
        for (const auto reference : references) {
            if (reference == id) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] static bool covers_host(
        const TopologyEndpoint& endpoint,
        ResourceRef host) noexcept {
        return endpoint.device == host.device &&
               (endpoint.kind == TopologyEndpointKind::DEVICE ||
                endpoint.resource == host.resource);
    }

    [[nodiscard]] static bool covers_pair(
        const TopologyLink& link,
        ResourceRef from,
        ResourceRef to) noexcept {
        const auto& description = link.description();

        if (covers_host(description.source, from) &&
            covers_host(description.destination, to)) {
            return true;
        }

        return description.direction ==
                   TopologyLinkDirection::BIDIRECTIONAL &&
               covers_host(description.source, to) &&
               covers_host(description.destination, from);
    }

    // Declared preference: bandwidth first, then latency, then identity.
    [[nodiscard]] static bool better_link(
        const TopologyLink* left,
        const TopologyLink* right) noexcept {
        if (right == nullptr) {
            return true;
        }

        const auto& a = left->description().attributes;
        const auto& b = right->description().attributes;

        if (a.bandwidth_bytes_per_second != b.bandwidth_bytes_per_second) {
            return a.bandwidth_bytes_per_second >
                   b.bandwidth_bytes_per_second;
        }

        if (a.latency_ns != b.latency_ns) {
            return a.latency_ns < b.latency_ns;
        }

        return left->description().id.value() <
               right->description().id.value();
    }

    [[nodiscard]] const TopologyLink* best_link(
        ResourceRef from,
        ResourceRef to) const noexcept {
        const TopologyLink* best = nullptr;

        topology_.for_each_link(
            [&](const TopologyLink* link) {
                if (covers_pair(*link, from, to) &&
                    better_link(link, best)) {
                    best = link;
                }
            });

        return best;
    }

    [[nodiscard]] bool host_available(ResourceRef host) const noexcept {
        const auto* device = devices_.find_device(host.device);

        if (device == nullptr) {
            return false;
        }

        const auto* resource = device->find_resource(host.resource);

        return resource != nullptr && resource->available();
    }

    // Engine-local consuming-copy choice: usable records with
    // available hosts, ranked engine-local first (home device among the
    // chosen compute engines — zero staging), then lowest identifiers.
    // Empty engines (no compute requirement) keep pure identifier order.
    [[nodiscard]] const DataResidency* choose_source(
        DataId id,
        const std::vector<DeviceId>& engines) const noexcept {
        const DataResidency* best = nullptr;

        data_.for_each_data(
            [&](const Data* datum) {
                if (datum->description().id != id) {
                    return;
                }

                datum->for_each_residency(
                    [&](const DataResidency* record) {
                        if (!record->usable() ||
                            !host_available(
                                record->description().resource)) {
                            return;
                        }

                        if (better_source(record, engines, best)) {
                            best = record;
                        }
                    });
            });

        return best;
    }

    [[nodiscard]] static bool source_local(
        const DataResidency* record,
        const std::vector<DeviceId>& engines) noexcept {
        for (const auto engine : engines) {
            if (record->description().resource.device == engine) {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] static bool better_source(
        const DataResidency* record,
        const std::vector<DeviceId>& engines,
        const DataResidency* best) noexcept {
        if (best == nullptr) {
            return true;
        }

        const bool record_local = source_local(record, engines);
        const bool best_local = source_local(best, engines);

        if (record_local != best_local) {
            return record_local;
        }

        return better_record(record, best);
    }

    [[nodiscard]] const DataResidency* choose_target(
        DataId id,
        const DataResidency* source) const noexcept {
        const DataResidency* best = nullptr;
        const TopologyLink* best_via = nullptr;

        data_.for_each_data(
            [&](const Data* datum) {
                if (datum->description().id != id) {
                    return;
                }

                datum->for_each_residency(
                    [&](const DataResidency* record) {
                        if (source != nullptr &&
                            record->description().id ==
                                source->description().id) {
                            return;
                        }

                        if (record->update_owner().valid() ||
                            !host_available(
                                record->description().resource)) {
                            return;
                        }

                        const auto* via =
                            source == nullptr
                                ? nullptr
                                : best_link(
                                      source->description().resource,
                                      record->description().resource);

                        // A distinct target must be reachable from the
                        // consuming record.
                        if (source != nullptr && via == nullptr) {
                            return;
                        }

                        if (better_target(
                                record,
                                via,
                                best,
                                best_via)) {
                            best = record;
                            best_via = via;
                        }
                    });
            });

        return best;
    }

    [[nodiscard]] static bool better_target(
        const DataResidency* left,
        const TopologyLink* left_via,
        const DataResidency* right,
        const TopologyLink* right_via) noexcept {
        if (right == nullptr) {
            return true;
        }

        if (left_via != nullptr && right_via != nullptr &&
            left_via != right_via) {
            return better_link(left_via, right_via);
        }

        return left->description().id.value() <
               right->description().id.value();
    }

    [[nodiscard]] std::size_t record_count(DataId id) const noexcept {
        std::size_t count = 0;

        data_.for_each_data(
            [&](const Data* datum) {
                if (datum->description().id == id) {
                    count = datum->residency_count();
                }
            });

        return count;
    }

    // The sole representation of a Data, or nullptr unless exactly one
    // exists — regardless of claims. Identity checks only; never
    // placement.
    [[nodiscard]] const DataResidency* sole_residency(DataId id) const
        noexcept {
        const DataResidency* sole = nullptr;
        std::size_t count = 0;

        data_.for_each_data(
            [&](const Data* datum) {
                if (datum->description().id != id) {
                    return;
                }

                datum->for_each_residency(
                    [&](const DataResidency* record) {
                        sole = record;
                        ++count;
                    });
            });

        return count == 1 ? sole : nullptr;
    }

    // The sole unclaimed representation with an available host, or
    // nullptr. Placement consults claims through this, never around it.
    [[nodiscard]] const DataResidency* free_sole_residency(DataId id) const
        noexcept {
        const DataResidency* sole = nullptr;
        std::size_t count = 0;

        data_.for_each_data(
            [&](const Data* datum) {
                if (datum->description().id != id) {
                    return;
                }

                datum->for_each_residency(
                    [&](const DataResidency* record) {
                        sole = record;
                        ++count;
                    });
            });

        if (count != 1 || sole == nullptr) {
            return nullptr;
        }

        if (sole->update_owner().valid() ||
            !host_available(sole->description().resource)) {
            return nullptr;
        }

        return sole;
    }

    [[nodiscard]] static bool better_record(
        const DataResidency* left,
        const DataResidency* right) noexcept {
        return right == nullptr ||
               left->description().id.value() <
                   right->description().id.value();
    }

    struct Candidate {
        ResourceRef ref;
        const TopologyLink* via{nullptr};
    };

    // Measurement-informed preference: discovery before exploitation.
    // Unmeasured mechanisms are sampled ahead of measured ones so that new
    // or never-tried mechanisms are not starved; once measured, candidates
    // rank by mean observed successful duration — measured behavior
    // outranks declared attributes. Evidence of failure is not speed
    // evidence and leaves a mechanism in the unmeasured class.
    [[nodiscard]] bool faster_mechanism(
        const Candidate& left,
        const Candidate& right) const noexcept {
        const auto left_summary = measurements_.summarize(
            left.ref, MeasurementQuantity::DURATION_NS);
        const auto right_summary = measurements_.summarize(
            right.ref, MeasurementQuantity::DURATION_NS);

        // Saturated totals cannot yield a meaningful mean: such evidence
        // falls back to the unmeasured class rather than lying about speed.
        const bool left_measured =
            left_summary.succeeded_observations > 0 &&
            left_summary.succeeded_total_value <
                std::numeric_limits<std::uint64_t>::max();
        const bool right_measured =
            right_summary.succeeded_observations > 0 &&
            right_summary.succeeded_total_value <
                std::numeric_limits<std::uint64_t>::max();

        if (left_measured != right_measured) {
            // Discovery: the unmeasured mechanism is sampled — including
            // later arrivals, which earn one bounded re-exploration
            // sample against measured incumbents before means decide.
            // Proven incumbents hold: past proof the measured behavior
            // stands and the newcomer waits (churn residual closed).
            const auto incumbent_successes =
                left_measured ? left_summary.succeeded_observations
                              : right_summary.succeeded_observations;

            if (incumbent_successes >= kProvenSuccesses) {
                return left_measured;
            }

            return !left_measured;
        }

        if (left_measured) {
            const auto left_mean =
                left_summary.succeeded_total_value /
                left_summary.succeeded_observations;
            const auto right_mean =
                right_summary.succeeded_total_value /
                right_summary.succeeded_observations;

            if (left_mean != right_mean) {
                return left_mean < right_mean;
            }
        }

        return better_candidate(left, right);
    }

    [[nodiscard]] static bool better_candidate(
        const Candidate& left,
        const Candidate& right) noexcept {
        if (left.via != nullptr || right.via != nullptr) {
            if (left.via == nullptr) {
                return false;
            }

            if (right.via == nullptr) {
                return true;
            }

            if (better_link(left.via, right.via)) {
                return true;
            }

            if (better_link(right.via, left.via)) {
                return false;
            }
        }

        return left.ref.device.value() < right.ref.device.value() ||
               (left.ref.device == right.ref.device &&
                left.ref.resource.value() < right.ref.resource.value());
    }

    // Fills the binding with mechanisms for one requirement and reports how
    // many distinct resources were selected.
    [[nodiscard]] std::size_t choose_mechanisms(
        const ResourceRequirement& requirement,
        const std::vector<const TopologyLink*>& hops,
        PhysicalBinding& binding) const noexcept {
        std::vector<Candidate> chosen;

        for (std::size_t count = 0; count < requirement.minimum;) {
            bool found = false;
            Candidate best_value{};

            devices_.for_each_device(
                [&](const Device* device) {
                    device->for_each_resource(
                        [&](const Resource* resource) {
                            const auto& description =
                                resource->description();

                            if (!resource->available() ||
                                !is_mechanism(
                                    requirement.role,
                                    description.kind)) {
                                return;
                            }

                            const Candidate candidate{
                                ResourceRef{
                                    description.owner,
                                    description.id,
                                },
                                reachable_via(
                                    hops,
                                    ResourceRef{
                                        description.owner,
                                        description.id,
                                    }),
                            };

                            if (!hops.empty() &&
                                requirement.role ==
                                    ResourceBindingRole::TRANSFER &&
                                candidate.via == nullptr) {
                                return;
                            }

                            for (const auto& existing : chosen) {
                                if (existing.ref == candidate.ref) {
                                    return;
                                }
                            }

                            if (!found ||
                                faster_mechanism(
                                    candidate,
                                    best_value)) {
                                best_value = candidate;
                                found = true;
                            }
                        });
                });

            if (!found) {
                return count;
            }

            chosen.push_back(best_value);
            binding.resources.push_back(
                ResourceBinding{
                    requirement.role,
                    best_value.ref,
                });

            ++count;
        }

        return chosen.size();
    }

    [[nodiscard]] static const TopologyLink* reachable_via(
        const std::vector<const TopologyLink*>& hops,
        ResourceRef ref) noexcept {
        const TopologyLink* best = nullptr;

        for (const auto* hop : hops) {
            const auto& description = hop->description();

            // A mechanism drives a hop when it sits on either endpoint
            // device: link endpoints are the data locations, and the
            // engines that move data between them live on the devices at
            // the ends of the link.
            const bool on_path =
                description.source.device == ref.device ||
                description.destination.device == ref.device;

            if (on_path && better_link(hop, best)) {
                best = hop;
            }
        }

        return best;
    }

    const DeviceRegistry& devices_;
    const DataRegistry& data_;
    const Topology& topology_;
    const MeasurementRegistry& measurements_;
};

} // namespace gerdos