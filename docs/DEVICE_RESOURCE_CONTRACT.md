# GERDOS Device and Resource Contract

## Status

Architecture contract v0.1.

This document defines ownership, lifetime, identity, description, and runtime
state semantics for Devices and Resources.

## 1. Device

A Device is a physical or logical hardware ownership boundary.

A Device has:

- a stable `DeviceId`
- an immutable `DeviceDescription`
- zero or more Resources

The Resource Registry owns active Device objects. A Device owns its active
Resource objects. External components do not own Devices or Resources merely
by obtaining an identifier or lookup result.

A Device is not necessarily a scheduling unit.

A Device must not contain model-specific execution concepts, scheduler state,
vendor API handles, or telemetry policy.

## 2. Resource

A Resource represents a schedulable or constraining execution resource.

A Resource has:

- a stable `ResourceId`
- an owning `DeviceId`
- an immutable `ResourceDescription`
- runtime state

Resource kinds currently include:

- `COMPUTE`
- `MEMORY`
- `STORAGE`
- `TRANSFER`
- `SYNCHRONIZATION`

Physical connectivity such as PCIe or NVLink belongs to Topology, not
automatically to Resource.

## 3. Ownership

The runtime ownership hierarchy is:

    ResourceRegistry
        |
        +-- Device
              |
              +-- Resource

The Resource Registry owns active Devices.

A Device owns its active Resources.

Device lifetime dominates Resource lifetime.

A Resource cannot remain registered after its owning Device has been removed.

A `DeviceId` or `ResourceId` is an identifier, not an ownership token.

Holding an identifier does not keep the corresponding object alive.

## 4. Description

Descriptions represent relatively stable identity and classification.

`DeviceDescription` and `ResourceDescription` are logically immutable after
registration. Implementations must not expose APIs that mutate their identity,
ownership, or classification after registration.

`ResourceDescription` currently contains:

- resource identity
- owning device identity
- resource kind
- human-readable name

Descriptions must not be used as mutable runtime state.

## 5. Runtime State

Runtime state represents transient execution availability.

The initial runtime state model should remain deliberately small.

Resource-specific quantities such as bandwidth, utilization, queue depth,
capacity, latency, or throughput must not be forced into one universal state
structure merely for convenience.

Such observations belong to appropriate resource-specific state or
measurement/telemetry mechanisms.

## 6. Availability

The runtime distinguishes at least these concepts:

- identifier existence
- registration
- active participation
- availability for new work

These concepts must not be represented as one overloaded boolean.

A resource may be:

- available
- unavailable
- initializing
- draining
- failed

The exact representation must remain explicit and testable.

Availability is distinct from capacity, utilization, and health. The initial
implementation must not force those concepts into one universal state value.

Unavailable resources must not be selected for new execution unless an
explicit future policy permits it.

## 7. Registry

The Resource Registry is an ownership and discovery boundary.

It is responsible for:

- registering Devices
- registering Resources under Devices
- resolving identifiers
- removing Devices
- invalidating dependent Resource registrations

The registry is not:

- a scheduler
- a planner
- a telemetry database
- a backend implementation
- a global policy engine

## 8. Removal

Device removal must invalidate all Resources owned by that Device.

Previously issued identifiers may remain as historical identifiers, but must
not resolve to active resources after removal.

Device and Resource identifiers are not reused during the lifetime of a GERDOS
runtime instance.

The system must distinguish:

- identifier existence
- active registration
- resource availability

These are different concepts.

## 9. Backend Independence

Backends discover hardware and provide backend-specific implementation.

The core registry must not depend on:

- CUDA
- HIP
- SYCL
- vendor handles
- driver-specific object types

A simulated backend must be capable of populating the same core Device and
Resource abstractions.

## 10. No Hidden Ownership

Core APIs must make ownership explicit.

A successful lookup normally returns a borrowed/non-owning view or reference.
It must not silently transfer ownership to the caller.

Callers must not retain borrowed references across registry operations that may
remove or replace the referenced Device or Resource.

The exact synchronization mechanism is an implementation concern, but the
lifetime guarantee must remain explicit.

Raw pointers and references must not silently imply ownership.

## 11. Concurrency and Synchronization

The Device Registry, Device resource ownership, and their runtime state are
not internally synchronized in the initial implementation.

Concurrent access therefore requires external synchronization by the owning
runtime or integration layer.

Operations that create, remove, or otherwise mutate Devices or Resources must
be externally synchronized against other accesses to the affected registry or
Device.

Runtime availability/state changes follow the same external-synchronization
rule.

Const lookup and const enumeration do not provide an independent thread-safety
guarantee. They are safe for concurrent use only when the underlying object or
collection is not concurrently mutated.

Borrowed Device and Resource pointers remain subject to the lifetime of their
owning object. Removal or destruction invalidates previously obtained borrowed
access.

Enumeration does not create a snapshot and does not change these
synchronization or lifetime requirements.

The initial contract does not prescribe mutexes, atomics, lock-free structures,
or another particular synchronization mechanism.

## 12. Topology Boundary

Topology is a separate runtime graph and is not part of the Device ownership
hierarchy.

Topology links may refer to Devices and Resources by non-owning identifier,
but topology does not own or directly validate those objects.

Device and Resource lifetime therefore remains authoritative for object
ownership. A topology relationship must not be interpreted as keeping an
endpoint alive.

When a Device or Resource is removed, topology relationships referring to
that endpoint become invalid for execution. The future runtime integration
layer is responsible for reconciling active topology with active resources.

Topology identifiers are separate from DeviceId and ResourceId and must not
be used as ownership tokens.

## 13. Enumeration

The Device Registry exposes enumeration of active Devices without transferring
ownership.

A Device exposes enumeration of its active Resources without transferring
ownership.

Enumeration callbacks receive borrowed pointers. Const owners expose const
borrowed pointers.

Enumeration order is unspecified because active objects are stored in
unordered ownership containers. Consumers must identify objects by their
stable runtime identifiers rather than by enumeration position.

Enumeration does not create a snapshot. Callbacks must not add or remove
entries from the collection currently being enumerated.

Enumeration does not expose or transfer ownership of the underlying
containers.

Topology exposes enumeration of its active topology links without transferring
ownership.

Topology enumeration callbacks receive borrowed pointers. Const topology
owners expose const borrowed pointers.

Topology-link enumeration order is unspecified. Consumers must identify links
by their stable TopologyLinkId rather than by enumeration position.

Topology enumeration does not create a snapshot. Callbacks must not add or
remove topology links from the collection currently being enumerated.

Topology enumeration does not create an ownership relationship between the
Topology graph and its endpoint Devices or Resources.

## 14. Architectural Invariants

The implementation must preserve these invariants:

1. Every active Resource has exactly one owning Device.
2. Every active Resource has exactly one ResourceId.
3. `DeviceId` and `ResourceId` values are not reused during the lifetime of
   a GERDOS runtime instance.
4. Device removal removes active ownership of its Resources.
5. Resource descriptions are not runtime state.
6. Runtime state does not redefine resource identity.
7. Registry does not perform scheduling.
8. Core does not depend on vendor APIs.
9. Simulated hardware uses the same core contracts.
10. No global mutable singleton registry is required.

## 15. Test Requirements

The implementation must eventually test:

- Device creation
- Resource creation
- Resource ownership
- Device/resource lookup
- Device removal
- Resource invalidation after device removal
- identifier stability
- duplicate registration rejection
- simulated backend compatibility
- absence of vendor dependencies in core

The tests must not require physical hardware.
