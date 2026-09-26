# GERDOS Core Contract

## Status

Architecture contract v0.2.

This document defines the conceptual boundaries of the GERDOS runtime before
implementation begins.

The central architectural model is:

    Resource Graph
    + Data Residency
    + Topology Graph
    + Operation / Execution separation

GERDOS is an execution runtime, not a model implementation.

---

# 1. Fundamental Rule

GERDOS must remain independent of:

- individual model families
- model architectures
- GPU vendors
- CPU vendors
- accelerator vendors
- quantization formats
- storage vendors
- specific interconnect implementations

The core understands:

- resources
- capabilities
- data
- residency
- topology
- operations
- execution
- measurements
- constraints

Hardware and model-specific knowledge belongs behind explicit boundaries.

---

# 2. Resource Graph

The Resource Graph is the central representation of the execution environment.

A resource represents something that can participate in execution or constrain
execution.

Examples:

- CPU compute capacity
- GPU compute capacity
- VRAM
- HBM
- system RAM
- pinned host memory
- NVMe storage capacity
- transfer engine
- communication/transfer capacity
- accelerator execution engine

Physical connectivity such as PCIe, NVLink, or other interconnect paths is
represented by the Topology model rather than automatically being represented
as a Resource.

The runtime must not assume that a physical device corresponds to exactly one
schedulable resource.

A physical device may expose multiple resources.

Conceptually:

                    Physical Device
                           |
          +----------------+----------------+
          |                |                |
       Compute           Memory          Transfer
       Resource          Resource         Resource

---

# 3. Device

A Device represents a physical or logical hardware ownership boundary.

A device exposes one or more resources and capabilities.

For example:

    GPU0
      |
      +-- compute resource
      +-- VRAM resource
      +-- transfer resource
      +-- synchronization capabilities

A Device is therefore not necessarily the unit being scheduled.

Its resources may be scheduled independently or jointly.

The Device abstraction must not expose vendor-specific API objects to the core.

---

# 4. Resource Types

Initial conceptual resource categories include:

    COMPUTE
    MEMORY
    STORAGE
    TRANSFER
    SYNCHRONIZATION

These categories are extensible.

A resource must have:

- stable identity
- type
- owning device, where applicable
- capacity, where meaningful
- current availability
- capabilities
- measurements
- lifecycle state

Capacity and availability are distinct.

Example:

    VRAM capacity      = 32 GiB
    current available  = 7 GiB

The scheduler must not confuse them.

---

# 5. Compute Resource

A compute resource is a resource through which compatible operations may execute.

Examples:

- CPU execution capacity
- GPU execution units
- tensor engines
- accelerator engines

Compute resources expose capabilities rather than vendor APIs.

Capability information may include:

- supported operation classes
- supported data types
- parallelism
- vector width
- acceleration features
- concurrency
- measured throughput

The runtime must not assume that all compute resources execute all operations
equally.

---

# 6. Memory Resource

A memory resource represents addressable storage participating in execution.

Examples:

- GPU VRAM
- HBM
- system RAM
- pinned host memory
- unified memory
- mapped memory

Relevant properties include:

- capacity
- current occupancy
- bandwidth
- latency
- addressability
- accessibility
- persistence
- coherency
- supported data representations

Memory is not automatically equivalent to cache.

---

# 7. Storage Resource

A storage resource represents persistent or substantially higher-latency storage.

Examples:

- NVMe
- SSD
- persistent memory
- other persistent storage

Relevant properties include:

- capacity
- read bandwidth
- write bandwidth
- random-access characteristics
- latency
- queue depth
- access mechanism

Storage may participate in execution as a lower memory tier.

However:

    Storage != Memory

The distinction remains explicit because latency, persistence, and access
semantics differ.

---

# 8. Topology

Topology describes the physical and logical connectivity through which
resources and other runtime endpoints can communicate or move data.

Topology is not itself a `ResourceKind`.

Examples include:

- PCIe
- NVLink
- Infinity Fabric
- CPU memory fabrics
- storage connectivity
- other accelerator interconnects

A topology link or path may describe:

- source endpoint
- destination endpoint
- directionality
- static bandwidth or capacity
- latency characteristics
- accessibility
- peer-access capability
- transfer mechanisms
- measured effective throughput
- contention characteristics

Physical connectivity such as PCIe or NVLink therefore belongs to Topology,
rather than being represented automatically as an `INTERCONNECT` resource.

A `TRANSFER` resource represents a schedulable mechanism that performs data
movement, such as a DMA engine, copy engine, or storage I/O engine.

Topology describes the connectivity through which that movement can occur.

The runtime must represent actual topology rather than assuming a fixed
machine architecture.

## Topology Identity and Lifetime

Topology relationships have identity independent of the identities of their
endpoints.

Each topology relationship has a `TopologyLinkId`.

`TopologyLinkId` values are unique within their owning topology graph and are
not reused by that graph after a topology link is removed. A future
runtime-level identity authority may establish a broader runtime-wide
identity scope.

A topology link owns its relationship description but does not own its
endpoint Devices, Resources, or other runtime objects.

Topology endpoints contain non-owning runtime identifiers rather than
transferring ownership. The topology graph validates endpoint identifier
structure, but does not own or directly validate the referenced Device or
Resource objects. Endpoint existence and reconciliation with active runtime
objects belong to a future runtime integration layer.

The topology graph owns active topology links. A future runtime-level owner
may own the topology graph itself.

Topology does not own the Device Registry and must not become an alternative
Device or Resource ownership mechanism.

Removing a topology link retires its `TopologyLinkId`.

Removing a Device or Resource invalidates the applicability of topology
relationships that reference it. The mechanism that reconciles topology with
the active Device/Resource registry is a future runtime integration concern;
the topology graph itself must not silently acquire ownership of those
objects.

Topology identity is distinct from measurements. Observed bandwidth,
latency, contention, and throughput must not redefine topology-link identity.

The initial topology implementation is deliberately single-threaded and
requires external synchronization if accessed concurrently. Concurrency
machinery is not part of the topology contract at this stage.

## Enumeration

Runtime-owned collections may expose enumeration of their active objects
without transferring ownership.

Enumeration APIs expose borrowed pointers or references to objects owned by
the corresponding runtime owner. Const owners expose const borrowed access.

Enumeration order is unspecified unless a future contract explicitly defines
an ordering guarantee. Consumers must identify runtime objects by their stable
identifiers rather than by enumeration position.

Enumeration does not create a snapshot. The collection currently being
enumerated must not be structurally modified by the enumeration callback.

Enumeration does not expose or transfer ownership of the underlying
containers.

Device enumeration is owned by the Device Registry. Resource enumeration is
owned by its Device. Topology-link enumeration is owned by the Topology graph.
These enumeration boundaries must not create alternative ownership paths.

Borrowed enumeration results follow the same lifetime rules as other borrowed
runtime access: removal of an object invalidates previously obtained borrowed
access to that object.

## Concurrency and Synchronization

The initial core runtime objects and ownership containers are not internally
synchronized.

Concurrent access to a core object or ownership container therefore requires
external synchronization by the owning runtime or integration layer.

Any operation that may mutate ownership, registration, removal, runtime state,
or topology must be externally synchronized against other accesses that could
observe or modify the affected object or collection.

Read-only access is valid concurrently only when the underlying object or
collection is not concurrently mutated.

A `const` API provides const access semantics but does not constitute a
thread-safety guarantee.

Borrowed pointers and references do not extend object lifetime and must not be
used after the owning object has been removed or destroyed.

Enumeration follows the same synchronization and lifetime rules as direct
lookup. Enumeration must not be treated as a concurrent snapshot.

Topology access follows the same external-synchronization requirement as
Device and Resource ownership access.

The initial contract deliberately does not mandate mutexes, atomics, lock-free
structures, reader-writer locks, or any other particular synchronization
mechanism. Such mechanisms belong to a future runtime or integration layer.

# 9. Transfer Resource

Data movement is itself a schedulable resource.

Examples:

- DMA engine
- copy engine
- storage I/O engine
- host-to-device transfer capability

Data movement may consume or contend for:

- bandwidth
- copy engines
- interconnect capacity
- memory bandwidth
- storage queues

Therefore transfer-related work must remain visible to the execution planner.

---

# 10. Topology Graph

The Topology Graph describes connectivity and communication relationships
between runtime endpoints.

Endpoints may include devices, resources, memory regions, storage resources,
or other addressable execution/runtime entities.

Topology therefore describes how the runtime environment is connected rather
than defining another category of Resource.

Conceptually:

    CPU
     |
    RAM
     |
    PCIe
     |
    GPU
     |
    VRAM

Or:

    GPU0
      |
    NVLink
      |
    GPU1

Or:

    NVMe
      |
    PCIe
      |
    RAM
      |
    PCIe
      |
    GPU

A topology relationship may contain:

- source
- destination
- bandwidth
- latency
- directionality
- accessibility
- transfer mechanism
- peer-access capability
- contention characteristics

Topology is runtime knowledge, not a hardcoded architecture assumption.

---

# 11. Data

Data represents logical material consumed or produced by operations.

Examples:

- tensor
- model weight
- activation
- KV-cache block
- intermediate buffer
- metadata

Logical data identity is independent of physical residency.

For example:

    Weight X

may simultaneously exist as:

    NVMe copy
    RAM copy
    pinned-RAM copy
    VRAM copy

These are physical representations of one logical data object.

---

# 12. Data Residency

Residency describes where physical representations of logical data currently
exist.

GERDOS must distinguish:

    logical identity
    physical location
    validity
    freshness
    accessibility
    representation

Multiple copies may exist.

The runtime must track whether each copy is:

- valid
- stale
- being transferred
- unavailable

The scheduler must be able to reason about the cost of obtaining usable data
at a particular execution resource.

---

# 13. Data Identity and Residency Contract

The implementation separates logical Data identity from physical residency.

A `DataId` identifies one logical Data object. It does not identify a physical
allocation, memory address, storage location, representation, or Device.

Data objects are runtime-owned objects. The owning Data Registry is responsible
for their lifetime and identity management.

A Data object does not own Devices or Resources. It may refer to runtime
resources through non-owning identifiers.

Logical Data identity remains stable while physical representations are
created, transferred, replaced, or removed.

A logical Data object may have multiple simultaneous physical representations.

Each physical representation is represented by a distinct Data Residency
record. A residency record identifies the Resource on which the representation
exists through a non-owning `ResourceRef`.

`ResourceRef` identifies a Resource within its owning Device by combining the
owning `DeviceId` and the `ResourceId`. A bare `ResourceId` is not sufficient
to identify a Resource outside its owning Device scope.

A `DataResidencyId` identifies one residency record and is distinct from
`DataId` and `ResourceId`.

Within one Data object, at most one Data Residency may exist for a given
`(ResourceRef, representation)` pair. Multiple representations may coexist on
the same Resource, and the same representation may exist on different
Resources. A duplicate `(ResourceRef, representation)` pair must be rejected.

Data Residency identifiers are owned by their containing Data object and are
not reused after the corresponding residency record is removed. The residency
identifier namespace is scoped to the containing Data object.

A `DataResidencyRef` identifies a residency within its containing Data object
by combining the owning `DataId` and the `DataResidencyId`. A bare
`DataResidencyId` is not sufficient to identify a residency outside its
containing Data scope.

A `DataResidencyRef` is a non-owning identity reference. It does not transfer
ownership of the Data object or the residency record.

Data Residency records do not transfer or imply ownership of the referenced
Resource.

A residency record describes runtime state associated with one physical
representation, including:

- residency identity
- logical Data identity
- referenced ResourceRef
- representation information
- residency state

The initial residency state model is deliberately small:

- `VALID` — the representation is currently usable
- `STALE` — the representation exists but is not the current usable version
- `TRANSFERRING` — the representation is involved in an in-progress movement
  or update and must not be assumed usable
- `UNAVAILABLE` — the representation exists but cannot currently be used

Residency state changes are constrained by an explicit transition predicate.
Self-transitions are legal and idempotent. The authoritative transitions are:

| From | Allowed destinations |
|---|---|
| `UNAVAILABLE` | `UNAVAILABLE`, `VALID`, `TRANSFERRING` |
| `VALID` | `VALID`, `STALE`, `TRANSFERRING` |
| `STALE` | `STALE`, `TRANSFERRING`, `UNAVAILABLE` |
| `TRANSFERRING` | `TRANSFERRING`, `VALID`, `UNAVAILABLE` |

Any transition not listed above must be rejected without changing the current
state. Transition validity is enforced by the core rather than being left to
callers.

Residency state is distinct from `ResourceAvailability`. Resource availability
describes whether a Resource can accept or support work; residency state
describes the condition of a particular physical representation of Data.

The initial implementation does not define a separate `CREATING` residency
state. Creation of a residency is represented by later operation and execution
semantics rather than by expanding the initial residency state machine.

Representation information is intentionally opaque at the core level. The
core must not introduce model-specific tensor types, datatype enumerations,
quantization formats, or backend allocation handles merely to represent
residency.

Multiple representations of the same logical Data object may coexist.

A representation may become stale without invalidating the logical Data
identity. Removing a physical representation removes that residency; it does
not remove the logical Data object.

A residency that is being transferred must remain distinguishable from a
usable valid residency.

An unavailable or stale residency must not be treated as usable merely because
the residency record still exists.

The initial implementation does not define automatic coherence, eviction,
prefetch, placement, transfer scheduling, or planner policy. Those behaviors
belong to later runtime layers.

Data and residency ownership follows the same external-synchronization and
borrowed-access rules established by the core concurrency contract.

Ownership-registration functions use conditional ownership transfer semantics.
A registration request first validates the supplied object and rejects invalid,
conflicting, or retired identities without transferring ownership. A successful
registration transfers ownership to the containing runtime object.

Callers must not rely on the state of a successfully registered source object
after ownership transfer. Registration failure during pre-transfer validation
does not consume the supplied object.

The core uses `bool` registration results for these ownership-adoption
operations. `true` means that ownership was accepted by the containing object;
`false` means that the object was not registered.

Data identity and residency identity must not be inferred from enumeration
position, ResourceId, DeviceId, memory address, or backend allocation handle.

The core Data abstraction must remain independent of model architecture,
tensor semantics, quantization scheme, storage format, and backend vendor.

---

# 14. Operation Identity and Contract

An `OperationId` identifies one logical Operation. Operation identity is
distinct from Data identity, Data Residency identity, Resource identity,
Device identity, and Execution identity.

An Operation represents declarative executable work. It describes what work
must occur without selecting the physical resources, data residencies,
execution backend, execution queue, or execution attempt.

An Operation may reference logical Data objects through `DataId`. These
references do not identify a physical Data Residency and do not imply any
specific memory, storage, Device, or backend placement.

An Operation may declare input and output Data references.

An Operation may declare dependencies on other Operations through
`OperationId`. A dependency expresses a logical prerequisite relationship and
does not by itself define scheduling, synchronization primitives, queue
selection, or physical execution ordering.

An Operation may declare capability requirements. Capability requirements are
opaque to model architecture and backend implementation at the core level.
The core must not introduce model-specific operation types or vendor-specific
capability identifiers merely to represent an Operation requirement.

An Operation may declare resource requirements. Resource requirements express
what classes or capabilities of runtime resources are required without
selecting a concrete Resource or Device.

An Operation does not own referenced Data, Resources, Devices, or other
runtime objects. References are non-owning identity references.

An Operation does not contain execution state. Running, completed, failed,
cancelled, retried, timing, measurements, and other attempt-specific state
belong to Execution.

An Operation does not contain backend handles, device pointers, memory
addresses, physical placement decisions, transfer paths, scheduler decisions,
or execution timestamps.

An Operation remains logically distinct from each Execution that attempts to
perform it. A single Operation may therefore correspond to zero, one, or
multiple Executions over its lifetime.

Operation identity must not be inferred from enumeration position, DataId,
ResourceId, DeviceId, memory address, backend allocation handle, or execution
attempt.

Operation ownership follows the core ownership and external-synchronization
rules established by this contract.

Examples include:

- matrix multiplication
- attention
- normalization
- reduction
- dequantization
- data transfer
- storage read
- synchronization

An Operation may contain generic execution requirements such as:

- inputs
- outputs
- dependencies
- capability requirements
- resource requirements
- constraints
- estimated cost

An Operation does not select a vendor backend.

---

# 15. Execution

Execution is a concrete runtime instance of an Operation.

Conceptually:

    Operation
        |
        +-- selected resources
        +-- selected data residency
        +-- dependencies
        +-- execution state
        +-- timing
        +-- measurements
        +-- result
        +-- failure state

Therefore:

    Operation = what should happen

    Execution = a particular attempt to make it happen

This distinction is fundamental to scheduling, retries, profiling, and
reproducibility.

The initial core implementation establishes Execution identity, its
association with an Operation, and its lifecycle state only.

The initial Execution object does not yet select concrete Resources or Data
Residencies and does not yet contain backend handles, physical placement,
transfer paths, execution queues, timing, measurements, result payloads, or
failure metadata. Those concerns require subsequent contracts.

A subsequent physical execution binding may be associated with an Execution.
The binding belongs to the Execution and contains non-owning runtime identity
references. It does not transfer ownership of referenced Data, Data Residency,
Resource, Device, or Topology objects.

Physical binding is optional while an Execution is PENDING. Once a physical
binding has been established, it is immutable for the lifetime of that
Execution. An existing Execution must not be rebound to a different physical
realization. If a different physical realization is required, a new Execution
must be created for the same Operation.

Under the subsequent physical binding contract, an Execution entering RUNNING
must have an established physical binding. A PENDING Execution may instead
be cancelled before physical binding is established.

A physical binding records the runtime identities selected for one concrete
execution attempt. It does not cache or assert current runtime availability.
The validity of a binding's identifiers, resolution of those identifiers
against current runtime registries, and current execution admissibility are
distinct concerns.

If a referenced Resource or Data Residency becomes unavailable or is removed
after binding, the historical binding of the Execution is not rewritten.
Execution-time validation determines whether the bound attempt can proceed.
A failed Execution retains the physical binding that was selected for that
attempt.

Physical binding does not alter Operation identity or Data identity. A retry
is represented by a new Execution and may select different physical bindings.

The initial physical binding contract does not encode backend handles, device
addresses, queues, events, vendor-specific objects, or other backend execution
state. Topology paths are also not part of the initial physical binding
contract; topology remains a separate description of connectivity and path
constraints.

The core does not yet mandate separate binding types for computation,
movement, storage, or synchronization. Such role-specific semantics require
later contracts. In particular, a transfer Resource identifies a runtime
resource through which movement may execute; it is not itself the movement
operation.

Physical bindings contain two distinct role domains: Data binding
roles and Resource binding roles. A Data binding role refers to a
Data Residency through a `DataResidencyRef`. A Resource binding role refers
to a Resource through a `ResourceRef`. A role from one domain must not be used
to classify a reference from the other domain.

The initial Data binding roles are `INPUT`, `OUTPUT`, `SOURCE`, and
`DESTINATION`. `INPUT` and `OUTPUT` describe data representations consumed or
produced by an execution. `SOURCE` and `DESTINATION` describe directional
relationships involving physical data representations. The Operation contract
determines the meaning of that relationship. These role meanings are not
interchangeable and do not by themselves determine the kind of Operation being
executed.

The initial Resource binding roles are `COMPUTE` and `TRANSFER`. `COMPUTE`
identifies a Resource through which computation executes. `TRANSFER`
identifies a Resource through which movement executes. Resource kinds remain
defined by the Resource contract and must not be duplicated unnecessarily as
binding roles.

A physical binding may contain multiple references with the same role. A role
does not imply uniqueness, cardinality, or exclusivity. Whether a particular
combination or repetition of roles is semantically valid is determined by the
Operation and its execution contract, not by the physical binding identity
structure alone.

Physical binding validation is separated into structural validity, runtime
resolution, and semantic execution admissibility. Structural validity
establishes that the binding contains well-formed role/reference pairs.
Runtime resolution establishes whether the referenced runtime identities can
currently be resolved. Semantic execution admissibility establishes whether
the resolved binding is appropriate for the Operation and current execution
conditions. These are distinct checks and must not be collapsed into one
binding-validity state.

A physical binding does not determine the kind or semantics of the Operation.
The Operation establishes the logical work being performed, while the
physical binding describes one concrete physical realization of that work.

Execution state transitions are explicit and must be validated by the
Execution state machine. An Execution may transition from PENDING to RUNNING,
and from RUNNING to COMPLETED, FAILED, or CANCELLED. PENDING may transition
directly to CANCELLED. State-preserving transitions are permitted. Terminal
states are not resurrected into non-terminal states.

A retry is represented by a new Execution associated with the same Operation;
retry policy and retry limits are outside the initial Execution contract.

---

# 16. Workload

A Workload describes the computation submitted to GERDOS.

A workload may contain:

- operations
- data
- dependencies
- constraints
- performance requirements
- quality requirements

A model is one possible producer of a workload.

GERDOS must not require workloads to originate from models.

---

# 17. Model Boundary

Models exist outside the core runtime.

Conceptually:

    Model Adapter
         |
         v
      Workload
         |
         v
    GERDOS Core

A model adapter may understand:

- model architecture
- tensor layout
- graph structure
- weight format
- quantization
- model-specific execution constraints

The core must not require this knowledge.

Therefore:

    scheduler -> model architecture

is forbidden.

While:

    model adapter -> generic workload

is allowed.

---

# 18. Backend Boundary

Backends translate generic execution requirements into hardware-specific
operations.

Conceptually:

    Core Runtime
         |
    Backend Interface
         |
    +----+--------+--------+
    |             |        |
   CPU           GPU    Storage
    |             |        |
 implementation implementation

The core requests capabilities.

The backend supplies implementation.

The core must not directly depend on:

- CUDA
- HIP
- ROCm
- SYCL
- vendor runtime handles
- vendor-specific synchronization objects

Those belong behind backend boundaries.

---

# 18. Execution Planning

The planner receives:

    Workload
    Resource Graph
    Topology Graph
    Data Residency
    Measurements
    Constraints

The planner produces an execution plan.

Conceptually:

    Workload
        |
        v
      Planner
        |
        +-- resource selection
        +-- data placement
        +-- transfer planning
        +-- operation ordering
        +-- synchronization
        +-- overlap opportunities
        |
        v
    Execution Plan

The planner must be able to consider computation and movement together.

---

# 19. Computation and Movement

Computation and movement are separate but interacting resources.

For example:

    Compute A
       +
    NVMe -> RAM
       +
    RAM -> VRAM

may execute concurrently if dependencies and resource constraints allow it.

GERDOS should support:

- asynchronous transfers
- prefetching
- double buffering
- pipelining
- eviction
- overlap
- demand loading

These are execution mechanisms, not assumptions about a particular model.

---

# 20. Resource Capacity and Performance

GERDOS must distinguish:

    capacity
    availability
    bandwidth
    latency
    throughput
    contention
    utilization

For example:

    theoretical PCIe bandwidth
        !=
    measured effective transfer bandwidth

Similarly:

    GPU theoretical FLOPS
        !=
    measured operation throughput

Measurements must be represented independently from static specifications.

---

# 21. Measurement

Measurement represents observed system behavior.

Examples:

- memory bandwidth
- transfer bandwidth
- transfer latency
- compute throughput
- operation duration
- cache hit rate
- queue latency
- storage throughput
- utilization

Measurements should retain sufficient context to establish:

- what was measured
- where it was measured
- under what conditions
- when it was measured
- measurement confidence

The scheduler should increasingly prefer measured behavior when reliable data
exists.

---

# 22. Resource Graph vs Topology Graph

These concepts are related but distinct.

The Resource Graph describes:

    what resources exist

The Topology Graph describes:

    how resources relate and communicate

A runtime may therefore know:

    Resource:
        GPU0.VRAM

    Resource:
        GPU1.VRAM

    Relationship:
        GPU0.VRAM -> GPU1.VRAM
        mechanism = NVLink
        bandwidth = measured value

The separation allows topology to change without redefining resource identity.

---

# 23. No Hidden Global State

Core runtime state must have explicit ownership and lifetime.

Avoid hidden mutable process-wide state.

This is necessary for:

- multiple runtime instances
- testing
- reproducibility
- servers
- isolation
- debugging

---

# 24. No Vendor Leakage

Forbidden:

    Core -> CUDA
    Core -> ROCm
    Core -> SYCL
    Core -> vendor-specific API

Allowed:

    Core -> generic backend interface
    Backend -> vendor implementation

---

# 25. No Model Leakage

Forbidden:

    Core -> MiMo
    Core -> Qwen
    Core -> DeepSeek
    Core -> expert count
    Core -> transformer layer count
    Core -> attention-head assumptions

Allowed:

    Model Adapter -> generic Workload

---

# 26. Dependency Direction

The intended dependency direction is:

    Public API
         |
      Runtime
         |
    Workload / Execution
         |
    Resource / Data / Topology
         |
    Backend Interfaces
         |
    Backend Implementations

Model adapters feed workloads into the runtime.

They do not redefine the runtime's fundamental abstractions.

---

# 27. Architectural Test

Before introducing a subsystem, ask:

1. Is this a generic runtime concept?
2. Can it exist without a specific model?
3. Can it exist without a specific vendor?
4. Can it be tested without physical hardware?
5. Does it represent state, capability, or an operation?
6. Does it belong in the core or an adapter?
7. What measurable behavior justifies it?
8. What dependency direction does it introduce?

If these questions cannot be answered, the abstraction is not ready.

---

# 28. Initial Non-Goals

The following are deliberately not part of the first implementation:

- model loader
- tokenizer
- CUDA backend
- ROCm backend
- SYCL backend
- quantization implementation
- MoE implementation
- attention implementation
- KV cache
- speculative decoding
- serving API
- distributed execution
- NVMe expert streaming

These will be introduced only after the core contracts are established.

---

# 29. Architectural Objective

The resulting runtime should be capable of representing an execution environment
such as:

                  GERDOS
                    |
        +-----------+-----------+
        |                       |
   Resource Graph           Topology
        |                       |
   +----+----+----+----+    PCIe / NVLink
   |    |    |    |    |
Compute Memory Storage Transfer Synchronization
   |      |      |      |
  CPU  RAM/VRAM   NVMe   DMA / Copy

while allowing a workload to move through that environment without the core
being aware of which model, vendor, accelerator generation, or storage device
produced the workload.

This is the foundation required for GERDOS to dynamically exploit heterogeneous
hardware rather than merely implementing another fixed offload strategy.
