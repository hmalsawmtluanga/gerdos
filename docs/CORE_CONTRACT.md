# GERDOS Core Contract

## Status

Architecture contract v0.3.

This document defines the conceptual boundaries of the GERDOS runtime. The core
implementation currently establishes identity, ownership, resource and data
residency state, topology identity, operations, executions, physical binding,
structural validation, runtime resolution, semantic admissibility, execution
admission, execution effects, execution results, the backend execution
interface with its simulated implementation, runtime integration, and
measurement evidence.

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

Resource availability is represented by `ResourceAvailability`:

- `AVAILABLE` — the resource may currently accept or support work.
- `UNAVAILABLE` — the resource exists but cannot currently accept or support work.
- `INITIALIZING` — the resource is not yet available for work.
- `DRAINING` — the resource is being withdrawn and is not available for new work.
- `FAILED` — the resource is not currently available because it has entered a failure state.

`Resource::set_availability()` currently records the runtime availability state without imposing a core-level transition table. In particular, recovery semantics for `FAILED` and the operational conditions governing transitions between availability states are not yet defined by the core contract. Those lifecycle and recovery policies belong to a later runtime layer.

Availability is not part of resource identity. A resource may become unavailable while remaining the same resource, and references to it remain structurally valid. Resolution therefore does not imply current availability.

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

A topology link declares:

- source endpoint
- destination endpoint
- directionality
- static declared attributes: bandwidth and latency, where zero means
  uncharacterized

Static attributes are declared expectations, never evidence. Observed link
behavior — measured effective throughput, latency, and contention — belongs
to Measurement evidence through a future measurement subject generalization,
and never redefines link identity. Runtime capacity accounting is derived,
not stored: the capacity a link has committed is computed from the in-flight
attempts that traverse it, and belongs to a future planning and execution
accounting layer with visibility into in-flight attempts; the version-zero
planner performs no capacity accounting. Topology links carry no mutable
runtime state.

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
describes whether a Resource can accept or support work; residency state describes
the condition of a particular physical representation of Data.

`DataResidency::usable()` reports only residency-state usability: it is true when
the residency state is `VALID`. It does not establish that the Resource referenced
by that residency is currently available. Effective runtime usability therefore
requires both a usable residency state and an available referenced Resource; that
combined evaluation is performed by the execution-admission gate.

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
`false` means that the object was not registered. Registration functions that
take their argument by value consume it unconditionally; registration
functions that take an rvalue reference consume it only on acceptance.

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

An Operation may declare input and output Data references. Every stored
`DataId` reference must be a valid identity. The referenced Data object need
not already exist when the Operation is registered; reference validity and
object existence are separate concerns.

An Operation may declare dependencies on other Operations through
`OperationId`. Every stored dependency reference must be a valid identity. A
referenced Operation need not already exist when the Operation is registered;
forward references are permitted. A dependency expresses a logical
prerequisite relationship and does not by itself define scheduling,
synchronization primitives, queue selection, or physical execution ordering.
Dependency graphs are not validated at registration: cycles and
self-references are representable today, and graph validation and ordering
semantics belong to a future planning contract.

An Operation may declare capability requirements. Capability requirements are
opaque to model architecture and backend implementation at the core level.
The core must not introduce model-specific operation types or vendor-specific
capability identifiers merely to represent an Operation requirement.

An Operation may declare resource requirements. Resource requirements express
what runtime participation the work needs — as resource binding roles and
minimum counts — without selecting a concrete Resource or Device.

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

## Operation Requirements

An Operation declares its requirements; it does not classify itself. There is
no operation-kind taxonomy in the core. What an operation needs is expressed
only through its declared data references, its declared resource requirements,
and — in the future — capability requirements.

Data requirements are derived from the declared data references:

- every `DataId` declared as an input must be bound in a consuming role
  (`INPUT` or `SOURCE`) by an admissible physical realization
- every `DataId` declared as an output must be bound in a producing role
  (`OUTPUT` or `DESTINATION`) by an admissible physical realization
- a `DataId` may be declared as both input and output; an admissible physical
  realization then binds it in both a consuming and a producing role

Resource requirements declare required runtime participation:

- a resource requirement names a resource binding role and a minimum number
  of resource entries of that role
- an admissible physical realization binds at least the required number of
  resources in each required role
- at most one requirement exists per resource binding role; a requirement must
  have a minimum of at least one
- resources bound beyond the declared requirements are permitted
- an operation with no resource requirements places no resource-role demand

Resource requirements speak the current resource binding role domain and grow
with it. They do not express resource kinds, vendor features, placement, or
model semantics.

Capability requirements are reserved as a future extension of this requirement
model. They will be introduced only together with defined capability semantics
on Resources; the core does not represent capabilities as free-form strings.

Requirements are declarative. They do not select placement, order execution,
imply dependencies, or encode scheduling or planner policy.

Semantic execution admissibility evaluates a physical binding against these
requirements.

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
association with an Operation, and its lifecycle state only. The associated
`OperationId` must be a valid identity, but the referenced Operation need not
already exist at Execution registration time.

The initial Execution object does not yet select concrete Resources or Data
Residencies and does not yet contain backend handles, physical placement,
transfer paths, execution queues, timing, measurements, or result payloads.
Attempt outcomes are recorded through the execution result.

A subsequent physical execution binding may be associated with an Execution.
The binding belongs to the Execution and contains non-owning runtime identity
references. It does not transfer ownership of referenced Data, Data Residency,
Resource, Device, or Topology objects.

Physical binding is optional while an Execution is PENDING. Once a physical
binding has been established, it is immutable for the lifetime of that
Execution. An existing Execution must not be rebound to a different physical
realization. If a different physical realization is required, a new Execution
must be created for the same Operation.

Establishing a physical binding consumes the supplied binding value. If
establishment is rejected, the supplied value is discarded; it is not retained
or restored. Callers must not rely on the state of the supplied binding value
after the call.

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
Execution-time validation, currently established by the execution-admission
gate, determines whether the bound attempt can proceed. A failed Execution
retains the physical binding that was selected for that attempt.

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

## Execution Admission

The execution-admission gate establishes whether an Execution and its
established physical binding form an executable physical realization under the
current runtime state. It composes structural validity, runtime resolution, and
current usability. It does not evaluate semantic execution admissibility.

Admission requires all of the following:

- the Execution is `PENDING`
- a physical binding has been established
- the binding is non-empty: at least one data or resource entry exists
- the binding is structurally valid
- every binding reference resolves against the current registries
- every referenced Resource is currently available
- every data residency referenced in a consuming role (`INPUT`, `SOURCE`) is
  usable

A data residency referenced in a producing role (`OUTPUT`, `DESTINATION`) is
not required to be usable: a representation that is about to be created or
rewritten is not yet usable. Its referenced Resource must still be available.

Effective residency usability is evaluated as defined by the Data Residency
contract: a usable residency state and an available referenced Resource.

The admission verdict is observational. It does not mutate registries, the
Execution, its binding, or runtime state. On success it produces an
execution-scoped evidence token that identifies the admitted Execution; only
this gate can produce such evidence. Recording the evidence on the attempt —
which enables its `RUNNING` transition — is a separate step performed at the
commit point, after the backend has accepted the work.

An Execution may transition from `PENDING` to `RUNNING` only after admission has
been established for it. The state machine's binding requirement is structural:
a binding value has been established. Admission is stricter and defines the
executable physical realization. Both requirements apply.

Admission is mechanically enforced, not advisory: the `RUNNING` transition
is rejected unless admission evidence has been recorded on the attempt, and
that record can be established only from admission-gate evidence, for that
attempt, exactly once. Rejected admission never establishes evidence, and a
begin rejected before the commit point leaves no evidence behind.

Admission also enforces role and kind coherence: a `COMPUTE` resource binding
must reference a compute resource, and a `TRANSFER` resource binding must
reference a transfer resource. Memory and storage resources are places,
reached through residencies, and are never bound as mechanisms.

Admission evidence is established against the runtime state observed at
admission time. If a referenced object is removed or its runtime state changes
after admission, the evidence is not rewritten; re-evaluation is obtained by
requesting admission again. Concurrent mutation remains subject to the core
external-synchronization rules.

Semantic execution admissibility — whether the resolved binding is appropriate
for the Operation being attempted — is established separately by the
binding-admissibility gate. Admission does not inspect Operation semantics,
topology paths, measurements, or planner state.

## Semantic Execution Admissibility

Semantic execution admissibility establishes whether a physical binding is an
appropriate realization of a specific Operation. It evaluates the Operation's
declared requirements against the binding. It does not consult registries,
runtime state, topology, measurements, or planner policy.

A physical binding is semantically admissible for an Operation when:

- every data binding in a consuming role (`INPUT`, `SOURCE`) references a
  `DataId` that the Operation declares as an input
- every data binding in a producing role (`OUTPUT`, `DESTINATION`) references
  a `DataId` that the Operation declares as an output
- every declared input `DataId` is referenced by at least one consuming data
  binding
- every declared output `DataId` is referenced by at least one producing data
  binding
- for every declared resource requirement, the binding contains at least the
  required number of distinct resources in the required role; one resource
  listed twice does not satisfy a minimum of two. A resource bound in two
  roles is structurally well-formed but cannot pass admission: mechanism
  coherence pins one resource to one role, so it never satisfies two
  requirements in practice

A binding containing a resource role value outside the binding role domains
is inadmissible.

A binding containing a role value outside the binding role domains is
inadmissible. Bindings beyond the declared coverage and requirements are
permitted: additional data bindings and additional resource bindings do not
make a binding inadmissible when the rules above hold.

Semantic execution admissibility is distinct from structural validity, runtime
resolution, and execution admission. Structural validity establishes
well-formed role/reference pairs. Runtime resolution establishes that the
referenced identities currently exist. Execution admission establishes that
the bound attempt is executable under current runtime state. Semantic
execution admissibility establishes that the realization matches the declared
work. These checks compose and must not be collapsed into one binding-validity
state.

Semantic execution admissibility does not select placement, order execution,
encode retry or scheduling policy, or classify the Operation.

## Execution Result

A terminal Execution records one result. The result identifies the outcome of
the attempt, records whether the attempt's execution effects were applied
coherently, and may carry an opaque non-semantic diagnostic. The result is
recorded once and is immutable afterwards.

- a result is recorded only while the Execution is in a terminal state
- the recorded outcome must equal the Execution's terminal state
- at most one result exists per Execution

An attempt whose finishing effects were rejected is recorded as
`EFFECTS_REJECTED` rather than silently presented as coherent: its result and
its residency state may then disagree, and the disagreement is part of the
attempt's history. A rejected effects verdict is never discarded; it is
reported through the result and through the runtime integration layer's
completion reporting.

The result does not carry measurements; timing, duration, and throughput are
Measurement concerns. The result does not carry output data; produced and
rewritten representations are governed by Data Residency and the execution
effects rules. The result does not carry backend state.

A failed Execution retains its physical binding and its result as the history
of that attempt. Result recording is independent of retry policy: a retry is a
new Execution with its own result.

The runtime effects of an attempt — the residency state changes applied
through its binding roles as the attempt starts, completes, or fails — are
defined by the execution effects rules.

## Execution Effects

The execution effects rules define the runtime state changes an attempt applies
to the physical representations it produces. Effects are expressed through the
binding roles of the attempt and apply only to data bindings in producing roles
(`OUTPUT`, `DESTINATION`). Consuming bindings are not modified by the attempt.

The effects lifecycle is paired to the attempt:

- when the attempt starts, every producing residency enters the
  update-in-progress state (`TRANSFERRING`); the representation must not be
  assumed usable while it is being created or rewritten
- when the attempt completes, every producing residency becomes `VALID`
- when the attempt fails or is cancelled after it started, every producing
  residency becomes `UNAVAILABLE`

Failure and cancellation effects are deliberately pessimistic: after an
interrupted rewrite the runtime cannot assert that the previous representation
is intact, so the residency is recorded as existing but not usable.

Effects application is transactional: a call either applies all of its state
changes or leaves every residency unchanged and reports rejection. Starting
effects are rejected when the attempt is already terminal, when a producing
residency cannot be resolved, when another live attempt holds a claim on a
producing residency, or when a binding role value lies outside the binding
role domains. Finishing effects are rejected when the attempt is not
terminal, when a producing residency cannot be resolved, when a producing
residency is not in the update-in-progress state — the update did not start or
has already finished — when the claim is held by another attempt, or when a
binding role value lies outside the binding domain. Starting effects are
idempotent for the claiming attempt; finishing effects are single-shot.

Effects do not apply to attempts that never started: a cancelled PENDING
attempt has not touched any representation.

Effects are applied by the runtime integration layer in the intended
sequence: admissibility, admission, backend submission, start effects,
`RUNNING`, backend work, terminal transition, finish effects, result
recording, measurement capture. Backend submission precedes the start effects
and the state transition so that rejection stays atomic; backend work is
therefore in flight briefly before its producing representations are marked
update-in-progress, within one uninterruptible integration step under the
external-synchronization rules. Effects do not record results, do not change
execution state, and do not consult availability, admissibility, topology, or
measurements.

An update-in-progress is owned by exactly one attempt. Starting effects
claim the attempt's producing residencies and are rejected while another
live attempt holds a claim on one of them; repeated starting effects for the
claiming attempt itself remain idempotent. Finishing effects apply only to
residencies claimed by the attempt. A claim held by a terminal attempt is
stale and may be claimed by a later attempt, whose update then governs the
representation. Leaving the update-in-progress state releases the claim.

## Representation Movement and Lifetime

Movement is expressed by binding one logical Data object in a consuming role
at a source residency and in a producing role at a destination residency. The
execution effects rules then treat the destination as being created or
rewritten while the source representation remains usable; a movement is
therefore a copy at the effects level.

Whether a particular Operation also invalidates or removes its source
representation is part of that Operation's semantics and requires later
contracts. The effects rules do not infer it from the binding roles.

Representation records are destroyed only by explicit removal from their
containing Data object. Runtime state changes never destroy records, and a
record's identity is retired on removal as defined by the Data Residency
contract. Eviction — deciding when a representation should no longer be
retained — is future policy and is not part of execution effects.

## Backend Execution Interface

The backend execution interface is the single seam through which the core
runtime causes work to happen. Backends receive the work to perform — the
Operation of an attempt and the attempt's established physical binding — and
report when attempts complete.

- submitting an attempt begins backend-side work; a backend may reject the
  attempt
- polling appends attempts completed since the previous call; polling never
  blocks the runtime on hardware
- a completion reports the attempt identity, whether the work succeeded, and
  the observed duration of the work; the observed duration is evidence
  measured where the work happened. Terminal state transitions, execution
  effects, result recording, and measurement capture remain the
  responsibility of the runtime integration layer

Backend-specific state — vendor handles, queues, streams, events, device
pointers, and addresses — remains behind this interface and must never appear
in core types.

The simulated backend implements this interface with a deterministic
execution model so that the complete runtime — admission, effects, lifecycle,
and overlap — can be exercised without physical accelerator hardware.

Operation work descriptions — the generic semantics a real backend needs in
order to perform computation — require a later contract. Until then, backends
perform abstract work.

## Binding Planning

Binding planning produces one physical binding for an attempt of an Operation
from the current runtime state. Planning is deterministic and fail-closed: it
returns no binding rather than one that cannot pass structural validation,
runtime resolution, semantic admissibility, and execution admission.

Version-zero selection rules:

- a declared input binds a consuming entry; a declared output binds a
  producing entry; data declared as both consumes one record and produces to
  a distinct reachable record (movement: `SOURCE`/`DESTINATION`), falling
  back to in-place realization (`INPUT`/`OUTPUT`) only when the consuming
  record is the Data's sole representation; with sibling records that are
  unreachable or busy, planning fails loudly
- consuming entries use usable records with available hosts, lowest
  identifiers first
- producing entries prefer distinct unclaimed records with available hosts,
  reachable from the consuming record by a covering topology link and ranked
  by that link's declared bandwidth, then latency, then identifier order
- resource requirements are satisfied by distinct, available mechanisms whose
  kind matches the required role; movement transfer mechanisms must sit on
  the devices at the ends of a covering link — link endpoints are the data
  locations, and the engines that drive hops live on those devices — with the
  same declared-attribute preference
- movement between distinct records requires a covering topology link

Claimed producing records are never planned onto. Planning does not order
work, resolve dependencies, produce multi-attempt plans, or reserve capacity;
those are later planning concerns.

Measurement-informed preference is discovery before exploitation, bounded
by an exploration budget. While the evidence log is small, a mechanism with
no successful evidence is sampled ahead of measured ones, so new and
never-tried mechanisms are not starved by a first impression. Beyond the
budget, measured behavior outranks unmeasured mechanisms entirely: later
arrivals wait for a future exploration policy rather than starving proven
engines. Once measured, candidates rank by mean observed successful
duration: measured behavior outranks declared attributes, and faster
outranks slower. Failure evidence is not speed evidence, and saturated
totals cannot yield a meaningful mean — both leave a mechanism in the
unmeasured class. The evidence scope is the mechanism's proven behavior
across operations; means pool work shapes, so mechanisms must not be
compared across divergent work shapes in this version. Work-shape-scoped
comparison is a future refinement of the requirement model. Record and link
selection remain declared-attribute based until measurement subjects
generalize beyond mechanisms.

## Real Backends: CPU

The CPU backend is the first backend performing real work on real hardware.
It owns real allocations for the representations it touches — the core's
data plane lives behind the seam — and executes real kernels over them:
movement pairs are copied between representations, compute pairs run
arithmetic between them. Observed durations are real wall-clock nanoseconds
measured where the work happens, and flow through the same completion and
evidence path as every other backend.

Work runs on real background threads and poll never blocks. Allocations are
resolved at submission; worker threads touch only the buffers they are
given. A backend failure is not an incoherence: the attempt's effects apply
and its completion is reported with its outcome.

## Operation Work Descriptions

An Operation declares its computation as a work description: generic
data-parallel semantics over the bound representations. The elementwise
form transforms each producing representation toward `dst *
destination_scale + source_scale * src + constant`, repeated `passes`
times over `elements` float elements. Exact copying is the destination_scale
0, source_scale 1, constant 0, passes 1 case; a producing representation
without a consuming source is transformed without the source term. The
consuming record of the same Data supplies the source — or, for compute
shapes, the first consuming entry.

Two further elementwise forms complete the normalization vocabulary. The
exponential form applies the same affine wrapper around the natural
exponential: `dst * destination_scale + source_scale * exp(src) +
constant`, repeated `passes` times. The maximum-reduction form folds an
operand into one element: `dst[0] = dst[0] * destination_scale +
source_scale * max(src) + constant`, repeated `passes` times. Both forms
apply elementwise to every producing entry; the maximum-reduction form
writes only its first element and leaves the rest alone, exactly as the
sum-reduction form does.

Comparison and selection close the remaining gap the roadmap names. The
minimum-reduction form mirrors the maximum: `dst[0] = dst[0] *
destination_scale + source_scale * min(src) + constant`. Two elementwise
selection forms take the first two consuming entries as operands A and B:
`dst * destination_scale + source_scale * min(A, B) + constant` and the
same wrapper around `max(A, B)`. A three-operand selection form takes the
first three consuming entries as predicate P, value A, and value B:
`dst * destination_scale + source_scale * (P != 0 ? A : B) + constant`.
A two-operand gather form takes the first two consuming entries as the
value table V and the index table I: `dst * destination_scale +
source_scale * V[clamp(I)] + constant`, where each index is truncated
toward zero and clamped into the table range, so hostile index content
can never read out of bounds. All comparison and selection forms apply
elementwise to every producing entry; the minimum-reduction form writes
only its first element, exactly as the other reductions do.

The work description is parameters, not kinds: there is no operation type
anywhere in the core, and no gate inspects the work. Structural validation,
runtime resolution, semantic admissibility, and execution admission judge an
identical realization identically regardless of the work it declares —
pinned by test. Backends interpret the work at the seam; extending the
algebra changes no gate.

Well-formed work is executable work: the seam rejects work descriptions
with zero elements, zero passes, or sizing that cannot be allocated —
hostile arithmetic must never reach an allocation or a kernel, and a
vacuous success is not speed evidence. Matrix shapes additionally require
non-zero rows, inner, and columns with products that cannot overflow.
Identity registration accepts any declared work because registration is not
execution.

When the source is the destination record, the transform is defined as the
iterated form: each pass composes over the previous result. This holds for
the single-source elementwise forms, including the exponential: an
in-place pass reads the value the previous pass wrote. All three
reduction forms are also iterated: each pass re-folds the unchanged
source into the running first element. Multi-operand selections
(elementwise min/max, predicate selection, gather) require operand Data
records distinct from the destination Data: an attempt that aliases an
operand Data with its destination Data writes nothing, exactly as a
compute shape with fewer than two consuming entries writes nothing
rather than fabricating an operand. Distinct records of the same Data
still alias — the same-Data consuming entry is the source by definition.

Model adapters translate model semantics into this vocabulary; the core
never sees model terms. The element type is declared per work
description, one of three generic dtypes: `F32` (the default — every
existing construction site keeps its meaning), `F16`, or `I8`. Dtype is
a parameter beside form, never a form split: no gate inspects it, and
extending the dtype set changes no gate. Storage sizing counts elements
dtype-agnostically, but byte sizing multiplies by the dtype width with
the same overflow guard as element counts — combinations whose byte
product would overflow are seam-rejected before any allocation (width
1 cannot overflow, so SIZE_MAX I8 elements are well-formed by the
rule). Allocations keep their own dtype across attempts: a record
touched by one dtype then another keeps its storage and converts per
operation in F32 — retargeting never wipes an operand. Only a missing
record or an element-count mismatch reinitializes (fresh storage takes
the work's dtype). Mixed-dtype
attempts convert explicitly at the seam with round-half-away-from-zero
semantics, verified bit-exactly on values both dtypes represent
exactly; there is no silent reinterpretation. Every (form, dtype) pair
executes on CPU loops and accelerator kernels with identical semantics —
the CPU is the reference; I8 compares exact, F16 compares within a
documented tolerance against host float. Conversion is structural, not
per-kernel: both engines compute in F32 and convert once per direction
at the home boundary with shared helpers, so engine pairs cannot
disagree on conversion — only on F32 arithmetic in the last ulp, which
is printed and tolerance-documented. Dtype
names are generic
vocabulary allowed in the core; vendor-intrinsic spellings stay behind
the backend seam. The algebra grows by declared forms as backends learn
them — forms are backend-interpreted content, and no gate inspects
which form is declared.

## Model Adapters

The adapter layer is the documented model boundary: the one place model
terms may exist. Adapters translate model-semantic descriptions into the
core's generic work vocabulary, and translation is fail-closed — a step the
algebra cannot express exactly is refused by name with its reason, never
approximated silently. The leakage guard enforces the boundary: model
vocabulary is forbidden everywhere except the adapter layer, and platform
vocabulary is forbidden in the core.

## Real Backends: CPU

## Real Backends: Heterogeneous Routing

One backend seam may execute an attempt's work on more than one physical
engine. The engine is chosen by the devices that own the attempt's bound
mechanisms: attempts binding accelerator mechanisms execute real OpenCL
kernels on that device, all other attempts execute the CPU engine. Which
devices are accelerators is backend configuration, never vocabulary.

The data plane is home-respecting: every representation has exactly one
real allocation, living where its residency record says it lives — host
memory for host-homed records, device memory for accelerator-homed ones.
Work crossing homes performs real staging exactly as transfer engines
mediate between host and device memory; movement is engine-independent and
follows where the data lives.

Kernel semantics are engine-independent: movement pairs are copied between
representations, and compute pairs run the same arithmetic on either engine,
reading the consuming record of the same Data — or, for compute shapes, the
first consuming entry. An unpaired producing representation is written in
place. Durations are real wall-clock nanoseconds measured where the work
happens, on real background threads, and poll never blocks.

The heterogeneous backend test requires an OpenCL toolchain and device; it
is registered at configure time when the toolchain is found. All other tests
remain accelerator-free.

Comparative claims on real hardware are established by structural assertions
— which engines actually ran — plus wall-clock comparison with the ratio
printed as evidence. Benchmarks print before asserting, so no aborted
assertion can hide its numbers. Work shapes are chosen where the engines
genuinely diverge; where hardware is equal, evidence shows equality, and the
claimed speedup belongs to the machine, never to the contract.

## Runtime Integration

The runtime integration layer — the executor — is the component that
advances an attempt through its lifecycle. The `RUNNING` transition is
mechanically gated on admission evidence; direct structural transitions
outside the executor remain possible but cannot start an unadmitted
attempt, and any bypass that skips finishing effects leaves a stale update
claim for a later attempt to take over. It composes the established gates
as mandatory steps rather than optional advice:

    semantic admissibility
        -> execution admission
        -> update-claim arbitration
        -> backend submission
        -> admission evidence recording
        -> start effects
        -> RUNNING
        -> backend work
        -> terminal transition
        -> finish effects
        -> result recording
        -> measurement capture

Beginning an attempt is rejection-atomic: unless the backend accepts the
attempt, no gate verdict — including admission evidence — residency state,
execution state, or result is changed. Given recorded admission evidence and
free update claims, the RUNNING transition cannot be rejected.

Completion is applied in the fixed sequence: terminal transition, finish
effects, result recording, measurement capture. Completions for attempts that
are not `RUNNING` are discarded. Every applied completion is reported with
its integrity: a completion whose finishing effects were rejected is reported
as incoherent and its result is recorded as `EFFECTS_REJECTED`; nothing in the
completion path discards an effects verdict.

Cancellation of an attempt is performed by the executor: a `PENDING` attempt
is cancelled without effects, and an in-flight attempt receives the
failed-or-cancelled finishing effects before its result is recorded.

Runtime object removal is preconditioned on lifecycle state:

- an Execution may be removed only in a terminal state; an in-flight attempt
  must be cancelled first
- a Data Residency whose update is claimed by an attempt cannot be removed;
  the claim must be resolved or released first. An update-in-progress state
  without a claim does not protect the record
- a Data object cannot be removed while one of its residencies holds a
  claimed update

These preconditions keep the finishing effects applicable: removing the
objects an attempt must finalize would strand their state silently.

The executor holds no scheduling, placement, or retry policy: it advances
attempts that have already been given a physical binding. Choosing bindings
and ordering work belongs to planning.

---

# 16. Workload

A Workload describes the computation submitted to GERDOS.

At the core level a workload is a named stream of work units: operation
declarations carrying generic data references and resource requirements.
Model adapters produce workloads; the core consumes them without knowing
their origin, and no model vocabulary appears in the workload shape. The
ordering of the stream belongs to the submitting layer until the planning
contract owns dependency ordering. Repetition, streaming, and staged
workloads are expressed by more units, not by new core concepts.

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

# 19. Execution Planning

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

# 20. Computation and Movement

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

# 21. Resource Capacity and Performance

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

# 22. Measurement

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

## Measurement Records

A measurement record is evidence of one observation. Records are append-only
and immutable: the measurement registry exposes no removal or mutation of
recorded evidence.

A record establishes:

- what was measured — the measured quantity, the attempted Operation, the
  attempt it belongs to, and whether the attempt's work succeeded
- where it was measured — the observed Resource
- under what conditions — the measurement conditions, currently the number
  of attempts running concurrently with the subject attempt: the peers
  sharing the runtime when its work finished, sampled once per completion
  batch so co-completing attempts observe identical conditions
- when it was measured — the record's position in observation order
- measurement confidence — the number and recency of supporting
  observations, reported by queries rather than stored per record; queries
  count successful evidence separately from time-to-failure evidence

Queries accumulate values with saturating arithmetic; totals never wrap.
Evidence is retained without limit; retention policy is future work.

The observed Resource is the subject form of this contract. Observations of
topology links or other entities require a future subject generalization and
are not representable today.

`MeasurementId` values are allocated by the owning Measurement Registry in
observation order and are never reused; identifier order is observation order.
Measurement identity is distinct from all other runtime identity.

The first measured quantity is the observed duration of one execution attempt,
expressed in nanoseconds. Further quantities — such as byte counts — are
introduced together with the producers that can measure them. Static
specifications never appear as measurement records.

Measurement capture observes completed attempts: one duration observation is
recorded for each distinct Resource named by the attempt's resource bindings
— the mechanisms the attempt used — carrying the Operation that was
attempted, the attempt identity and outcome, and the conditions observed at
completion. Resources that merely host a bound residency are not subjects of
this observation. The observed duration is the whole attempt's duration,
repeated per subject; summing values across subjects therefore double-counts
and must not be read as total work. The backend execution interface supplies
the observed duration, because work is measured where it is performed. The
measured duration semantics are defined by the backend: for the simulated
backend it is the configured work of the attempt.

Queries summarize observations by subject and quantity and report the
supporting observation count, the accumulated value, and the most recent
observation. Planning may prefer measured behavior where sufficient evidence
exists.

---

# 23. Resource Graph vs Topology Graph

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

# 24. No Hidden Global State

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

# 25. No Vendor Leakage

Forbidden:

    Core -> CUDA
    Core -> ROCm
    Core -> SYCL
    Core -> vendor-specific API

Allowed:

    Core -> generic backend interface
    Backend -> vendor implementation

---

# 26. No Model Leakage

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

# 27. Dependency Direction

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

# 28. Architectural Test

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

# 29. Initial Non-Goals

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

# 30. Architectural Objective

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
