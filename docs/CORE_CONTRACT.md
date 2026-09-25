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

# 9. Transfer Resource

Data movement is itself a schedulable resource.

Examples:

- DMA engine
- copy engine
- storage I/O engine
- host-to-device transfer capability

Transfers may compete for:

- bandwidth
- copy engines
- interconnect capacity
- memory bandwidth
- storage queues

Therefore transfers must be visible to the execution planner.

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
- being created
- being transferred
- unavailable

The scheduler must be able to reason about the cost of obtaining usable data
at a particular execution resource.

---

# 13. Operation

An Operation represents executable work.

Examples:

- matrix multiplication
- attention
- normalization
- reduction
- dequantization
- data transfer
- storage read
- synchronization

An Operation contains generic execution requirements such as:

- inputs
- outputs
- dependencies
- capability requirements
- resource requirements
- constraints
- estimated cost

An Operation does not select a vendor backend.

---

# 14. Execution

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

---

# 15. Workload

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

# 16. Model Boundary

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

# 17. Backend Boundary

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
