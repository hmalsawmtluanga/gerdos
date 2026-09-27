# GERDOS Architecture

## 1. Purpose

GERDOS is an inference runtime whose primary abstraction is not a particular
model or accelerator.

The runtime manages execution across heterogeneous compute, memory, storage,
and transfer resources, while representing physical and logical connectivity
through a separate topology model.

The architecture must therefore remain independent of any particular:

- model family
- model architecture
- GPU vendor
- CPU vendor
- accelerator generation
- quantization format
- storage device
- interconnect
- operating environment

## 2. Global execution view

Conceptually:

                    GERDOS
                       |
              Global Execution View
                       |
        +--------------+--------------+
        |              |              |
      Compute        Memory         Storage
        |              |              |
    CPU / GPU     VRAM / RAM       NVMe / SSD
        |              |              |
        +--------------+--------------+
                       |
                Planner
                       |
                Execution Plan

This is a conceptual model, not a fixed implementation.

## 3. Resource classes

GERDOS will eventually represent at least:

### Compute

- CPU
- GPU
- accelerator
- heterogeneous compute engines

### Memory

- device-local memory
- host memory
- pinned host memory
- shared/unified memory where available
- cache layers

### Storage

- NVMe
- SSD
- other persistent storage

### Transfer

- DMA engines
- copy engines
- storage I/O engines
- host-to-device transfer mechanisms

### Synchronization

- synchronization mechanisms required to coordinate execution

## 4. Topology

GERDOS represents physical and logical connectivity separately from Resource
identity.

Examples include:

- PCIe
- NVLink
- Infinity Fabric
- CPU memory fabrics
- storage connectivity
- other accelerator interconnects

Topology describes endpoints, paths, accessibility, bandwidth, latency, and
measured behavior. It is not itself a ResourceKind.

## 5. Model abstraction

Models are external workloads.

The core runtime must not contain architecture-specific assumptions about
individual model families.

Model support should be implemented through explicit adapters that translate
model-specific representations into generic GERDOS workloads.

## 6. Execution

GERDOS should eventually be able to construct an execution plan based on:

- available compute
- available memory
- memory bandwidth
- interconnect bandwidth
- storage bandwidth
- transfer latency
- topology
- workload characteristics
- workload state
- cache state
- runtime measurements

Static heuristics may provide initial behavior, but runtime measurement should
be preferred wherever practical.

## 7. Data movement

Computation and data movement are separate but interacting execution concerns.

The runtime should support:

- asynchronous transfers
- double buffering
- prefetching
- eviction
- overlap of transfer and computation
- topology-aware movement
- demand-driven loading

## 8. Hardware independence

Hardware-specific functionality belongs behind backend/device interfaces.

The core runtime must not depend directly on CUDA, ROCm, SYCL, or any individual
vendor API.

## 9. Performance philosophy

GERDOS does not assume that maximum theoretical FLOPS determine inference
performance.

Relevant constraints can include:

- memory bandwidth
- PCIe bandwidth
- accelerator utilization
- CPU execution capacity
- storage latency
- data locality
- cache hit rate
- transfer overlap
- synchronization
- kernel launch overhead
- scheduling overhead

The runtime should measure these factors rather than assuming them.

## 10. Validation

Specific models and hardware configurations will be introduced later as
validation workloads.

They must not become architectural dependencies of GERDOS itself.

## 11. Heterogeneous execution adversarial scenario

Before implementing higher-level scheduling and execution abstractions, GERDOS
must be tested against a synthetic heterogeneous workload that exercises
capacity, residency, movement, topology, resource availability, and repeated
execution without depending on any particular model architecture.

### 11.1 Scenario topology

The adversarial environment contains two devices.

Host Device:
- Resource 100: host RAM, kind MEMORY
- Resource 101: NVMe storage, kind STORAGE
- Resource 102: host DMA engine, kind TRANSFER

Accelerator Device:
- Resource 200: device-local memory, kind MEMORY
- Resource 201: compute engine, kind COMPUTE
- Resource 202: transfer engine, kind TRANSFER

The conceptual movement path is:

NVMe -> RAM -> device-local memory -> accelerator compute

Topology and Resource identity remain separate. A topology path describes
connectivity and movement possibilities; it does not become a Resource merely
because it carries traffic.

### 11.2 Logical data and physical residency

The workload contains one logical Data object, identified independently from
where it is stored or which representation is currently usable.

The same Data may have multiple physical residencies:
- NVMe residency with a storage-oriented representation
- RAM residency with a host-accessible representation
- device-local memory residency with a compute-ready representation

A residency identifies the logical Data, the concrete ResourceRef, and its
representation. Different representations may coexist for the same Data.

The scenario must therefore preserve these distinctions:

Data != Data Residency
Data Residency != Resource
Resource != Device

A Data residency in TRANSFERRING state indicates that the residency is
currently involved in a transfer-related runtime state. The transfer
mechanism itself remains a higher-level concern and does not change the
identity of the logical Data object.

### 11.3 Operation and execution pressure

The scenario contains a logical Operation that requires Data to be available to
a suitable compute resource. The Operation identifies what work is required,
but does not select the accelerator, residency, transfer path, queue, backend,
or execution attempt.

The same Operation must be able to produce multiple Executions. For example,
one execution may fail after a transfer failure while a later execution retries
after establishing another usable residency.

Execution-specific information such as running, completed, failed, retried,
timing, measurements, selected resources, and selected physical residencies
belongs to Execution rather than Operation.

### 11.4 Adversarial runtime conditions

The scenario must exercise the following conditions independently:

1. Multiple residencies of the same Data exist simultaneously.
2. Different representations of the same Data coexist.
3. A required residency is TRANSFERRING while computation is waiting.
4. Another residency can satisfy the Operation instead of waiting for the first.
5. Transfer activity competes for shared movement capacity.
6. Topology constrains which movement paths are available.
7. Prefetch may create a residency before it is immediately required.
8. Eviction may remove a residency while preserving the logical Data.
9. A Resource may become unavailable while a residency still exists.
10. A transfer may fail and require a later Execution to retry.
11. One residency may become UNAVAILABLE while another remains VALID.
12. Capacity pressure may force placement or eviction decisions.

Current status of each condition:

| # | Condition | Status |
|---|-----------|--------|
| 1 | Multiple simultaneous residencies | exercised (identity, executor, and slice suites) |
| 2 | Coexisting representations | exercised (adversarial suite) |
| 3 | TRANSFERRING residency while work waits | exercised (slice, admission suites) |
| 4 | Alternate residency substitution | partial: admission accepts any usable residency; substitution is a planner decision (deferred) |
| 5 | Shared movement capacity contention | deferred before planning; single-writer update claims cover concurrent updates today |
| 6 | Topology-constrained movement | deferred before planning; topology has no consumer yet |
| 7 | Prefetch | deferred; the effects model supports creating residencies ahead of use |
| 8 | Eviction preserving logical Data | exercised (identity suite: residency removal leaves Data) |
| 9 | Resource unavailable while residency exists | exercised (adversarial suite) |
| 10 | Transfer failure and retry by a later Execution | exercised (executor suite) |
| 11 | One residency UNAVAILABLE while another VALID | exercised (admission and executor suites) |
| 12 | Capacity pressure | deferred before planning |

Conditions marked deferred remain open obligations for the planning layer and
must be closed or re-scoped before measurement-informed planning is claimed.

### 11.5 Architectural invariants under test

The scenario is successful only if the abstractions continue to preserve the
following boundaries:

- Data remains a logical identity independent of physical placement.
- Data Residency remains physical runtime state rather than logical Data
  identity.
- Resource remains an execution resource rather than a model-specific object.
- Device remains an ownership boundary rather than automatically becoming the
  scheduling unit.
- Topology remains separate from Resource identity.
- Operation remains declarative and contains no execution state.
- Execution remains the identity of a concrete attempt to perform an Operation.
- Backend-specific handles and vendor APIs remain outside the core contracts.
- Model-specific concepts remain outside the core contracts.

The scenario must not add missing concepts merely to make the scenario
executable. Any failure must instead be classified as a missing identity,
state, resource description, capability, topology, data-movement, operation,
execution, or planning concept.

### 11.6 Validation questions

The scenario should force explicit answers to these questions before higher-level
execution machinery is implemented:

- Can the current identity model represent every physical residency
  unambiguously within its defined ownership scopes?
- Can a logical Operation remain independent of its eventual Resource and residency choices?
- Can multiple Executions of one Operation be represented without changing Operation identity?
- Can resource disappearance be represented without corrupting Data identity?
- Can movement, contention, and topology constraints be represented without conflating them with computation?
- Which missing concepts belong in identity, state, resource description, capability, topology, data movement, Operation, Execution, or planning?
