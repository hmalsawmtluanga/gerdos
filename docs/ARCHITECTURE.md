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
                Global Scheduler
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

Model support should be implemented through explicit model/graph/weight
interfaces.

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
- model state
- cache state
- runtime measurements

Static heuristics may provide initial behavior, but runtime measurement should
be preferred wherever practical.

## 7. Data movement

Computation and data movement are separate resources.

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
- expert locality
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
