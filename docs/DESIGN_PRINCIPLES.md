# GERDOS Design Principles

## P1 — Model agnostic

No individual model is the reason GERDOS exists.

Models are workloads.

## P2 — Hardware is fixed; execution strategy is not

GERDOS attempts to improve useful hardware utilization through software.

## P3 — Global visibility

The runtime should reason about the complete execution environment rather than
treating CPU, GPU, RAM, VRAM, and storage as unrelated systems.

## P4 — Compute and movement are distinct

Moving data consumes bandwidth and time.

Execution planning must account for both computation and movement.

## P5 — CPU is a compute resource

The CPU must not be treated merely as an overflow location for GPU memory.

## P6 — Storage can become a memory tier

Persistent storage may participate in execution when capacity requirements
exceed available volatile memory and the resulting latency is acceptable.

## P7 — Locality matters

The cost of obtaining data can dominate the cost of computing with it.

## P8 — Measure before optimizing

Runtime decisions should increasingly be based on measured properties of the
actual machine and workload.

## P9 — No accidental vendor lock-in

Vendor-specific implementations must remain behind explicit interfaces.

## P10 — No model-specific architecture pollution

Validation targets may be highly specialized.

The GERDOS core must remain general.

## P11 — Reproducibility

Performance claims must record:

- hardware
- software versions
- compiler
- driver
- runtime configuration
- model
- quantization
- context length
- workload
- batch configuration
- relevant cache state

## P12 — Evidence over assumption

Architecture decisions should be supported by source inspection, experiments,
measurements, or reproducible external evidence.
