# GERDOS

## Global Execution Runtime for Dynamic Open Systems

GERDOS is a model-agnostic inference runtime designed to extract useful
performance from heterogeneous and constrained hardware.

Its purpose is to make modern open models practical across hardware
configurations that conventional inference runtimes may not exploit efficiently.

GERDOS treats compute, memory, storage, interconnects, and model state as
resources available to a single global execution system.

### Core principle

> Revive old hardware. Run new models.

### Architectural principles

- Model-agnostic core
- Hardware-aware execution
- Global resource visibility
- Dynamic scheduling
- Explicit memory hierarchy
- CPU as a compute resource
- GPU as a compute and memory resource
- Storage as an executable memory tier when beneficial
- Overlapped computation and data movement
- Measurement-driven decisions
- Backend independence
- Reproducible benchmarking

Specific models are validation workloads, not architectural dependencies.

---

## Status

Early architectural development.

The project is intentionally being built from a minimal foundation rather than
starting with model-specific or hardware-specific assumptions.
