# GERDOS Dependency Rules

## Rule 1 — Core before adapters

Generic runtime abstractions must not depend on model or hardware adapters.

## Rule 2 — Interfaces before implementations

Define the behavior required by the core before implementing a backend.

## Rule 3 — Data identity before residency

Logical data identity must not depend on where the data happens to reside.

## Rule 4 — Planning before execution

The scheduler plans execution.

Backends execute the plan.

## Rule 5 — Measurement is evidence

Hardware specifications may initialize the runtime's knowledge, but measured
behavior must be representable independently.

## Rule 6 — No vendor vocabulary in core interfaces

Core APIs should not expose CUDA streams, HIP handles, SYCL queues, CUDA events,
or equivalent vendor-specific objects.

Such objects belong behind backend boundaries.

## Rule 7 — No model vocabulary in core interfaces

Core APIs should not require concepts such as:

- expert
- attention head
- KV head
- transformer layer
- router
- token

unless those concepts are represented through a generic workload abstraction.

## Rule 8 — Explicit ownership

Resources, operations, executions, and runtime state must have explicit
ownership and lifetime.

## Rule 9 — No premature abstraction

Do not introduce an interface merely because it might become useful later.

Interfaces must correspond to an identified architectural boundary.

## Rule 10 — Hardware must be discoverable

The runtime should discover actual capabilities and topology rather than relying
on a fixed machine description.

## Rule 11 — Optimization must remain observable

Performance optimizations must expose enough telemetry to determine whether they
actually improved execution.

## Rule 12 — Experimental code is isolated

Experiments may violate normal architecture rules when necessary, but they must
not silently become dependencies of the production core.
