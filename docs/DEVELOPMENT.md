# GERDOS Development

## Development rule

Do not add a subsystem merely because a conventional inference runtime has one.

First establish:

1. the problem,
2. the required abstraction,
3. the measurable behavior,
4. the dependency boundary,
5. the implementation.

## Architecture rule

Dependencies should point toward generic runtime abstractions.

Hardware-specific and model-specific code must not become dependencies of the
core runtime.

## Validation rule

Every performance-oriented optimization should eventually have:

- a reproducible benchmark,
- a baseline,
- instrumentation,
- measurable success criteria.

## Research rule

Important architectural decisions should be backed by source code, papers,
documentation, experiments, or hardware measurements.

## Build and test discipline

The canonical build-and-run path lives in `GETTING_STARTED.md` —
follow it, not this section. In short: binaries run directly (hosted
`ctest` hits an environmental `LastTest.log.tmp` read-only error
unrelated to results, so `ctest` is CI-only); both configurations
must pass with zero warnings before any commit. Tests are registered
through `gerdos_add_test` in `CMakeLists.txt`; the
runtime-integration policy is external synchronization, so tests are
single-threaded by design.
