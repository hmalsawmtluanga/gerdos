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

Configure and build:

    cmake -S . -B build
    cmake --build build

Run the full suite in the default configuration and again in Release; the test
harness is always-on, so both configurations must pass:

    ctest --test-dir build
    cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
    cmake --build build-release
    ctest --test-dir build-release

Every change must leave both configurations green and warning-clean before it
is committed. Tests are registered through `gerdos_add_test` in
`CMakeLists.txt`; the runtime-integration policy is external synchronization,
so tests are single-threaded by design.
