# GERDOS

## Global Execution Runtime for Dynamic Open Systems

GERDOS is a model-agnostic execution runtime designed to extract useful
performance from heterogeneous and constrained hardware.

Its purpose is to make modern open models practical across hardware
configurations that conventional inference runtimes may not exploit efficiently.

GERDOS treats compute, memory, storage, transfer mechanisms, and model state
as resources within a single global execution system, while representing
physical connectivity through a separate topology model.

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

---

## Quickstart

Requires CMake 3.20+, a C++20 compiler, and Ninja (or your generator of
choice). Python 3.10+ only for the `gerdos` driver/wheels.

### Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Both configs must be green with zero warnings under
`-Wall -Wextra -Wpedantic`. Hardware suites print `SKIP: ...` and exit
green when no device is present — that is the hardware gate working,
not a failure. Full walkthrough: `docs/GETTING_STARTED.md`.

### Windows (MSVC developer prompt)

`cmake`/`ctest` are not on a plain PowerShell `PATH`; run everything
inside one `VsDevCmd` session:

```bat
VsDevCmd.bat -arch=amd64
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S . -B build-win
cmake --build build-win
ctest --test-dir build-win --output-on-failure
```

With the Vulkan SDK installed (`VULKAN_SDK` set), the Vulkan compute
backend test registers and runs; without it, it is absent-by-design.
OpenCL needs a vendor ICD or it stays unregistered. Same green-tree
rule, `/W4`, zero warnings.

### Python wheels

```bash
pip install gerdos
python -m gerdos selftest
```

Manylinux x86_64/aarch64 wheels (CPython 3.10–3.13) self-test on the
spot with no repo, no compiler, no env vars.

---

## License

MIT — see [LICENSE](LICENSE). Version is single-sourced from
`include/gerdos/version.hpp` (currently 0.1.2).
