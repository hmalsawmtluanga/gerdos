# Getting Started with GERDOS

Build, run the demo, and read the story in its output. Then read the
contract, which is law.

## 1. Build

```bash
gerdos_opencl_root="$HOME/ocltmp/extracted/usr"  # only if OpenCL headers live there
GERDOS_OPENCL_ROOT="$gerdos_opencl_root" cmake -S . -B build
GERDOS_OPENCL_ROOT="$gerdos_opencl_root" cmake --build build -j
```

Omit `GERDOS_OPENCL_ROOT` when the toolchain is installed system-wide.
Configure with `-DCMAKE_BUILD_TYPE=Release` into `build-release/` for the
second required configuration.

## 2. Run the demo

```bash
./build/gerdos_demo
```

Without an OpenCL GPU the demo prints `SKIP: no OpenCL GPU device` and
exits green — that is the hardware gate working, not a failure. With one,
it prints two runs (declared order vs measurement-informed planning) and
a closing verdict with the machine-scoped ratio. Ratios belong to the
machine; the portable claim is structural: measured evidence changed
the choices.

## 3. Run the tests

Binaries run directly (the house norm — `ctest` may hit an environmental
`LastTest.log.tmp` read-only error unrelated to results):

```bash
for t in build/gerdos_core_*_test build/gerdos_adapter_*_test build/gerdos_core_vk_compute_test; do
  ./$t ./workloads > /dev/null 2>&1 || ./$t > /dev/null 2>&1 || echo "FAIL: $(basename $t)"
done
./build/gerdos_core_no_leakage_test include
```

The `no_leakage` binary takes the include directory as its only argument;
the workload artifact test takes the workloads directory. The three
hardware suites and the demo need real devices; both configs must be
green with zero warnings under `-Wall -Wextra -Wpedantic`.

## 4. Read the contract

`docs/CORE_CONTRACT.md` settles every design argument. Then
`docs/ARCHITECTURE.md` (component map), `docs/ROADMAP.md` (where the
project is going), and `docs/DEVICE_RESOURCE_CONTRACT.md` (test and
leakage rules).

## 5. Next steps

- Add a model term: `docs/ADAPTER_GUIDE.md` (fail-closed translation).
- Add a backend: `docs/BACKEND_GUIDE.md` (the seam contract).
- Declare a workload by hand: `workloads/*.gwd` with the format in the
  contract's workload section; `tests/core_workload_artifact.cpp` shows
  the load-plan-execute path.
