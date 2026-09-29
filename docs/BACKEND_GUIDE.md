# Backend Authoring Guide: The Seam Contract

A backend implements `ExecutionBackend` (`include/gerdos/core/execution_backend.hpp`):
`submit()` + `poll()` + inspection (`allocation_count`, `sample`,
`allocations_equal`). Read `include/gerdos/cpu/cpu_backend.hpp` first
(simplest engine), then `include/gerdos/hw/heterogeneous_backend.hpp`
(staging across homes).

## 1. Atomicity

Beginning an attempt is rejection-atomic: unless the backend accepts,
nothing changes. Keep the undo pattern — snapshot replaced allocations
before touching the map, restore on launch failure, and size-check
before allocating so hostile sizing never reaches a kernel.

## 2. Ownership

Workers hold shared ownership of their storage (buffers via
`shared_ptr`, device handles via refcounted deleters through the
still-live device). A replacement must never dangle under a live job.
Teardown clears allocations before destroying the device. Submit work
through the shared `WorkPool` (`include/gerdos/core/work_pool.hpp`) —
refusal runs the undo path; destructors join then drain.

## 3. Measurement

Durations are real wall-clock nanoseconds measured where the work
happens, on background threads; `poll()` never blocks. Warm up
one-time driver costs in the constructor so they never appear in
measured work. Print numbers before asserting; machine-scoped ratios
only, never portable claims.

## 4. Staging and semantics

Compute in F32 with once-per-direction boundary conversion through
`include/gerdos/core/dtype.hpp` (the one semantic home — never
reimplement conversion). Device buffers hold F32 working copies;
allocations keep their own dtype and convert per operation. Honor the
aliasing rules: single-source iterated form, multi-operand Data-identity
guards, gather truncate-and-clamp. Match the CPU engine value-for-value
(exact for integer shapes, documented tolerance for float reductions).

## 5. Vocabulary and gating

Backend API spellings live in the backend directory (the documented
vendor boundary) — never in `include/gerdos/core/`. Report
availability (`*_available()`); without a device, dependents skip with
a printed reason and exit green. Register the suite at configure time
only when the toolchain is found (see the OpenCL/Vulkan blocks in
`CMakeLists.txt`).

## 6. Device fallback (OpenCL)

The heterogeneous backend prefers a GPU device and falls back to a
CPU device on the same platform when no GPU exists. The fallback
verifies the engine (submission, staging, kernels, dispatch) — it
never substantiates silicon claims: every runner prints the device
name and kind, and hardware-matrix rows record which kind ran.

## 7. Checklist

- [ ] rejection-atomic submit with undo
- [ ] shared storage ownership, teardown order
- [ ] F32-compute structure via shared helpers
- [ ] exact-value chains on the new engine (CPU reference)
- [ ] availability gate with skip-with-reason
- [ ] device identity printed (name + GPU / CPU-fallback)
- [ ] both configs green, zero warnings, leakage green
