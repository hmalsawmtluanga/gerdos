# GERDOS Machine Profiles — Floor and Ceiling

Two characterized machines run the same suite at the same revision.
The floor keeps the mission honest (constrained hardware); the ceiling
proves the machinery (real silicon, reference numbers). Ratios and
timings belong to their machine and must be re-measured elsewhere —
the portable claims are structural, per `docs/STUDY_CONSUMERS.md`.

## Floor — constrained reference (Linux)

Two records, newest first.

**Study machine** (transcribed from `docs/STUDY_CONSUMERS.md`, 2026-09-28):

| Field | Value |
| --- | --- |
| Machine | Intel i5-13420H, Intel Graphics (RPL-P), `/dev/dri/renderD128` |
| OS / toolchain | Linux, GCC 13, C++20, `-Wall -Wextra -Wpedantic` zero warnings |
| GPU drivers | OpenCL via `GERDOS_OPENCL_ROOT=$HOME/ocltmp/extracted/usr` (apt-extracted, no sudo); Vulkan offline SPIR-V |
| Config | default + Release (`build/`, `build-release/`); workload dirs `./workloads` |

**Original build machine** (owner-reported, not independently verified
in this session): Intel i3-8100, no discrete GPU, 2x8 GB DDR4-3200,
Linux Mint Xfce. The repo was built and tested there and pushed to
GitHub from that box.

## Ceiling — proving ground (Windows)

First real discrete GPU this project has run against.

| Field | Value |
| --- | --- |
| Date (UTC) | 2026-09-29 |
| Machine | Intel i5-12400F (no iGPU), AMD Radeon RX 6700 XT 12 GB (Navi 22, `0x1002:0x73df`) |
| RAM | 16 GB (2x8 GB DDR4-3200) |
| OS | Microsoft Windows 11 Pro, build 26200 |
| GPU driver | AMD proprietary 32.0.21030.2001; Vulkan API 1.4.315; loader `vulkan-1.dll` + `vulkaninfo` present |
| Vulkan device shape | `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`, dedicated async-compute queue family (compute + transfer, no graphics) beside the graphics family |
| Toolchain | MSVC 19.51 + Ninja (VsDevCmd `-arch=amd64`), CMake 4.3.1, Git 2.55.0, PowerShell 5.1 shell |
| Repo | `main` at `b85e947`, tree clean |
| Baseline (no SDKs) | Release + Debug build with zero warnings under `/W4`; `ctest` 28/30 in both (entries 29–30 need `sh.exe`; version-parity needs Python — both absent, documented, not fixed) |

### Explicit non-claim

Nothing run on the 6700 XT proves the revival thesis. This machine is
comfortable, modern hardware: its numbers are the **ceiling**, not the
mission. The floor stays the honest proxy for revival-class hardware.

### OpenCL verdict (this machine, closed)

Skipped with reason, timeboxed shut: no OpenCL headers, no import
library, and no ICD registered
(`HKLM\SOFTWARE\Khronos\OpenCL\Vendors` absent — `OpenCL.dll` exists
but points at nothing). The `GERDOS_OPENCL_ROOT` fallback only
searches Linux library paths, and AMD's Windows OpenCL is frozen
legacy. Consequence: the heterogeneous backend test, the
real-benchmark binary, and `gerdos_demo` (all inside the
`OpenCL_FOUND` block in `CMakeLists.txt`) stay unregistered here.
**Vulkan is this machine's GPU story; OpenCL is documented absence.**

### Vulkan path (this machine, SDK installed, first silicon run green)

Vulkan SDK 1.4.363.0 installed 2026-09-29; `gerdos_core_vk_compute`
registers and passes in Release and Debug with zero warnings under
`/W4`: suite 29/31, vk test 0.54 s, values byte-identical across
configs. No shader rebuild was needed — all 13 compute shaders are
pre-embedded as SPIR-V in `include/gerdos/vk/vk_shaders.hpp`. Census
reports: `build-win/scratch/vulkaninfo-{summary,full}.txt`,
`vk-first-run.log`, `vk-debug-run.log` (ignored scratch, not committed).

### Silicon dispatch record (RX 6700 XT, 2026-09-29)

Dispatched and verified on silicon: `FillVec` (fill seeds), affine via
`TransformVec` (exact 2.0), `ReduceSum` (f32 and I8, exact 6.0),
`ReduceMinmax` (max and min, exact), `ExponentialVec`,
`MatmulTile` for all three tested shapes (2x3x2, 2x3x1, tiled
17x17x17 — all exact), `ElementwiseMinmax`, `MaskSelect`, `Gather`,
and the composed tiny-classifier chain (`W=2 scores=6 biased=6.5
layer2=14 stats=14`).

NOT dispatched on this run (pipelines created at init, never
selected): the scalar `Transform` / `Exponential` / `Fill` shaders —
dispatch always takes the `_vec` variants — and the plain `Matmul`
shader, because the tiled branch is always taken while
`maxWorkGroup (1024) >= 256`. They are compiled, not executed: a
regression that breaks them would surface only on hardware taking
those branches.

Parity note: every exact-equality check bit-matches. The exponential
lives in its documented lane — the test computes the expectation with
host `std::exp` and allows 1e-4 relative for GPU `exp`
implementation differences. Observed: `404.428680` vs `404.428802`
(3e-7 relative, ~330x inside the allowance).

### Ceiling benchmark (this machine, 2026-09-29, wall time)

`gerdos_core_cpu_backend`: ~10 ms Release (12.0 / 9.2 / 9.5),
40.3 ms Debug. `gerdos_core_vk_compute`: ~530-560 ms in both
configs (558.6 / 531.8 / 529.3 Release, 551.7 Debug) — dominated
by Vulkan instance/device/pipeline setup, which is why the host
build type barely moves it. Full `ctest`: 29/31 in ~1.1 s in
both configs; the 2 gaps are the known environmental Not-Runs
(no `sh.exe`, no Python), identical in Release and Debug.

Floor-vs-ceiling note: the Mint i3-8100 comparison is deferred —
that machine is not available. When it is: pull `db54a8a`,
rebuild, and record the same three numbers for a two-column
table here.
