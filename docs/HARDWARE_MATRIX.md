# GERDOS Hardware Matrix — Compatibility Record

One row per tested machine. A row is: machine census, commit hash,
`ctest` result line, and which engines registered. Contributors:
see `docs/COMMUNITY_TESTING.md` for the 5-minute path; native
contributors paste the census block below plus their full `ctest`
tail. Maintainers append rows; never edit a landed row.

## Census block (paste with every submission)

```text
CPU / RAM / OS:
GPU / driver:
Toolchain (compiler, CMake, Vulkan SDK if any):
OpenCL vendors (`HKLM:\SOFTWARE\Khronos\OpenCL\Vendors`, or `ls /etc/OpenCL/vendors`):
Commit:
ctest: _/31 in _ s (list Not-Run numbers with reasons)
Engines registered: cpu always; vk iff Vulkan device; hetero iff OpenCL ICD
```

## Rows

### Row 1 — ceiling / proving ground (maintainer, 2026-09-29)

```text
CPU / RAM / OS: i5-12400F / 2x8GB DDR4-3200 / Win11 Pro build 26200
GPU / driver: RX 6700 XT 12 GB / 32.0.21030.2001
Toolchain: MSVC 19.51, CMake 4.3.1, Ninja 1.13.2, Vulkan SDK 1.4.363.0
OpenCL vendors: none (OpenCL.dll loader present, zero ICDs)
Commit: 6db04f4
ctest: 29/31 in 1.21 s Release (30, 31 Not-Run: no sh.exe, no Python)
Engines registered: cpu + vk; hetero absent (OpenCL_FOUND false)
Notes: vk_compute 0.75 s; tiny + tiny-gated chains exact on CPU and Vulkan.
```

### Row 2 — floor reference (Mint i3-8100, reserved)

Machine unavailable since before Phase B. When it returns: pull the
matrix tip, rebuild, paste a census block here. Expected delta vs
row 1: hetero registers (UHD 630 OpenCL), Vulkan absent.

### Row 3+ — contributor machines

Appended as submissions arrive. Most-wanted: any pre-2016 iGPU
(small `maxWorkGroup` takes the plain-Matmul branch our ceiling
never touches), any OpenCL 1.2-only box, any 32-bit or <4 GB
machine, any ARM board (CPU-engine coverage only).
