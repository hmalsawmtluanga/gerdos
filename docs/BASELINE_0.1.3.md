# GERDOS baseline at 0.1.3 (`2b87ff4`)

Phase 0 record (P0-04). Everything below was read off the tree or measured
by the 0.1.3 audit; re-cut this file if HEAD moves before Phase 1 starts.

## Version pins

- `include/gerdos/version.hpp`: 0.1.3
- `CMakeLists.txt` `PROJECT_VERSION`: 0.1.3
- `setup.py`: reads the version from `version.hpp`

## Operation set (11 `WorkForm`s, `include/gerdos/core/operation.hpp`)

`ELEMENTWISE_AFFINE`, `MATRIX_PRODUCT`, `REDUCE_SUM`, `EXPONENTIAL`,
`REDUCE_MAX`, `REDUCE_MIN`, `ELEMENTWISE_MIN`, `ELEMENTWISE_MAX`,
`MASK_SELECT`, `GATHER`, `ELEMENTWISE_DIVIDE`.

Dtypes (`WorkDtype`): `F32` (default), `F16`, `I8`.

## Refusal list (5 names, `adapters/model_adapter.hpp`)

`SOFTMAX`, `ATTENTION`, `RESIDUAL_ADD`, `LAYER_NORM`, `ARGMAX`
— refused by name with a stated reason, never approximated.

## Test counts

- 32 ctest entries (26 C++ suites + hardware/consumer/artifact tests +
  2 shell scripts: `version_parity.sh`, `smoke_consumer.sh`).
- Audit measurement on the Owner machine (i5-12400F, RX 6700 XT, MSVC
  19.51): **30/32 pass** in Release and Debug, zero warnings under `/W4`.
  The 2 failures are the shell-script tests (no `sh`/`python3` on Windows).
- AddressSanitizer: full suite clean (30/30).
- Fuzzing: 9,000 mutated `.gwd` files through the C API under ASan, clean.
- Shaders: all 14 embedded SPIR-V arrays match `shaders/spv/*.spv`.

## Known baseline defects (Phase 1 intake)

The 14 audit findings (1 wrong-result Vulkan bug, 1 memory-safety bug,
load-validation gaps, Windows/packaging gaps) are tracked as issues
`P1-01`–`P1-13` plus workspace clutter; see the Phase 1 board.

## Toolchain matrix (P0-05)

- Linux CI: `gcc` (default), `gcc-13`, `clang-18` on `ubuntu-24.04`.
- CMake minimum: 3.20.
- Windows/MSVC lane: defined, pending (P1-05). Owner machine reference:
  MSVC 19.51, CMake 4.4.3, Ninja 1.13.2, Vulkan SDK 1.4.363.0.
