# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
The C ABI is pre-1.0: error-code changes are recorded here and bump the
minor version so they stay visible.

## [Unreleased]

### Fixed

- Chained-sum test compares exact-or-within-4-ulp: left-to-right F32
  accumulation is not bit-identical to the regrouped closed form on every
  toolchain/libm (1-ulp CI failure on the newer ubuntu-24.04 image).
- Vulkan backend: host-homed outputs now download their transient result
  after the queue completes instead of copying stale pre-dispatch data
  (F-01). All work forms funnel through the single deferred readback.
- `sample()` bounds check compares the index against the element count
  instead of multiplying first, closing a `size_t` wrap that turned
  hostile indices into out-of-bounds reads (F-02, all backends).

## [0.1.3]

### Added

- `ELEMENTWISE_DIVIDE` work form (`dst = dst*dscale + sscale*(A[i]/B[0]) + c`,
  IEEE-754 division, F32-first).
- Three-way version parity test (header, `setup.py`, CMake pin).
- manylinux wheels for CPython 3.10–3.13 on x86_64 and aarch64.

### Changed

- README status lists division as supported and the five refused-by-name
  operations explicitly; poster claims demoted to direction statements.

## [0.1.2]

- Prior release. See git history for details.
