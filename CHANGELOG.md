# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
The C ABI is pre-1.0: error-code changes are recorded here and bump the
minor version so they stay visible.

## [Unreleased]

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
