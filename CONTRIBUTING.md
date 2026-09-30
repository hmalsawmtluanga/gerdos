# Contributing to GERDOS

## Laws first

Every PR follows L1–L9 of the Master Plan (§1). The short version:

1. Behaviour changes start as a `docs:` commit (contract section or ADR).
2. GPU forms need a CPU reference form and parity tests.
3. Refuse by name with a reason; never silently approximate.
4. Label evidence `real-device` / `software-device` / `simulated`.
5. Keep model knowledge in `adapters/` and drivers.
6. Check capability, never device identity (quirks live in `hw/quirks`).
7. Every fix ships a regression test that fails before the fix.
8. Zero warnings (`-Wall -Wextra -Wpedantic`, `/W4`); warnings are errors in CI.
9. Small typed commits (`docs:`, `feat:`, `fix:`, `test:`, `chore:`, `ci:`).

## Branches and reviews

Short-lived branches named `<lane>/P<phase>-<id>-<slug>` (e.g.
`a/P1-03-atomic-load`). Every PR needs the Orchestrator plus one
cross-lane reviewer; public claims need Owner sign-off.

## Hardware floor

Vulkan 1.0 compute or OpenCL 1.2; x86-64 SSE2 or ARMv8 NEON; 4 GB RAM;
GPUs down to 1–2 GB VRAM. Nothing may raise this floor without Owner
approval. Never commit models or weights to git (`/models/` is ignored;
fetch scripts pin SHA-256 and check licences).
