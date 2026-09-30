#!/bin/sh
# P0-02: file the 14 audit-of-0.1.3 findings as GitHub issues.
# OWNER-RUN ONLY (writes to the remote). Usage from the repo root:
#   sh scripts/p0-02-file-issues.sh
# Labels are created idempotently. Issues are created fresh on every run,
# so run it once; if you must re-run, close duplicates by hand first.
# The last step locks main (solo-owner rule: PRs only, CI must be green).
set -e
cd "$(dirname "$0")/.."
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "target repo: $REPO"
gh label create "sev:high" --color B60205 --description "audit severity high" 2>/dev/null || true
gh label create "sev:medium" --color D93F0B --description "audit severity medium" 2>/dev/null || true
gh label create "sev:low" --color FBCA04 --description "audit severity low" 2>/dev/null || true
ISSUE() {
  sev="$1"; title="$2"; body="$3"
  gh issue create --title "[$sev][$title" --label "sev:$sev" --body "$body"
}
ISSUE high "F-01] Vulkan host-homed output reads stale data" "Location: include/gerdos/vk/vk_compute_backend.hpp (download_transient, near :1281). Repro P0-03 probe_b on RX 6700 XT: affine 0.5x+1.5 over device-homed 1.0 with the OUTPUT homed on the host prints 1.0, expect 2.0. Device-homed twin prints 2.0. Fix: order the host download after the device write completes, then add the host-homed twin as a regression test (fails before, passes after)."
ISSUE high "F-02] gerdos_sample reads the wrong element on huge 64-bit indexes" "Location: include/gerdos/cpu/cpu_backend.hpp:230 - the (index+1)*width check wraps past the bound. Repro P0-03 probe_a via the C API: sample(901, 9002, 2^62) returns 0.75 (element 0), expect 0.0; under ASan index=SIZE_MAX is a heap-buffer-overflow. Fix: overflow-safe bounds check plus a past-the-end regression test."
ISSUE high "F-03] refused load leaves partial state behind" "Location: load path via src/gerdos_c.cpp gerdos_load into the artifact loader. Repro: load data 970 then a duplicate data 900 - load returns the refusal line but 970 stays resident. Fix: all-or-nothing load (validate first, commit after) with a test asserting empty state after refusal."
ISSUE high "F-04] absurd element counts allocate gigabytes before refusing" "Location: include/gerdos/core/workload_artifact.hpp:598-638. Repro: elements=4000000000 peaks near 13.3 GB resident and returns -1 only after ~39 s. Fix: sanity-cap element counts against a documented limit before allocating; regression test stays flat in memory."
ISSUE medium "F-05] README Windows-green claim is false" "Location: README.md status claim; tests/version_parity.sh:15 (python3). Repro P0-03: sh tests/version_parity.sh exits 127 on Windows - python3 is absent and the two shell tests cannot run, so 30/32 pass, not green. Fix in P1-05: qualify the claim by lane until the Windows lane exists."
ISSUE medium "F-06] source install fails on non-Linux" "Location: setup.py:52-56,77,82. Repro: pip install from source on Windows builds an MSVC .dll but the driver expects libgerdos_c.so, and non-Linux CMake flags assume a Unix toolchain. Fix: per-platform library name and flags, with a documented source-install test per OS."
ISSUE medium "F-07] CMake install omits the C ABI" "Location: CMakeLists.txt:239-243 installs the C++ headers but not include/gerdos/gerdos.h and no gerdos_c target. Fix: install the C header and export the C target so a system install serves C consumers."
ISSUE medium "F-08] Vulkan dispatch dimensions unchecked" "Location: include/gerdos/vk/vk_compute_backend.hpp:1037 (vkCmdDispatch). Fix: validate group counts against device limits and refuse by name when out of range; regression test with an oversized dispatch."
ISSUE medium "F-09] Vulkan VkResult returns unchecked" "Location: include/gerdos/vk/vk_compute_backend.hpp:1076 plus sibling call sites. Fix: check every Vulkan return and fail loud with the call name; no silent continuation."
ISSUE low "F-10] empty-string load is inconsistent" "Location: load path (src/gerdos_c.cpp gerdos_load vs artifact transcript rules). Fix: one documented rule for empty input, same on every entry point, with a test."
ISSUE low "F-11] machine header is blank on Windows" "Location: python/gerdos/__init__.py:90,129,167 (_machine_header). Fix: populate the header on Windows with a test."
ISSUE low "F-12] docs index covers 6 of 15 guides" "Location: docs/README.md. Fix: index every guide under P1-12 docs-refresh; keep the index complete as guides are added."
ISSUE low "F-13] CI has no permissions block and tag-only action pins" "Location: .github/workflows/ci.yml. Fix: least-privilege permissions block plus SHA-pinned actions."
ISSUE low "F-14] workspace re-clutter guard" "Location: .gitignore (fixed for .venv/egg-info/uv.lock at 2b87ff4). Fix: a CI check that fails on committed build artifacts so the clutter cannot return."
echo "14 issues filed. Locking main: PRs only, CI must be green."
PROT_JSON=$(mktemp)
trap "rm -f $PROT_JSON" EXIT
cat > "$PROT_JSON" <<JSON
{
  "required_status_checks": {"strict": true, "contexts": ["build-and-test", "build-wheels"]},
  "required_pull_request_reviews": null,
  "enforce_admins": true,
  "restrictions": {"users": [], "teams": []},
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
gh api "repos/$REPO/branches/main/protection" -X PUT --input "$PROT_JSON" > /dev/null
echo "main is locked. Phase 0 gate: confirm 14 issues plus labels."
