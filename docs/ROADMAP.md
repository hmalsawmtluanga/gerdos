# GERDOS Roadmap — Phases 1–6 Complete, Next Steps Open

Status: all six phases landed on main (see per-phase commits below).
Phases ran in order; each phase's contract-first `docs:` commit
preceded its `feat:` commit, every change stayed green in `build/`
and `build-release/` with zero warnings under `-Wall -Wextra
-Wpedantic`, and every checkpoint left `git status --porcelain`
empty on main. The phase sections below are the executed record —
read them as history, not as plan.

## Landed summary

- **Phase 1 (algebra):** dtype breadth (F32/F16/I8, F32-compute
  structure), comparison/selection forms, exponentials, reductions,
  composite refusal with named gaps.
- **Phase 2 (planning):** dependency scheduling, multi-attempt plans,
  derived capacity, locality, per-comparison discovery with the
  proven-incumbent bound.
- **Phase 3 (backends):** Vulkan second family, tiled/vectorized
  kernels, bounded worker pools on all engines.
- **Phase 4 (workloads):** artifact format v1, staged weight rounds,
  decoder-prefix translation with reference verification, thesis
  verdict (proven where the algebra reaches).
- **Phase 5 (hardening):** CI on hosted runners, packaging with
  consumer smoke test, guides, recovery policy, skip-with-reason
gates, runtime config.
- **Phase 6 (ecosystem):** second consumer (signal chains), published
  study with reproduction records, `pip install gerdos` 0.1.2 with
  bundled native library.

## Next (open)

- Platform wheels (remove the compiler requirement for testers).
- Physical small-board record (or community replies via
  `docs/COMMUNITY_TESTING.md`).
- First outside contribution through the documented path.
- `gerdos-micro` profile: v1 cut list defined structurally by Phase B
  (both tiny models are members); trimmed by floor/board data when
  hardware appears.

Standing laws (not repeated per phase): no `OperationKind` — work stays
parameters, never kinds; no gate inspects the work (pinned by test); no
vendor vocabulary in `include/gerdos/core/`, no model vocabulary outside
`include/gerdos/adapters/` (enforced by `core_no_leakage`); core stays
testable without accelerator hardware; assertion strength matches
measured reality — print evidence before asserting; self-caught bugs are
reported, not hidden.

---

## Phase 1 (remainder) — Algebra completeness

Goal: the work vocabulary expresses a real decoder layer's elementwise,
normalization, and selection needs at real dtypes, so Phase 4 has
something to translate through.

### 1a. Dtype breadth — F16 / I8 + mixed-precision moves (next step)

Contract-first questions to settle in `docs/CORE_CONTRACT.md`:

- Where the element type lives: a `dtype` field on `WorkDescription`
  (parameters, not kinds — one new enum beside `WorkForm`, never a
  form split per dtype), with `float` remaining the default so every
  existing construction site keeps compiling.
- Well-formedness per dtype: storage sizing in *elements* stays
  dtype-agnostic, but byte sizing (`elements * dtype_bytes`) must be
  overflow-checked per dtype exactly as float sizing is today; hostile
  dtype/element combinations are seam-rejected, never allocated.
- Mixed-precision moves: a consuming record in one dtype producing to a
  record in another is an explicit conversion at the seam (round-half-away
  vs truncation decided in the contract, documented, tested bit-exactly
  on values both dtypes represent exactly), never a silent reinterpret.
- Kernel parity rule: every `(form, dtype)` pair executes on CPU loops
  and OpenCL kernels with identical semantics; CPU is the reference,
  GPU compares exact where the dtype is exact (I8) and within a
  documented tolerance where it is not (F16 rounding vs host float).
- Leakage: dtype names (`f16`, `i8`, `fp16`, `int8`) are generic
  vocabulary, allowed in core; any vendor-intrinsic spelling
  (`half4`, `__nv_bfloat16`, `cl_half`) stays behind the backend seam.

Implementation order: `operation.hpp` dtype field + `valid()`/sizing →
CPU kernels + allocation byte-sizing → OpenCL kernels (scalar `half`
/ `char` paths first; vectorized later in Phase 3) → adapter dtype
carriage → exact-value CPU + GPU tests chained off discriminating
seeds (same pattern as §7–15 of `core_work_description.cpp`) → gate
pin extended to a dtype-carrying form.

Exit criteria: all existing forms run at all supported dtypes on both
engines; mixed-dtype copies convert exactly on round-trippable values;
leakage guard green; no test needs accelerator hardware except the two
gated suites + demo.

### 1b. Composite work — multiple steps per operation

Contract-first questions:

- Whether composition is an ordered step list inside `WorkDescription`
  or a planner-level multi-operation chain (prefer the latter unless a
  fusion argument forces the former — premature abstraction is
  prohibited, but so is scattering one semantic across two homes).
- If in-work: pass/composition interaction (sequential application,
  intermediate aliasing rules identical to cross-operation chaining),
  well-formedness of each step independently, seam rejection of any
  hostile step rejecting the whole operation atomically.
- Backend execution order guarantees (sequential, deterministic) and
  measurement attribution (one duration per operation, not per step).

Exit criteria: a fused softmax-shaped chain (exp → reduce → normalize)
executes as declared on both engines with the same values as the
unfused chain; gates still blind; adapters may emit fused or unfused
steps with identical refusal behavior.

---

## Phase 2 — Planning completeness

Goal: the planner schedules real dependency graphs well, not just
single attempts. Each item is contract-first in the Binding Planning
section, implemented in `binding_planner.hpp` (one semantic home per
rule), and verified with deterministic simulation tests plus real
wall-clock confirmation where hardware diverges.

### 2a. Dependency-graph scheduling

- Topological ordering over operation dependencies; fail-closed on
  cycles (return no plan, never deadlock).
- Readiness defined by residency usability + claim state, not by wall
  clock; the planner never blocks, it returns the ready subset.
- Tests: diamond graphs, long chains, and unready-blocked graphs in
  the simulator; adversarial cases (dependency on failed attempt's
  output, dependency on claimed record) pinned.

### 2b. Multi-attempt plans (pipelining)

- Plans covering more than one attempt: stage-then-compute pairing
  (the demo's manual pairing becomes planner output), transfer/compute
  overlap where topology allows.
- Plan representation: an ordered list of bindings with attempt
  identities, each individually gate-passable; rejection-atomicity
  extends per attempt (a refused later attempt does not undo earlier
  recorded evidence).
- Tests: the demo workload planned, not hand-paired; simulator
  verifies overlap structure deterministically.

### 2c. Derived capacity accounting

- The named owner/model from the contract is coded: resources expose
  derived capacity (how much is free *now*), updated by claims and
  releases, never by guessing.
- Over-subscription is refused loudly at plan time, not discovered at
  submission; capacity evidence joins the measurement log.
- Tests: contention scenarios (two attempts, one memory budget) planned
  deterministically in simulation.

### 2d. Locality-aware consuming-copy choice

- Today consuming entries take the lowest-identifier usable record;
  the refinement chooses the usable copy *closest to the chosen engine*
  (topology distance, then measured transfer cost).
- This matters hugely on real hardware (staging across the link vs
  computing where the data already lives); verify with wall-clock on
  the iGPU: same work, near copy vs far copy, measured preference
  visible in the evidence log.
- Tests: dual-residency records (host + device homes), planner picks
  the engine-local copy; simulator pins the rule, hardware confirms
  the payoff.

### 2e. Principled exploration

- Replace the blunt `kExplorationBudget = 32` with a policy the contract
  can state precisely: per-mechanism confidence (observations + variance)
  instead of a global log-size counter; later arrivals get bounded
  re-exploration instead of waiting forever.
- Constraints: discovery-before-exploitation preserved (new mechanisms
  still sampled, never starved); failure evidence still not speed
  evidence; determinism preserved (seeded tie-breaking, simulator
  reproducibility).
- Tests: arrival-mid-run scenarios, high-variance mechanism scenarios,
  and the existing demo/benchmark shapes all still green.

Exit criteria: the §11 adversarial conditions marked deferred (4, 5, 6,
7, 12 — substitution, contention, topology-constrained movement,
prefetch, capacity pressure) are exercised, not deferred; the demo's
manual orchestration is planner output; wall-clock still favors measured
evidence on real hardware with margins physical noise cannot flip.

---

## Phase 3 — Backend breadth

Goal: prove the vendor boundary is real by adding a second GPU family,
then scale performance and throughput. Each backend lives behind the
execution-backend seam; the core never learns a new vocabulary word.

### 3a. Second GPU family — Vulkan compute

- Why Vulkan: the Mesa driver is already present on this machine, so
  no installation and no privileged access are needed; success proves
  the backend seam, not a vendor relationship.
- Scope: the same work algebra, the same home-respecting data plane
  (host memory vs device memory, real staging), the same wall-clock
  measurement path, the same rejection atomicity.
- Contract-first: the Heterogeneous Routing section gains a
  device-family paragraph (engine chosen by bound mechanisms, family
  is configuration, kernels per family but semantics per contract).
- Tests: the exact-value chains rerun on Vulkan where the device
  exists; `gpu_available()`-style gating follows the established
  configure-time pattern; everything else stays hardware-free.
- Do NOT attempt CUDA/ROCm on this machine (no such hardware — claiming
  results without measuring would violate "never make up stuff").

### 3b. Tiled / vectorized kernels (still generic)

- Within each backend: work-group tiling for matrix shapes, vector
  loads/stores for streaming elementwise work. Still generic (no model
  shapes baked in, no autotuned constants claiming portability).
- Verification: same exact-value tests (tiling must not change values),
  wall-clock improvement printed as evidence with the machine-named
  caveat, never asserted as a portable ratio.

### 3c. Work pool instead of one thread per attempt

- Replace per-attempt `std::async` launch with a bounded worker pool:
  queue depth, backpressure policy (block vs refuse loudly — decided in
  contract), shutdown-drains-in-flight (existing destructor guarantee
  preserved), inspection-surface-valid-only-while-idle preserved.
- Tests: the resize-during-flight test and the adversarial suites rerun
  unchanged; add a pool-saturation test (more attempts than workers,
  all complete coherently, no dangling buffers).

Exit criteria: two GPU families execute the same algebra with the same
values; tiled kernels print faster wall-clock on shapes where the
machine genuinely diverges; the pool passes every existing concurrency
test plus saturation.

---

## Phase 4 — Real workload integration (highest uncertainty)

Goal: translate an actual decoder layer through the adapter and execute
it — scoped to *prove or break the thesis*, not to ship inference.
Explicit non-goal: GERDOS does not become a model implementation; if any
step below requires model semantics in the core, the phase fails loudly
and the contract says why.

### 4a. Workload artifact format (documented)

- A file format for declaring a workload: operations with data refs,
  dependencies, resource requirements, and work descriptions — the
  on-disk form of what the adapter emits today in memory.
- Versioned, fail-closed parsing (unknown fields refused, never
  ignored); a checked-in example artifact (tiny: tens of ops) that the
  test suite loads and executes end to end.
- No model vocabulary in the format: tensors are data, shapes are
  element counts, weights are initial-residency payloads.

### 4b. Real tensor / memory management

- Weight-scale residency: many Data objects, real byte pressure,
  eviction and prefetch actually exercised (closing §11 conditions 7
  and 12 for real instead of synthetically).
- Derived capacity (Phase 2c) meets real allocations: over-budget
  workloads refuse at plan time with a named reason.
- Tests: an artifact sized above device memory stages through in
  chunks; eviction preserves logical Data; residency removal rules
  (claims, terminal-state preconditions) hold under pressure.

### 4c. Decoder-layer translation through the adapter

- One attention-adjacent layer shape (e.g. QK-projection → score →
  softmax-composed-from-primitives → value-mix → output projection →
  residual add → normalization-shaped reductions) expressed *only* with
  the Phase-1 algebra; every op translates exactly or is refused loudly
  by name with its reason.
- Adapter vocabulary growth is per-op and fail-closed: each new model
  term gets a translation or a refusal string, never a silent
  approximation. The leakage guard stays green throughout.
- Verification: numerical exactness against a CPU reference computed
  outside GERDOS (hand-rolled, small, checked in) — bit-exact for
  integer shapes, tolerance-documented for float reductions; the
  comparison is printed evidence, not a hidden fixture.

### 4d. Thesis verdict

- The question Phase 4 answers: does measurement-informed planning beat
  declared order on a real translated layer on real hardware, with the
  same structural-plus-wall-clock evidence discipline as the demo and
  the benchmark?
- Either answer ships: a win extends the demo story to real work; a
  loss is reported with numbers and root-caused (wrong granularity?
  transfer-dominated? no engine divergence at this shape?) and re-scopes
  Phases 5–6. "Should work" is never reported as "works".

Exit criteria: a checked-in artifact + adapter path executes a real
layer shape end to end with numerical verification and a printed,
machine-scoped performance verdict — win or loss, with numbers.

---

## Phase 5 — Production hardening

Goal: GERDOS becomes operable by someone other than its author. Items
are largely independent; order by need, each with its own contract
paragraph where behavior is user-visible.

- **Config:** file + environment resolution for backend selection,
  device identity, exploration policy knobs, pool sizing; fail-closed
  on unknown keys; documented defaults matching today's hardcoded
  behavior.
- **Observability:** structured per-attempt trace (gates verdicts,
  chosen mechanisms, durations, evidence deltas); a log format with a
  versioned schema; never leaks backend handles or model terms outside
  their layers.
- **Error-recovery policy:** retry budgets, failure-class routing
  (transient vs structural), poison-record quarantine — decided in
  contract, pinned by adversarial tests, deterministic in simulation.
- **Packaging:** installable headers + CMake package config; versioned
  ABI note (header-only today — state the guarantee explicitly).
- **CI:** both build configs, leakage guard, demo, and the gated-hardware
  policy (skip-with-reason where no device exists, never fake-green).
- **Benchmark regression suite:** checked-in baselines are machine-local
  (never portable ratios); CI asserts *structure* (engines chosen,
  evidence recorded) and prints numbers; regressions are
  human-judged against local history, not hard-aborted on noise.
- **User docs:** a getting-started path (build → run demo → read the
  story in its output), the contract as law, the adapter authoring
guide (how to add a model term fail-closed), backend authoring guide
  (the seam contract: atomicity, ownership, measurement, staging).

Exit criteria: a second human (the owner) builds, runs, reads output,
and adds one adapter term + one backend stub from the docs alone.

---

## Phase 6 — Ecosystem

Goal: prove GERDOS is infrastructure, not a demo.

- **A second consumer built on GERDOS:** a distinct workload family
  (not decoder-adjacent — e.g. a data-parallel analytics pipeline or an
  image-processing chain through the same work algebra) with its own
  adapter, its own artifact, and its own printed verdict. Reuses the
  runtime, the planner, and the backends without touching them.
- **Published comparative studies:** measurement-informed vs declared
  order across the consumers, with full P11 reproduction records
  (hardware, software, driver, config, workload, batch, cache state);
  machine-scoped numbers, structural claims portable.
- **Upstream posture:** contribution path for new backends/adapters
  (review checklist: leakage scan, gate-blindness pin, exact-value
  chains on both engines, both configs green, contract-first docs).

Exit criteria: two consumers, published numbers with reproduction
records, and at least one outside contribution merged through the
checklist — or a written verdict on why not, with numbers.

---

## Appendix — Phase 1 remainder detail (dtype breadth, starting now)

Step 1 (contract-first, this session): `dtype` field semantics,
well-formedness per dtype, mixed-precision conversion rule, kernel
parity rule, leakage-safe dtype vocabulary — committed as `docs:`
before any header changes.

Step 2 (`feat:`): `operation.hpp` dtype field + `valid()`/sizing,
CPU allocation byte-sizing + kernels, OpenCL scalar dtype kernels,
adapter dtype carriage.

Step 3 (`test:`): exact-value CPU + GPU chains per dtype (I8 exact,
F16 tolerance-documented), mixed-dtype conversion tests on
round-trippable values, gate pin on a dtype-carrying form, adapter
refusal tests for unknown dtypes.

Step 4: both configs + leakage + demo, checkpoint commits, clean tree.
