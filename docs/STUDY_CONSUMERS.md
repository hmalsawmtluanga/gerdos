# Comparative Study: Measurement-Informed Planning Across Two Consumers

Machine-scoped numbers with a full reproduction record. The portable
claim is structural (measured evidence changed the choices); ratios
belong to this machine and must be re-measured elsewhere.

## Reproduction record

| Field | Value |
| --- | --- |
| Date (UTC) | 2026-09-28 |
| Machine | Intel i5-13420H, Intel Graphics (RPL-P), /dev/dri/renderD128 |
| OS / toolchain | Linux, GCC 13, C++20, `-Wall -Wextra -Wpedantic` zero warnings |
| GPU drivers | OpenCL via `GERDOS_OPENCL_ROOT=$HOME/ocltmp/extracted/usr` (apt-extracted, no sudo); Vulkan offline SPIR-V |
| Config | default + Release (`build/`, `build-release/`); workload dirs `./workloads` |
| Workloads | `workloads/decoder_prefix.gwd` (7 ops), `workloads/signal_chain.gwd` (3 ops) |
| Batch | one attempt per operation, sequenced through plan/execute |
| Cache state | cold per process; evidence registries start empty per test |

## Observed (this machine, Release)

- Decoder prefix: 7/7 translated ops coherent; reference-exact values
  (`E0=403.428802`, `mean=69.503380`); evidence 7 observations.
- Signal chain: 3/3 translated ops coherent; `0.75` bit-exact across
  normalize/select/gather; evidence 3 observations.
- Staged weights (memory pressure): 4 ops in 2 capacity-fitting rounds
  against a 2 MiB budget; partials exact at `262144`; evidence 4.
- Demo (heterogeneous): measurement-informed planning beat declared
  order structurally on the run date (see demo output; ratio printed
  there is machine-local, not a portable claim).

## Refusals bounding the scope

SOFTMAX, ATTENTION, DIVIDE, LAYER_NORM, RESIDUAL_ADD (decoder side);
SIGNAL_THRESHOLD (signal side). Each names its missing form. No silent
approximation anywhere in either consumer.

## Constrained-profile bring-up (revival proving)

The revival claim is proven by constraint, not by new hardware: the
same suites run under `GERDOS_NO_THREADS` (inline pool, no threads)
and through the pure-C consumer (`cc`, never `c++`) plus the Python
driver (no compute in Python). Observed this machine:

- `GERDOS_NO_THREADS` probe: inline execution `42`, `pending=0`,
  config parse intact;
- pure-C consumer: `0.75` exact across normalize/select/gather,
  3 coherent, evidence 3, refusals loud (garbage, null runtime);
- Python driver: `coherent=3 evidence=3` on the signal chain;
- thread budget pinned: 3 pools x 4 workers, 192 bounded slots max.

A physical small board (Pi-class / 2 GB x86) with this table filled in
is the next proving step — the constraint modes above are the
dress rehearsal, not the performance.

## Standing caveats

CI (`.github/workflows/ci.yml`) is proven by clean-tree simulation
on this machine: 30/30 green without toolchains (hardware suites
absent-by-registration, reported visibly), both configs, plus the
Python driver step. Hosted-runner execution stays open until the
first push runs it. The second consumer
is in-repo: it proves the runtime composes across workload families,
not that an outside party has integrated. The outside-contribution
line stays open until one merges through the documented path.
