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
