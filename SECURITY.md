# Security policy

## Supported versions

Only the latest `main` and the latest published minor release receive
security fixes.

## Reporting

Open a **private** report to the Owner (do not file a public issue for
weight-file crashes, fuzzer findings, or anything with a potential
attack surface). Weight files and `.gwd` artifacts are untrusted input:
loaders must stay bounds-checked and fuzzed.

## Runner hygiene

Self-hosted runners execute **only trusted `main` pushes, never fork PRs**.
See the Master Plan risk register.
