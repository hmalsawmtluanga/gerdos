# GERDOS pull request

## Ticket

ID / Title:

Lane / Owner:

Reviewers: M + <cross-lane junior>

## Contract

- [ ] Behaviour change started as a `docs:` commit (CORE_CONTRACT.md section or ADR link):
- [ ] Accuracy class (EXACT / BOUNDED / ORDERED_SUM) stated, or N/A:

## Definition of Done

- [ ] Follows L1–L9 (§1 of the Master Plan)
- [ ] Green on the Linux and Windows lanes (plus sanitizer lanes where applicable)
- [ ] Reviewed by the Orchestrator **and** one junior from a different lane
- [ ] Regression test fails before the fix (for `fix:` PRs — reviewer verified)
- [ ] Docs, CHANGELOG entry and README status line updated where relevant
- [ ] Under ~400 changed lines excluding generated files (or split with reason)
- [ ] GPU forms have a CPU reference form and parity tests (L2)
- [ ] Refusals state a reason and have refusal tests (L3)
- [ ] Evidence carries a `real-device` / `software-device` / `simulated` label (L4)
- [ ] No model knowledge outside `adapters/` and drivers (L5 — `gerdos_core_no_leakage` green)
- [ ] No device-identity branches outside `hw/quirks` (L6)

## Public claims

- [ ] N/A, or Owner sign-off obtained (L10)
