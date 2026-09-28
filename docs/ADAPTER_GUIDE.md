# Adapter Authoring Guide: Add a Model Term Fail-Closed

Model vocabulary lives in exactly one place: `include/gerdos/adapters/`.
The leakage guard (`core_no_leakage`) fails the build if model terms
appear anywhere else. Read `include/gerdos/adapters/model_adapter.hpp`
first — every existing term is a worked example.

## 1. Name the term

Add a `ModelOp` enumerator with a one-line comment stating its model
semantics. If `ModelStep` lacks a field your term needs (the third
operand `operand_c` was added for mask selection), append the field
with a default so existing construction sites keep compiling.

## 2. Translate exactly or refuse loudly

Add a `case` that either emits core work units (forms, dtypes, scales,
shapes — generic vocabulary only) or pushes a refusal string naming
the term and the exact missing capability:

- expressible prefix + missing closer → refuse the whole step (never
  emit a partial that masquerades as progress; see SOFTMAX).
- single-source limits: the affine form reads one source per producing
  entry — two-source sums are refused (see RESIDUAL_ADD).
- zero counts, incomplete shapes, vacuous minimums → refuse by name.

Dtype carriage is automatic (post-switch stamping) — but verify your
new case pushes exactly one operation so the stamp lands correctly.

## 3. Pin it in the test

Extend the fragment in `tests/adapter_model_translation.cpp`: the new
step, its translated shape assertions, and — for refusals — the exact
refusal prefix. Run both configs; leakage must stay green.

## 4. Checklist

- [ ] term documented in the enum comment
- [ ] translation exact or refusal names the gap
- [ ] no approximation, no silent defaulting
- [ ] fragment test covers translate + refuse paths
- [ ] `core_no_leakage` green (model terms only under `adapters/`)
