# Associative-reader component interventions

This is an exploratory checkpoint analysis of the three trained
`associative_read_depth_kda` seeds. It does not alter or overwrite the archived
softmax-versus-associative confirmatory comparison.

Under `archived_strict`, the evaluator reuses the archived `normal` and
gamma-zero `memory_output_off` records. Under
`recompute_current_environment`, it recomputes both references on the active
GPU. Both policies evaluate the following three new modes on the same frozen
validation blocks:

- `history_only`: read the decayed incoming state and retain the current write
  for later depth transitions, but do not read that write immediately.
- `current_correction_only`: read only the current gated-delta correction. Its
  innovation remains history-conditioned, so this is not a history-free model.
- `first_current_off`: suppress the current correction read only after the
  first attention sublayer. Because the incoming state is zero there, this
  removes an unambiguously same-layer branch while retaining its write.

Every mode preserves the full recurrent state update. The interventions change
later hidden states and therefore later writes, so their whole-model NLL effects
are not additive component attributions. In particular, `history_only` tests a
write-then-read-trained checkpoint under read-then-write-like inference; it does
not tell us how a read-then-write model would train.

## Integrity policy

Both policies validate the frozen manifest, preflight, analysis-plan hash,
shared-source hashes, and all six checkpoint identities. They also strictly
load each direct-reader checkpoint, prove installation leaves every state-dict
tensor unchanged, and exercise logits and diagnostics for all nine novel
seed/mode evaluations.

`archived_strict` additionally requires the archived BF16 runtime/hardware core,
loads the archived metric records, and reproduces the first frozen normal and
memory-off block within `1e-5` per sequence.

`recompute_current_environment` is the hardware-migration policy. It never
opens `comparison.json` or archived checkpoint-result JSON files, and therefore
cannot import their NLLs. It uses only frozen provenance from `preflight.json`,
then fully recomputes normal and gamma-zero for all three seeds on the active
GPU. Every resumable result is pinned to the reference policy, environment,
checkpoint hash, source hash, and physical GPU UUID.

Do not disable a failed integrity check. Use
`recompute_current_environment` when the archived hardware is unavailable.

## Run

For a hardware migration, from the repository root:

```bash
python att-residual-exp/test_associative_interventions.py
python att-residual-exp/test_associative_intervention_integrity.py
python att-residual-exp/test_associative_reference_policy.py
python att-residual-exp/test_associative_reference_resume.py
python att-residual-exp/test_evaluator.py

python -u att-residual-exp/evaluate_associative_interventions.py \
  --reference-policy recompute_current_environment --preflight-only

set -o pipefail
python -u att-residual-exp/evaluate_associative_interventions.py \
  --reference-policy recompute_current_environment --resume 2>&1 \
  | tee -a /workspace/attres-associative-interventions-rtx4090.log
```

This performs 15 full passes: five modes times three seeds, containing
62,914,560 scored target tokens. Results are written separately under:

```text
att-residual-exp/runs/associative_interventions_recomputed_4m_seed424242/
```

Completion check:

```bash
test -f \
  att-residual-exp/runs/associative_interventions_recomputed_4m_seed424242/summary.json \
  && echo COMPLETE || echo INCOMPLETE
```

The primary quantity is `mode_minus_normal_nll`: positive means suppressing
that read pathway hurt quality. `memory_off_minus_mode_nll` measures how much
of the full memory-off penalty the mode recovers. Training seeds—not validation
blocks—remain the independent architecture replicates.

The wrapper deliberately delegates the state update to the frozen transition
and then recomputes read components for measurement. Consequently, the novel
mode throughput and VRAM fields include instrumentation overhead and must not
be used as architecture-efficiency measurements.
