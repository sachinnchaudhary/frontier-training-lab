# Associative-reader component interventions

This is an exploratory checkpoint analysis of the three trained
`associative_read_depth_kda` seeds. It does not alter or overwrite the archived
softmax-versus-associative confirmatory comparison.

The evaluator reuses the archived `normal` and gamma-zero `memory_output_off`
records, then computes three new modes on the same frozen validation blocks:

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

Before evaluating, the script:

- validates the archived manifest, comparison, preflight, analysis, source,
  checkpoint, normal, and memory-off hashes;
- requires the same BF16 runtime/hardware core and byte-identical shared source
  files as the confirmatory evaluation;
- re-evaluates the first frozen block for normal and memory-off in every seed,
  requiring all archived per-sequence NLL sums to reproduce within `1e-5`;
- strictly loads every checkpoint with weights-only deserialization;
- proves intervention installation leaves every state-dict tensor unchanged;
- exercises logits and diagnostics for all nine novel seed/mode evaluations.

Do not bypass a failed compatibility or numerical-anchor check. If the old
environment cannot be reproduced, revise the analysis plan to recompute full
normal and memory-off references in the new environment.

## Run

From the repository root:

```bash
python att-residual-exp/test_associative_interventions.py
python att-residual-exp/test_associative_intervention_integrity.py
python att-residual-exp/test_evaluator.py

python -u att-residual-exp/evaluate_associative_interventions.py --preflight-only

set -o pipefail
python -u att-residual-exp/evaluate_associative_interventions.py --resume 2>&1 \
  | tee -a /workspace/attres-associative-interventions.log
```

The nine new passes contain 37,748,736 scored target tokens. The six numerical
anchors add 98,304 target tokens. Results are written separately under:

```text
att-residual-exp/runs/associative_interventions_4m_seed424242/
```

Completion check:

```bash
test -f \
  att-residual-exp/runs/associative_interventions_4m_seed424242/summary.json \
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
