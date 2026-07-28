# Shared 100M Checkpoint Evaluation

The in-training validation estimate used 20 batches of 4 sequences, or 40,960
target tokens. It is paired within a training seed, but each training seed uses
a different fixed validation sample. Before comparing the final checkpoints,
evaluate all six on one larger shared manifest.

From the repository root:

```bash
python att-residual-exp/test_evaluator.py
python -u att-residual-exp/evaluate_checkpoints.py --preflight-only
nohup python -u att-residual-exp/evaluate_checkpoints.py --resume > /workspace/attres-common-eval.log 2>&1 &
tail -f /workspace/attres-common-eval.log
```

The evaluator expects `latest.pt` for seeds 1337, 2027, and 3407 under both
architecture directories. It reconstructs each model from the checkpoint's
embedded configuration, requires a strict state-dict load, and evaluates one
model at a time.

The original training harness repeatedly monitored the first 10M tokens of
`fineweb_val_000000.bin`. This evaluator freezes a deterministic sample of 256
non-overlapping 16,384-target blocks from the previously untouched raw
validation slice `[10M, 20M)`. This gives exactly 4,194,304 target tokens per
checkpoint and 25,165,824 across all six. The source shard, tokenizer, pool,
selected block IDs, and checkpoints are SHA-256 verified.

Re-running the identical command with `--resume` skips only complete results
whose checkpoint, manifest, environment, and evaluation hashes match. Outputs
are written crash-safely under
`att-residual-exp/runs/common_eval_4m_seed424242/`:

- `eval_manifest.json`: exact data identity and selected block IDs;
- `checkpoint_results/*.json`: per-checkpoint totals and paired block/sequence
  losses;
- `results.csv`: compact checkpoint-level table;
- `comparison.json`: paired seed deltas, training-seed uncertainty, conditional
  block-bootstrap uncertainty, and the predeclared 0.01-NLL decision rule.

Validation blocks are not independent training replicates. The primary
architecture interval in `comparison.json` is computed across the three paired
training seeds, never across the 256 blocks.

To package the completed evaluation:

```bash
tar -czf /workspace/attres-common-eval-results.tar.gz att-residual-exp/runs/common_eval_4m_seed424242
runpodctl send /workspace/attres-common-eval-results.tar.gz
```
