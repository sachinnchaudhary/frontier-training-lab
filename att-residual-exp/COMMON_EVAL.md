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

For the softmax-reader versus direct associative-reader comparison, use the
separate frozen analysis plan and output directory:

```bash
python att-residual-exp/test_evaluator.py
python -u att-residual-exp/evaluate_checkpoints.py --comparison reader --preflight-only
nohup python -u att-residual-exp/evaluate_checkpoints.py --comparison reader --resume > /workspace/attres-reader-eval.log 2>&1 &
tail -f /workspace/attres-reader-eval.log
```

The evaluator expects `latest.pt` for seeds 1337, 2027, and 3407 under both
architecture directories. It reconstructs each model from the checkpoint's
embedded configuration, requires a strict state-dict load, and evaluates one
model at a time.

In `--comparison reader` mode, the two architecture directories are
`softmax_read_gated_delta_depth_memory` and `associative_read_depth_kda`. The
evaluator additionally requires identical backbones and writer settings and
verifies that the direct model kept FFN width 949 rather than spending its
reader parameter savings on a wider FFN.

The original training harness repeatedly monitored the first 10M tokens of
`fineweb_val_000000.bin`. The AttnRes comparison froze `[10M, 20M)`. Reader
mode instead freezes `[20M, 30M)`, which was untouched by training and by the
prior common evaluation. Each plan samples 256 non-overlapping
16,384-target-token blocks, giving exactly 4,194,304 target tokens per
checkpoint and 25,165,824 across all six. The source shard, tokenizer, pool,
selected block IDs, and checkpoints are SHA-256 verified.

Reader mode evaluates each checkpoint twice: normally and with every
depth-memory output gamma set to zero. It therefore scores 50,331,648 target
tokens across twelve checkpoint evaluations. The difference
`memory_off_mean_nll - normal_mean_nll` is causal evidence that the trained
memory path is useful rather than ignored.

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

Reader-mode outputs use
`att-residual-exp/runs/common_eval_reader_4m_seed424242/`. Its CSV and
`comparison.json` also include memory-off penalties for both readers.

Mechanism diagnostics in each checkpoint result are explicitly
`diagnostics_first_batch`: they describe one fixed 2,048-token probe batch,
not all 4.19M evaluation tokens. Quality NLL and the memory-off penalty use the
full manifest.

Validation blocks are not independent training replicates. The primary
architecture interval in `comparison.json` is computed across the three paired
training seeds, never across the 256 blocks.

To package the completed evaluation:

```bash
tar -czf /workspace/attres-common-eval-results.tar.gz att-residual-exp/runs/common_eval_4m_seed424242
runpodctl send /workspace/attres-common-eval-results.tar.gz
```

For reader mode, substitute
`common_eval_reader_4m_seed424242` and a distinct archive name.
