# Attention Residual vs. Softmax-Read Gated-Delta Depth Memory

This directory contains a controlled language-model experiment with two entry
points:

- `attention_residual_baseline.py`: Full Attention Residuals over Transformer
  depth.
- `softmax_read_depth_kda.py`: the proposed Softmax-Read Gated-Delta Depth
  Memory. It is KDA-inspired, but it is not a claim to implement the complete
  Kimi Delta Attention sequence layer. The gated delta recurrence is applied
  across Transformer sublayers, while ordinary causal attention still mixes
  tokens along the sequence.

Both models share the tokenizer, data loader, causal self-attention, RoPE,
normalization, feed-forward implementation, optimizer, scheduler, evaluation,
and checkpoint harness. The architectural residual path is the intended
difference.

## Architecture definitions

Each Transformer block contributes two depth steps: one self-attention
sublayer and one FFN sublayer. Therefore a model with `L` blocks has `D = 2L`
depth steps.

### Full Attention Residual baseline

Let `v_0` be the token embedding. At depth step `j`, the model computes a raw
sublayer output and appends it to the history:

```text
v_j = f_j(h_{j-1})
V_j = stack(v_0, ..., v_j)
h_j = sum_i softmax_i(w_j^T RMSNorm_j(V_j[i])) V_j[i]
```

The history contains raw attention/FFN branch outputs, not cumulative hidden
states. Every post-sublayer router has its own learned static pseudo-query
`w_j`, initialized to exactly zero, so routing begins uniformly over the
available history. There is no ordinary `hidden + delta` residual in this
baseline. The router after the last FFN is also evaluated, so the final model
state aggregates all `1 + 2L` sources.

### Softmax-Read Gated-Delta Depth Memory (KDA-inspired)

The proposal keeps ordinary Transformer residual additions and a per-token
slot state

```text
Z_j: [batch, sequence, S, r].
```

For sublayer `j`, first compute its raw branch delta and ordinary residual:

```text
delta_j = f_j(h_{j-1})
u_j = h_{j-1} + delta_j
```

At every live boundary (`j < D`), write that raw delta into the depth state:

```text
Z_tilde = alpha_j * Z_{j-1}
e_j     = v_j - Z_tilde^T k_j
Z_j     = Z_tilde + beta_j k_j e_j^T
```

Here `k_j`, `v_j`, `alpha_j`, and `beta_j` are projected independently for
each token from `RMSNorm(delta_j)`. The updated slots are then read with
softmax attention to prepare the next sublayer input:

```text
A_j = softmax_slots(q(u_j) K(Z_j)^T / sqrt(read_key_dim))
h_j = u_j + gamma_j W_o(A_j V(Z_j)).
```

Thus the implemented indexing is **compute -> residual add -> write -> read
for the next sublayer**. This is equivalent to saying that sublayer `j + 1`
reads the state written through sublayer `j`. The first read from an empty
state and the final write with no consumer are omitted, leaving exactly
`D - 1 = 2L - 1` live transitions.

The state is initialized to zero on every forward pass. It recurs only over
model depth, never over batches or sequence positions. Softmax mixes the `S`
slots belonging to the same token; it does not pool the sequence axis, so it
does not introduce future-token leakage. Storage is fixed in layer count at
`O(batch * sequence * S * r)`, and each new depth step performs a fixed-size
write and read.

Writes are dense in this first experiment: every channel of every token's
normalized sublayer delta is available to the learned write projections. There
is no `SparseCompress`, `DenseCompress`, or top-k content filter. Top-k is
reserved for a later, isolated ablation.

Initial writer settings are `alpha ~= 0.99` and `beta ~= 0.119`; the memory
read gate starts at `gamma = 1e-3`. The reference state and recurrence are kept
in FP32 even when the surrounding model uses BF16 autocast.

## Parameter matching

By default, the proposal constructs the corresponding Full Attention Residual
model, measures its trainable parameter count, and selects the nearest FFN
hidden width for the proposal. This absorbs the memory-module parameters so a
comparison is not won merely by adding capacity. The exact target, selected
width, and remaining parameter difference are written to each run's config.

Use `--no-param-match` only for the explicit equal-backbone-width ablation.

The included presets use a 1,024-token vocabulary and SwiGLU FFNs:

| Mode | Width | Blocks / depth steps | Heads | Sequence | Baseline FFN / params | Matched proposal FFN / params | Training target |
|---|---:|---:|---:|---:|---:|---:|---:|
| `smoke` | 64 | 3 / 6 | 4 | 24 | 128 / 256,128 | 48 / 256,146 | synthetic check |
| `pilot` | 256 | 8 / 16 | 4 | 256 | 768 / 7,366,912 | 693 / 7,369,446 | 20M sampled tokens |
| `full` | 384 | 12 / 24 | 6 | 512 | 1,024 / 22,077,312 | 949 / 22,083,550 | 100M sampled tokens |

The 12-block, roughly 22M-parameter preset is the recommended first serious
comparison: it is large enough to expose a meaningful depth bottleneck while
remaining practical on one rented consumer GPU. The 7.4M pilot should be used
to validate throughput and learning behavior before spending on it.

## Correctness checks

Run the architecture-level test first:

```powershell
python att-residual-exp/test_architectures.py
```

It checks both models for the expected output shape, finite forward/backward
values, finite gradients, and causal prefix invariance: changing the suffix of
an input must not change logits over its unchanged prefix. It also verifies
uniform zero-query AttnRes routing, the gated-delta initialization, and the
vectorized recurrence equation.

Each entry point also has a lightweight built-in smoke mode:

```powershell
python att-residual-exp/attention_residual_baseline.py --mode smoke
python att-residual-exp/softmax_read_depth_kda.py --mode smoke
```

## Training commands

`parameter_golf_sp1024` is the dataset key used by the shared loader; it maps
to the local `data/datasets/fineweb10B_sp1024` files.

Paired pilot, one seed:

```powershell
python att-residual-exp/attention_residual_baseline.py --mode pilot --dataset parameter_golf_sp1024 --seed 1337 --run-name pilot_seed1337
python att-residual-exp/softmax_read_depth_kda.py --mode pilot --dataset parameter_golf_sp1024 --seed 1337 --run-name pilot_seed1337
```

Before a full run, benchmark 500 optimization steps on the exact rented GPU:

```powershell
python att-residual-exp/attention_residual_baseline.py --mode full --dataset parameter_golf_sp1024 --max-steps 500 --run-name throughput_baseline
python att-residual-exp/softmax_read_depth_kda.py --mode full --dataset parameter_golf_sp1024 --max-steps 500 --run-name throughput_proposal
```

Then run the paired full comparison. Repeat the same commands with at least
three seeds for a result intended to support a claim.

```powershell
python att-residual-exp/attention_residual_baseline.py --mode full --dataset parameter_golf_sp1024 --seed 1337 --run-name full_seed1337
python att-residual-exp/softmax_read_depth_kda.py --mode full --dataset parameter_golf_sp1024 --seed 1337 --run-name full_seed1337
```

`--max-encoded-tokens` limits the corpus loaded into memory.
`--target-train-tokens` controls how many sampled training tokens are actually
processed; these are deliberately separate quantities. For example, after a
positive 100M-token result, a 200M-token confirmation uses
`--target-train-tokens 200000000`.

Before interpreting the final 100M-token checkpoint losses, run the larger
paired evaluation described in [COMMON_EVAL.md](COMMON_EVAL.md). It evaluates
all six checkpoints on one deterministic manifest drawn from a previously
untouched validation slice.

## Budget guidance

Do not translate the $200 budget directly into a large grid before measuring
throughput. For a target run,

```text
estimated cost = target tokens / (measured tokens/sec * 3600) * hourly GPU price.
```

A sensible staged allocation is:

1. Run all synthetic and smoke checks locally.
2. Spend at most about $10 on the paired 20M-token pilot and throughput probes.
3. Use roughly $45 for one-seed, one-change-at-a-time ablation screening.
4. Reserve roughly $65 for the main baseline/proposal comparison at three seeds.
5. Reserve roughly $60 for a deeper confirmation if the main signal is real.
6. Keep roughly $20 for failed jobs, profiling, and reruns.

Those are spending caps, not cost predictions; actual capacity depends on the
rented GPU and measured throughput. As of 2026-07-28, Runpod's published Pod
rates list an RTX 4090 at $0.69/hour and an H100 PCIe at $2.89/hour; verify the
[current pricing](https://www.runpod.io/pricing) and compare measured
tokens-per-dollar before choosing. Keep top-k, alternate optimizers, slot-count
sweeps, and other architecture changes out of the first paired comparison.

## Run artifacts

Runs are written under `att-residual-exp/runs/<architecture>/<run-name>/`:

- `config.json`: resolved model/training configuration and architecture metadata;
- `logs.jsonl`: training loss, validation loss, throughput, and memory-routing diagnostics;
- `checkpoints/best.pt`: best validation checkpoint;
- `checkpoints/latest.pt`: rotating resumable checkpoint;
- `summary.json`: best result, timing, token throughput, and peak CUDA memory.

The `runs` directory ignores generated artifacts in Git while retaining its
placeholder `.gitignore`.
