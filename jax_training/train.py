from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_training.data import get_batch, load_cached_lm_dataset
from jax_training.model import JaxLMConfig, init_lm_params, loss_fn, loss_with_moe_aux


LATENT_SWEEP_DIMS = {
    "small": 96,
    "medium": 192,
    "large": 384,
}

ARCHITECTURE_PRESETS = {
    "csa_hca_moe": {
        "attention_type": "deepseek_csa_hca",
        "experiment_name": "jax_standard_training",
        "run_prefix": "deepseek_csa_hca_moe",
        "top_k": 8,
    },
    "csa_hca_mhc_moe": {
        "attention_type": "deepseek_csa_hca_mhc",
        "experiment_name": "jax_standard_training",
        "run_prefix": "deepseek_csa_hca_mhc_moe",
        "top_k": 8,
    },
    "kimi_deltanet_moe": {
        "attention_type": "kimi_deltanet",
        "experiment_name": "jax_standard_training",
        "run_prefix": "kimi_deltanet_moe",
        "top_k": 2,
    },
    "deepseek_sparse_moe": {
        "attention_type": "deepseek_sparse",
        "experiment_name": "jax_standard_training",
        "run_prefix": "deepseek_sparse_moe",
        "top_k": 8,
        "use_moe": True,
    },
    "deepseek_sparse": {
        "attention_type": "deepseek_sparse",
        "experiment_name": "jax_standard_training",
        "run_prefix": "deepseek_sparse",
        "top_k": 8,
        "use_moe": False,
    },
    "lightning_sparse_moe": {
        "attention_type": "lightning_sparse",
        "experiment_name": "jax_standard_training",
        "run_prefix": "lightning_sparse_moe",
        "top_k": 8,
        "use_moe": True,
    },
    "lightning_sparse": {
        "attention_type": "lightning_sparse",
        "experiment_name": "jax_standard_training",
        "run_prefix": "lightning_sparse",
        "top_k": 8,
        "use_moe": False,
    },
    "deepseek_mhla_moe": {
        "attention_type": "mhla",
        "experiment_name": "jax_standard_training",
        "run_prefix": "deepseek_mhla_moe",
        "top_k": 2,
    },
}






@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1337
    dataset_name: str = "parameter_golf_sp1024"
    max_encoded_tokens: int = 150_000_000
    batch_size: int = 4
    seq_len: int = 512
    max_steps: int = 30_000
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    optimizer_type: str = "muon"
    muon_lr: float = 0.003
    adamw_lr: float = 3e-4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1_000
    min_lr_ratio: float = 0.1
    ema_beta: float = 0.98
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    experiment_name: str = "jax_standard_training"
    architecture: str = "csa_hca_moe"
    latent_variant: str = "medium"
    run_name: str = ""
    log_path: str = ""
    preflight: bool = False
    fail_fast: bool = True
    max_grad_norm: float = 1_000.0
    grad_clip_norm: float = 0.0
    moe_aux_loss_weight: float = 0.01


class MuonAdamWState(NamedTuple):
    step: jnp.ndarray
    muon_momentum: object
    adam_m: object
    adam_v: object


def param_count(params) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def tree_l2_norm(tree):
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree))
    )


def make_muon_mask(params):
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    mask_leaves = []
    for path, leaf in leaves_with_path:
        path_text = "/".join(str(part) for part in path)
        is_matrix = getattr(leaf, "ndim", 0) == 2
        is_embedding = "token_embedding" in path_text or "lm_head" in path_text
        mask_leaves.append(bool(is_matrix and not is_embedding))
    return jax.tree_util.tree_unflatten(treedef, mask_leaves)


def init_muon_adamw_state(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return MuonAdamWState(
        step=jnp.asarray(0, dtype=jnp.int32),
        muon_momentum=zeros,
        adam_m=zeros,
        adam_v=zeros,
    )


def learning_rate_schedule(step, base_lr, warmup_steps, max_steps, min_lr_ratio):
    step_f = jnp.asarray(step, dtype=jnp.float32)
    warmup_f = jnp.asarray(max(warmup_steps, 1), dtype=jnp.float32)
    max_f = jnp.asarray(max(max_steps, warmup_steps + 1), dtype=jnp.float32)

    warmup = base_lr * step_f / warmup_f
    progress = (step_f - warmup_f) / jnp.maximum(max_f - warmup_f, 1.0)
    progress = jnp.clip(progress, 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    decay = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
    return jnp.where(step_f < warmup_f, warmup, decay)


def zeropower_via_newtonschulz5(g, steps=5, eps=1e-7):
    x = g.astype(jnp.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    x = x / (jnp.linalg.norm(x) + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transposed:
        x = x.T
    return x.astype(g.dtype)


def muon_adamw_update(params, grads, state, mask, train_config: TrainConfig):
    step = state.step + 1
    muon_lr = learning_rate_schedule(
        step,
        train_config.muon_lr,
        train_config.warmup_steps,
        train_config.max_steps,
        train_config.min_lr_ratio,
    )
    adamw_lr = learning_rate_schedule(
        step,
        train_config.adamw_lr,
        train_config.warmup_steps,
        train_config.max_steps,
        train_config.min_lr_ratio,
    )

    new_mu_m = jax.tree_util.tree_map(
        lambda mu_m, g, is_muon: (
            train_config.muon_momentum * mu_m + g if is_muon else mu_m
        ),
        state.muon_momentum,
        grads,
        mask,
    )
    new_adam_m = jax.tree_util.tree_map(
        lambda adam_m, g, is_muon: (
            adam_m
            if is_muon
            else train_config.adam_beta1 * adam_m + (1.0 - train_config.adam_beta1) * g
        ),
        state.adam_m,
        grads,
        mask,
    )
    new_adam_v = jax.tree_util.tree_map(
        lambda adam_v, g, is_muon: (
            adam_v
            if is_muon
            else train_config.adam_beta2 * adam_v
            + (1.0 - train_config.adam_beta2) * jnp.square(g)
        ),
        state.adam_v,
        grads,
        mask,
    )

    def update_param_leaf(p, g, is_muon, mu_m_new, adam_m_new, adam_v_new):
        if is_muon:
            update = zeropower_via_newtonschulz5(
                mu_m_new,
                steps=train_config.muon_ns_steps,
            )
            scale = jnp.sqrt(jnp.maximum(1.0, p.shape[0] / p.shape[1]))
            p_new = p * (1.0 - muon_lr * train_config.weight_decay)
            p_new = p_new - muon_lr * scale * update
            return p_new

        m_hat = adam_m_new / (1.0 - train_config.adam_beta1 ** step)
        v_hat = adam_v_new / (1.0 - train_config.adam_beta2 ** step)
        update = m_hat / (jnp.sqrt(v_hat) + train_config.adam_eps)
        p_new = p * (1.0 - adamw_lr * train_config.weight_decay)
        p_new = p_new - adamw_lr * update
        return p_new

    new_params = jax.tree_util.tree_map(
        update_param_leaf,
        params,
        grads,
        mask,
        new_mu_m,
        new_adam_m,
        new_adam_v,
    )

    return (
        new_params,
        MuonAdamWState(
            step=step,
            muon_momentum=new_mu_m,
            adam_m=new_adam_m,
            adam_v=new_adam_v,
        ),
        muon_lr,
        adamw_lr,
    )


def make_train_step(model_config: JaxLMConfig, train_config: TrainConfig, mask):
    def make_metrics(loss):
        expert_count = model_config.num_routed_experts
        return {
            "ce_loss": loss,
            "moe_aux_loss": jnp.asarray(0.0, dtype=loss.dtype),
            "loss_total": loss,
            "moe_router_entropy": jnp.asarray(0.0, dtype=loss.dtype),
            "moe_router_logit_std": jnp.asarray(0.0, dtype=loss.dtype),
            "moe_expert_frac": jnp.zeros((expert_count,), dtype=loss.dtype),
            "moe_router_prob_frac": jnp.zeros((expert_count,), dtype=loss.dtype),
            "moe_expert_frac_max": jnp.asarray(0.0, dtype=loss.dtype),
            "moe_expert_frac_min": jnp.asarray(0.0, dtype=loss.dtype),
            "moe_expert_frac_std": jnp.asarray(0.0, dtype=loss.dtype),
            "moe_dead_experts": jnp.asarray(0.0, dtype=loss.dtype),
        }

    def loss_and_metrics(params, xb, yb):
        if model_config.use_moe:
            return loss_with_moe_aux(params, xb, yb, model_config)
        loss = loss_fn(params, xb, yb, model_config)
        return loss, make_metrics(loss)

    @jax.jit
    def train_step(params, opt_state, xb, yb):
        (loss, metrics), grads = jax.value_and_grad(
            loss_and_metrics,
            has_aux=True,
        )(params, xb, yb)
        raw_grad_norm = tree_l2_norm(grads)
        clip_norm = jnp.asarray(train_config.grad_clip_norm, dtype=raw_grad_norm.dtype)
        clip_scale = jnp.where(
            (clip_norm > 0.0) & (raw_grad_norm > clip_norm),
            clip_norm / (raw_grad_norm + 1e-6),
            1.0,
        )
        grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)
        grad_norm = tree_l2_norm(grads)
        params, opt_state, muon_lr, adamw_lr = muon_adamw_update(
            params,
            grads,
            opt_state,
            mask,
            train_config,
        )
        return (
            params,
            opt_state,
            loss,
            metrics,
            grad_norm,
            raw_grad_norm,
            muon_lr,
            adamw_lr,
        )

    return train_step


def make_eval_step(model_config: JaxLMConfig):
    @jax.jit
    def eval_step(params, xb, yb):
        return loss_fn(params, xb, yb, model_config)

    return eval_step


def evaluate(params, dataset, model_config, train_config, rng, eval_step):
    losses = []
    for _ in range(train_config.eval_batches):
        xb_np, yb_np = get_batch(
            "val",
            dataset,
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
            rng=rng,
        )
        loss = eval_step(params, jnp.asarray(xb_np), jnp.asarray(yb_np))
        losses.append(float(loss))
    return float(np.mean(losses))


def write_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run standard JAX LM training.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Run a tiny stability/correctness pass suitable for a small GPU. "
            "This overrides model/training size, but keeps the selected architecture."
        ),
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Do not stop early on non-finite loss/grad or repeated instability.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1_000.0,
        help="Fail-fast threshold for raw gradient norm.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Clip global gradient norm before optimizer update; <=0 disables clipping.",
    )
    parser.add_argument(
        "--architecture",
        choices=tuple(ARCHITECTURE_PRESETS),
        default="csa_hca_moe",
        help=(
            "Architecture preset. DeepSeekMoE is applied after every attention "
            "block by jax_training.model."
        ),
    )
    parser.add_argument(
        "--latent-variant",
        choices=tuple(LATENT_SWEEP_DIMS),
        default="medium",
        help="Latent width preset used by compressed-attention variants.",
    )
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--rope-dim", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--index-dim", type=int, default=64)
    parser.add_argument("--index-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--lightning-lse-alpha", type=float, default=4.0)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--csa-compress-rate", type=int, default=8)
    parser.add_argument("--hca-compress-rate", type=int, default=64)
    parser.add_argument("--local-window-size", type=int, default=64)
    parser.add_argument("--num-mhc-streams", type=int, default=4)
    parser.add_argument("--mhc-hidden-dim", type=int, default=1536)
    parser.add_argument("--mhc-sinkhorn-iters", type=int, default=8)
    parser.add_argument("--num-routed-experts", type=int, default=8)
    parser.add_argument("--num-shared-experts", type=int, default=1)
    parser.add_argument("--expert-hidden-dim", type=int, default=3072)
    parser.add_argument("--deltanet-key-dim", type=int, default=None)
    parser.add_argument("--deltanet-value-dim", type=int, default=None)
    parser.add_argument(
        "--deltanet-gate-type",
        choices=("none", "scalar", "vector"),
        default="vector",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-encoded-tokens", type=int, default=150_000_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--moe-aux-loss-weight",
        type=float,
        default=0.01,
        help="Coefficient for the MoE router load-balancing auxiliary loss.",
    )
    return parser.parse_args()


def arg_was_set(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def cli_value(args, flag: str, default):
    attr = flag.removeprefix("--").replace("-", "_")
    return getattr(args, attr) if arg_was_set(flag) else default


def build_train_config(args) -> TrainConfig:
    preset = ARCHITECTURE_PRESETS[args.architecture]
    latent_dim = args.latent_dim or LATENT_SWEEP_DIMS[args.latent_variant]
    batch_size = args.batch_size
    seq_len = args.seq_len
    max_steps = args.max_steps if args.max_steps is not None else 30_000
    log_interval = args.log_interval
    eval_interval = args.eval_interval
    eval_batches = args.eval_batches
    max_encoded_tokens = args.max_encoded_tokens

    if args.preflight:
        batch_size = cli_value(args, "--batch-size", 1)
        seq_len = cli_value(args, "--seq-len", 64)
        max_steps = args.max_steps if args.max_steps is not None else 1000
        log_interval = cli_value(args, "--log-interval", 1)
        eval_interval = cli_value(args, "--eval-interval", 10)
        eval_batches = cli_value(args, "--eval-batches", 2)
        max_encoded_tokens = cli_value(args, "--max-encoded-tokens", 250_000)

    run_name = (
        f"{preset['run_prefix']}_latent{latent_dim}_muon_"
        f"{args.model_dim}d_{args.num_layers}l_seq{args.seq_len}"
    )
    if args.preflight:
        run_name = f"preflight_{args.architecture}"
    log_path = f"experiment/jax_standard_training/{args.architecture}/summary.jsonl"
    return replace(
        TrainConfig(),
        seed=args.seed,
        experiment_name=preset["experiment_name"],
        architecture=args.architecture,
        latent_variant=args.latent_variant,
        run_name=run_name,
        log_path=log_path,
        max_steps=max_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        max_encoded_tokens=max_encoded_tokens,
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_batches=eval_batches,
        preflight=args.preflight,
        fail_fast=not args.no_fail_fast,
        max_grad_norm=args.max_grad_norm,
        grad_clip_norm=args.grad_clip_norm,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
    )


def build_model_config(dataset, train_config: TrainConfig, args) -> JaxLMConfig:
    preset = ARCHITECTURE_PRESETS[train_config.architecture]
    latent_dim = args.latent_dim or LATENT_SWEEP_DIMS[train_config.latent_variant]
    top_k = args.top_k if args.top_k is not None else preset["top_k"]
    model_dim = args.model_dim
    num_layers = args.num_layers
    num_heads = args.num_heads
    head_dim = args.head_dim
    rope_dim = args.rope_dim
    chunk_size = args.chunk_size
    index_dim = args.index_dim
    index_heads = args.index_heads
    csa_compress_rate = args.csa_compress_rate
    hca_compress_rate = args.hca_compress_rate
    local_window_size = args.local_window_size
    num_mhc_streams = args.num_mhc_streams
    mhc_hidden_dim = args.mhc_hidden_dim
    mhc_sinkhorn_iters = args.mhc_sinkhorn_iters
    num_routed_experts = args.num_routed_experts
    num_shared_experts = args.num_shared_experts
    expert_hidden_dim = args.expert_hidden_dim
    moe_top_k = args.moe_top_k

    if train_config.preflight:
        model_dim = cli_value(args, "--model-dim", 256)
        num_layers = cli_value(args, "--num-layers", 2)
        num_heads = cli_value(args, "--num-heads", 4)
        head_dim = cli_value(args, "--head-dim", 64)
        latent_dim = cli_value(args, "--latent-dim", 64)
        rope_dim = cli_value(args, "--rope-dim", 16)
        chunk_size = cli_value(args, "--chunk-size", 8)
        index_dim = cli_value(args, "--index-dim", 32)
        index_heads = cli_value(args, "--index-heads", 2)
        top_k = cli_value(args, "--top-k", min(top_k, 4))
        csa_compress_rate = cli_value(args, "--csa-compress-rate", 4)
        hca_compress_rate = cli_value(args, "--hca-compress-rate", 32)
        local_window_size = cli_value(args, "--local-window-size", 32)
        num_mhc_streams = cli_value(args, "--num-mhc-streams", 2)
        mhc_hidden_dim = cli_value(args, "--mhc-hidden-dim", 512)
        mhc_sinkhorn_iters = cli_value(args, "--mhc-sinkhorn-iters", 4)
        num_routed_experts = cli_value(args, "--num-routed-experts", 2)
        num_shared_experts = cli_value(args, "--num-shared-experts", 1)
        moe_top_k = cli_value(args, "--moe-top-k", min(args.moe_top_k, num_routed_experts))
        expert_hidden_dim = cli_value(args, "--expert-hidden-dim", 512)
        top_k = max(top_k, 1)
        num_routed_experts = max(num_routed_experts, 1)
        num_shared_experts = max(num_shared_experts, 0)
        moe_top_k = min(max(moe_top_k, 1), num_routed_experts)

    return JaxLMConfig(
        vocab_size=dataset.vocab_size,
        max_seq_len=train_config.seq_len,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        attention_type=preset["attention_type"],
        chunk_size=chunk_size,
        deltanet_key_dim=args.deltanet_key_dim,
        deltanet_value_dim=args.deltanet_value_dim,
        deltanet_gate_type=args.deltanet_gate_type,
        index_dim=index_dim,
        index_heads=index_heads,
        csa_compress_rate=csa_compress_rate,
        hca_compress_rate=hca_compress_rate,
        local_window_size=local_window_size,
        num_mhc_streams=num_mhc_streams,
        mhc_hidden_dim=mhc_hidden_dim,
        mhc_sinkhorn_iters=mhc_sinkhorn_iters,
        num_routed_experts=num_routed_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        lightning_lse_alpha=args.lightning_lse_alpha,
        use_moe=bool(preset.get("use_moe", True)),
        moe_top_k=moe_top_k,
        expert_hidden_dim=expert_hidden_dim,
        moe_aux_loss_weight=train_config.moe_aux_loss_weight,
    )


def build_deepseek_mla_model_config(dataset, train_config: TrainConfig) -> JaxLMConfig:
    class Args:
        architecture = "deepseek_mhla_moe"
        model_dim = 768
        num_layers = 6
        num_heads = 12
        head_dim = 64
        latent_dim = None
        rope_dim = 32
        chunk_size = 16
        index_dim = 64
        index_heads = 4
        top_k = 2
        moe_top_k = 3
        csa_compress_rate = 8
        hca_compress_rate = 64
        local_window_size = 64
        num_routed_experts = 8
        num_shared_experts = 1
        expert_hidden_dim = 3072
        deltanet_key_dim = None
        deltanet_value_dim = None
        deltanet_gate_type = "vector"

    return build_model_config(
        dataset,
        replace(train_config, architecture="deepseek_mhla_moe"),
        Args,
    )


def first_block_mla_stats(params):
    attn = params["blocks"][0]["attn"]
    stats = {}
    for name in ("q_down", "kv_down", "q_proj", "k_proj", "v_proj", "out_proj"):
        if name in attn:
            stats[f"{name}_norm"] = float(jnp.linalg.norm(attn[name]))
    return stats


def run_training(train_config: TrainConfig, model_config: JaxLMConfig, dataset=None):
    if dataset is None:
        dataset = load_cached_lm_dataset(
            train_config.dataset_name,
            max_encoded_tokens=train_config.max_encoded_tokens,
        )
    key = jax.random.PRNGKey(train_config.seed)
    params = init_lm_params(key, model_config)
    params_total = param_count(params)
    mask = make_muon_mask(params)
    opt_state = init_muon_adamw_state(params)

    train_step = make_train_step(model_config, train_config, mask)
    eval_step = make_eval_step(model_config)

    rng = np.random.default_rng(train_config.seed)
    val_rng = np.random.default_rng(train_config.seed + 1)
    last_time = time.time()
    start_time = last_time
    train_ema = None
    val_ema = None
    last_val_loss = math.nan

    run_header = {
        "event": "run_start",
        "experiment_name": train_config.experiment_name,
        "run_name": train_config.run_name,
        "train_config": asdict(train_config),
        "model_config": asdict(model_config),
        "params": params_total,
        "param_million": params_total / 1_000_000,
        "latent_ratio": model_config.latent_dim / model_config.model_dim,
        "compression_ratio": model_config.model_dim / model_config.latent_dim,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }
    config_path = Path(train_config.log_path).with_name("config.json")
    write_json(config_path, run_header)
    write_jsonl(train_config.log_path, run_header)
    print(
        f"experiment={train_config.experiment_name} "
        f"run={train_config.run_name} "
        f"params={params_total} "
        f"optimizer={train_config.optimizer_type} "
        f"attn={model_config.attention_type} "
        f"latent_dim={model_config.latent_dim} "
        f"num_routed_experts={model_config.num_routed_experts} "
        f"use_moe={model_config.use_moe} "
        f"moe_top_k={model_config.moe_top_k} "
        f"top_k={model_config.top_k} "
        f"batch_size={train_config.batch_size} "
        f"seq_len={train_config.seq_len} "
        f"max_steps={train_config.max_steps}"
    )

    for step in range(1, train_config.max_steps + 1):
        xb_np, yb_np = get_batch(
            "train",
            dataset,
            batch_size=train_config.batch_size,
            seq_len=train_config.seq_len,
            rng=rng,
        )
        (
            params,
            opt_state,
            train_loss,
            train_metrics,
            grad_norm,
            raw_grad_norm,
            muon_lr,
            adamw_lr,
        ) = train_step(
            params,
            opt_state,
            jnp.asarray(xb_np),
            jnp.asarray(yb_np),
        )

        train_loss_f = float(train_loss)
        ce_loss_f = float(train_metrics["ce_loss"])
        moe_aux_loss_f = float(train_metrics["moe_aux_loss"])
        moe_router_entropy_f = float(train_metrics["moe_router_entropy"])
        moe_router_logit_std_f = float(train_metrics["moe_router_logit_std"])
        moe_expert_frac_max_f = float(train_metrics["moe_expert_frac_max"])
        moe_expert_frac_min_f = float(train_metrics["moe_expert_frac_min"])
        moe_expert_frac_std_f = float(train_metrics["moe_expert_frac_std"])
        moe_expert_frac = [
            float(x) for x in np.asarray(train_metrics["moe_expert_frac"])
        ]
        moe_router_prob_frac = [
            float(x) for x in np.asarray(train_metrics["moe_router_prob_frac"])
        ]
        moe_expert_frac_s = "[" + ",".join(f"{x:.2f}" for x in moe_expert_frac) + "]"
        moe_dead_experts_f = float(train_metrics["moe_dead_experts"])
        grad_norm_f = float(grad_norm)
        raw_grad_norm_f = float(raw_grad_norm)
        loss_is_finite = math.isfinite(train_loss_f)
        grad_is_finite = math.isfinite(grad_norm_f) and math.isfinite(raw_grad_norm_f)
        grad_is_too_large = raw_grad_norm_f > train_config.max_grad_norm

        if train_config.fail_fast and (
            not loss_is_finite or not grad_is_finite or grad_is_too_large
        ):
            reason = []
            if not loss_is_finite:
                reason.append("non_finite_loss")
            if not grad_is_finite:
                reason.append("non_finite_grad_norm")
            if grad_is_too_large:
                reason.append("grad_norm_above_threshold")
            stop_event = {
                "event": "run_stop",
                "status": "failed",
                "reason": ",".join(reason),
                "step": step,
                "train_loss": train_loss_f,
                "grad_norm": grad_norm_f,
                "raw_grad_norm": raw_grad_norm_f,
                "max_grad_norm": train_config.max_grad_norm,
                "grad_clip_norm": train_config.grad_clip_norm,
                "run_name": train_config.run_name,
                "architecture": train_config.architecture,
                "attn": model_config.attention_type,
                "preflight": train_config.preflight,
            }
            write_jsonl(train_config.log_path, stop_event)
            raise RuntimeError(
                "fail-fast stopped training: "
                f"reason={stop_event['reason']} step={step} "
                f"loss={train_loss_f} grad_norm={grad_norm_f} "
                f"raw_grad_norm={raw_grad_norm_f}"
            )

        if train_ema is None:
            train_ema = train_loss_f
        else:
            train_ema = train_config.ema_beta * train_ema + (1.0 - train_config.ema_beta) * train_loss_f

        should_eval = step == 1 or step % train_config.eval_interval == 0
        if should_eval:
            last_val_loss = evaluate(
                params,
                dataset,
                model_config,
                train_config,
                val_rng,
                eval_step,
            )
            if val_ema is None:
                val_ema = last_val_loss
            else:
                val_ema = train_config.ema_beta * val_ema + (1.0 - train_config.ema_beta) * last_val_loss

        if step == 1 or step % train_config.log_interval == 0 or should_eval:
            now = time.time()
            elapsed = now - last_time
            steps = 1 if step == 1 else train_config.log_interval
            step_time_ms = 1000.0 * elapsed / max(steps, 1)
            tokens_per_sec = (
                train_config.batch_size
                * train_config.seq_len
                * steps
                / max(elapsed, 1e-8)
            )
            last_time = now
            tokens_seen = step * train_config.batch_size * train_config.seq_len
            elapsed_total = now - start_time
            param_stats = first_block_mla_stats(params)
            log = {
                "experiment_name": train_config.experiment_name,
                "run_name": train_config.run_name,
                "latent_variant": train_config.latent_variant,
                "step": step,
                "train_loss": train_loss_f,
                "ce_loss": ce_loss_f,
                "moe_aux_loss": moe_aux_loss_f,
                "moe_router_entropy": moe_router_entropy_f,
                "moe_router_logit_std": moe_router_logit_std_f,
                "moe_expert_frac_max": moe_expert_frac_max_f,
                "moe_expert_frac_min": moe_expert_frac_min_f,
                "moe_expert_frac_std": moe_expert_frac_std_f,
                "moe_expert_frac": moe_expert_frac,
                "moe_router_prob_frac": moe_router_prob_frac,
                "moe_dead_experts": moe_dead_experts_f,
                "val_loss": last_val_loss,
                "train_ema": train_ema,
                "val_ema": val_ema,
                "muon_lr": float(muon_lr),
                "adamw_lr": float(adamw_lr),
                "grad_norm": grad_norm_f,
                "raw_grad_norm": raw_grad_norm_f,
                "grad_clip_norm": train_config.grad_clip_norm,
                "loss_is_finite": loss_is_finite,
                "grad_is_finite": grad_is_finite,
                "grad_is_too_large": grad_is_too_large,
                "tokens_sec": tokens_per_sec,
                "tokens_seen": tokens_seen,
                "elapsed_sec": elapsed_total,
                "step_time_ms": step_time_ms,
                "batch_size": train_config.batch_size,
                "seq_len": train_config.seq_len,
                "params": params_total,
                "param_million": params_total / 1_000_000,
                "optimizer": train_config.optimizer_type,
                "layers": model_config.num_layers,
                "attn": model_config.attention_type,
                "residual_type": model_config.residual_type,
                "seed": train_config.seed,
                "pos": "rope",
                "norm": "rmsnorm",
                "model_dim": model_config.model_dim,
                "num_heads": model_config.num_heads,
                "head_dim": model_config.head_dim,
                "latent_dim": model_config.latent_dim,
                "latent_ratio": model_config.latent_dim / model_config.model_dim,
                "compression_ratio": model_config.model_dim / model_config.latent_dim,
                "rope_dim": model_config.rope_dim,
                "index_dim": model_config.index_dim,
                "index_heads": model_config.index_heads,
                "key_dim": model_config.deltanet_key_dim or model_config.head_dim,
                "value_dim": model_config.deltanet_value_dim or model_config.head_dim,
                "state_dim": model_config.deltanet_key_dim or model_config.head_dim,
                "chunk_size": model_config.chunk_size,
                "gate_type": model_config.deltanet_gate_type,
                "fine_grained_gate": model_config.deltanet_gate_type == "vector",
                "scalar_gate": model_config.deltanet_gate_type == "scalar",
                "delta_rule": model_config.attention_type == "kimi_deltanet",
                "max_steps": train_config.max_steps,
                "warmup_steps": train_config.warmup_steps,
                "weight_decay": train_config.weight_decay,
                "csa_compress_rate": model_config.csa_compress_rate,
                "hca_compress_rate": model_config.hca_compress_rate,
                "local_window_size": model_config.local_window_size,
                "num_mhc_streams": model_config.num_mhc_streams,
                "mhc_sinkhorn_iters": model_config.mhc_sinkhorn_iters,
                "num_routed_experts": model_config.num_routed_experts,
                "num_shared_experts": model_config.num_shared_experts,
                "use_moe": model_config.use_moe,
                "top_k": model_config.top_k,
                "lightning_lse_alpha": model_config.lightning_lse_alpha,
                "moe_top_k": model_config.moe_top_k,
                "moe_aux_loss_weight": model_config.moe_aux_loss_weight,
                "expert_hidden_dim": model_config.expert_hidden_dim,
            }
            log.update(param_stats)
            write_jsonl(train_config.log_path, log)
            print(
                f"run={train_config.run_name} "
                f"step={step} "
                f"train_loss={train_loss_f:.4f} "
                f"ce_loss={ce_loss_f:.4f} "
                f"moe_aux={moe_aux_loss_f:.4f} "
                f"moe_dead={moe_dead_experts_f:.0f} "
                f"moe_frac_minmax=[{moe_expert_frac_min_f:.2f},{moe_expert_frac_max_f:.2f}] "
                f"moe_frac_by_expert={moe_expert_frac_s} "
                f"val_loss={last_val_loss:.4f} "
                f"train_ema={train_ema:.4f} "
                f"val_ema={(val_ema if val_ema is not None else math.nan):.4f} "
                f"muon_lr={float(muon_lr):.2e} "
                f"adamw_lr={float(adamw_lr):.2e} "
                f"grad_norm={grad_norm_f:.4f} "
                f"raw_grad_norm={raw_grad_norm_f:.4f} "
                f"tokens_sec={tokens_per_sec:.0f} "
                f"tokens_seen={tokens_seen} "
                f"batch_size={train_config.batch_size} "
                f"seq_len={train_config.seq_len} "
                f"params={params_total} "
                f"optimizer={train_config.optimizer_type} "
                f"layers={model_config.num_layers} "
                f"attn={model_config.attention_type} "
                f"num_routed_experts={model_config.num_routed_experts} "
                f"use_moe={model_config.use_moe} "
                f"moe_top_k={model_config.moe_top_k} "
                f"top_k={model_config.top_k} "
                f"latent_dim={model_config.latent_dim} "
                f"compression={model_config.model_dim / model_config.latent_dim:.2f} "
                f"seed={train_config.seed} "
                f"pos=rope "
                f"norm=rmsnorm"
            )

    write_jsonl(
        train_config.log_path,
        {
            "event": "run_stop",
            "status": "passed",
            "step": train_config.max_steps,
            "run_name": train_config.run_name,
            "architecture": train_config.architecture,
            "attn": model_config.attention_type,
            "preflight": train_config.preflight,
            "last_train_loss": train_ema,
            "last_val_loss": last_val_loss,
            "last_grad_norm": grad_norm_f if train_config.max_steps > 0 else math.nan,
            "last_raw_grad_norm": (
                raw_grad_norm_f if train_config.max_steps > 0 else math.nan
            ),
        },
    )


def main():
    args = parse_args()
    train_config = build_train_config(args)

    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    dataset = load_cached_lm_dataset(
        train_config.dataset_name,
        max_encoded_tokens=train_config.max_encoded_tokens,
    )
    model_config = build_model_config(dataset, train_config, args)
    run_training(train_config, model_config, dataset)


if __name__ == "__main__":
    main()
