from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_training.data import get_batch, load_cached_lm_dataset
from jax_training.model import JaxLMConfig, init_lm_params, loss_fn


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1337
    dataset_name: str = "parameter_golf_sp1024"
    max_encoded_tokens: int = 150_000_000
    batch_size: int = 8
    seq_len: int = 512
    max_steps: int = 30_000
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    optimizer_type: str = "muon"
    muon_lr: float = 1e-2
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
    log_path: str = "experiment/jax_csa_hca_mhc_muon/summary.jsonl"


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
    @jax.jit
    def train_step(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb, model_config)
        grad_norm = tree_l2_norm(grads)
        params, opt_state, muon_lr, adamw_lr = muon_adamw_update(
            params,
            grads,
            opt_state,
            mask,
            train_config,
        )
        return params, opt_state, loss, grad_norm, muon_lr, adamw_lr

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


def main():
    train_config = TrainConfig()

    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    dataset = load_cached_lm_dataset(
        train_config.dataset_name,
        max_encoded_tokens=train_config.max_encoded_tokens,
    )

    model_config = JaxLMConfig(
        vocab_size=dataset.vocab_size,
        max_seq_len=512,
        model_dim=768,
        num_layers=6,
        num_heads=12,
        head_dim=64,
        latent_dim=192,
        rope_dim=32,
        attention_type="deepseek_csa_hca_mhc",
        chunk_size=16,
        index_dim=64,
        index_heads=4,
        csa_compress_rate=8,
        hca_compress_rate=64,
        local_window_size=64,
        num_mhc_streams=4,
        mhc_hidden_dim=1536,
        mhc_sinkhorn_iters=8,
        num_routed_experts=8,
        num_shared_experts=1,
        top_k=2,
        expert_hidden_dim=3072,
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
        "train_config": asdict(train_config),
        "model_config": asdict(model_config),
        "params": params_total,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }
    write_jsonl(train_config.log_path, run_header)
    print(
        f"params={params_total} "
        f"optimizer={train_config.optimizer_type} "
        f"attn={model_config.attention_type} "
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
        params, opt_state, train_loss, grad_norm, muon_lr, adamw_lr = train_step(
            params,
            opt_state,
            jnp.asarray(xb_np),
            jnp.asarray(yb_np),
        )

        train_loss_f = float(train_loss)
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
            tokens_per_sec = (
                train_config.batch_size
                * train_config.seq_len
                * steps
                / max(elapsed, 1e-8)
            )
            last_time = now
            tokens_seen = step * train_config.batch_size * train_config.seq_len
            elapsed_total = now - start_time
            log = {
                "step": step,
                "train_loss": train_loss_f,
                "val_loss": last_val_loss,
                "train_ema": train_ema,
                "val_ema": val_ema,
                "muon_lr": float(muon_lr),
                "adamw_lr": float(adamw_lr),
                "grad_norm": float(grad_norm),
                "tokens_sec": tokens_per_sec,
                "tokens_seen": tokens_seen,
                "elapsed_sec": elapsed_total,
                "batch_size": train_config.batch_size,
                "seq_len": train_config.seq_len,
                "params": params_total,
                "optimizer": train_config.optimizer_type,
                "layers": model_config.num_layers,
                "attn": model_config.attention_type,
                "seed": train_config.seed,
                "pos": "rope",
                "norm": "rmsnorm",
                "latent_dim": model_config.latent_dim,
                "csa_compress_rate": model_config.csa_compress_rate,
                "hca_compress_rate": model_config.hca_compress_rate,
                "local_window_size": model_config.local_window_size,
                "num_mhc_streams": model_config.num_mhc_streams,
                "mhc_sinkhorn_iters": model_config.mhc_sinkhorn_iters,
                "num_routed_experts": model_config.num_routed_experts,
                "num_shared_experts": model_config.num_shared_experts,
                "top_k": model_config.top_k,
                "expert_hidden_dim": model_config.expert_hidden_dim,
            }
            write_jsonl(train_config.log_path, log)
            print(
                f"step={step} "
                f"train_loss={train_loss_f:.4f} "
                f"val_loss={last_val_loss:.4f} "
                f"train_ema={train_ema:.4f} "
                f"val_ema={(val_ema if val_ema is not None else math.nan):.4f} "
                f"muon_lr={float(muon_lr):.2e} "
                f"adamw_lr={float(adamw_lr):.2e} "
                f"grad_norm={float(grad_norm):.4f} "
                f"tokens_sec={tokens_per_sec:.0f} "
                f"tokens_seen={tokens_seen} "
                f"batch_size={train_config.batch_size} "
                f"seq_len={train_config.seq_len} "
                f"params={params_total} "
                f"optimizer={train_config.optimizer_type} "
                f"layers={model_config.num_layers} "
                f"attn={model_config.attention_type} "
                f"seed={train_config.seed} "
                f"pos=rope "
                f"norm=rmsnorm"
            )


if __name__ == "__main__":
    main()
