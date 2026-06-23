from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class LightningSparseConfig:
    model_dim: int
    num_heads: int
    latent_dim: int
    rope_dim: int
    index_dim: int
    index_heads: int
    top_k: int
    lse_alpha: float = 4.0
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    expert_hidden_dim: int = 2048
    eps: float = 1e-6


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _rms_normalize(x, eps):
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * rms


def validate_lightning_sparse_config(config):
    if config.model_dim < 1:
        raise ValueError("model_dim must be >= 1")
    if config.num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if config.latent_dim < 1:
        raise ValueError("latent_dim must be >= 1")
    if config.rope_dim < 1:
        raise ValueError("rope_dim must be >= 1")
    if config.rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")
    if config.index_dim < 1:
        raise ValueError("index_dim must be >= 1")
    if config.index_heads < 1:
        raise ValueError("index_heads must be >= 1")
    if config.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if config.lse_alpha <= 0:
        raise ValueError("lse_alpha must be > 0")


def validate_lightning_sparse_params(params, config):
    D = config.model_dim
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    Ih = config.index_heads
    I = config.index_dim

    expected_shapes = {
        "q_down": (D, C),
        "kv_down": (D, C),
        "k_rope": (D, R),
        "q_absorb": (C, H * C),
        "q_rope": (C, H * R),
        "idx_q": (D, Ih * I),
        "idx_k": (D, I),
        "idx_router": (D, Ih),
        "idx_log_tau": (Ih,),
        "out_proj": (H * C, D),
    }

    for name, expected_shape in expected_shapes.items():
        if name not in params:
            raise KeyError(f"missing Lightning sparse attention param: {name}")
        if params[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {params[name].shape}, expected {expected_shape}"
            )


def validate_lightning_sparse_inputs(x, params, config):
    validate_lightning_sparse_config(config)
    if x.ndim != 3:
        raise ValueError(f"x must be [B, T, D], got {x.shape}")
    if x.shape[-1] != config.model_dim:
        raise ValueError(
            f"x last dim is {x.shape[-1]}, expected model_dim={config.model_dim}"
        )
    if config.top_k > x.shape[1]:
        raise ValueError(
            f"top_k={config.top_k} cannot exceed sequence length T={x.shape[1]}"
        )
    validate_lightning_sparse_params(params, config)


def init_lightning_sparse_params(key, config):
    validate_lightning_sparse_config(config)
    keys = jax.random.split(key, 9)
    D = config.model_dim
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    Ih = config.index_heads
    I = config.index_dim

    return {
        "q_down": _xavier(keys[0], (D, C)),
        "kv_down": _xavier(keys[1], (D, C)),
        "k_rope": _xavier(keys[2], (D, R)),
        "q_absorb": _xavier(keys[3], (C, H * C)),
        "q_rope": _xavier(keys[4], (C, H * R)),
        "idx_q": _xavier(keys[5], (D, Ih * I)),
        "idx_k": _xavier(keys[6], (D, I)),
        "idx_router": _xavier(keys[7], (D, Ih)),
        "idx_log_tau": jnp.zeros((Ih,), dtype=jnp.float32),
        "out_proj": _xavier(keys[8], (H * C, D)),
    }


def lightning_index_scores(x, params, config):
    B, T, _ = x.shape
    Ih = config.index_heads
    I = config.index_dim
    alpha = jnp.asarray(config.lse_alpha, dtype=x.dtype)

    idx_q = jnp.matmul(x, params["idx_q"])
    idx_q = jnp.reshape(idx_q, [B, T, Ih, I])
    idx_q = _rms_normalize(idx_q, config.eps)

    idx_k = jnp.matmul(x, params["idx_k"])
    idx_k = _rms_normalize(idx_k, config.eps)

    gates = jax.nn.softmax(jnp.matmul(x, params["idx_router"]), axis=-1)
    tau = jnp.exp(params["idx_log_tau"]).astype(x.dtype)

    raw = jnp.einsum("bqhi,bki->bqkh", idx_q, idx_k)
    raw = raw / jnp.sqrt(jnp.asarray(I, dtype=x.dtype))
    raw = jax.nn.relu(raw * tau[None, None, None, :])

    z = jnp.log(gates[:, :, None, :] + config.eps) + alpha * raw
    scores = jax.nn.logsumexp(z, axis=-1) / alpha

    return apply_causal_mask(scores)


def lightning_sparse_attention(x, params, config):
    """
    DeepSeek sparse/DSA-style MLA attention with Lightning Indexer scoring.

    The selected KV interface is unchanged: one top-k KV set per query token.
    Only the indexer compression changes from linear head collapse to
    router-weighted logsumexp over calibrated retrieval perspectives.
    """
    validate_lightning_sparse_inputs(x, params, config)

    B, T, _ = x.shape
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    Ktop = config.top_k

    c_q = jnp.matmul(x, params["q_down"])
    c_kv = jnp.matmul(x, params["kv_down"])
    k_rope = jnp.matmul(x, params["k_rope"])

    q_a = jnp.matmul(c_q, params["q_absorb"])
    q_r = jnp.matmul(c_q, params["q_rope"])
    q_a = jnp.reshape(q_a, [B, T, H, C])
    q_r = jnp.reshape(q_r, [B, T, H, R])
    q = jnp.concatenate([q_a, apply_rope(q_r)], axis=-1)

    kv_key = jnp.concatenate([c_kv, apply_rope_mqa(k_rope)], axis=-1)
    kv_val = c_kv

    index_score = lightning_index_scores(x, params, config)
    _, top_indices = jax.lax.top_k(index_score, Ktop)

    selected_keys = gather_topk(kv_key, top_indices)
    selected_vals = gather_topk(kv_val, top_indices)

    scores = jnp.einsum("bqhd,bqkd->bqhk", q, selected_keys)
    scores = scores / jnp.sqrt(jnp.asarray(C + R, dtype=x.dtype))
    selected_valid = selected_causal_valid(top_indices)
    probs = safe_masked_softmax(scores, selected_valid[:, :, None, :], axis=-1)

    out = jnp.einsum("bqhk,bqkc->bqhc", probs, selected_vals)
    out = jnp.reshape(out, [B, T, H * C])
    return jnp.matmul(out, params["out_proj"])


def apply_rope(x):
    B, T, H, R = x.shape
    if R % 2 != 0:
        raise ValueError("rope_dim must be even")

    half = R // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    positions = jnp.arange(T, dtype=x.dtype)
    freqs = 1.0 / (10000.0 ** (jnp.arange(half, dtype=x.dtype) / half))
    angles = positions[:, None] * freqs[None, :]

    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]

    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def apply_rope_mqa(x):
    B, T, R = x.shape
    if R % 2 != 0:
        raise ValueError("rope_dim must be even")

    half = R // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    positions = jnp.arange(T, dtype=x.dtype)
    freqs = 1.0 / (10000.0 ** (jnp.arange(half, dtype=x.dtype) / half))
    angles = positions[:, None] * freqs[None, :]

    cos = jnp.cos(angles)[None, :, :]
    sin = jnp.sin(angles)[None, :, :]

    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def apply_causal_mask(scores):
    _, T, _ = scores.shape
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    return jnp.where(mask[None, :, :], scores, -jnp.inf)


def selected_causal_valid(top_indices):
    _, T, _ = top_indices.shape
    query_pos = jnp.arange(T)[None, :, None]
    return top_indices <= query_pos


def safe_masked_softmax(scores, mask, axis=-1):
    masked_scores = jnp.where(mask, scores, -jnp.inf)
    has_valid = jnp.any(mask, axis=axis, keepdims=True)
    safe_scores = jnp.where(has_valid, masked_scores, 0.0)
    probs = jax.nn.softmax(safe_scores, axis=axis)
    return jnp.where(mask, probs, 0.0)


def gather_topk(values, indices):
    B = values.shape[0]
    batch_indices = jnp.arange(B)[:, None, None]
    return values[batch_indices, indices]


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    config = LightningSparseConfig(
        model_dim=32,
        num_heads=4,
        latent_dim=8,
        rope_dim=4,
        index_dim=8,
        index_heads=3,
        top_k=2,
        lse_alpha=4.0,
    )

    param_key, x_key = jax.random.split(key)
    params = init_lightning_sparse_params(param_key, config)
    x = jax.random.normal(x_key, (2, 12, config.model_dim))

    y = lightning_sparse_attention(x, params, config)
    scores = lightning_index_scores(x, params, config)

    def tiny_loss(p):
        return jnp.mean(jnp.square(lightning_sparse_attention(x, p, config)))

    loss, grads = jax.value_and_grad(tiny_loss)(params)
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    )

    print("input:", x.shape)
    print("output:", y.shape)
    print("index scores:", scores.shape)
    print("loss:", loss)
    print("grad norm:", grad_norm)
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    np.testing.assert_equal(y.shape, (2, 12, config.model_dim))
    np.testing.assert_equal(scores.shape, (2, 12, 12))
