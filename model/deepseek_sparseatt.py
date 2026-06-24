from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DeepSeekSparseConfig:
    model_dim: int
    num_heads: int
    latent_dim: int
    rope_dim: int
    index_dim: int
    index_heads: int
    top_k: int

    # Use True for fair comparison with your Lightning shared-key indexer.
    # Use False for per-index-head keys.
    shared_index_key: bool = True

    # Differentiable indexer learning.
    index_aux_weight: float = 0.02
    teacher_temp: float = 1.0
    student_temp: float = 1.0

    # Optional: give selected index score a differentiable path into attention logits.
    # Keep 0.0 for clean baseline.
    index_score_bias_beta: float = 0.0

    # Plain SwiGLU FFN.
    expert_hidden_dim: int = 2048

    eps: float = 1e-6


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def _rms_normalize(x, eps=1e-6):
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * rms


def rms_norm(x, weight, eps=1e-6):
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * rms * weight


def validate_deepseek_sparse_config(config):
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
    if config.teacher_temp <= 0:
        raise ValueError("teacher_temp must be > 0")
    if config.student_temp <= 0:
        raise ValueError("student_temp must be > 0")
    if config.expert_hidden_dim < 1:
        raise ValueError("expert_hidden_dim must be >= 1")


def expected_idx_k_shape(config):
    D = config.model_dim
    I = config.index_dim
    Ih = config.index_heads

    if config.shared_index_key:
        return (D, I)

    return (D, Ih * I)


def validate_deepseek_sparse_params(params, config):
    D = config.model_dim
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    I = config.index_dim
    Ih = config.index_heads
    F = config.expert_hidden_dim

    expected_shapes = {
        "q_down": (D, C),
        "kv_down": (D, C),
        "k_rope": (D, R),

        "q_absorb": (C, H * C),
        "q_rope": (C, H * R),

        "idx_q": (D, Ih * I),
        "idx_k": expected_idx_k_shape(config),
        "idx_w": (D, Ih),

        "out_proj": (H * C, D),

        "ffn_norm": (D,),
        "ffn_gate": (D, F),
        "ffn_up": (D, F),
        "ffn_down": (F, D),
    }

    for name, shape in expected_shapes.items():
        if name not in params:
            raise KeyError(f"missing DeepSeek sparse attention param: {name}")
        if params[name].shape != shape:
            raise ValueError(
                f"{name} has shape {params[name].shape}, expected {shape}"
            )


def validate_deepseek_sparse_inputs(x, params, config):
    validate_deepseek_sparse_config(config)

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

    validate_deepseek_sparse_params(params, config)


def init_deepseek_sparse_params(key, config):
    validate_deepseek_sparse_config(config)

    keys = jax.random.split(key, 12)

    D = config.model_dim
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    I = config.index_dim
    Ih = config.index_heads
    F = config.expert_hidden_dim

    return {
        "q_down": _xavier(keys[0], (D, C)),
        "kv_down": _xavier(keys[1], (D, C)),
        "k_rope": _xavier(keys[2], (D, R)),

        "q_absorb": _xavier(keys[3], (C, H * C)),
        "q_rope": _xavier(keys[4], (C, H * R)),

        "idx_q": _xavier(keys[5], (D, Ih * I)),
        "idx_k": _xavier(keys[6], expected_idx_k_shape(config)),
        "idx_w": _xavier(keys[7], (D, Ih)),

        "out_proj": _xavier(keys[8], (H * C, D)),

        "ffn_norm": jnp.ones((D,), dtype=jnp.float32),
        "ffn_gate": _xavier(keys[9], (D, F)),
        "ffn_up": _xavier(keys[10], (D, F)),
        "ffn_down": _xavier(keys[11], (F, D)),
    }


def causal_mask(T):
    return jnp.tril(jnp.ones((T, T), dtype=bool))


def apply_causal_mask(scores):
    # scores: [B, T, T]
    _, T, _ = scores.shape
    mask = causal_mask(T)
    return jnp.where(mask[None, :, :], scores, -jnp.inf)


def selected_causal_valid(top_indices):
    # top_indices: [B, T, K]
    _, T, _ = top_indices.shape
    query_pos = jnp.arange(T)[None, :, None]
    return top_indices <= query_pos


def safe_masked_softmax(scores, mask, axis=-1):
    masked_scores = jnp.where(mask, scores, -jnp.inf)
    has_valid = jnp.any(mask, axis=axis, keepdims=True)
    safe_scores = jnp.where(has_valid, masked_scores, 0.0)
    probs = jax.nn.softmax(safe_scores, axis=axis)
    return jnp.where(mask, probs, 0.0)


def safe_masked_log_softmax(scores, mask, axis=-1):
    masked_scores = jnp.where(mask, scores, -jnp.inf)
    has_valid = jnp.any(mask, axis=axis, keepdims=True)
    safe_scores = jnp.where(has_valid, masked_scores, 0.0)
    log_probs = jax.nn.log_softmax(safe_scores, axis=axis)
    return jnp.where(mask, log_probs, 0.0)


def gather_topk(values, indices):
    # values:  [B, T, D]
    # indices: [B, Q, K]
    # return:  [B, Q, K, D]
    B = values.shape[0]
    batch_indices = jnp.arange(B)[:, None, None]
    return values[batch_indices, indices]


def normalize_selected_bias(bias, valid_mask, eps=1e-6):
    # bias:       [B, T, K]
    # valid_mask: [B, T, K]
    bias = jnp.where(valid_mask, bias, 0.0)

    count = jnp.maximum(jnp.sum(valid_mask, axis=-1, keepdims=True), 1)
    mean = jnp.sum(bias, axis=-1, keepdims=True) / count

    centered = jnp.where(valid_mask, bias - mean, 0.0)
    var = jnp.sum(jnp.square(centered), axis=-1, keepdims=True) / count

    normed = centered * jax.lax.rsqrt(var + eps)
    return jnp.where(valid_mask, normed, 0.0)


def apply_rope(x):
    # x: [B, T, H, R]
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

    return jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        axis=-1,
    )


def apply_rope_mqa(x):
    # x: [B, T, R]
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

    return jnp.concatenate(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        axis=-1,
    )


def build_mla_qkv(x, params, config):
    B, T, _ = x.shape

    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim

    c_q = jnp.matmul(x, params["q_down"])       # [B, T, C]
    c_kv = jnp.matmul(x, params["kv_down"])     # [B, T, C]
    k_rope = jnp.matmul(x, params["k_rope"])    # [B, T, R]

    q_a = jnp.matmul(c_q, params["q_absorb"])   # [B, T, H*C]
    q_r = jnp.matmul(c_q, params["q_rope"])     # [B, T, H*R]

    q_a = jnp.reshape(q_a, [B, T, H, C])
    q_r = jnp.reshape(q_r, [B, T, H, R])

    q = jnp.concatenate([q_a, apply_rope(q_r)], axis=-1)

    kv_key = jnp.concatenate([c_kv, apply_rope_mqa(k_rope)], axis=-1)
    kv_val = c_kv

    return q, kv_key, kv_val


def deepseek_index_scores(x, params, config):
    """
    DeepSeek-style indexer score.

    Baseline compression:
        score(t, k) = sum_i gate_i(t) * ReLU(<q_i(t), k_i(k)> / sqrt(index_dim))

    If shared_index_key=True:
        k_i(k) is shared across indexer heads.
    If shared_index_key=False:
        each indexer head has its own key projection.
    """
    B, T, _ = x.shape

    Ih = config.index_heads
    I = config.index_dim

    idx_q = jnp.matmul(x, params["idx_q"])
    idx_q = jnp.reshape(idx_q, [B, T, Ih, I])
    idx_q = _rms_normalize(idx_q, config.eps)

    idx_k = jnp.matmul(x, params["idx_k"])

    if config.shared_index_key:
        # idx_k: [B, T, I]
        idx_k = _rms_normalize(idx_k, config.eps)
        raw = jnp.einsum("bqhi,bki->bqkh", idx_q, idx_k)
    else:
        # idx_k: [B, T, Ih, I]
        idx_k = jnp.reshape(idx_k, [B, T, Ih, I])
        idx_k = _rms_normalize(idx_k, config.eps)
        raw = jnp.einsum("bqhi,bkhi->bqkh", idx_q, idx_k)

    raw = raw / jnp.sqrt(jnp.asarray(I, dtype=x.dtype))
    raw = jax.nn.relu(raw)

    gates = jax.nn.softmax(jnp.matmul(x, params["idx_w"]), axis=-1)

    index_score = jnp.einsum("bqkh,bqh->bqk", raw, gates)
    return apply_causal_mask(index_score)


def indexer_auxiliary_loss(index_score, q, kv_key, config):
    """
    Differentiable teacher loss for the sparse indexer.

    Because hard top-k integer indices do not give gradient to indexer parameters,
    this loss trains the indexer to approximate dense MLA attention geometry.
    """
    B, T, H, Dq = q.shape
    mask = causal_mask(T)

    teacher_logits = jnp.einsum("bqhd,bkd->bqhk", q, kv_key)
    teacher_logits = teacher_logits / jnp.sqrt(jnp.asarray(Dq, dtype=q.dtype))

    teacher_probs_h = safe_masked_softmax(
        teacher_logits / config.teacher_temp,
        mask[None, :, None, :],
        axis=-1,
    )

    teacher_probs = jnp.mean(teacher_probs_h, axis=2)
    teacher_probs = jax.lax.stop_gradient(teacher_probs)

    student_log_probs = safe_masked_log_softmax(
        index_score / config.student_temp,
        mask[None, :, :],
        axis=-1,
    )

    aux = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
    return jnp.mean(aux)


def feedforward(x, params):
    gate = jnp.matmul(x, params["ffn_gate"])
    up = jnp.matmul(x, params["ffn_up"])
    hidden = jax.nn.silu(gate) * up
    return jnp.matmul(hidden, params["ffn_down"])


def deepseek_sparse_attention(x, params, config, return_aux=False, return_info=False):
    """
    Corrected DeepSeek-style sparse MLA baseline.

    Output interface:
        if return_aux=False:
            return block_out

        if return_aux=True:
            return block_out, index_aux_loss

        if return_info=True:
            return block_out, index_aux_loss, info
    """
    validate_deepseek_sparse_inputs(x, params, config)

    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    Ktop = config.top_k

    q, kv_key, kv_val = build_mla_qkv(x, params, config)

    index_score = deepseek_index_scores(x, params, config)

    top_values, top_indices = jax.lax.top_k(index_score, Ktop)

    selected_keys = gather_topk(kv_key, top_indices)  # [B, T, K, C+R]
    selected_vals = gather_topk(kv_val, top_indices)  # [B, T, K, C]

    attn_logits = jnp.einsum("bqhd,bqkd->bqhk", q, selected_keys)
    attn_logits = attn_logits / jnp.sqrt(jnp.asarray(C + R, dtype=x.dtype))

    selected_valid = selected_causal_valid(top_indices)

    if config.index_score_bias_beta != 0.0:
        selected_index_score = gather_topk(index_score, top_indices)
        selected_index_score = normalize_selected_bias(
            selected_index_score,
            selected_valid,
            eps=config.eps,
        )
        attn_logits = (
            attn_logits
            + config.index_score_bias_beta * selected_index_score[:, :, None, :]
        )

    probs = safe_masked_softmax(
        attn_logits,
        selected_valid[:, :, None, :],
        axis=-1,
    )

    out = jnp.einsum("bqhk,bqkc->bqhc", probs, selected_vals)
    out = jnp.reshape(out, [x.shape[0], x.shape[1], H * C])
    out = jnp.matmul(out, params["out_proj"])

    # This preserves your simple block-local FFN style.
    # If your outer transformer already does residual/pre-norm, adapt this there.
    block_out = out + feedforward(rms_norm(out, params["ffn_norm"], config.eps), params)

    aux_loss = indexer_auxiliary_loss(index_score, q, kv_key, config)

    if return_info:
        info = {
            "index_score": index_score,
            "top_values": top_values,
            "top_indices": top_indices,
            "selected_valid": selected_valid,
            "valid_fraction": jnp.mean(selected_valid.astype(jnp.float32)),
        }
        return block_out, aux_loss, info

    if return_aux:
        return block_out, aux_loss

    return block_out


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    config = DeepSeekSparseConfig(
        model_dim=32,
        num_heads=4,
        latent_dim=8,
        rope_dim=4,
        index_dim=8,
        index_heads=3,
        top_k=4,
        shared_index_key=True,
        index_aux_weight=0.02,
        index_score_bias_beta=0.0,
        expert_hidden_dim=64,
    )

    param_key, x_key = jax.random.split(key)
    params = init_deepseek_sparse_params(param_key, config)
    x = jax.random.normal(x_key, (2, 12, config.model_dim))

    y, aux, info = deepseek_sparse_attention(
        x,
        params,
        config,
        return_aux=True,
        return_info=True,
    )

    def tiny_loss(p):
        y, index_aux = deepseek_sparse_attention(x, p, config, return_aux=True)
        main_loss = jnp.mean(jnp.square(y))
        return main_loss + config.index_aux_weight * index_aux

    loss, grads = jax.value_and_grad(tiny_loss)(params)

    print("input:", x.shape)
    print("output:", y.shape)
    print("aux:", float(aux))
    print("loss:", float(loss))
    print("valid_fraction:", float(info["valid_fraction"]))
    print("total_grad_norm:", float(tree_l2_norm(grads)))

    for name in ["idx_q", "idx_k", "idx_w"]:
        print(name, "grad_norm:", float(tree_l2_norm(grads[name])))

    np.testing.assert_equal(y.shape, (2, 12, config.model_dim))

    # Make sure selected masking is causal.
    selected_valid = info["selected_valid"]
    top_indices = info["top_indices"]
    query_pos = jnp.arange(x.shape[1])[None, :, None]
    causal_ok = jnp.all(jnp.where(selected_valid, top_indices <= query_pos, True))
    np.testing.assert_equal(bool(np.asarray(causal_ok)), True)

    print("backend:", jax.default_backend())
    print("devices:", jax.devices())