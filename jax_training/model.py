from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from model.mhlatent_attention import (
    MHLAConfig,
    deepseek_moe,
    init_deepseek_moe_params,
    init_mhla_params,
    mhlatent_attention,
    rms_norm,
)
from model.kimi_deltanet import (
    KimiDeltaNetConfig,
    init_kimi_deltanet_params,
    kimi_deltanet_parallel_chunkwise,
)
from model.deepseek_sparseatt import (
    DeepSeekSparseConfig,
    deepseek_sparse_attention,
    init_deepseek_sparse_params,
)
from model.deepseek_csa import (
    DeepSeekCSAConfig,
    deepseek_hybrid_attention,
    init_deepseek_csa_params,
    init_deepseek_hca_params,
)
from model.deepseek_mhc import (
    MHCConfig,
    init_mhc_params,
    mhc_block,
    mhc_readout,
)


@dataclass(frozen=True)
class JaxLMConfig:
    vocab_size: int
    max_seq_len: int
    model_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    latent_dim: int
    rope_dim: int
    attention_type: str = "mhla"
    chunk_size: int = 16
    deltanet_key_dim: int | None = None
    deltanet_value_dim: int | None = None
    deltanet_gate_type: str = "vector"
    index_dim: int = 32
    index_heads: int = 2
    csa_compress_rate: int = 4
    hca_compress_rate: int = 128
    local_window_size: int = 128
    num_mhc_streams: int = 4
    mhc_hidden_dim: int = 256
    mhc_sinkhorn_iters: int = 20
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 2048
    eps: float = 1e-6


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_mha_params(key, config: JaxLMConfig):
    keys = jax.random.split(key, 4)
    D = config.model_dim
    return {
        "q_proj": _xavier(keys[0], (D, D)),
        "k_proj": _xavier(keys[1], (D, D)),
        "v_proj": _xavier(keys[2], (D, D)),
        "out_proj": _xavier(keys[3], (D, D)),
    }


def apply_rope_to_heads(x, rope_dim):
    if rope_dim <= 0:
        return x
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")

    rope_dim = min(rope_dim, x.shape[-1])
    x_content = x[..., :-rope_dim]
    x_rope = x[..., -rope_dim:]
    half = rope_dim // 2
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]

    T = x.shape[1]
    positions = jnp.arange(T, dtype=x.dtype)
    freqs = 1.0 / (10000.0 ** (jnp.arange(half, dtype=x.dtype) / half))
    angles = positions[:, None] * freqs[None, :]
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    rope = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return jnp.concatenate([x_content, rope], axis=-1)


def apply_causal_mask(scores):
    T = scores.shape[-1]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    return jnp.where(mask[None, None, :, :], scores, -jnp.inf)


def mha_attention(x, params, config: JaxLMConfig):
    B, T, D = x.shape
    H = config.num_heads
    Dh = config.head_dim

    q = jnp.matmul(x, params["q_proj"])
    k = jnp.matmul(x, params["k_proj"])
    v = jnp.matmul(x, params["v_proj"])

    q = jnp.reshape(q, (B, T, H, Dh))
    k = jnp.reshape(k, (B, T, H, Dh))
    v = jnp.reshape(v, (B, T, H, Dh))

    q = apply_rope_to_heads(q, config.rope_dim)
    k = apply_rope_to_heads(k, config.rope_dim)

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    scores = scores / jnp.sqrt(jnp.asarray(Dh, dtype=x.dtype))
    scores = apply_causal_mask(scores)

    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(weights, v)
    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.reshape(out, (B, T, D))
    return jnp.matmul(out, params["out_proj"])


def _attention_config(config: JaxLMConfig) -> MHLAConfig:
    return MHLAConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        latent_dim=config.latent_dim,
        rope_dim=config.rope_dim,
        num_experts=config.num_routed_experts,
        num_routed_experts=config.num_routed_experts,
        num_shared_experts=config.num_shared_experts,
        top_k=config.top_k,
        expert_hidden_dim=config.expert_hidden_dim,
        eps=config.eps,
    )


def _kimi_config(config: JaxLMConfig) -> KimiDeltaNetConfig:
    key_dim = config.deltanet_key_dim or config.head_dim
    value_dim = config.deltanet_value_dim or config.head_dim
    return KimiDeltaNetConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=config.chunk_size,
        gate_type=config.deltanet_gate_type,
        eps=config.eps,
        num_routed_experts=config.num_routed_experts,
        num_shared_experts=config.num_shared_experts,
        top_k=config.top_k,
        expert_hidden_dim=config.expert_hidden_dim,
    )


def _sparse_config(config: JaxLMConfig) -> DeepSeekSparseConfig:
    return DeepSeekSparseConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        latent_dim=config.latent_dim,
        rope_dim=config.rope_dim,
        index_dim=config.index_dim,
        index_heads=config.index_heads,
        top_k=config.top_k,
        num_routed_experts=config.num_routed_experts,
        num_shared_experts=config.num_shared_experts,
        expert_hidden_dim=config.expert_hidden_dim,
        eps=config.eps,
    )


def _csa_config(config: JaxLMConfig) -> DeepSeekCSAConfig:
    return DeepSeekCSAConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        latent_dim=config.latent_dim,
        rope_dim=config.rope_dim,
        index_dim=config.index_dim,
        index_heads=config.index_heads,
        csa_compress_rate=config.csa_compress_rate,
        top_k=config.top_k,
        hca_compress_rate=config.hca_compress_rate,
        local_window_size=config.local_window_size,
        num_routed_experts=config.num_routed_experts,
        num_shared_experts=config.num_shared_experts,
        expert_hidden_dim=config.expert_hidden_dim,
        eps=config.eps,
    )


def _mhc_config(config: JaxLMConfig) -> MHCConfig:
    return MHCConfig(
        model_dim=config.model_dim,
        num_streams=config.num_mhc_streams,
        hidden_dim=config.mhc_hidden_dim,
        sinkhorn_iters=config.mhc_sinkhorn_iters,
        eps=config.eps,
    )


def init_lm_params(key, config: JaxLMConfig):
    if config.model_dim != config.num_heads * config.head_dim:
        raise ValueError("model_dim must equal num_heads * head_dim")
    if config.num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    keys = jax.random.split(key, 3 + 4 * config.num_layers)
    attn_config = _attention_config(config)
    kimi_config = _kimi_config(config)
    sparse_config = _sparse_config(config)
    csa_config = _csa_config(config)
    mhc_config = _mhc_config(config)

    blocks = []
    offset = 2
    for _ in range(config.num_layers):
        if config.attention_type == "mha":
            attn_params = init_mha_params(keys[offset], config)
        elif config.attention_type == "mhla":
            attn_params = init_mhla_params(keys[offset], attn_config)
        elif config.attention_type == "kimi_deltanet":
            attn_params = init_kimi_deltanet_params(keys[offset], kimi_config)
        elif config.attention_type == "deepseek_sparse":
            attn_params = init_deepseek_sparse_params(keys[offset], sparse_config)
        elif config.attention_type == "deepseek_csa_hca":
            csa_key, hca_key = jax.random.split(keys[offset])
            attn_params = {
                "csa": init_deepseek_csa_params(csa_key, csa_config),
                "hca": init_deepseek_hca_params(hca_key, csa_config),
            }
        elif config.attention_type == "deepseek_csa_hca_mhc":
            csa_key, hca_key, mhc_key = jax.random.split(keys[offset], 3)
            attn_params = {
                "csa": init_deepseek_csa_params(csa_key, csa_config),
                "hca": init_deepseek_hca_params(hca_key, csa_config),
                "mhc": init_mhc_params(mhc_key, mhc_config),
            }
        else:
            raise ValueError(f"unknown attention_type: {config.attention_type}")

        blocks.append({
            "attn_norm": jnp.ones((config.model_dim,), dtype=jnp.float32),
            "attn": attn_params,
            "moe_norm": jnp.ones((config.model_dim,), dtype=jnp.float32),
            "moe": init_deepseek_moe_params(keys[offset + 1], attn_config),
        })
        offset += 2

    return {
        "token_embedding": _xavier(keys[0], (config.vocab_size, config.model_dim)),
        "blocks": tuple(blocks),
        "final_norm": jnp.ones((config.model_dim,), dtype=jnp.float32),
        "lm_head": _xavier(keys[1], (config.model_dim, config.vocab_size)),
    }


def transformer_block(x, block_params, config: JaxLMConfig):
    attn_config = _attention_config(config)
    kimi_config = _kimi_config(config)
    sparse_config = _sparse_config(config)
    csa_config = _csa_config(config)
    mhc_config = _mhc_config(config)

    h = rms_norm(x, block_params["attn_norm"], eps=config.eps)
    if config.attention_type == "mha":
        h = mha_attention(h, block_params["attn"], config)
    elif config.attention_type == "mhla":
        h = mhlatent_attention(h, block_params["attn"], attn_config)
    elif config.attention_type == "kimi_deltanet":
        h = kimi_deltanet_parallel_chunkwise(h, block_params["attn"], kimi_config)
    elif config.attention_type == "deepseek_sparse":
        h = deepseek_sparse_attention(h, block_params["attn"], sparse_config)
    elif config.attention_type == "deepseek_csa_hca":
        h = deepseek_hybrid_attention(h, block_params["attn"], csa_config)
    elif config.attention_type == "deepseek_csa_hca_mhc":
        h_streams = jnp.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], config.num_mhc_streams, config.model_dim),
        )

        def layer_fn(h_in):
            hybrid_params = {
                "csa": block_params["attn"]["csa"],
                "hca": block_params["attn"]["hca"],
            }
            return deepseek_hybrid_attention(h_in, hybrid_params, csa_config)

        h_streams = mhc_block(
            h_streams,
            block_params["attn"]["mhc"],
            mhc_config,
            layer_fn,
        )
        x = mhc_readout(h_streams, block_params["attn"]["mhc"])

        h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
        x = x + deepseek_moe(h, block_params["moe"], attn_config)
        return x
    else:
        raise ValueError(f"unknown attention_type: {config.attention_type}")
    x = x + h

    h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
    x = x + deepseek_moe(h, block_params["moe"], attn_config)

    return x


def lm_forward(params, token_ids, config: JaxLMConfig):
    x = params["token_embedding"][token_ids]

    for block_params in params["blocks"]:
        x = transformer_block(x, block_params, config)

    x = rms_norm(x, params["final_norm"], eps=config.eps)
    return jnp.matmul(x, params["lm_head"])


def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1,
    )
    return -jnp.mean(target_log_probs)


def loss_fn(params, token_ids, targets, config: JaxLMConfig):
    logits = lm_forward(params, token_ids, config)
    return cross_entropy_loss(logits, targets)
