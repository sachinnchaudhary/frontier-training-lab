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
from model.multi_lightening_index import (
    LightningSparseConfig,
    init_lightning_sparse_params,
    lightning_sparse_attention,
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
    residual_type: str = "ordinary"
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
    deepseek_shared_index_key: bool = True
    deepseek_index_aux_weight: float = 0.02
    deepseek_teacher_temp: float = 1.0
    deepseek_student_temp: float = 1.0
    deepseek_index_score_bias_beta: float = 0.0
    lightning_lse_alpha: float = 4.0
    lightning_index_aux_weight: float = 0.02
    lightning_teacher_temp: float = 1.0
    lightning_student_temp: float = 1.0
    lightning_index_score_bias_beta: float = 0.0
    use_moe: bool = True
    moe_top_k: int = 2
    expert_hidden_dim: int = 2048
    moe_aux_loss_weight: float = 0.0
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
        top_k=config.moe_top_k,
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
        top_k=config.moe_top_k,
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
        shared_index_key=config.deepseek_shared_index_key,
        index_aux_weight=config.deepseek_index_aux_weight,
        teacher_temp=config.deepseek_teacher_temp,
        student_temp=config.deepseek_student_temp,
        index_score_bias_beta=config.deepseek_index_score_bias_beta,
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


def _lightning_config(config: JaxLMConfig) -> LightningSparseConfig:
    return LightningSparseConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        latent_dim=config.latent_dim,
        rope_dim=config.rope_dim,
        index_dim=config.index_dim,
        index_heads=config.index_heads,
        top_k=config.top_k,
        lse_alpha=config.lightning_lse_alpha,
        index_aux_weight=config.lightning_index_aux_weight,
        teacher_temp=config.lightning_teacher_temp,
        student_temp=config.lightning_student_temp,
        index_score_bias_beta=config.lightning_index_score_bias_beta,
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
    lightning_config = _lightning_config(config)
    mhc_config = _mhc_config(config)

    blocks = []
    offset = 2
    for _ in range(config.num_layers):
        if config.attention_type == "mha":
            attn_params = init_mha_params(keys[offset], config)
        elif config.attention_type == "mha_mhc":
            mha_key, mhc_key = jax.random.split(keys[offset])
            attn_params = {
                "mha": init_mha_params(mha_key, config),
                "mhc": init_mhc_params(mhc_key, mhc_config),
            }
        elif config.attention_type == "mhla":
            attn_params = init_mhla_params(keys[offset], attn_config)
        elif config.attention_type == "kimi_deltanet":
            attn_params = init_kimi_deltanet_params(keys[offset], kimi_config)
        elif config.attention_type == "deepseek_sparse":
            attn_params = init_deepseek_sparse_params(keys[offset], sparse_config)
        elif config.attention_type == "lightning_sparse":
            attn_params = init_lightning_sparse_params(keys[offset], lightning_config)
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

        block = {
            "attn_norm": jnp.ones((config.model_dim,), dtype=jnp.float32),
            "attn": attn_params,
        }
        if config.use_moe:
            block.update({
                "moe_norm": jnp.ones((config.model_dim,), dtype=jnp.float32),
                "moe": init_deepseek_moe_params(keys[offset + 1], attn_config),
            })
        blocks.append(block)
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
    lightning_config = _lightning_config(config)
    mhc_config = _mhc_config(config)

    h = rms_norm(x, block_params["attn_norm"], eps=config.eps)
    if config.attention_type == "mha":
        h = mha_attention(h, block_params["attn"], config)
    elif config.attention_type == "mha_mhc":
        h_streams = jnp.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], config.num_mhc_streams, config.model_dim),
        )

        def layer_fn(h_in):
            return mha_attention(h_in, block_params["attn"]["mha"], config)

        h_streams = mhc_block(
            h_streams,
            block_params["attn"]["mhc"],
            mhc_config,
            layer_fn,
        )
        x = mhc_readout(h_streams, block_params["attn"]["mhc"])

        if not config.use_moe:
            return x
        h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
        x = x + deepseek_moe(h, block_params["moe"], attn_config)
        return x
    elif config.attention_type == "mhla":
        h = mhlatent_attention(h, block_params["attn"], attn_config)
    elif config.attention_type == "kimi_deltanet":
        h = kimi_deltanet_parallel_chunkwise(h, block_params["attn"], kimi_config)
    elif config.attention_type == "deepseek_sparse":
        h = deepseek_sparse_attention(h, block_params["attn"], sparse_config)
    elif config.attention_type == "lightning_sparse":
        h = lightning_sparse_attention(h, block_params["attn"], lightning_config)
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

        if not config.use_moe:
            return x
        h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
        x = x + deepseek_moe(h, block_params["moe"], attn_config)
        return x
    else:
        raise ValueError(f"unknown attention_type: {config.attention_type}")
    x = x + h

    if not config.use_moe:
        return x
    h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
    x = x + deepseek_moe(h, block_params["moe"], attn_config)

    return x


def moe_router_aux_stats(x, moe_params, config: JaxLMConfig):
    attn_config = _attention_config(config)
    router_logits = jnp.matmul(x, moe_params["router"])
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    _, top_indices = jax.lax.top_k(router_logits, k=attn_config.top_k)

    E = attn_config.num_routed_experts
    K = attn_config.top_k
    one_hot = jax.nn.one_hot(top_indices, E, dtype=x.dtype)
    selected = jnp.sum(one_hot, axis=-2)
    token_fraction = jnp.mean(selected, axis=(0, 1)) / K
    prob_fraction = jnp.mean(router_probs, axis=(0, 1))
    aux_loss = E * jnp.sum(token_fraction * prob_fraction)

    entropy = -jnp.sum(router_probs * jnp.log(router_probs + config.eps), axis=-1)
    dead_experts = jnp.sum(token_fraction <= 0.0)
    return {
        "moe_aux_loss": aux_loss,
        "moe_router_entropy": jnp.mean(entropy),
        "moe_router_logit_std": jnp.std(router_logits),
        "moe_expert_frac": token_fraction,
        "moe_router_prob_frac": prob_fraction,
        "moe_expert_frac_max": jnp.max(token_fraction),
        "moe_expert_frac_min": jnp.min(token_fraction),
        "moe_expert_frac_std": jnp.std(token_fraction),
        "moe_dead_experts": dead_experts,
    }


def transformer_block_with_moe_stats(x, block_params, config: JaxLMConfig):
    attn_config = _attention_config(config)
    kimi_config = _kimi_config(config)
    sparse_config = _sparse_config(config)
    csa_config = _csa_config(config)
    lightning_config = _lightning_config(config)
    mhc_config = _mhc_config(config)

    h = rms_norm(x, block_params["attn_norm"], eps=config.eps)
    if config.attention_type == "mha":
        h = mha_attention(h, block_params["attn"], config)
        x = x + h
    elif config.attention_type == "mha_mhc":
        h_streams = jnp.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], config.num_mhc_streams, config.model_dim),
        )

        def layer_fn(h_in):
            return mha_attention(h_in, block_params["attn"]["mha"], config)

        h_streams = mhc_block(
            h_streams,
            block_params["attn"]["mhc"],
            mhc_config,
            layer_fn,
        )
        x = mhc_readout(h_streams, block_params["attn"]["mhc"])
    elif config.attention_type == "mhla":
        h = mhlatent_attention(h, block_params["attn"], attn_config)
        x = x + h
    elif config.attention_type == "kimi_deltanet":
        h = kimi_deltanet_parallel_chunkwise(h, block_params["attn"], kimi_config)
        x = x + h
    elif config.attention_type == "deepseek_sparse":
        h = deepseek_sparse_attention(h, block_params["attn"], sparse_config)
        x = x + h
    elif config.attention_type == "lightning_sparse":
        h = lightning_sparse_attention(h, block_params["attn"], lightning_config)
        x = x + h
    elif config.attention_type == "deepseek_csa_hca":
        h = deepseek_hybrid_attention(h, block_params["attn"], csa_config)
        x = x + h
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
    else:
        raise ValueError(f"unknown attention_type: {config.attention_type}")

    h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
    moe_stats = moe_router_aux_stats(h, block_params["moe"], config)
    x = x + deepseek_moe(h, block_params["moe"], attn_config)
    return x, moe_stats


def lm_forward(params, token_ids, config: JaxLMConfig):
    x = params["token_embedding"][token_ids]

    for block_params in params["blocks"]:
        x = transformer_block(x, block_params, config)

    x = rms_norm(x, params["final_norm"], eps=config.eps)
    return jnp.matmul(x, params["lm_head"])


def transformer_block_with_deepseek_sparse_aux(x, block_params, config: JaxLMConfig):
    attn_config = _attention_config(config)
    sparse_config = _sparse_config(config)

    h = rms_norm(x, block_params["attn_norm"], eps=config.eps)
    h, index_aux = deepseek_sparse_attention(
        h,
        block_params["attn"],
        sparse_config,
        return_aux=True,
    )
    x = x + h

    if config.use_moe:
        h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
        x = x + deepseek_moe(h, block_params["moe"], attn_config)

    return x, index_aux


def transformer_block_with_lightning_aux(x, block_params, config: JaxLMConfig):
    attn_config = _attention_config(config)
    lightning_config = _lightning_config(config)

    h = rms_norm(x, block_params["attn_norm"], eps=config.eps)
    h, index_aux = lightning_sparse_attention(
        h,
        block_params["attn"],
        lightning_config,
        return_aux=True,
    )
    x = x + h

    if config.use_moe:
        h = rms_norm(x, block_params["moe_norm"], eps=config.eps)
        x = x + deepseek_moe(h, block_params["moe"], attn_config)

    return x, index_aux


def lm_forward_with_deepseek_sparse_aux(params, token_ids, config: JaxLMConfig):
    if config.attention_type != "deepseek_sparse":
        raise ValueError("sparse auxiliary loss is only valid for deepseek_sparse")

    x = params["token_embedding"][token_ids]
    index_aux_total = jnp.asarray(0.0, dtype=x.dtype)

    for block_params in params["blocks"]:
        x, index_aux = transformer_block_with_deepseek_sparse_aux(
            x,
            block_params,
            config,
        )
        index_aux_total = index_aux_total + index_aux

    x = rms_norm(x, params["final_norm"], eps=config.eps)
    logits = jnp.matmul(x, params["lm_head"])
    return logits, index_aux_total / config.num_layers


def lm_forward_with_lightning_aux(params, token_ids, config: JaxLMConfig):
    if config.attention_type != "lightning_sparse":
        raise ValueError("lightning auxiliary loss is only valid for lightning_sparse")

    x = params["token_embedding"][token_ids]
    index_aux_total = jnp.asarray(0.0, dtype=x.dtype)

    for block_params in params["blocks"]:
        x, index_aux = transformer_block_with_lightning_aux(x, block_params, config)
        index_aux_total = index_aux_total + index_aux

    x = rms_norm(x, params["final_norm"], eps=config.eps)
    logits = jnp.matmul(x, params["lm_head"])
    return logits, index_aux_total / config.num_layers


def lm_forward_with_moe_stats(params, token_ids, config: JaxLMConfig):
    x = params["token_embedding"][token_ids]
    stats_accum = None

    for block_params in params["blocks"]:
        x, block_stats = transformer_block_with_moe_stats(x, block_params, config)
        if stats_accum is None:
            stats_accum = block_stats
        else:
            stats_accum = {
                name: stats_accum[name] + block_stats[name]
                for name in stats_accum
            }

    x = rms_norm(x, params["final_norm"], eps=config.eps)
    logits = jnp.matmul(x, params["lm_head"])
    stats = {
        name: value / config.num_layers
        for name, value in stats_accum.items()
    }
    return logits, stats


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


def loss_with_deepseek_sparse_index_aux(params, token_ids, targets, config: JaxLMConfig):
    logits, index_aux_loss = lm_forward_with_deepseek_sparse_aux(
        params,
        token_ids,
        config,
    )
    ce_loss = cross_entropy_loss(logits, targets)
    total_loss = ce_loss + config.deepseek_index_aux_weight * index_aux_loss
    return total_loss, {
        "ce_loss": ce_loss,
        "index_aux_loss": index_aux_loss,
        "index_aux_weight": jnp.asarray(
            config.deepseek_index_aux_weight,
            dtype=ce_loss.dtype,
        ),
        "loss_total": total_loss,
    }


def loss_with_lightning_index_aux(params, token_ids, targets, config: JaxLMConfig):
    logits, index_aux_loss = lm_forward_with_lightning_aux(params, token_ids, config)
    ce_loss = cross_entropy_loss(logits, targets)
    total_loss = ce_loss + config.lightning_index_aux_weight * index_aux_loss
    return total_loss, {
        "ce_loss": ce_loss,
        "index_aux_loss": index_aux_loss,
        "index_aux_weight": jnp.asarray(
            config.lightning_index_aux_weight,
            dtype=ce_loss.dtype,
        ),
        "loss_total": total_loss,
    }


def loss_with_moe_aux(params, token_ids, targets, config: JaxLMConfig):
    logits, stats = lm_forward_with_moe_stats(params, token_ids, config)
    ce_loss = cross_entropy_loss(logits, targets)
    aux_loss = stats["moe_aux_loss"]
    total_loss = ce_loss + config.moe_aux_loss_weight * aux_loss
    metrics = {
        "ce_loss": ce_loss,
        "moe_aux_loss": aux_loss,
        "loss_total": total_loss,
        **stats,
    }
    return total_loss, metrics
