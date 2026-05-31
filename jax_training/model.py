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
    index_dim: int = 32
    index_heads: int = 2
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 2048
    eps: float = 1e-6


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


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
    return KimiDeltaNetConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        key_dim=config.head_dim,
        value_dim=config.head_dim,
        chunk_size=config.chunk_size,
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


def init_lm_params(key, config: JaxLMConfig):
    if config.model_dim != config.num_heads * config.head_dim:
        raise ValueError("model_dim must equal num_heads * head_dim")
    if config.num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    keys = jax.random.split(key, 3 + 4 * config.num_layers)
    attn_config = _attention_config(config)
    kimi_config = _kimi_config(config)
    sparse_config = _sparse_config(config)

    blocks = []
    offset = 2
    for _ in range(config.num_layers):
        if config.attention_type == "mhla":
            attn_params = init_mhla_params(keys[offset], attn_config)
        elif config.attention_type == "kimi_deltanet":
            attn_params = init_kimi_deltanet_params(keys[offset], kimi_config)
        elif config.attention_type == "deepseek_sparse":
            attn_params = init_deepseek_sparse_params(keys[offset], sparse_config)
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

    h = rms_norm(x, block_params["attn_norm"], eps=config.eps)
    if config.attention_type == "mhla":
        h = mhlatent_attention(h, block_params["attn"], attn_config)
    elif config.attention_type == "kimi_deltanet":
        h = kimi_deltanet_parallel_chunkwise(h, block_params["attn"], kimi_config)
    elif config.attention_type == "deepseek_sparse":
        h = deepseek_sparse_attention(h, block_params["attn"], sparse_config)
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
