from dataclasses import dataclass

import jax 
import jax.numpy as jnp   
import numpy as np

"""
h_t -> compress -> c_t^Q, c_t^KV, k_t^R

query side:
    c_t^Q -> W_Q_A (absorbed W_K) -> q_t^A   [num_heads, latent_dim]
    c_t^Q -> W_Q_R                -> q_t^R   [num_heads, rope_dim]
    q_t = concat(q_t^A, q_t^R)               [num_heads, latent_dim + rope_dim]

indexer side:
    h_t -> W_I_Q -> q_t^I        [index_heads, index_dim]
    h_t -> W_I_K -> k_t^I        [index_heads, index_dim]
    h_t -> W_I_W -> w_t^I        [index_heads]
    
    score = sum_h w_t^I[h] * ReLU(q_t^I[h] @ past_idx_k[h].T)
    top-k indices are selected from causal indexer scores

KV fetching:
    For each selected index j:
    Key_j = concat(c_j^KV, k_j^R)       [latent_dim + rope_dim]
    Val_j = c_j^KV                      [latent_dim]
    no K/V decompression before sparse attention

attention per head (MQA mode):
    scores = q_i @ Key_selected.T      [top_k]
    probs  = softmax(scores)
    o_i    = probs @ Val_selected      [latent_dim]

output:
    concat(o_1...o_H) -> [num_heads * latent_dim]
    output = concat_heads @ W_O        [model_dim]

"""

@dataclass(frozen=True)
class DeepSeekSparseConfig:
    model_dim: int
    num_heads: int
    latent_dim: int
    rope_dim: int
    index_dim: int
    index_heads: int
    top_k: int
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    expert_hidden_dim: int = 256
    eps: float = 1e-6


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


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
    if config.num_routed_experts < 1:
        raise ValueError("num_routed_experts must be >= 1")
    if config.num_shared_experts < 1:
        raise ValueError("num_shared_experts must be >= 1")


def validate_deepseek_sparse_params(params, config):
    D = config.model_dim
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    I = config.index_dim
    Ih = config.index_heads

    expected_shapes = {
        "q_down": (D, C),
        "kv_down": (D, C),
        "k_rope": (D, R),
        "q_absorb": (C, H * C),
        "q_rope": (C, H * R),
        "idx_q": (D, Ih * I),
        "idx_k": (D, Ih * I),
        "idx_w": (D, Ih),
        "out_proj": (H * C, D),
    }

    for name, expected_shape in expected_shapes.items():
        if name not in params:
            raise KeyError(f"missing DeepSeek sparse attention param: {name}")
        if params[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {params[name].shape}, expected {expected_shape}"
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
    keys = jax.random.split(key, 9)
    D = config.model_dim
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    I = config.index_dim
    Ih = config.index_heads

    return {
        "q_down": _xavier(keys[0], (D, C)),
        "kv_down": _xavier(keys[1], (D, C)),
        "k_rope": _xavier(keys[2], (D, R)),
        "q_absorb": _xavier(keys[3], (C, H * C)),
        "q_rope": _xavier(keys[4], (C, H * R)),
        "idx_q": _xavier(keys[5], (D, Ih * I)),
        "idx_k": _xavier(keys[6], (D, Ih * I)),
        "idx_w": _xavier(keys[7], (D, Ih)),
        "out_proj": _xavier(keys[8], (H * C, D)),
    }


def deepseek_sparse_attention(x, params, config):  
    """
    Dense-reference sparse attention with token-level lightning indexer.

    x: [B, T, D]

    returns:
      out: [B, T, D]

    largest reference intermediate:
      idx_raw: [B, T, T, index_heads]
    """
    validate_deepseek_sparse_inputs(x, params, config)

    B, T, D = x.shape 

    H = config.num_heads
    C = config.latent_dim  
    R = config.rope_dim  
    I = config.index_dim  
    Ktop= config.top_k  

    c_q = jnp.matmul(x, params["q_down"])        #[B, T, C]
    c_kv = jnp.matmul(x, params["kv_down"])      #[B, T, C] 
    k_rope = jnp.matmul(x, params["k_rope"])     #[B, T, R]

    #multi query.  
    q_a = jnp.matmul(c_q, params["q_absorb"])   #[B, T, H * C]
    q_r = jnp.matmul(c_q, params["q_rope"])     #[B, T, H * R]
                                                  
    q_a = jnp.reshape(q_a, [B, T, H, C])  
    q_r = jnp.reshape(q_r, [B, T, H, R])  

    q_r = apply_rope(q_r)   
    k_rope = apply_rope_mqa(k_rope)   
    
    q= jnp.concatenate([q_a, q_r], axis=-1)  

    # MQA key/value cache
    kv_key = jnp.concatenate([c_kv, k_rope], axis=-1) 
    kv_val = c_kv  


    #lightning indexer.  

    idx_q = jnp.matmul(x, params["idx_q"])          # [B, T, H_i * I]
    idx_k = jnp.matmul(x, params["idx_k"])          # [B, T, H_i * I]
    idx_w = jnp.matmul(x, params["idx_w"])          # [B, T, H_i]

    idx_q = jnp.reshape(idx_q, [B, T, config.index_heads, I])
    idx_k = jnp.reshape(idx_k, [B, T, config.index_heads, I])
    idx_w = jnp.reshape(idx_w, [B, T, config.index_heads])

    #dence reference indexer score.  
    idx_raw = jnp.einsum("bqhi,bkhi->bqkh", idx_q, idx_k)    
    idx_raw = jax.nn.relu(idx_raw)  

    index_score = jnp.einsum("bqkh,bqh->bqk", idx_raw, idx_w)

    index_score = apply_causal_mask(index_score)  
    top_values, top_indices = jax.lax.top_k(index_score, Ktop)


    #selected compressed KV  
    selected_keys = gather_topk(kv_key, top_indices)   # [B, T, Ktop, C + R]
    selected_vals = gather_topk(kv_val, top_indices)   # [B, T, Ktop, C]


    #sparse MQA attention.  

    scores = jnp.einsum("bqhd,bqkd->bqhk", q, selected_keys) 
    scores = scores / jnp.sqrt(jnp.asarray(C + R, dtype=x.dtype))
    scores = apply_selected_causal_mask(scores, top_indices)

    probs = jax.nn.softmax(scores, axis=-1) 

    out = jnp.einsum("bqhk,bqkc->bqhc", probs, selected_vals)  # [B, T, H, C]
    out = jnp.reshape(out, (B, T, H * C))
    
    #absorbed value/output projection  

    out = jnp.matmul(out, params["out_proj"]) 

    return out 
 


def deepseek_moe(x, params, config):  

    B, T, D = x.shape
    
    E = config.num_routed_experts
    S = config.num_shared_experts
    K = config.top_k  

    shared_out = jnp.zeros_like(x)  

    for shared_id in range(S):  
      shared_out += expert_mlp(x, params["shared_experts"][shared_id]) 

    shared_out = shared_out / S  
    router_logits = jnp.matmul(x, params["router"]) 
    
    top_values, top_indices = jax.lax.top_k(router_logits, k=K)  

    router_weights = jax.nn.softmax(top_values, axis=-1) 

    routed_out = jnp.zeros_like(x)

    for expert_id in range(E):  
       expert_out = expert_mlp(x, 
                               params["routed_experts"][expert_id],
                               )
       expert_weight = jnp.zeros([B, T])

       for slot in range(K):  
          is_selected = top_indices[..., slot] == expert_id
          slot_weight = jnp.where(is_selected, router_weights[..., slot], 
                           0.0) 
          expert_weight += slot_weight  
       routed_out += expert_out * expert_weight[..., None]  

    out = shared_out + routed_out  
    
    return out   


def expert_mlp(x, expert_params): 

    gate = jnp.matmul(x, expert_params["gate_proj"]) 
    up = jnp.matmul(x, expert_params["up_proj"])  

    hidden = jax.nn.silu(gate) * up  

    out = jnp.matmul(hidden, expert_params['down_proj'])  

    return out  
     

def apply_rope(x):  
    
    B, T, H, R = x.shape 

    assert R % 2 == 0 

    half = R // 2  

    x1 = x[..., :half]
    x2 = x[..., half:]  

    position =  jnp.arange(T)  

    freqs = 1.0 / (10000 ** (jnp.arange(half) / half)) 

    angles = position[:, None] * freqs[None, :]  

    cos = jnp.cos(angles)[None, :, None, :] 
    sin = jnp.sin(angles)[None, :, None, :]  

    rotated = jnp.concatenate([
        x1 * cos - x2 * sin, 
        x1 * sin + x2 * cos,
    ], axis=-1)  

    return rotated


def apply_rope_mqa(x):  
    
    B, T, R = x.shape 

    assert R % 2 == 0

    half = R // 2 

    x1 = x[..., :half]
    x2 = x[...,half:] 

    positions = jnp.arange(T)  

    freqs = 1.0 / (10000 ** (jnp.arange(half) / half)) 

    angles = positions[: , None] * freqs[None, :]  

    cos = jnp.cos(angles)[None, : , :]  
    sin = jnp.sin(angles)[None, :, :]  

    rotated = jnp.concatenate([
         x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,   
    ], axis=-1)   

    return rotated 
 


def apply_causal_mask(scores):

    B, T, _ = scores.shape    
    
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))

    scores = jnp.where(
        mask[None, :, :], 
        scores, 
        -jnp.inf,
    )

    return scores 


def apply_selected_causal_mask(scores, top_indices):
    # scores: [B, T, H, Ktop]
    # top_indices: [B, T, Ktop]
    selected_valid = selected_causal_valid(top_indices)

    return jnp.where(
        selected_valid[:, :, None, :],
        scores,
        -jnp.inf,
    )


def selected_causal_valid(top_indices):
    # top_indices: [B, T, Ktop]
    _, T, _ = top_indices.shape
    query_pos = jnp.arange(T)[None, :, None]
    return top_indices <= query_pos


def gather_topk(values, indices):
    # values: [B, T, D]
    # indices: [B, T, K]
    B = values.shape[0]
    batch_indices = jnp.arange(B)[:, None, None]
    return values[batch_indices, indices]


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    config = DeepSeekSparseConfig(
        model_dim=32,
        num_heads=4,
        latent_dim=8,
        rope_dim=4,
        index_dim=8,
        index_heads=2,
        top_k=4,
    )

    param_key, x_key = jax.random.split(key)
    params = init_deepseek_sparse_params(param_key, config)
    x = jax.random.normal(x_key, (2, 12, config.model_dim))

    y = deepseek_sparse_attention(x, params, config)

    idx_q = jnp.reshape(
        jnp.matmul(x, params["idx_q"]),
        (2, 12, config.index_heads, config.index_dim),
    )
    idx_k = jnp.reshape(
        jnp.matmul(x, params["idx_k"]),
        (2, 12, config.index_heads, config.index_dim),
    )
    idx_w = jnp.reshape(
        jnp.matmul(x, params["idx_w"]),
        (2, 12, config.index_heads),
    )
    idx_raw = jax.nn.relu(jnp.einsum("bqhi,bkhi->bqkh", idx_q, idx_k))
    index_score = jnp.einsum("bqkh,bqh->bqk", idx_raw, idx_w)
    index_score = apply_causal_mask(index_score)
    _, top_indices = jax.lax.top_k(index_score, config.top_k)
    selected_valid = selected_causal_valid(top_indices)
    topk_filler_count = jnp.sum(~selected_valid)
    dummy_scores = jnp.zeros((2, 12, config.num_heads, config.top_k))
    dummy_scores = apply_selected_causal_mask(dummy_scores, top_indices)
    unmasked_invalid_count = jnp.sum(
        jnp.isfinite(dummy_scores) & ~selected_valid[:, :, None, :]
    )

    def tiny_loss(p):
        return jnp.mean(jnp.square(deepseek_sparse_attention(x, p, config)))

    loss, grads = jax.value_and_grad(tiny_loss)(params)
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    )

    print("input:", x.shape)
    print("output:", y.shape)
    print("top-k filler invalid count:", topk_filler_count)
    print("unmasked invalid count:", unmasked_invalid_count)
    print("loss:", loss)
    print("grad norm:", grad_norm)
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    np.testing.assert_equal(y.shape, (2, 12, config.model_dim))
    np.testing.assert_equal(int(np.asarray(unmasked_invalid_count)), 0)

  

"""
Make sparse attention reference correct:
- no future token leakage
- selected KV shapes explicit
- top-k behavior valid at early tokens
- output matches expected [B,T,D]
- works under jit/value_and_grad
"""
