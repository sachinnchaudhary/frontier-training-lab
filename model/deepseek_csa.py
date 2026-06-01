from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DeepSeekCSAConfig:
    model_dim: int
    num_heads: int
    latent_dim: int
    rope_dim: int
    index_dim: int
    index_heads: int
    csa_compress_rate: int
    top_k: int
    hca_compress_rate: int = 128
    local_window_size: int = 128
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    expert_hidden_dim: int = 2048
    eps: float = 1e-6


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_deepseek_csa_params(key, config):
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
        "q_absorb": _xavier(keys[2], (C, H * C)),
        "q_rope": _xavier(keys[3], (C, H * R)),
        "fusion_proj": _xavier(keys[4], (D, 1)),
        "idx_q": _xavier(keys[5], (D, Ih * I)),
        "idx_w": _xavier(keys[6], (D, Ih)),
        "idx_k": _xavier(keys[7], (D, Ih * I)),
        "out_proj": _xavier(keys[8], (H * C, D)),
    }


def init_deepseek_hca_params(key, config):
    keys = jax.random.split(key, 6)
    D = config.model_dim
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim

    return {
        "q_down": _xavier(keys[0], (D, C)),
        "kv_down": _xavier(keys[1], (D, C)),
        "q_absorb": _xavier(keys[2], (C, H * C)),
        "q_rope": _xavier(keys[3], (C, H * R)),
        "fusion_proj": _xavier(keys[4], (D, 1)),
        "out_proj": _xavier(keys[5], (H * C, D)),
    }


def init_deepseek_moe_params(key, config):
    D = config.model_dim
    E = config.num_routed_experts
    S = config.num_shared_experts
    hidden = config.expert_hidden_dim
    keys = jax.random.split(key, 1 + 3 * (E + S))

    def init_expert(offset):
        return {
            "gate_proj": _xavier(keys[offset], (D, hidden)),
            "up_proj": _xavier(keys[offset + 1], (D, hidden)),
            "down_proj": _xavier(keys[offset + 2], (hidden, D)),
        }

    shared_experts = []
    for shared_id in range(S):
        shared_experts.append(init_expert(1 + 3 * shared_id))

    routed_experts = []
    routed_base = 1 + 3 * S
    for expert_id in range(E):
        routed_experts.append(init_expert(routed_base + 3 * expert_id))

    return {
        "router": _xavier(keys[0], (D, E)),
        "shared_experts": tuple(shared_experts),
        "routed_experts": tuple(routed_experts),
    }


def init_deepseek_hybrid_params(key, config):
    csa_key, hca_key, moe_key = jax.random.split(key, 3)
    return {
        "attn": {
            "csa": init_deepseek_csa_params(csa_key, config),
            "hca": init_deepseek_hca_params(hca_key, config),
        },
        "moe": init_deepseek_moe_params(moe_key, config),
        "attn_norm": jnp.ones((config.model_dim,)),
        "moe_norm": jnp.ones((config.model_dim,)),
    }


"""

=====================================================================
PHASE 1: PER-TOKEN PREPARATION (The Current Word)
=====================================================================
h_t -> compress -> c_t^Q, c_t^KV  
(Notice: Memory position and index keys are NOT generated here anymore)

query side:
    c_t^Q -> W_Q_A (absorbed W_K) -> q_t^A   [num_heads, latent_dim]
    c_t^Q -> W_Q_R                -> q_t^R   [num_heads, rope_dim]
    q_t = concat(q_t^A, q_t^R)               [num_heads, latent_dim + rope_dim]

indexer scout (Querying the past):
    h_t -> W_I_Q -> q_t^I        [index_heads, index_dim]
    h_t -> W_I_W -> w_t^I        [index_heads]

=====================================================================
PHASE 2: CSA SLIDING WINDOW COMPRESSION (The V4 Upgrade)
=====================================================================
Wait for 'w' tokens (e.g., 4) to gather in the short-term buffer.
Buffer = [h_1, h_2, ..., h_w] and their latents [c_1^KV, ..., c_w^KV]

fusion_weights = Softmax(Linear(Buffer))     [w]

// 1. Fuse the individual latents into a single Super-Block
c_chunk^KV = sum_{m=1 to w} (fusion_weights[m] * c_m^KV)  [latent_dim]

// 2. Generate Block-Level Position and Block-Level Index Key
k_chunk^R = W_K_R(Block_Position_ID)                      [rope_dim]
k_chunk^I = W_I_K(Fused_Input_Buffer)                     [index_heads, index_dim]

// Push to Vault
Save (c_chunk^KV, k_chunk^R, k_chunk^I) to the Global Cache Vault.

=====================================================================
PHASE 3: INDEXER FILTERING & TOP-K
=====================================================================
// The indexer now sweeps chunks, not tokens.
score = sum_h w_t^I[h] * ReLU(q_t^I[h] @ past_idx_k_chunk[h].T)

top-k indices are selected from causal indexer scores (selecting Super-Blocks!)

=====================================================================
PHASE 4: KV FETCHING & ATTENTION
=====================================================================
KV fetching:
    For each selected Super-Block index j:
    Key_chunk_j = concat(c_chunk_j^KV, k_chunk_j^R)     [latent_dim + rope_dim]
    Val_chunk_j = c_chunk_j^KV                          [latent_dim]
    (Still no K/V decompression before sparse attention)

attention per head (MQA mode):
    scores = q_i @ Key_chunk_selected.T      [top_k]
    probs  = softmax(scores)
    o_i    = probs @ Val_chunk_selected      [latent_dim]

=====================================================================
PHASE 5: OUTPUT
=====================================================================
output:
    concat(o_1...o_H) -> [num_heads * latent_dim]
    output = concat_heads @ W_O              [model_dim]



phase 1 — per token (all w tokens parallel):
    h_t → compress → c_t^Q, c_t^KV
    c_t^Q → queries q_t (N heads) + rope q_t^R
    h_t → W_I_Q → q_t^I   (indexer query)
    h_t → W_I_W → w_t^I   (indexer weights)

phase 2 — chunk fusion (once per w tokens):
    linear(buffer) → softmax → fusion_weights   [w]
    c_chunk^KV = sum(fusion_weights * latents)
    k_chunk^R = W_K_R(block_position)
    k_chunk^I = W_I_K(fused_buffer)
    save to vault

phase 3 — indexing:
    score_j = sum_h w_t^I[h] * ReLU(q_t^I[h] @ k_chunk_j^I[h].T)
    pick top-k chunks

phase 4 — fetch and attend:
    Key_j   = concat(c_chunk_j^KV, k_chunk_j^R)
    Val_j   = c_chunk_j^KV
    each query head attends over top-k KV blocks

phase 5 — output:
    concat heads → W_O → u_t

"""


"""
HCA(Higly compressed attention): 

=====================================================================
PHASE 1: PER-TOKEN PREPARATION (w = 128 tokens parallel)
=====================================================================
    h_t → compress → c_t^Q, c_t^KV
    c_t^Q → queries q_t (N heads) + rope q_t^R
    
    // [DELETED]: No Lightning Indexer. 
    // q_t^I and w_t^I are completely removed from the silicon.

=====================================================================
PHASE 2: MASSIVE CHUNK FUSION (once per w=128 tokens)
=====================================================================
    linear(buffer) → softmax → fusion_weights   [w=128]
    c_chunk^KV = sum(fusion_weights * latents)
    k_chunk^R = W_K_R(block_position)           [Anchor Timestamp]
    
    // [DELETED]: No Index Barcode. k_chunk^I is removed.
    
    save to Global Vault

=====================================================================
PHASE 3: INDEXING
=====================================================================
    // [DELETED]: Phase 3 vanishes entirely. 
    // Because 128 tokens are crushed into 1 chunk, the vault is 
    // tiny enough that the GPU can afford to skip filtering.

=====================================================================
PHASE 4: FETCH AND ATTEND (The Dense + Local Hybrid)
=====================================================================
    // Track A: The Distant Past (Dense sweep of ALL chunks)
    For EVERY chunk_j in the Global Vault:
        Key_chunk_j = concat(c_chunk_j^KV, k_chunk_j^R)
        Val_chunk_j = c_chunk_j^KV

    // Track B: The Immediate Past (Sliding Window Bypass)
    // The GPU keeps the most recent 128 tokens completely raw.
    For EVERY recent token_m in the Local Buffer:
        Key_local_m = concat(c_m^KV, k_m^R)   [Token-level RoPE]      
        Val_local_m = c_m^KV                  [Unfused latent]
     
    // The Math:
    Key_total = concat_sequence(Key_local, Key_chunk_ALL)
    Val_total = concat_sequence(Val_local, Val_chunk_ALL)
    
    
    at any moments its 128 different c_m^kv + past n highly compressed chunks.
    for token T we have c_T^kv and and its n head of queries which looks into this k and v of past n compressed chunks 
    and past sliding window contexts kv block differently for each token in that window.


    each query head attends over ALL keys in Key_total 
    (No top-k restriction)

=====================================================================
PHASE 5: OUTPUT
=====================================================================
    concat heads → W_O → u_t


"""


def deepseek_csa_attention(x, params, config):  
   
   B, T, D = x.shape 
   H = config.num_heads
   C = config.latent_dim  
   R = config.rope_dim  
   M = config.csa_compress_rate  
   Ktop = config.top_k  
  
  
   c_q = jnp.matmul(x, params["q_down"]) 
   c_kv = jnp.matmul(x, params["kv_down"])  

   q_a =  jnp.matmul(c_q, params["q_absorb"]) 
   q_r = jnp.matmul(c_q, params["q_rope"])  

   q_a = jnp.reshape(q_a, [B, T, H ,C])  
   q_r = jnp.reshape(q_r, [B, T, H, R])  

   q_r = apply_rope(q_r)  
   q = jnp.concatenate([q_a, q_r], axis=-1)  

   c_blocks, block_repr, block_rope = csa_compress_blocks(
       x, c_kv, params, config
   )
   B, N, C = c_blocks.shape

   block_rope = jnp.broadcast_to(block_rope[None, :, :], (B, N, R))
   block_keys = jnp.concatenate([
      c_blocks, 
      block_rope,
   ], axis=-1)  

   block_vals = c_blocks
   
   idx_q = jnp.matmul(x, params["idx_q"])  
   idx_w  = jnp.matmul(x, params["idx_w"]) 

   Ih = config.index_heads
   I = config.index_dim

   idx_q = jnp.reshape(idx_q, [B, T, Ih, I])
   idx_w = jnp.reshape(idx_w, [B, T, Ih])  

   idx_k = jnp.matmul(block_repr, params["idx_k"])  
   idx_k  = jnp.reshape(idx_k, [B, N, Ih, I])  

   raw = jnp.einsum("bthi,bnhi->btnh", idx_q, idx_k)
   raw = jax.nn.relu(raw) 

   index_score = jnp.einsum("btnh,bth->btn", raw, idx_w)  

   index_score = apply_causal_block_mask(index_score, config) 
   
   top_values, top_indices =  jax.lax.top_k(index_score, Ktop) 

   selected_keys = gather_blocks(block_keys, top_indices)
   selected_vals = gather_blocks(block_vals, top_indices)  
   
   scores = jnp.einsum("bthd,btkd->bthk", q, selected_keys)
   scores = scores / jnp.sqrt(jnp.asarray(C + R, dtype=x.dtype))

   probs = jax.nn.softmax(scores, axis=-1) 

   out = jnp.einsum("bthk,btkc->bthc", probs, selected_vals)

   out = jnp.reshape(out, [B, T, H * C])  
   out = jnp.matmul(out, params["out_proj"]) 

   return out  




def deepseek_hca_attention(x, params, config):

    B, T, D = x.shape 
    H = config.num_heads
    C = config.latent_dim
    R = config.rope_dim
    L = config.local_window_size

    c_q = jnp.matmul(x, params["q_down"]) 
    c_kv = jnp.matmul(x, params["kv_down"]) 

    q_a = jnp.matmul(c_q, params["q_absorb"])  
    q_r = jnp.matmul(c_q, params["q_rope"]) 

    q_a = jnp.reshape(q_a, [B, T, H, C]) 
    q_r = jnp.reshape(q_r, [B, T, H, R])  
    
    q_r = apply_rope(q_r)  

    q = jnp.concatenate([q_a, q_r], axis=-1)

    c_blocks, block_rope = hca_compress_blocks(x, c_kv, params, config)
    B, N, C = c_blocks.shape

    block_rope = jnp.broadcast_to(block_rope[None, :, :], (B, N, R))
    chunk_keys = jnp.concatenate([c_blocks, block_rope], axis=-1)
    chunk_vals = c_blocks  

    token_rope = token_rope_embedding(T, R)  
    token_rope = jnp.broadcast_to(token_rope[None, :, :], (B, T, R))

    local_keys = jnp.concatenate([c_kv, token_rope], axis=-1)
    local_vals = c_kv  

    chunk_scores = jnp.einsum("bthd,bnd->bthn", q, chunk_keys) 
    chunk_scores = chunk_scores / jnp.sqrt(jnp.asarray(C + R, dtype=x.dtype))

    chunk_scores = apply_causal_chunk_mask(
        chunk_scores,
        compress_rate=config.hca_compress_rate,
    )

    chunk_probs = jax.nn.softmax(chunk_scores, axis=-1)  
    chunk_out = jnp.einsum("bthn,bnc->bthc", chunk_probs, chunk_vals) 

    local_scores = jnp.einsum("bthd,bsd->bths", q, local_keys)
    local_scores = local_scores / jnp.sqrt(jnp.asarray(C + R, dtype=x.dtype))
    local_scores = apply_local_causal_mask(local_scores, L)
    local_probs = jax.nn.softmax(local_scores, axis=-1) 

    local_out = jnp.einsum("bths,bsc->bthc", local_probs, local_vals)
    
    out = local_out + chunk_out  

    out = jnp.reshape(out, [B, T, H*C]) 
    out = jnp.matmul(out, params["out_proj"])  

    return out  

def deepseek_hybrid_attention(x, params, config):
    csa = deepseek_csa_attention(x, params["csa"], config)
    hca = deepseek_hca_attention(x, params["hca"], config)

    return csa + hca



def rms_norm(x, weight=None, eps=1e-6):
     if weight is None:
          weight = 1.0
     rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
     return x * rms * weight
     


def deepseek_csa_hca_moe_block(x, params, config):
    h = rms_norm(x, params["attn_norm"], config.eps)

    attn_out = deepseek_hybrid_attention(h, params["attn"], config)

    x = x + attn_out

    h = rms_norm(x, params["moe_norm"], config.eps)

    moe_out = deepseek_moe(h, params["moe"], config)

    x = x + moe_out

    return x


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



def token_rope_embedding(T, rope_dim):
    positions = jnp.arange(T)
    return rope_embedding(positions, rope_dim)


def hca_compress_blocks(x, c_kv, params, config):
    B, T, D = x.shape
    C = config.latent_dim
    M = config.hca_compress_rate

    N = (T + M - 1) // M
    T_pad = N * M

    x_pad = pad_to_length(x, T_pad)
    c_pad = pad_to_length(c_kv, T_pad)

    x_blk = jnp.reshape(x_pad, [B, N, M, D])
    c_blk = jnp.reshape(c_pad, [B, N, M, C])

    logits = jnp.matmul(x_blk, params["fusion_proj"])
    logits = mask_invalid_block_position(logits, T, M)

    weights = jax.nn.softmax(logits, axis=2)
    c_blocks = jnp.sum(weights * c_blk, axis=2)

    block_positions = jnp.arange(N) * M
    block_rope = rope_embedding(block_positions, config.rope_dim)

    return c_blocks, block_rope


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

   
def csa_compress_blocks(x, c_kv, params, config):   
    B, T, D = x.shape  
    C = config.latent_dim  
    M = config.csa_compress_rate

    N = (T + M - 1) // M
    T_pad = N * M  

    x_pad = pad_to_length(x, T_pad)  
    c_pad = pad_to_length(c_kv, T_pad)

    x_blk = jnp.reshape(x_pad, [B, N, M, D])  
    c_blk = jnp.reshape(c_pad, [B, N, M, C])  

    logits = jnp.matmul(x_blk, params["fusion_proj"]) 
    logits = mask_invalid_block_position(logits, T, M)

    weights = jax.nn.softmax(logits, axis=2)  
    c_blocks = jnp.sum(weights * c_blk, axis=2) 
    block_repr = jnp.sum(weights * x_blk, axis=2)  
    
    block_positions = jnp.arange(N) * M  
    block_rope = rope_embedding(block_positions,config.rope_dim) 

    return c_blocks, block_repr, block_rope 



def apply_causal_block_mask(index_score, config):  
  
  B, T, N = index_score.shape 
  M = config.csa_compress_rate  

  token_pos = jnp.arange(T)  
  block_start = jnp.arange(N) * M

  mask = block_start[None, :] <= token_pos[:, None]

  index_score = jnp.where(
      mask[None, :, :], 
      index_score, 
      -jnp.inf, 
  )

  return index_score


def apply_causal_chunk_mask(scores, compress_rate):
  B, T, H, N = scores.shape

  token_pos = jnp.arange(T)
  block_start = jnp.arange(N) * compress_rate
  mask = block_start[None, :] <= token_pos[:, None]

  return jnp.where(mask[None, :, None, :], scores, -jnp.inf)


def apply_local_causal_mask(scores, local_window_size):
  B, T, H, S = scores.shape

  query_pos = jnp.arange(T)[:, None]
  key_pos = jnp.arange(S)[None, :]
  causal = key_pos <= query_pos
  in_window = key_pos >= (query_pos - local_window_size + 1)
  mask = causal & in_window

  return jnp.where(mask[None, :, None, :], scores, -jnp.inf)
     


def gather_blocks(values, indices):
     
     B = values.shape[0]  
     batch_idx = jnp.arange(B)[:, None, None]  

     gathered = values[batch_idx, indices] 

     return gathered  




def pad_to_length(x, target_len):  
     
     B, T, D = x.shape  

     pad_len = target_len - T  

     if pad_len == 0:  
         return x  
     
     padding = jnp.zeros([B, pad_len, D], dtype=x.dtype)  

     x_pad = jnp.concatenate([x, padding], axis=1)  

     return x_pad  
 

def mask_invalid_block_position(logits, T, M):  
     
     B, N, M, _ = logits.shape 

     block_ids = jnp.arange(N)[:, None] 
     inner_ids = jnp.arange(M)[None, :]  

     token_pos = block_ids* M + inner_ids

     valid = token_pos < T 

     logits = jnp.where(
         valid[None, :, :, None], 
         logits, 
         -jnp.inf
                        )
     
     return logits 


def rope_embedding(positions, rope_dim):  

    assert rope_dim % 2 == 0 

    half = rope_dim // 2 
    freqs = 1.0 / (10000 ** (jnp.arange(half) / half)) 

    angles = positions[:, None]  *freqs[None, :]  

    emb = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1) 

    return emb  


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    config = DeepSeekCSAConfig(
        model_dim=32,
        num_heads=4,
        latent_dim=8,
        rope_dim=4,
        index_dim=8,
        index_heads=2,
        csa_compress_rate=4,
        top_k=2,
        hca_compress_rate=6,
        local_window_size=4,
        num_routed_experts=4,
        num_shared_experts=1,
        expert_hidden_dim=64,
    )

    csa_key, hca_key, hybrid_key, x_key = jax.random.split(key, 4)
    csa_params = init_deepseek_csa_params(csa_key, config)
    hca_params = init_deepseek_hca_params(hca_key, config)
    hybrid_params = init_deepseek_hybrid_params(hybrid_key, config)

    x = jax.random.normal(x_key, (2, 12, config.model_dim))

    y_csa = deepseek_csa_attention(x, csa_params, config)
    y_hca = deepseek_hca_attention(x, hca_params, config)
    y_block = deepseek_csa_hca_moe_block(x, hybrid_params, config)

    def tiny_loss(p):
        return jnp.mean(jnp.square(deepseek_csa_hca_moe_block(x, p, config)))

    loss, grads = jax.value_and_grad(tiny_loss)(hybrid_params)
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
    )

    print("input:", x.shape)
    print("csa output:", y_csa.shape)
    print("hca output:", y_hca.shape)
    print("block output:", y_block.shape)
    print("loss:", loss)
    print("grad norm:", grad_norm)
    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    np.testing.assert_equal(y_csa.shape, (2, 12, config.model_dim))
    np.testing.assert_equal(y_hca.shape, (2, 12, config.model_dim))
    np.testing.assert_equal(y_block.shape, (2, 12, config.model_dim))


