from dataclasses import dataclass

import jax  
import jax.numpy as jnp   
import numpy as np


"""
step 1: token t1 arrives
        KV_1 = k1 ⊗ v1         [D,D]
        o1   = q1 @ KV_1

step 2: token t2 arrives
        KV_2 = KV_1 + k2 ⊗ v2  [D,D]  ← accumulates past
        o2   = q2 @ KV_2

step 3: token t3 arrives
        KV_3 = KV_2 + k3 ⊗ v3  [D,D]
        o3   = q3 @ KV_3


DeltaNet: KV_t = KV_{t-1} - k_t ⊗ (k_t @ KV_{t-1}) + k_t ⊗ v_t       

k_t ⊗ (k_t @ KV_{t-1}): eraser and  writer: k_t ⊗ v_t

this gives us value of previous KV cache vector's V: k_t @ KV_{t-1}
this gives us entire same matrices of previous KV cache for subtraction and erasing entirely:  k_t ⊗ (k_t @ KV_{t-1})    

clean new write with new_v:  k_t ⊗ v_t  


kimi deltanet: KV_t = KV_{t-1} - k_t ⊗ (β_t*(k_t @ KV_{t-1})) + k_t ⊗ (β_t*v_t) 

deltanet chunkwise algorithm: 
step 1:  cumulative fade per token
step 2:  how each token sees old memory
step 3:  what old memory says at each key
step 4:  how much each token wants to change memory
step 5:  relative fade between any two tokens
step 6:  do two tokens write to overlapping slots
step 7:  contamination matrix — how much earlier tokens affect later ones
step 8:  correct the writes — remove double counting between tokens
step 9:  how much each query benefits from each earlier write after fade
step 10: output = old memory contribution + within-chunk contribution
step 11: updated memory = faded old memory + all corrected writes faded to end

"""




@dataclass(frozen=True)
class KimiDeltaNetConfig:
    model_dim: int
    num_heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    eps: float = 1e-6
    num_routed_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 256


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_kimi_deltanet_params(key, config):
    keys = jax.random.split(key, 6)
    D = config.model_dim
    H = config.num_heads
    Kd = config.key_dim
    Vd = config.value_dim
    return {
        "q_proj": _xavier(keys[0], (D, H * Kd)),
        "k_proj": _xavier(keys[1], (D, H * Kd)),
        "v_proj": _xavier(keys[2], (D, H * Vd)),
        "alpha_proj": _xavier(keys[3], (D, H * Kd)),
        "beta_proj": _xavier(keys[4], (D, H)),
        "out_proj": _xavier(keys[5], (H * Vd, D)),
    }


def kimi_deltanet_stepwise(x, params, config):  

    B, T, D= x.shape    
    H =  config.num_heads
    Kd = config.key_dim  
    Vd = config.value_dim  
    
    q, k, v, alpha, beta = project_inputs(x, params, config)

    S = jnp.zeros([B, H, Kd, Vd]) 
    outputs = []  

    for t in range(T):  
        q_t  = q[:, t]
        k_t  = k[:, t]
        v_t  = v[:, t]
        a_t = alpha[:, t]
        b_t  = beta[:, t] 
    
        S = S  * a_t[..., : , None]  

        read_t = jax.numpy.einsum("bhk,bhkv->bhv", k_t, S)
        
        delta_t =  v_t - read_t 

        S = S + b_t[..., None, None] * jnp.einsum("bhk,bhv->bhkv", k_t, delta_t)
        
        out_t = jnp.einsum("bhk,bhkv->bhv", q_t, S)

        outputs.append(out_t)  

    out = jnp.stack(outputs, axis=1) 
    out = jnp.reshape(out, [B, T, H* Vd])  
    out = jnp.matmul(out, params["out_proj"])  
    return out


def kimi_deltanet_chunckwise(x, params, config):  

      B, T, D = x.shape
      
      q, k, v, alpha, beta = project_inputs(x, params, config)
      H = config.num_heads
      Kd = config.key_dim
      Vd = config.value_dim

      S = jnp.zeros([B, H, Kd, Vd])
      chunk_outputs = [] 

      for start in range(0, T, config.chunk_size):  
            end = min(start + config.chunk_size, T)

            q_c = q[:, start:end] 
            k_c = k[:, start:end]  
            v_c = v[:, start:end]  

            alpha_c = alpha[:, start:end]  
            beta_c = beta[:, start:end]  

            out_c, S = deltanet_chunk_scan(q_c,
            k_c,
            v_c,
            alpha_c,
            beta_c,
            S,)

            chunk_outputs.append(out_c)  
      
      out = jnp.concatenate(chunk_outputs, axis=1)
      out = jnp.reshape(out, [B, T, H * Vd])
      out = jnp.matmul(out, params['out_proj']) 
      
      return out  


def kimi_deltanet_parallel_chunkwise(x, params, config):
      B, T, D = x.shape

      q, k, v, alpha, beta = project_inputs(x, params, config)
      H = config.num_heads
      Kd = config.key_dim
      Vd = config.value_dim

      S = jnp.zeros([B, H, Kd, Vd])
      chunk_outputs = []

      for start in range(0, T, config.chunk_size):
            end = min(start + config.chunk_size, T)

            out_c, S = deltanet_chunk_fine_gated(
                  q[:, start:end],
                  k[:, start:end],
                  v[:, start:end],
                  alpha[:, start:end],
                  beta[:, start:end],
                  S,
            )
            chunk_outputs.append(out_c)

      out = jnp.concatenate(chunk_outputs, axis=1)
      out = jnp.reshape(out, [B, T, H * Vd])
      out = jnp.matmul(out, params["out_proj"])
      return out


def merge_heads(x):
   # x: [B, H, T, Dh] -> [B, T, H * Dh]
   B, H, T, Dh = x.shape
   x = jnp.transpose(x, (0, 2, 1, 3))
   return jnp.reshape(x, (B, T, H * Dh))


def deltanet_chunk_scan(q_c, k_c, v_c, alpha_c, beta_c, S):  

    B, C, H, Kd = q_c.shape
    Vd = S.shape[-1]

    outputs = []  

    for i in range(C):  
         q_i = q_c[:, i]  
         k_i = k_c[:, i] 
         v_i = v_c[:, i] 

         a_i = alpha_c[:, i] 
         b_i = beta_c[:, i]  
    
         S = S * a_i[..., :, None]  

         read_i  = jnp.einsum("bhk,bhkv->bhv", k_i, S) 
         delta_i = v_i - read_i  

         S = S + b_i[..., None, None] * jnp.einsum(
              "bhk,bhv->bhkv",
            k_i,
            delta_i,
         )

         out_i = jnp.einsum("bhk,bhkv->bhv", q_i, S)
         outputs.append(out_i) 

    return jnp.stack(outputs, axis=1), S  



def deltanet_chunk_fine_gated(q_c, k_c, v_c, alpha_c, beta_c, S0):  

    q = jnp.transpose(q_c, [0, 2, 1, 3])  
    k = jnp.transpose(k_c, [0, 2, 1, 3]) 
    v = jnp.transpose(v_c, [0, 2, 1, 3]) 

    alpha = jnp.transpose(alpha_c, [0, 2, 1, 3])
    beta = jnp.transpose(beta_c, [0, 2, 1]) 

    B, H, C, Kd = q.shape  
    Vd = v.shape[-1] 

    prefix = jnp.cumprod(alpha, axis=2) 

    decay = build_pairwise_decay(prefix)  
     
    S0_per_token = prefix[..., None] * S0[:, :, None, :, :] 
    k_s0 = jnp.einsum("bhck,bhckv->bhcv", k, S0_per_token)      
    q_s0 = jnp.einsum("bhck,bhckv->bhcv", q, S0_per_token) 

    u = beta[..., None] * (v - k_s0)

    kk_decay = jnp.einsum("bhid,bhijd,bhjd->bhij", k, decay, k) 
    lower = jnp.tril(jnp.ones([C, C]), k =-1) 

    kk_lower = jnp.where(lower, kk_decay, 0.0) 

    A = jnp.eye(C)[None, None, :, :] + beta[:, :, :, None] * kk_lower

    W = solve_triangular(A, u, lower=True)

    qk_decay = jnp.einsum("bhid,bhijd,bhjd->bhij", q, decay, k)
    causal = jnp.tril(jnp.ones((C, C)), k=0)
    qk_causal = jnp.where(causal, qk_decay, 0.0)  

    out = q_s0 + jnp.einsum("bhij,bhjv->bhiv", qk_causal, W) 

    chunk_decay = prefix[:, :, -1, :] 
    S_end = chunk_decay[..., :, None] * S0  

    end_decay = build_end_decay(prefix)
    S_end = S_end + jnp.einsum("bhck,bhcv->bhkv", k * end_decay, W) 

    out = jnp.transpose(out, [0, 2, 1, 3])
    
    return out, S_end 

def kimi_deltanet_moe_block(x, params, config):  
     
     h = rms_norm(x, params["attn_norm"], config.eps) 
     h = kimi_deltanet_chunckwise(h, params["attn"], config)
     x = x + h 

     h = rms_norm(x, params["moe_norm"], config.eps)  
     h = deepseek_moe(h, params["moe"], config)
  
     x = x + h  

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

     
def build_pairwise_decay(prefix):  
    
    B, H, C, Kd = prefix.shape
    
    prefix_i = prefix[:, :,:,  None, :]  
    prefix_j = prefix[:, :, None, :, :]
    
    decay = prefix_i / prefix_j

    eye = jnp.identity(C)  
    decay = jnp.where(
        eye[None, None, :, : , None], 
        1.0, 
        decay,    
        )

    causal = jnp.tril(jnp.ones([C,C])) 
    decay = jnp.where(
        causal[None, None, :, :, None],
        decay,
        0.0,
    )   
    
    return decay  


def solve_triangular(A, U, lower=True):

     def solve_one_head(A_bh, U_bh):  
         return jax.scipy.linalg.solve_triangular(
             A_bh, 
             U_bh, 
             lower=lower,
         ) 
     
     solve_heads = jax.vmap(solve_one_head, in_axes=(0,0)) 
     solve_batch = jax.vmap(solve_heads, in_axes=(0,0))
     
     W = solve_batch(A, U)  

     return W

def build_end_decay(prefix): 
    
    B, H, C, Kd = prefix.shape
    end_prefix = prefix[:, : , -1, :] 

    end_decay = end_prefix[:, :, None, :] / prefix
    last = jax.nn.one_hot(C - 1, C).astype(bool)  

    end_decay = jnp.where(last[None, None, :, None], 
                          1.0, 
                          end_decay, 
                          )  

    return end_decay  


def rms_norm(x, weight=None, eps=1e-6):
     if weight is None:
          weight = 1.0
     rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
     return x * rms * weight
     

def project_inputs(x, params, config):
     B, T, D = x.shape
     H = config.num_heads
     Kd = config.key_dim
     Vd = config.value_dim

     q = jnp.matmul(x, params["q_proj"])
     k = jnp.matmul(x, params["k_proj"])
     v = jnp.matmul(x, params["v_proj"])
     alpha = jnp.matmul(x, params["alpha_proj"])
     beta = jnp.matmul(x, params["beta_proj"])

     q = jnp.reshape(q, (B, T, H, Kd))
     k = jnp.reshape(k, (B, T, H, Kd))
     v = jnp.reshape(v, (B, T, H, Vd))
     alpha = jax.nn.sigmoid(jnp.reshape(alpha, (B, T, H, Kd)))
     beta = jax.nn.sigmoid(jnp.reshape(beta, (B, T, H)))

     q = l2_normalize(q)
     k = l2_normalize(k)

     return q, k, v, alpha, beta


def l2_normalize(x, eps=1e-6):
     denom = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps)
     return x / denom


if __name__ == "__main__":
     key = jax.random.PRNGKey(0)
     config = KimiDeltaNetConfig(
          model_dim=32,
          num_heads=4,
          key_dim=8,
          value_dim=8,
          chunk_size=4,
     )

     param_key, x_key = jax.random.split(key)
     params = init_kimi_deltanet_params(param_key, config)
     x = jax.random.normal(x_key, (2, 12, config.model_dim))

     out_stepwise = kimi_deltanet_stepwise(x, params, config)
     out_chunkwise = kimi_deltanet_chunckwise(x, params, config)
     out_parallel = kimi_deltanet_parallel_chunkwise(x, params, config)
     scan_error = jnp.max(jnp.abs(out_stepwise - out_chunkwise))
     parallel_error = jnp.max(jnp.abs(out_stepwise - out_parallel))

     def tiny_loss(p):
          return jnp.mean(jnp.square(kimi_deltanet_parallel_chunkwise(x, p, config)))

     loss, grads = jax.value_and_grad(tiny_loss)(params)
     grad_norm = jnp.sqrt(
          sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(grads))
     )

     print("stepwise:", out_stepwise.shape)
     print("scan chunkwise:", out_chunkwise.shape)
     print("parallel chunkwise:", out_parallel.shape)
     print("scan max error:", scan_error)
     print("parallel max error:", parallel_error)
     print("tiny loss:", loss)
     print("grad norm:", grad_norm)
     print("devices:", jax.devices())
     print("backend:", jax.default_backend())

     np.testing.assert_allclose(out_stepwise, out_chunkwise, atol=1e-5, rtol=1e-5)
     np.testing.assert_allclose(out_stepwise, out_parallel, atol=1e-5, rtol=1e-5)
