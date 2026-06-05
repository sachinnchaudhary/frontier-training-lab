from dataclasses import dataclass

import jax
import jax.numpy as jnp


"""
 Input: [10, 1024]   
   weight_down: [1024, 256]
 latent compressed Q/KV vector:  [10, 256]
 for query unzipping time:  [10,1024] 
   weight_up: [256, 128] -> dim per head
 [B, 8, 10, 128] 
 unzipped: [10,128] 
  
  score[1,10]: Q1[1 * 256] * ckv[256 * 10]  


  We have input [10, 1024].
  We apply W_down[1024, 256] to compress it.
  This makes compressed Q and KV = [10, 256].
  For each head during training, we apply W_up [256, 128].
  This expands the latents into [10, 128] (Content) + [10, 64] (RoPE).
  In Inference:For the new Query [1, 128], we first fuse it by multiplying it by W_up[128, 256].
  This makes a Fused Query [1, 256], converting the raw query directly into the compressed latent space.
  We multiply the Fused Query [1, 256] by the Compressed KV Cache transposed [256, 10], which gives Scores [1, 10].
  Finally, the Scores [1, 10] multiply against the Compressed KV Cache [10, 256] to give a Mixed Latent [1, 256].
  We pass this through the Value weights W_up[256, 128] to get the final output of [1, 128]. 
"""


@dataclass(frozen=True)
class MHLAConfig:
   model_dim: int
   num_heads: int
   head_dim: int
   latent_dim: int
   rope_dim: int
   num_experts: int = 4
   num_routed_experts: int = 4
   num_shared_experts: int = 1
   top_k: int = 2
   expert_hidden_dim: int = 2048
   eps: float = 1e-6


def validate_mhla_config(config):
   if config.model_dim != config.num_heads * config.head_dim:
      raise ValueError("model_dim must equal num_heads * head_dim")
   if config.rope_dim % 2 != 0:
      raise ValueError("rope_dim must be even")
   if config.top_k > config.num_routed_experts:
      raise ValueError("top_k must be <= num_routed_experts")
   if config.num_shared_experts < 1:
      raise ValueError("num_shared_experts must be >= 1")


def validate_mhla_params(params, config):
   H = config.num_heads
   Dh = config.head_dim
   R = config.rope_dim
   D = config.model_dim
   C = config.latent_dim

   expected_shapes = {
      "q_down": (D, C),
      "kv_down": (D, C),
      "q_up_content": (C, H * Dh),
      "q_up_rope": (C, H * R),
      "k_up_content": (C, H * Dh),
      "k_up_rope": (C, H * R),
      "v_up": (C, H * Dh),
      "out_proj": (H * Dh, D),
   }

   for name, expected_shape in expected_shapes.items():
      if name not in params:
         raise KeyError(f"missing MHLA param: {name}")
      if params[name].shape != expected_shape:
         raise ValueError(
            f"{name} has shape {params[name].shape}, expected {expected_shape}"
         )


def validate_mhla_inputs(x, params, config):
   validate_mhla_config(config)
   if x.ndim != 3:
      raise ValueError(f"x must be [B, T, D], got {x.shape}")
   if x.shape[-1] != config.model_dim:
      raise ValueError(
         f"x last dim is {x.shape[-1]}, expected model_dim={config.model_dim}"
      )
   validate_mhla_params(params, config)

def _xavier(key, shape):
   fan_in, fan_out = shape[0], shape[-1]
   limit = jnp.sqrt(6.0 / (fan_in + fan_out))
   return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def mhlatent_attention(x, params, config):
   """
   Dense causal MHLA/MLA reference path.

   x: [B, T, D]

   params:
      q_down:       [D, C]
      kv_down:      [D, C]
      q_up_content: [C, H * Dh]
      q_up_rope:    [C, H * R]
      k_up_content: [C, H * Dh]
      k_up_rope:    [C, H * R]
      v_up:         [C, H * Dh]
      out_proj:     [H * Dh, D]

   returns:
      out: [B, T, D]

   largest intermediate:
      scores/weights: [B, H, T, T]
   """
   validate_mhla_inputs(x, params, config)

   B, T, D = x.shape 
   H = config.num_heads
   Dh = config.head_dim 
   C = config.latent_dim 
   R =config.rope_dim
   
   # Compress model states into query and KV latent streams.
   q_latent = jnp.matmul(x, params["q_down"])  #[B, T, C] 
   kv_latent = jnp.matmul(x , params["kv_down"])  #[B, T, C] 
   
   # Expand latent streams into per-head content and RoPE components.
   q_content = jnp.matmul(q_latent , params["q_up_content"]) #[B, T, H * Dh]
   q_rope = jnp.matmul(q_latent, params["q_up_rope"])        #[B, T, H * R]
                                                
   q_content = jnp.reshape(q_content, [B, T, H, Dh]) #[B, T, H, Dh] 
   q_rope = jnp.reshape(q_rope, [B, T, H, R])        #[B, T, H ,R] 

   
   k_content = jnp.matmul(kv_latent, params["k_up_content"]) #[B, T, H * Dh]
   v = jnp.matmul(kv_latent, params["v_up"])                 #[B, T, H * Dh]
   k_rope = jnp.matmul(kv_latent, params["k_up_rope"])       #[B, T, H * R]

   k_content = jnp.reshape(k_content, [B, T, H, Dh])         #[B, T, H, Dh]
   v = jnp.reshape(v, [B, T, H, Dh])                         #[B, T, H, Dh]
   k_rope = jnp.reshape(k_rope, [B, T, H, R])                #[B, T, H, R]
   

   # Apply RoPE only to the RoPE subspace; content dimensions stay unrotated.
   positions = jnp.arange(T)
   q_rope = apply_rope(q_rope, positions) 
   k_rope = apply_rope(k_rope, positions) 

   q = jnp.concatenate([q_content, q_rope], axis=-1)
   k = jnp.concatenate([k_content, k_rope], axis=-1) 

   # Move heads before sequence: [B, T, H, *] -> [B, H, T, *].
   q = jnp.transpose(q, (0, 2, 1, 3))
   k = jnp.transpose(k, (0, 2, 1, 3))
   v = jnp.transpose(v, (0, 2, 1, 3))

   
   scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) 
   scores = scores / jnp.sqrt(jnp.asarray(Dh + R, dtype=x.dtype))

   scores =  apply_causal_mask(scores)
   
   weights = jax.nn.softmax(scores, axis=-1)
   out = jnp.matmul(weights, v) 

   out = jnp.transpose(out, (0, 2, 1, 3))
   out = jnp.reshape(out, [B, T, H * Dh]) 

   out = jnp.matmul(out, params["out_proj"])  

   return out 

   

def mhlatent_attention_step(x_new, past_kv_latent_cache, params, config):
   """
   Single-token reference decode helper.

   x_new: [B, 1, D]
   past_kv_latent_cache: [B, T_past, C], previous tokens only.

   returns:
      out: [B, 1, D]
      new_kv_latent_cache: [B, T_past + 1, C]
   """
   validate_mhla_inputs(x_new, params, config)
   if x_new.shape[1] != 1:
      raise ValueError(f"x_new must have sequence length 1, got {x_new.shape}")
   if past_kv_latent_cache.ndim != 3:
      raise ValueError(
         "past_kv_latent_cache must be [B, T_past, C], "
         f"got {past_kv_latent_cache.shape}"
      )
   if past_kv_latent_cache.shape[0] != x_new.shape[0]:
      raise ValueError("x_new and past_kv_latent_cache batch dimensions must match")
   if past_kv_latent_cache.shape[-1] != config.latent_dim:
      raise ValueError(
         f"past_kv_latent_cache last dim is {past_kv_latent_cache.shape[-1]}, "
         f"expected latent_dim={config.latent_dim}"
      )

   B, _, D = x_new.shape
   T_past = past_kv_latent_cache.shape[1]
   H = config.num_heads
   Dh = config.head_dim
   R = config.rope_dim
   
   q_latent = jnp.matmul(x_new, params["q_down"])
   current_kv_latent = jnp.matmul(x_new, params["kv_down"])
   kv_latent_cache = jnp.concatenate(
      [past_kv_latent_cache, current_kv_latent],
      axis=1,
   )
   T_cache = T_past + 1

   q_content = jnp.matmul(q_latent, params["q_up_content"])
   q_rope = jnp.matmul(q_latent, params["q_up_rope"])

   k_content = jnp.matmul(kv_latent_cache, params["k_up_content"])
   k_rope = jnp.matmul(kv_latent_cache, params["k_up_rope"])
   v = jnp.matmul(kv_latent_cache, params["v_up"])

   q_content = jnp.reshape(q_content, (B, 1, H, Dh))
   q_rope = jnp.reshape(q_rope, (B, 1, H, R))
   k_content = jnp.reshape(k_content, (B, T_cache, H, Dh))
   k_rope = jnp.reshape(k_rope, (B, T_cache, H, R))
   v = jnp.reshape(v, (B, T_cache, H, Dh))

   # Causality comes from constructing cache as past tokens plus current token.
   key_positions = jnp.arange(T_cache)
   query_position = jnp.asarray([T_past])
   q = jnp.concatenate([q_content, apply_rope(q_rope, query_position)], axis=-1)
   k = jnp.concatenate([k_content, apply_rope(k_rope, key_positions)], axis=-1)

   scores = jnp.einsum("bqhd,bkhd->bhqk", q, k)
   scores = scores / jnp.sqrt(jnp.asarray(Dh + R, dtype=x_new.dtype))
   weights = jax.nn.softmax(scores, axis=-1)
   out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
   out = jnp.reshape(out, (B, 1, H * Dh))
   out = jnp.matmul(out, params["out_proj"])
   return out, kv_latent_cache


 
def _validate_expert_params(expert_params, config, expert_name):
    D = config.model_dim
    hidden = config.expert_hidden_dim
    expected_shapes = {
      "gate_proj": (D, hidden),
      "up_proj": (D, hidden),
      "down_proj": (hidden, D),
    }

    for name, expected_shape in expected_shapes.items():
      if name not in expert_params:
         raise KeyError(f"missing {expert_name} param: {name}")
      if expert_params[name].shape != expected_shape:
         raise ValueError(
            f"{expert_name}.{name} has shape {expert_params[name].shape}, "
            f"expected {expected_shape}"
         )


def validate_deepseek_moe_params(params, config):
    D = config.model_dim
    E = config.num_routed_experts
    S = config.num_shared_experts

    if "router" not in params:
      raise KeyError("missing DeepSeekMoE param: router")
    if params["router"].shape != (D, E):
      raise ValueError(
         f"router has shape {params['router'].shape}, expected {(D, E)}"
      )
    if "shared_experts" not in params:
      raise KeyError("missing DeepSeekMoE param: shared_experts")
    if "routed_experts" not in params:
      raise KeyError("missing DeepSeekMoE param: routed_experts")
    if len(params["shared_experts"]) != S:
      raise ValueError(
         f"got {len(params['shared_experts'])} shared experts, expected {S}"
      )
    if len(params["routed_experts"]) != E:
      raise ValueError(
         f"got {len(params['routed_experts'])} routed experts, expected {E}"
      )

    for shared_id, expert_params in enumerate(params["shared_experts"]):
      _validate_expert_params(expert_params, config, f"shared_experts[{shared_id}]")
    for expert_id, expert_params in enumerate(params["routed_experts"]):
      _validate_expert_params(expert_params, config, f"routed_experts[{expert_id}]")


def validate_deepseek_moe_inputs(x, params, config):
    validate_mhla_config(config)
    if x.ndim != 3:
      raise ValueError(f"x must be [B, T, D], got {x.shape}")
    if x.shape[-1] != config.model_dim:
      raise ValueError(
         f"x last dim is {x.shape[-1]}, expected model_dim={config.model_dim}"
      )
    validate_deepseek_moe_params(params, config)


def deepseek_moe(x, params, config):
    """
    Dense-reference DeepSeekMoE.

    x: [B, T, D]

    params:
      router: [D, E]
      shared_experts[S]:
         gate_proj: [D, hidden]
         up_proj: [D, hidden]
         down_proj: [hidden, D]
      routed_experts[E]:
         gate_proj: [D, hidden]
         up_proj: [D, hidden]
         down_proj: [hidden, D]

    returns:
      out: [B, T, D]
    """
    validate_deepseek_moe_inputs(x, params, config)

    B, T, D = x.shape
    
    E = config.num_routed_experts
    S = config.num_shared_experts
    K = config.top_k  

    # Shared experts are always active for every token.
    shared_out = jnp.zeros_like(x)  

    for shared_id in range(S):  
      shared_out += expert_mlp(x, params["shared_experts"][shared_id]) 

    shared_out = shared_out / S  

    # Router selects K routed experts independently for each token.
    router_logits = jnp.matmul(x, params["router"]) 
    
    top_values, top_indices = jax.lax.top_k(router_logits, k=K)  

    # Normalize only the selected expert logits, so selected weights sum to 1.
    router_weights = jax.nn.softmax(top_values, axis=-1) 

    routed_out = jnp.zeros_like(x)

    # Reference path: compute all routed experts, then zero unselected experts.
    # This is correct but intentionally not expert-dispatch optimized.
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

  
     
      
def init_mhla_params(key, config):
   keys = jax.random.split(key, 8)
   H = config.num_heads
   Dh = config.head_dim
   R = config.rope_dim
   D = config.model_dim
   C = config.latent_dim

   return {
      "q_down": _xavier(keys[0], (D, C)),
      "kv_down": _xavier(keys[1], (D, C)),
      "q_up_content": _xavier(keys[2], (C, H * Dh)),
      "q_up_rope": _xavier(keys[3], (C, H * R)),
      "k_up_content": _xavier(keys[4], (C, H * Dh)),
      "k_up_rope": _xavier(keys[5], (C, H * R)),
      "v_up": _xavier(keys[6], (C, H * Dh)),
      "out_proj": _xavier(keys[7], (H * Dh, D)),
   }


def init_moe_params(key, config):
   D = config.model_dim
   E = config.num_experts
   hidden = config.expert_hidden_dim
   keys = jax.random.split(key, 1 + 3 * E)

   experts = []
   for expert_id in range(E):
      base = 1 + 3 * expert_id
      experts.append({
         "gate_proj": _xavier(keys[base], (D, hidden)),
         "up_proj": _xavier(keys[base + 1], (D, hidden)),
         "down_proj": _xavier(keys[base + 2], (hidden, D)),
      })

   return {
      "router": _xavier(keys[0], (D, E)),
      "experts": tuple(experts),
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


def rms_norm(x, weight, eps=1e-6):
   rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
   return x * rms * weight


def apply_rope(x, positions=None):
   # x: [B, T, H, R], R must be even.
   B, T, H, R = x.shape
   if R % 2 != 0:
      raise ValueError("rope_dim must be even")

   half = R // 2
   if positions is None:
      positions = jnp.arange(T)
   positions = positions.astype(x.dtype)
   freqs = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=x.dtype) / half))
   angles = positions[:, None] * freqs[None, :]
   cos = jnp.cos(angles)[None, :, None, :]
   sin = jnp.sin(angles)[None, :, None, :]

   x1 = x[..., :half]
   x2 = x[..., half:]
   return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def apply_causal_mask(scores):
   # scores: [B, H, T, T]
   T = scores.shape[-1]
   mask = jnp.tril(jnp.ones((T, T), dtype=bool))
   return jnp.where(mask[None, None, :, :], scores, -jnp.inf)


def merge_heads(x):
   # x: [B, H, T, Dh] -> [B, T, H * Dh]
   B, H, T, Dh = x.shape
   x = jnp.transpose(x, (0, 2, 1, 3))
   return jnp.reshape(x, (B, T, H * Dh))



if __name__ == "__main__":  

    key = jax.random.PRNGKey(0)
    config = MHLAConfig(
       model_dim=1024, 
       num_heads=8, 
       head_dim=128, 
       latent_dim=256, 
       rope_dim=16, 
       num_experts=4,
       num_routed_experts=4,
       num_shared_experts=1,
       top_k=2, 
       expert_hidden_dim=2048,
    ) 

    attn_key, moe_key, x_key = jax.random.split(key, 3)
    params = {
       "attn": init_mhla_params(attn_key, config),
       "moe": init_deepseek_moe_params(moe_key, config),
    }

    x = jax.random.normal(x_key, (1, 10, config.model_dim))
    y = mhlatent_attention(x, params["attn"], config)
    past_cache = jnp.matmul(x[:, :-1, :], params["attn"]["kv_down"])
    step_y, new_cache = mhlatent_attention_step(
       x[:, -1:, :],
       past_cache,
       params["attn"],
       config,
    )
    z = deepseek_moe(y, params["moe"], config)

    print("input shape", x.shape) 
    print("attention output shape", y.shape)
    print("step output shape", step_y.shape)
    print("new cache shape", new_cache.shape)
    print("step max error", jnp.max(jnp.abs(y[:, -1:, :] - step_y)))
    print("moe output shape", z.shape)
    print(jax.devices())
    print(jax.default_backend())


     

