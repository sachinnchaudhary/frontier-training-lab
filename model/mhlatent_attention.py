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


def _xavier(key, shape):
   fan_in, fan_out = shape[0], shape[-1]
   limit = jnp.sqrt(6.0 / (fan_in + fan_out))
   return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def mhlatent_attention(x, params, config): 
   
   B, T, D = x.shape 
   H = config.num_heads
   Dh = config.head_dim 
   C = config.latent_dim 
   R =config.rope_dim

   q_latent = jnp.matmul(x, params["q_down"]) 
   kv_latent = jnp.matmul(x , params["kv_down"]) 
   
   q_content = jnp.matmul(q_latent , params["q_up_content"]) 
   q_rope = jnp.matmul(q_latent, params["q_up_rope"]) 

   q_content = jnp.reshape(q_content, [B, T, H, Dh]) 
   q_rope = jnp.reshape(q_rope, [B, T, H, R]) 

   
   k_content = jnp.matmul(kv_latent, params["k_up_content"]) 
   v = jnp.matmul(kv_latent, params["v_up"]) 
   k_rope = jnp.matmul(kv_latent, params["k_up_rope"])  

   k_content = jnp.reshape(k_content, [B, T, H, Dh]) 
   v = jnp.reshape(v, [B, T, H, Dh]) 
   k_rope = jnp.reshape(k_rope, [B, T, H, R])
   

   #apply rope. 
   q_rope = apply_rope(q_rope) 
   k_rope = apply_rope(k_rope) 

   q = jnp.concatenate([q_content, q_rope], axis=-1)
   k = jnp.concatenate([k_content, k_rope], axis=-1) 

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

   

def mhla_attention_inference_step(x_new, kv_latent_cache, params, configs):  
    
    q_latent = jnp.matmul(x_new, params["q_down"]) 

    q_fused = jnp.matmul(q_latent, params["fused_qk"]) 

    scores = jnp.matmul(q_fused, jnp.transpose(kv_latent_cache, -1, -2)) 
    scores = scores / jnp.sqrt(configs.head_dim) 

    weights = jax.nn.softmax(scores, axis=-1) 

    mixed_latent = jnp.matmul(weights, kv_latent_cache) 

    value = jnp.matmul(mixed_latent, params["v_up"]) 
    out = merge_heads(value) 

    out = jnp.matmul(out, params["out_proj"])  

    return out 



def determinsitic_moe(x, params, config):  
   
   E = config.num_experts
   K = config.top_k  

   router_logits = jnp.matmul(x, params["router"]) 

   top_value, top_indices =  jax.lax.top_k(router_logits, k= K) 
   router_weights = jax.nn.softmax(top_value, axis=-1)  

   output = jnp.zeros_like(x)  

   for expert_id in range(E):  

       expert_mask = top_indices == expert_id  
       expert_out = expert_mlp(x, params["experts"][expert_id])  

       for slot in range(K):  
           weight = jnp.where(top_indices[...,slot] == expert_id, router_weights[..., slot] , 0.0)    
           output += expert_out * weight[..., None]  

   return output  


 
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


def apply_rope(x):
   # x: [B, T, H, R], R must be even.
   B, T, H, R = x.shape
   if R % 2 != 0:
      raise ValueError("rope_dim must be even")

   half = R // 2
   positions = jnp.arange(T, dtype=x.dtype)
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
    z = deepseek_moe(y, params["moe"], config)

    print("input shape", x.shape) 
    print("attention output shape", y.shape)
    print("moe output shape", z.shape)
    print(jax.devices())
    print(jax.default_backend())


     

