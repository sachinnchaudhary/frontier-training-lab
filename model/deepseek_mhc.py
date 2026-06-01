from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


"""

=====================================================================
SETUP: THE N-STREAM HIGHWAY
=====================================================================
    // Let n = 4 (Expansion Rate)
    // Let C = 2560 (Hidden Dimension)
    // Input: x_l (The massive memory highway, shape: [n, C])

=====================================================================
PHASE 1: DYNAMIC ROUTING GENERATION
=====================================================================
    // Flatten the entire memory highway to see the full context
    x_vec = flatten(x_l)                     // Shape: [1, n*C]
    x_norm = RMSNorm(x_vec)

    // Generate the raw routing parameters (Dynamic + Static biases)
    H_pre_raw  = Linear_pre(x_norm) + b_pre  // Shape: [1, n]
    H_post_raw = Linear_post(x_norm)+ b_post // Shape: [1, n]
    H_res_raw  = Linear_res(x_norm) + b_res  // Shape: [n, n]

=====================================================================
PHASE 2: THE MANIFOLD CONSTRAINTS (The mHC Fix)
=====================================================================
    // Constrain Read/Write to prevent signal cancellation
    H_pre = Sigmoid(H_pre_raw)
    H_post = 2 * Sigmoid(H_post_raw)

    // [THE CORE FIX]: The Sinkhorn-Knopp Projection
    // This stops the compounding explosion by forcing the mixing 
    // matrix to be strictly doubly stochastic (a convex combination).
    
    H_res = exp(H_res_raw)                   // Step 1: Make strictly positive
    
    Loop t from 1 to 20:                     // Step 2: Iterative normalization
        H_res = Normalize_Columns_To_Sum_To_1(H_res)
        H_res = Normalize_Rows_To_Sum_To_1(H_res)
        
    // H_res is now perfectly safe to multiply across 100 layers.

=====================================================================
PHASE 3: SQUEEZE & PROCESS (The Read Phase)
=====================================================================
    // Compress the 4 memory streams down to 1 stream for the layer
    h_in = matmul(H_pre, x_l)                // Shape: [1, n] x [n, C] -> [1, C]
    
    // The Heavy Lifting (Standard Attention or Multi-Layer Perceptron)
    h_out = Layer_F(h_in)                    // Shape: [1, C]

=====================================================================
PHASE 4: EXPAND, SHUFFLE, & MERGE (Write & Mix Phase)
=====================================================================
    // Track A: The Write (Expand processed data back to 4 streams)
    update = matmul(Transpose(H_post), h_out) // Shape: [n, 1] x [1, C] -> [n, C]

    // Track B: The Mix (Parallel shuffle of the original memory highway)
    highway_next = matmul(H_res, x_l)         // Shape: [n, n] x [n, C] -> [n, C]
    
    // The Merge (Combine the shuffled highway with the new thought)
    x_l+1 = highway_next + update             // Shape: [n, C]
    
    // Output x_l+1 is passed directly to the next block


"""


@dataclass(frozen=True)
class MHCConfig:
    model_dim: int
    num_streams: int
    hidden_dim: int
    sinkhorn_iters: int = 20
    eps: float = 1e-6


def _xavier(key, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def init_mhc_params(key, config):
    route_dim = config.num_streams * config.model_dim
    N = config.num_streams
    D = config.model_dim
    hidden = config.hidden_dim

    keys = jax.random.split(key, 7)

    return {
        "route_norm": jnp.ones((route_dim,), dtype=jnp.float32),
        "pre_proj": _xavier(keys[0], (route_dim, N)),
        "pre_bias": jnp.zeros((N,), dtype=jnp.float32),
        "post_proj": _xavier(keys[1], (route_dim, N)),
        "post_bias": jnp.zeros((N,), dtype=jnp.float32),
        "res_proj": _xavier(keys[2], (route_dim, N * N)),
        "res_bias": jnp.zeros((N * N,), dtype=jnp.float32),
        "layer_norm": jnp.ones((D,), dtype=jnp.float32),
        "layer_gate": _xavier(keys[3], (D, hidden)),
        "layer_up": _xavier(keys[4], (D, hidden)),
        "layer_down": _xavier(keys[5], (hidden, D)),
        "readout": _xavier(keys[6], (D, D)),
    }



def mhc_block(x_streams, params, config, layer_fn):  

    H_pre, H_post, H_res = generate_mhc_routes(x_streams, params, config)
    
    h_in = jnp.einsum("btn,btnd->btd", H_pre, x_streams) 

    h_out = layer_fn(h_in)

    update = H_post[..., :, None] * h_out[..., None, :]  

    highway_next = jnp.einsum("btij,btjd->btid", H_res, x_streams) 

    x_next = highway_next + update

    return x_next



def generate_mhc_routes(x_streams, params, config):  

    B, T, N, D = x_streams.shape  

    x_vec = jnp.reshape(x_streams, [B, T, N * D])  

    x_norm = rms_norm(x_vec, params["route_norm"], config.eps)   
    
    pre_raw = jnp.matmul(x_norm, params["pre_proj"]) + params["pre_bias"]
    post_raw = jnp.matmul(x_norm, params["post_proj"]) + params["post_bias"]
    res_raw = jnp.matmul(x_norm, params["res_proj"]) + params["res_bias"]

    res_raw = jnp.reshape(res_raw, [B, T, N, N])

    H_pre = jax.nn.sigmoid(pre_raw) 
    H_post = 2.0 * jax.nn.sigmoid(post_raw) 
    H_res = sinkhorn(res_raw, config.sinkhorn_iters, config.eps)

    return H_pre, H_post, H_res  



def sinkhorn(logits, iters, eps):  

    H = jnp.exp(logits)  

    for i in range(iters):  
        H = H / (jnp.sum(H, axis=-2, keepdims=True) + eps)  

        H = H / (jnp.sum(H, axis=-1, keepdims=True) + eps)  

    return H  
    

def rms_norm(x, weights, eps):  

    rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)

    return x * rms * weights  


def mhc_test_layer(h, params, config):
    h = rms_norm(h, params["layer_norm"], config.eps)
    gate = jnp.matmul(h, params["layer_gate"])
    up = jnp.matmul(h, params["layer_up"])
    hidden = jax.nn.silu(gate) * up
    return jnp.matmul(hidden, params["layer_down"])


def mhc_readout(x_streams, params):
    h = jnp.mean(x_streams, axis=2)
    return jnp.matmul(h, params["readout"])


def mhc_forward(x_streams, params, config):
    def layer_fn(h):
        return mhc_test_layer(h, params, config)

    y_streams = mhc_block(x_streams, params, config, layer_fn)
    y = mhc_readout(y_streams, params)
    return y_streams, y


def mse_loss(params, x_streams, target, config):
    _, y = mhc_forward(x_streams, params, config)
    return jnp.mean(jnp.square(y - target))


def tree_l2_norm(tree):
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree))
    )


def sgd_update(params, grads, learning_rate):
    return jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g,
        params,
        grads,
    )


def train_step(params, x_streams, target, config, learning_rate):
    loss, grads = jax.value_and_grad(mse_loss)(params, x_streams, target, config)
    params = sgd_update(params, grads, learning_rate)
    return params, loss, tree_l2_norm(grads)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    config = MHCConfig(
        model_dim=32,
        num_streams=4,
        hidden_dim=64,
        sinkhorn_iters=20,
    )

    param_key, x_key, target_key = jax.random.split(key, 3)
    params = init_mhc_params(param_key, config)

    x_streams = jax.random.normal(
        x_key,
        (2, 8, config.num_streams, config.model_dim),
    )
    target = jax.random.normal(target_key, (2, 8, config.model_dim))

    y_streams, y = mhc_forward(x_streams, params, config)
    loss, grads = jax.value_and_grad(mse_loss)(params, x_streams, target, config)

    H_pre, H_post, H_res = generate_mhc_routes(x_streams, params, config)
    row_error = jnp.max(jnp.abs(jnp.sum(H_res, axis=-1) - 1.0))
    col_error = jnp.max(jnp.abs(jnp.sum(H_res, axis=-2) - 1.0))

    print("input streams:", x_streams.shape)
    print("output streams:", y_streams.shape)
    print("readout:", y.shape)
    print("loss:", loss)
    print("grad norm:", tree_l2_norm(grads))
    print("route pre:", H_pre.shape)
    print("route post:", H_post.shape)
    print("route residual:", H_res.shape)
    print("sinkhorn row error:", row_error)
    print("sinkhorn col error:", col_error)

    learning_rate = 1e-3
    for step in range(1, 6):
        params, loss, grad_norm = train_step(
            params,
            x_streams,
            target,
            config,
            learning_rate,
        )
        print(f"step={step} loss={float(loss):.6f} grad_norm={float(grad_norm):.6f}")

    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    np.testing.assert_equal(
        y_streams.shape,
        (2, 8, config.num_streams, config.model_dim),
    )
    np.testing.assert_equal(y.shape, (2, 8, config.model_dim))

