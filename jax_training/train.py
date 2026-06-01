from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_training.data import get_batch, load_cached_lm_dataset
from jax_training.model import JaxLMConfig, init_lm_params, loss_fn


def make_train_step(config: JaxLMConfig, optimizer):
    @jax.jit
    def train_step(params, opt_state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb, config)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def main():
    seed = 1337
    dataset_name = "parameter_golf_sp1024"
    max_encoded_tokens = 1_000_000
    batch_size = 2
    seq_len = 64
    max_steps = 1000
    log_interval = 10
    learning_rate = 3e-4
    weight_decay = 0.01

    print("devices:", jax.devices())
    print("backend:", jax.default_backend())

    dataset = load_cached_lm_dataset(
        dataset_name,
        max_encoded_tokens=max_encoded_tokens,
    )

    config = JaxLMConfig(
        vocab_size=dataset.vocab_size,
        max_seq_len=seq_len,
        model_dim=128,
        num_layers=1,
        num_heads=4,
        head_dim=32,
        latent_dim=64,
        rope_dim=16,
        attention_type="deepseek_csa_hca_mhc",
        chunk_size=16,
        index_dim=32,
        index_heads=2,
        csa_compress_rate=4,
        hca_compress_rate=16,
        local_window_size=16,
        num_mhc_streams=4,
        mhc_hidden_dim=256,
        mhc_sinkhorn_iters=20,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
        expert_hidden_dim=256,
    )

    key = jax.random.PRNGKey(seed)
    params = init_lm_params(key, config)

    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    opt_state = optimizer.init(params)
    train_step = make_train_step(config, optimizer)

    rng = np.random.default_rng(seed)
    last_time = time.time()

    for step in range(1, max_steps + 1):
        xb_np, yb_np = get_batch(
            "train",
            dataset,
            batch_size=batch_size,
            seq_len=seq_len,
            rng=rng,
        )
        xb = jnp.asarray(xb_np)
        yb = jnp.asarray(yb_np)

        params, opt_state, loss = train_step(params, opt_state, xb, yb)

        if step == 1 or step % log_interval == 0:
            now = time.time()
            elapsed = now - last_time
            steps = 1 if step == 1 else log_interval
            tokens_per_sec = batch_size * seq_len * steps / max(elapsed, 1e-8)
            last_time = now
            print(
                f"step={step} "
                f"loss={float(loss):.4f} "
                f"tokens_sec={tokens_per_sec:.0f}"
            )


if __name__ == "__main__":
    main()
