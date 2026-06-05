import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from model.deepseek_csa import (
    DeepSeekCSAConfig,
    deepseek_csa_attention,
    deepseek_hca_attention,
    deepseek_hybrid_attention,
    init_deepseek_csa_params,
    init_deepseek_hca_params,
    init_deepseek_hybrid_params,
)
from model.deepseek_mhc import MHCConfig, init_mhc_params, mhc_forward
from model.deepseek_sparseatt import (
    DeepSeekSparseConfig,
    deepseek_sparse_attention,
    init_deepseek_sparse_params,
)
from model.kimi_deltanet import (
    KimiDeltaNetConfig,
    init_kimi_deltanet_params,
    kimi_deltanet_chunckwise,
    kimi_deltanet_parallel_chunkwise,
    kimi_deltanet_stepwise,
)
from model.mhlatent_attention import MHLAConfig, init_mhla_params, mhlatent_attention


PRESETS = {
    "tiny": {
        "batch_size": 1,
        "seq_len": 64,
        "model_dim": 128,
        "num_heads": 4,
        "head_dim": 32,
        "latent_dim": 32,
        "rope_dim": 16,
        "index_dim": 16,
        "index_heads": 2,
        "top_k": 2,
        "chunk_size": 16,
        "csa_compress_rate": 8,
        "hca_compress_rate": 32,
        "local_window_size": 32,
        "mhc_streams": 4,
        "mhc_hidden_dim": 256,
        "mhc_sinkhorn_iters": 8,
    },
    "small": {
        "batch_size": 1,
        "seq_len": 256,
        "model_dim": 256,
        "num_heads": 4,
        "head_dim": 64,
        "latent_dim": 64,
        "rope_dim": 16,
        "index_dim": 32,
        "index_heads": 2,
        "top_k": 4,
        "chunk_size": 16,
        "csa_compress_rate": 8,
        "hca_compress_rate": 64,
        "local_window_size": 64,
        "mhc_streams": 4,
        "mhc_hidden_dim": 512,
        "mhc_sinkhorn_iters": 8,
    },
    "medium": {
        "batch_size": 1,
        "seq_len": 256,
        "model_dim": 512,
        "num_heads": 4,
        "head_dim": 128,
        "latent_dim": 128,
        "rope_dim": 32,
        "index_dim": 64,
        "index_heads": 4,
        "top_k": 8,
        "chunk_size": 8,
        "csa_compress_rate": 16,
        "hca_compress_rate": 96,
        "local_window_size": 96,
        "mhc_streams": 6,
        "mhc_hidden_dim": 1024,
        "mhc_sinkhorn_iters": 12,
    },
}


def block_until_ready(value):
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def param_count(params):
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def make_input(key, cfg):
    return jax.random.normal(
        key,
        (cfg["batch_size"], cfg["seq_len"], cfg["model_dim"]),
        dtype=jnp.float32,
    )


def build_mhla(key, cfg):
    config = MHLAConfig(
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        head_dim=cfg["head_dim"],
        latent_dim=cfg["latent_dim"],
        rope_dim=cfg["rope_dim"],
    )
    params = init_mhla_params(key, config)
    return params, config, lambda p, x: mhlatent_attention(x, p, config)


def build_sparse(key, cfg):
    config = DeepSeekSparseConfig(
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        latent_dim=cfg["latent_dim"],
        rope_dim=cfg["rope_dim"],
        index_dim=cfg["index_dim"],
        index_heads=cfg["index_heads"],
        top_k=cfg["top_k"],
    )
    params = init_deepseek_sparse_params(key, config)
    return params, config, lambda p, x: deepseek_sparse_attention(x, p, config)


def build_csa(key, cfg):
    config = DeepSeekCSAConfig(
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        latent_dim=cfg["latent_dim"],
        rope_dim=cfg["rope_dim"],
        index_dim=cfg["index_dim"],
        index_heads=cfg["index_heads"],
        csa_compress_rate=cfg["csa_compress_rate"],
        top_k=cfg["top_k"],
        hca_compress_rate=cfg["hca_compress_rate"],
        local_window_size=cfg["local_window_size"],
        num_routed_experts=4,
        num_shared_experts=1,
        expert_hidden_dim=cfg["model_dim"] * 2,
    )
    params = init_deepseek_csa_params(key, config)
    return params, config, lambda p, x: deepseek_csa_attention(x, p, config)


def build_hca(key, cfg):
    config = DeepSeekCSAConfig(
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        latent_dim=cfg["latent_dim"],
        rope_dim=cfg["rope_dim"],
        index_dim=cfg["index_dim"],
        index_heads=cfg["index_heads"],
        csa_compress_rate=cfg["csa_compress_rate"],
        top_k=cfg["top_k"],
        hca_compress_rate=cfg["hca_compress_rate"],
        local_window_size=cfg["local_window_size"],
        num_routed_experts=4,
        num_shared_experts=1,
        expert_hidden_dim=cfg["model_dim"] * 2,
    )
    params = init_deepseek_hca_params(key, config)
    return params, config, lambda p, x: deepseek_hca_attention(x, p, config)


def build_csa_hca(key, cfg):
    config = DeepSeekCSAConfig(
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        latent_dim=cfg["latent_dim"],
        rope_dim=cfg["rope_dim"],
        index_dim=cfg["index_dim"],
        index_heads=cfg["index_heads"],
        csa_compress_rate=cfg["csa_compress_rate"],
        top_k=cfg["top_k"],
        hca_compress_rate=cfg["hca_compress_rate"],
        local_window_size=cfg["local_window_size"],
        num_routed_experts=4,
        num_shared_experts=1,
        expert_hidden_dim=cfg["model_dim"] * 2,
    )
    params = init_deepseek_hybrid_params(key, config)["attn"]
    return params, config, lambda p, x: deepseek_hybrid_attention(x, p, config)


def build_kimi_stepwise(key, cfg):
    return build_kimi(key, cfg, kimi_deltanet_stepwise)


def build_kimi_chunkwise(key, cfg):
    return build_kimi(key, cfg, kimi_deltanet_chunckwise)


def build_kimi_parallel(key, cfg):
    return build_kimi(key, cfg, kimi_deltanet_parallel_chunkwise)


def build_kimi(key, cfg, fn):
    config = KimiDeltaNetConfig(
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        key_dim=cfg["latent_dim"],
        value_dim=cfg["latent_dim"],
        chunk_size=cfg["chunk_size"],
        gate_type="vector",
        expert_hidden_dim=cfg["model_dim"] * 2,
    )
    params = init_kimi_deltanet_params(key, config)
    return params, config, lambda p, x: fn(x, p, config)


def build_mhc(key, cfg):
    config = MHCConfig(
        model_dim=cfg["model_dim"],
        num_streams=cfg["mhc_streams"],
        hidden_dim=cfg["mhc_hidden_dim"],
        sinkhorn_iters=cfg["mhc_sinkhorn_iters"],
    )
    params = init_mhc_params(key, config)

    def forward(p, x):
        x_streams = jnp.repeat(x[:, :, None, :], config.num_streams, axis=2)
        _, y = mhc_forward(x_streams, p, config)
        return y

    return params, config, forward


BUILDERS = {
    "mhla": build_mhla,
    "sparse": build_sparse,
    "csa": build_csa,
    "hca": build_hca,
    "csa_hca": build_csa_hca,
    "kimi_stepwise": build_kimi_stepwise,
    "kimi_chunkwise": build_kimi_chunkwise,
    "kimi_parallel": build_kimi_parallel,
    "mhc": build_mhc,
}

MODEL_GROUPS = {
    "all": list(BUILDERS.keys()),
    "core": ["mhla", "sparse", "csa", "hca", "csa_hca", "mhc"],
    "kimi": ["kimi_stepwise", "kimi_chunkwise", "kimi_parallel"],
}


def timed_call(fn):
    start = time.perf_counter()
    out = fn()
    block_until_ready(out)
    return (time.perf_counter() - start) * 1000.0, out


def mean_timed_call(fn, iters):
    times = []
    out = None
    for _ in range(iters):
        elapsed_ms, out = timed_call(fn)
        times.append(elapsed_ms)
    return sum(times) / len(times), min(times), max(times), out


def finite_float(value):
    return float(value)


def run_trace(
    model_name,
    preset_name,
    trace_mode,
    trace_dir,
    iters,
    jitted_forward,
    jitted_train,
    params,
    x,
):
    trace_path = trace_dir / f"{model_name}_{preset_name}_{trace_mode}"
    trace_path.mkdir(parents=True, exist_ok=True)

    if trace_mode == "forward":
        fn = lambda: jitted_forward(params, x)
    elif trace_mode == "train":
        fn = lambda: jitted_train(params, x)
    else:
        raise ValueError(f"unknown trace_mode: {trace_mode}")

    print(f"starting JAX trace: {trace_path}")
    jax.profiler.start_trace(str(trace_path))
    try:
        for _ in range(iters):
            block_until_ready(fn())
    finally:
        jax.profiler.stop_trace()
    print(f"stopped JAX trace: {trace_path}")
    return str(trace_path)


def profile_one(model_name, preset_name, warmup, iters, seed, trace, trace_mode, trace_dir):
    cfg = PRESETS[preset_name]
    key = jax.random.PRNGKey(seed)
    param_key, x_key = jax.random.split(key)

    params, model_config, forward = BUILDERS[model_name](param_key, cfg)
    x = make_input(x_key, cfg)

    jitted_forward = jax.jit(lambda p, batch: forward(p, batch))
    jitted_train = jax.jit(
        lambda p, batch: jax.value_and_grad(
            lambda pp: jnp.mean(jnp.square(forward(pp, batch)))
        )(p)
    )

    compile_fwd_ms, y = timed_call(lambda: jitted_forward(params, x))
    compile_train_ms, train_out = timed_call(lambda: jitted_train(params, x))
    loss, grads = train_out

    for _ in range(warmup):
        block_until_ready(jitted_forward(params, x))
        block_until_ready(jitted_train(params, x))

    trace_path = None
    if trace:
        trace_path = run_trace(
            model_name=model_name,
            preset_name=preset_name,
            trace_mode=trace_mode,
            trace_dir=trace_dir,
            iters=iters,
            jitted_forward=jitted_forward,
            jitted_train=jitted_train,
            params=params,
            x=x,
        )

    fwd_mean, fwd_min, fwd_max, y = mean_timed_call(
        lambda: jitted_forward(params, x),
        iters,
    )
    train_mean, train_min, train_max, train_out = mean_timed_call(
        lambda: jitted_train(params, x),
        iters,
    )
    loss, grads = train_out
    grad_norm = tree_l2_norm(grads)

    tokens = cfg["batch_size"] * cfg["seq_len"]
    row = {
        "model": model_name,
        "preset": preset_name,
        "trace_enabled": trace,
        "trace_mode": trace_mode if trace else None,
        "trace_path": trace_path,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "batch_size": cfg["batch_size"],
        "seq_len": cfg["seq_len"],
        "model_dim": cfg["model_dim"],
        "num_heads": cfg["num_heads"],
        "head_dim": cfg["head_dim"],
        "latent_dim": cfg["latent_dim"],
        "rope_dim": cfg["rope_dim"],
        "index_dim": cfg["index_dim"],
        "index_heads": cfg["index_heads"],
        "top_k": cfg["top_k"],
        "chunk_size": cfg["chunk_size"],
        "csa_compress_rate": cfg["csa_compress_rate"],
        "hca_compress_rate": cfg["hca_compress_rate"],
        "local_window_size": cfg["local_window_size"],
        "mhc_streams": cfg["mhc_streams"],
        "mhc_hidden_dim": cfg["mhc_hidden_dim"],
        "mhc_sinkhorn_iters": cfg["mhc_sinkhorn_iters"],
        "param_count": param_count(params),
        "output_shape": tuple(int(dim) for dim in y.shape),
        "loss": finite_float(loss),
        "grad_norm": finite_float(grad_norm),
        "loss_is_finite": bool(jnp.isfinite(loss)),
        "grad_norm_is_finite": bool(jnp.isfinite(grad_norm)),
        "compile_forward_ms": compile_fwd_ms,
        "compile_train_ms": compile_train_ms,
        "forward_ms_mean": fwd_mean,
        "forward_ms_min": fwd_min,
        "forward_ms_max": fwd_max,
        "train_ms_mean": train_mean,
        "train_ms_min": train_min,
        "train_ms_max": train_max,
        "tokens_per_sec_forward": tokens / (fwd_mean / 1000.0),
        "tokens_per_sec_train": tokens / (train_mean / 1000.0),
        "warmup": warmup,
        "iters": iters,
        "seed": seed,
    }
    return row


def write_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def print_row(row):
    print(
        " ".join(
            [
                f"model={row['model']}",
                f"preset={row['preset']}",
                f"backend={row['backend']}",
                f"B={row['batch_size']}",
                f"T={row['seq_len']}",
                f"D={row['model_dim']}",
                f"params={row['param_count']}",
                f"compile_fwd_ms={row['compile_forward_ms']:.2f}",
                f"fwd_ms={row['forward_ms_mean']:.2f}",
                f"train_ms={row['train_ms_mean']:.2f}",
                f"tok_s_fwd={row['tokens_per_sec_forward']:.0f}",
                f"tok_s_train={row['tokens_per_sec_train']:.0f}",
                f"loss={row['loss']:.6f}",
                f"grad_norm={row['grad_norm']:.6f}",
                f"finite={row['loss_is_finite'] and row['grad_norm_is_finite']}",
            ]
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Small JAX profiler for reference models.")
    parser.add_argument(
        "--model",
        choices=sorted(list(BUILDERS.keys()) + list(MODEL_GROUPS.keys())),
        default="all",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="tiny")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--trace-mode",
        choices=["forward", "train"],
        default="train",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("profiling/traces"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling/runs/jax_profile_summary.jsonl"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = MODEL_GROUPS[args.model] if args.model in MODEL_GROUPS else [args.model]
    if args.trace and len(models) != 1:
        raise ValueError("trace profiling should run one model at a time")

    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.default_backend())

    for model_name in models:
        row = profile_one(
            model_name=model_name,
            preset_name=args.preset,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            trace=args.trace,
            trace_mode=args.trace_mode,
            trace_dir=args.trace_dir,
        )
        print_row(row)
        write_jsonl(args.output, row)

    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
