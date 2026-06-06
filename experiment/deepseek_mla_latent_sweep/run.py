from __future__ import annotations

import argparse
from dataclasses import replace

import jax

from jax_training.data import load_cached_lm_dataset
from jax_training.model import JaxLMConfig
from jax_training.train import TrainConfig, run_training


LATENT_VARIANTS = {
    "small": 96,
    "medium": 192,
    "large": 384,
}
RUN_ORDER = ("small", "medium", "large", "mha")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek MLA latent sweep plus MHA baseline."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=RUN_ORDER,
        default=list(RUN_ORDER),
        help="Runs to launch in order.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-encoded-tokens", type=int, default=150_000_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    return parser.parse_args()


def build_train_config(args, run_id):
    run_name = build_run_name(run_id, args.seq_len)
    log_path = f"experiment/deepseek_mla_latent_sweep/{run_id}/summary.jsonl"
    return replace(
        TrainConfig(),
        seed=args.seed,
        experiment_name="deepseek_mla_latent_sweep",
        latent_variant=run_id,
        run_name=run_name,
        log_path=log_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_encoded_tokens=args.max_encoded_tokens,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
    )


def build_run_name(run_id, seq_len):
    if run_id == "mha":
        return f"mha_muon_768d_6l_seq{seq_len}"
    latent_dim = LATENT_VARIANTS[run_id]
    return f"deepseek_mla_latent{latent_dim}_muon_768d_6l_seq{seq_len}"


def build_model_config(dataset, train_config, run_id):
    is_mha = run_id == "mha"
    latent_dim = 192 if is_mha else LATENT_VARIANTS[run_id]
    return JaxLMConfig(
        vocab_size=dataset.vocab_size,
        max_seq_len=train_config.seq_len,
        model_dim=768,
        num_layers=6,
        num_heads=12,
        head_dim=64,
        latent_dim=latent_dim,
        rope_dim=32,
        attention_type="mha" if is_mha else "mhla",
        chunk_size=16,
        index_dim=64,
        index_heads=4,
        csa_compress_rate=8,
        hca_compress_rate=64,
        local_window_size=64,
        num_mhc_streams=4,
        mhc_hidden_dim=1536,
        mhc_sinkhorn_iters=8,
        num_routed_experts=8,
        num_shared_experts=1,
        top_k=2,
        expert_hidden_dim=3072,
    )


def main():
    args = parse_args()

    print("devices:", jax.devices())
    print("backend:", jax.default_backend())
    print("runs:", ", ".join(args.runs))

    dataset = load_cached_lm_dataset(
        "parameter_golf_sp1024",
        max_encoded_tokens=args.max_encoded_tokens,
    )

    for run_id in args.runs:
        train_config = build_train_config(args, run_id)
        model_config = build_model_config(dataset, train_config, run_id)
        print(f"starting run={train_config.run_name}")
        run_training(train_config, model_config, dataset)


if __name__ == "__main__":
    main()
