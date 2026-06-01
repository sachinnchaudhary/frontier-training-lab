from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import jax

from jax_training.data import load_cached_lm_dataset
from jax_training.model import JaxLMConfig
from jax_training.train import TrainConfig, run_training


CSA_COMPRESS_RATES = (4, 8, 16)
HCA_COMPRESS_RATES = (32, 64, 128)
RUN_IDS = tuple(
    f"csa{csa}_hca{hca}"
    for csa, hca in itertools.product(CSA_COMPRESS_RATES, HCA_COMPRESS_RATES)
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CSA/HCA compression balance sweep."
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help="Pilot uses 3k steps; full uses 30k steps.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=RUN_IDS,
        default=list(RUN_IDS),
        help="CSA/HCA compression run ids to launch.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-encoded-tokens", type=int, default=150_000_000)
    return parser.parse_args()


def parse_run_id(run_id):
    csa_part, hca_part = run_id.split("_")
    csa = int(csa_part.removeprefix("csa"))
    hca = int(hca_part.removeprefix("hca"))
    if csa not in CSA_COMPRESS_RATES:
        raise ValueError(f"unsupported csa_compress_rate in {run_id}")
    if hca not in HCA_COMPRESS_RATES:
        raise ValueError(f"unsupported hca_compress_rate in {run_id}")
    return csa, hca


def build_train_config(args, run_id):
    if args.mode == "pilot":
        max_steps = 3_000
        eval_batches = 10
        warmup_steps = 300
    else:
        max_steps = 30_000
        eval_batches = 20
        warmup_steps = 1_000

    seq_len = 512
    run_name = f"csa_hca_{run_id}_muon_768d_6l_seq{seq_len}"
    log_path = f"experiment/csa_hca_compression_sweep/{args.mode}/{run_id}/summary.jsonl"
    return replace(
        TrainConfig(),
        seed=args.seed,
        experiment_name="csa_hca_compression_sweep",
        latent_variant=run_id,
        run_name=run_name,
        log_path=log_path,
        max_encoded_tokens=args.max_encoded_tokens,
        batch_size=8,
        seq_len=seq_len,
        max_steps=max_steps,
        log_interval=10,
        eval_interval=250,
        eval_batches=eval_batches,
        warmup_steps=warmup_steps,
    )


def build_model_config(dataset, train_config, run_id):
    csa_compress_rate, hca_compress_rate = parse_run_id(run_id)
    return JaxLMConfig(
        vocab_size=dataset.vocab_size,
        max_seq_len=train_config.seq_len,
        model_dim=768,
        num_layers=6,
        num_heads=12,
        head_dim=64,
        latent_dim=192,
        rope_dim=32,
        attention_type="deepseek_csa_hca",
        chunk_size=16,
        index_dim=64,
        index_heads=4,
        csa_compress_rate=csa_compress_rate,
        hca_compress_rate=hca_compress_rate,
        local_window_size=64,
        num_mhc_streams=4,
        mhc_hidden_dim=1536,
        mhc_sinkhorn_iters=8,
        num_routed_experts=8,
        num_shared_experts=1,
        top_k=8,
        moe_top_k=2,
        expert_hidden_dim=3072,
    )


def main():
    args = parse_args()

    print("devices:", jax.devices())
    print("backend:", jax.default_backend())
    print("experiment=csa_hca_compression_sweep")
    print("mode:", args.mode)
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
