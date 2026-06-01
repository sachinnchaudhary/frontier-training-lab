from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import jax

from jax_training.data import load_cached_lm_dataset
from jax_training.model import JaxLMConfig
from jax_training.train import TrainConfig, run_training


STATE_DIMS = (32, 64, 128, 256)
CHUNK_SIZES = (8, 16, 32, 64, 128)
GATE_TYPES = ("nogate", "scalar", "vector")

PILOT_DEFAULT_RUNS = (
    "state64_chunk16_nogate",
    "state64_chunk16_scalar",
    "state64_chunk16_vector",
    "state64_chunk64_nogate",
    "state64_chunk64_scalar",
    "state64_chunk64_vector",
    "state128_chunk16_nogate",
    "state128_chunk16_scalar",
    "state128_chunk16_vector",
    "state128_chunk64_nogate",
    "state128_chunk64_scalar",
    "state128_chunk64_vector",
)


def all_run_ids():
    return tuple(
        f"state{state_dim}_chunk{chunk_size}_{gate_type}"
        for state_dim, chunk_size, gate_type in itertools.product(
            STATE_DIMS,
            CHUNK_SIZES,
            GATE_TYPES,
        )
    )


def parse_run_id(run_id):
    parts = run_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"invalid run id: {run_id}")
    state_dim = int(parts[0].removeprefix("state"))
    chunk_size = int(parts[1].removeprefix("chunk"))
    gate_type = parts[2]
    if state_dim not in STATE_DIMS:
        raise ValueError(f"unsupported state_dim in {run_id}")
    if chunk_size not in CHUNK_SIZES:
        raise ValueError(f"unsupported chunk_size in {run_id}")
    if gate_type not in GATE_TYPES:
        raise ValueError(f"unsupported gate_type in {run_id}")
    return state_dim, chunk_size, gate_type


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiment-2: Kimi DeltaNet memory capacity sweep."
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help="Pilot uses cheaper config; full uses larger config.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Run ids. Use 'all' for full grid.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-encoded-tokens", type=int, default=150_000_000)
    return parser.parse_args()


def selected_runs(args):
    if args.runs is None:
        return PILOT_DEFAULT_RUNS if args.mode == "pilot" else ()
    if args.runs == ["all"]:
        return all_run_ids()
    valid = set(all_run_ids())
    unknown = [run_id for run_id in args.runs if run_id not in valid]
    if unknown:
        raise ValueError(f"unknown runs: {unknown}")
    return tuple(args.runs)


def build_train_config(args, run_id):
    if args.mode == "pilot":
        batch_size = 8
        seq_len = 512
        max_steps = 3_000
        eval_batches = 10
        warmup_steps = 300
        model_tag = "512d_4l"
    else:
        batch_size = 8
        seq_len = 512
        max_steps = 30_000
        eval_batches = 20
        warmup_steps = 1_000
        model_tag = "768d_6l"

    run_name = f"kimi_deltanet_{run_id}_muon_{model_tag}_seq{seq_len}"
    log_path = f"experiment/kimi_deltanet_memory_sweep/{args.mode}/{run_id}/summary.jsonl"
    return replace(
        TrainConfig(),
        seed=args.seed,
        experiment_name="kimi_deltanet_memory_sweep",
        latent_variant=run_id,
        run_name=run_name,
        log_path=log_path,
        max_encoded_tokens=args.max_encoded_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        max_steps=max_steps,
        log_interval=10,
        eval_interval=250,
        eval_batches=eval_batches,
        warmup_steps=warmup_steps,
    )


def build_model_config(dataset, train_config, args, run_id):
    state_dim, chunk_size, gate_type = parse_run_id(run_id)
    if args.mode == "pilot":
        model_dim = 512
        num_layers = 4
        num_heads = 8
        expert_hidden_dim = 2048
    else:
        model_dim = 768
        num_layers = 6
        num_heads = 12
        expert_hidden_dim = 3072

    return JaxLMConfig(
        vocab_size=dataset.vocab_size,
        max_seq_len=train_config.seq_len,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=model_dim // num_heads,
        latent_dim=state_dim,
        rope_dim=32,
        attention_type="kimi_deltanet",
        chunk_size=chunk_size,
        deltanet_key_dim=state_dim,
        deltanet_value_dim=state_dim,
        deltanet_gate_type=gate_type,
        index_dim=64,
        index_heads=4,
        num_routed_experts=8,
        num_shared_experts=1,
        top_k=2,
        expert_hidden_dim=expert_hidden_dim,
    )


def main():
    args = parse_args()
    runs = selected_runs(args)
    if not runs:
        raise ValueError("full mode needs explicit --runs or --runs all")

    print("devices:", jax.devices())
    print("backend:", jax.default_backend())
    print("experiment=kimi_deltanet_memory_sweep")
    print("mode:", args.mode)
    print("runs:", ", ".join(runs))

    dataset = load_cached_lm_dataset(
        "parameter_golf_sp1024",
        max_encoded_tokens=args.max_encoded_tokens,
    )

    for run_id in runs:
        train_config = build_train_config(args, run_id)
        model_config = build_model_config(dataset, train_config, args, run_id)
        print(f"starting run={train_config.run_name}")
        run_training(train_config, model_config, dataset)


if __name__ == "__main__":
    main()
