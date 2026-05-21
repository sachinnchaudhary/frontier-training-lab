import json
import random
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pretokenized import load_pretokenized_ids
from data.tokenizer import get_batch
from model.training import (
    TinyLanguageModel,
    count_parameters,
    cosine_lr,
    estimate_loss,
    make_eval_batches,
    set_optimizer_lr,
)


@dataclass
class GQARunConfig:
    run_name: str
    attention_type: str
    num_kv_heads: int | None
    dataset_name: str = "parameter_golf_sp1024"
    max_encoded_tokens: int = 50_000_000
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    batch_size: int = 32
    eval_batch_size: int = 32
    seq_len: int = 256
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    use_cosine_decay: bool = True
    log_interval: int = 10
    eval_iters: int = 50
    seed: int = 1337
    use_rope: bool = True
    norm_type: str = "rmsnorm"


RUNS = [
    GQARunConfig(run_name="mha", attention_type="mha", num_kv_heads=None),
    GQARunConfig(run_name="gqa_kv8", attention_type="gqa", num_kv_heads=8),
    GQARunConfig(run_name="gqa_kv4", attention_type="gqa", num_kv_heads=4),
    GQARunConfig(run_name="gqa_kv2", attention_type="gqa", num_kv_heads=2),
    GQARunConfig(run_name="mqa_kv1", attention_type="gqa", num_kv_heads=1),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_kv_heads(config: GQARunConfig) -> int:
    return config.num_heads if config.attention_type == "mha" else int(config.num_kv_heads)


def metadata(config: GQARunConfig, param_count: int) -> dict:
    head_dim = config.dim // config.num_heads
    kv_heads = effective_kv_heads(config)
    kv_sharing_ratio = config.num_heads / kv_heads
    kv_cache_bytes_per_token = 2 * config.num_layers * kv_heads * head_dim * 2
    return {
        "num_heads": config.num_heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_sharing_ratio": kv_sharing_ratio,
        "model_parameters": param_count,
        "kv_cache_bytes_per_token_fp16": kv_cache_bytes_per_token,
        "kv_cache_mib_at_seq_len_fp16": (
            kv_cache_bytes_per_token * config.seq_len / (1024 * 1024)
        ),
    }


def run_experiment(config: GQARunConfig, output_dir: Path) -> dict:
    if config.dim % config.num_heads != 0:
        raise ValueError("dim must be divisible by num_heads")
    if config.attention_type == "gqa" and config.num_heads % int(config.num_kv_heads) != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_dir = output_dir / config.run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs.jsonl"
    summary_path = run_dir / "summary.json"

    train_ids, val_ids, vocab_size = load_pretokenized_ids(
        config.dataset_name,
        max_tokens=config.max_encoded_tokens,
    )

    model = TinyLanguageModel(
        vocab_size=vocab_size,
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.seq_len,
        use_rope=config.use_rope,
        norm_type=config.norm_type,
        attention_type=config.attention_type,
        num_kv_heads=config.num_kv_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        foreach=False,
        fused=False,
    )
    param_count = count_parameters(model)
    static_metadata = metadata(config, param_count)
    tokens_per_step = config.batch_size * config.seq_len
    total_training_tokens = config.max_steps * tokens_per_step

    run_config = {
        "type": "run_config",
        **asdict(config),
        **static_metadata,
        "device": str(device),
        "actual_vocab_size": vocab_size,
        "training_tokens_target": total_training_tokens,
        "checkpoint_dir": str(checkpoint_dir),
        "position_encoding": "rope" if config.use_rope else "learned_absolute",
    }
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(run_config) + "\n")
    print("run config:", run_config)

    eval_batches = {
        "train": make_eval_batches(
            "train",
            train_ids,
            val_ids,
            config.eval_batch_size,
            config.seq_len,
            device,
            config.eval_iters,
            seed=config.seed + 10_000,
        ),
        "val": make_eval_batches(
            "val",
            train_ids,
            val_ids,
            config.eval_batch_size,
            config.seq_len,
            device,
            config.eval_iters,
            seed=config.seed + 20_000,
        ),
    }

    best_val_loss = float("inf")
    best_step = 0
    best_train_loss = float("inf")
    last_log_time = time.time()

    for step in range(config.max_steps + 1):
        if config.use_cosine_decay:
            current_lr = cosine_lr(
                step,
                max(config.max_steps, 1),
                config.learning_rate,
                config.min_learning_rate,
            )
            set_optimizer_lr(optimizer, current_lr)
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        xb, yb = get_batch(
            "train",
            train_ids,
            val_ids,
            config.batch_size,
            config.seq_len,
            device=device,
        )

        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % config.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.time()
            elapsed = now - last_log_time
            logged_steps = 1 if step == 0 else config.log_interval
            tokens_per_sec = tokens_per_step * logged_steps / max(elapsed, 1e-8)
            last_log_time = now

            losses = estimate_loss(
                model,
                train_ids,
                val_ids,
                config.eval_batch_size,
                config.seq_len,
                vocab_size,
                device,
                eval_iters=config.eval_iters,
                eval_batches=eval_batches,
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_train_loss = losses["train"]
                best_step = step

            log_row = {
                "type": "train_log",
                "step": step,
                "train_loss": losses["train"],
                "validation_loss": losses["val"],
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "learning_rate": current_lr,
                "gradient_norm": grad_norm.item(),
                "tokens_per_second": tokens_per_sec,
                "training_tokens_seen": step * tokens_per_step,
                "attention_type": config.attention_type,
                **static_metadata,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

            print(
                f"{config.run_name} step={step} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"best_val={best_val_loss:.4f}@{best_step} "
                f"lr={current_lr:.2e} "
                f"grad_norm={grad_norm.item():.4f} "
                f"tokens_sec={tokens_per_sec:.0f} "
                f"heads={config.num_heads} "
                f"kv_heads={static_metadata['num_kv_heads']} "
                f"head_dim={static_metadata['head_dim']} "
                f"kv_ratio={static_metadata['kv_sharing_ratio']:.1f} "
                f"params={param_count}"
            )

    summary = {
        **asdict(config),
        **static_metadata,
        "actual_vocab_size": vocab_size,
        "training_tokens_target": total_training_tokens,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "best_train_loss": best_train_loss,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = ArgumentParser(description="Run MHA/GQA/MQA attention sweep.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps for every variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/gqa_runs"),
        help="Directory for logs, summaries, and checkpoints.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=[config.run_name for config in RUNS],
        help="Optional subset of run names to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("experiment/gqa_runs")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = RUNS
    if args.only:
        selected = set(args.only)
        runs = [config for config in runs if config.run_name in selected]
    if args.max_steps is not None:
        runs = [replace(config, max_steps=args.max_steps) for config in runs]

    summaries = []
    for config in runs:
        summaries.append(run_experiment(config, output_dir))

    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(json.dumps(summary) + "\n")

    print("wrote summary:", summary_path)


if __name__ == "__main__":
    main()
