import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from data.datasets import load_dataset_text
from data.pretokenized import load_pretokenized_ids
from data.tokenizer import get_batch, prepare_text_data_cached
from model.training import TinyLanguageModel, count_parameters, estimate_loss


@dataclass
class ScalingRunConfig:
    run_name: str
    dataset_name: str = "parameter_golf_sp1024"
    max_encoded_tokens: int = 1_000_000
    dim: int = 128
    num_heads: int = 8
    batch_size: int = 16
    eval_batch_size: int = 8
    seq_len: int = 128
    max_steps: int = 1500
    learning_rate: float = 3e-4
    log_interval: int = 10
    eval_iters: int = 50
    vocab_target_size: int = 9000
    tokenizer_train_chars: int = 2_000_000
    seed: int = 1337
    use_rope: bool = True


RUNS = [
    ScalingRunConfig(
        run_name="fineweb_pg_dim128_tokens100k",
        max_encoded_tokens=100_000,
        dim=128,
        num_heads=8,
        batch_size=16,
        eval_batch_size=8,
        max_steps=2000,
        vocab_target_size=9000,
    ),
    ScalingRunConfig(
        run_name="fineweb_pg_dim128_tokens200k",
        max_encoded_tokens=200_000,
        dim=128,
        num_heads=8,
        batch_size=16,
        eval_batch_size=8,
        max_steps=2000,
        vocab_target_size=9000,
    ),
    ScalingRunConfig(
        run_name="fineweb_pg_dim128_tokens400k",
        max_encoded_tokens=400_000,
        dim=128,
        num_heads=8,
        batch_size=16,
        eval_batch_size=8,
        max_steps=2000,
        vocab_target_size=9000,
    ),
    ScalingRunConfig(
        run_name="fineweb_pg_dim128_tokens800k",
        max_encoded_tokens=800_000,
        dim=128,
        num_heads=8,
        batch_size=16,
        eval_batch_size=8,
        max_steps=3000,
        vocab_target_size=9000,
    ),
    ScalingRunConfig(
        run_name="fineweb_pg_dim128_tokens1600k",
        max_encoded_tokens=1_600_000,
        dim=128,
        num_heads=8,
        batch_size=16,
        eval_batch_size=8,
        max_steps=5000,
        vocab_target_size=9000,
    ),
    ScalingRunConfig(
        run_name="fineweb_pg_dim128_tokens3200k",
        max_encoded_tokens=3_200_000,
        dim=128,
        num_heads=8,
        batch_size=16,
        eval_batch_size=8,
        max_steps=8000,
        vocab_target_size=9000,
    ),
    ScalingRunConfig(
        run_name="fineweb_pg_dim128_tokens6400k",
        max_encoded_tokens=6_400_000,
        dim=128,
        num_heads=8,
        batch_size=16,
        eval_batch_size=8,
        max_steps=12000,
        vocab_target_size=9000,
    ),
    ScalingRunConfig(
        run_name="fineweb_pg_dim192_tokens6400k",
        max_encoded_tokens=6_400_000,
        dim=192,
        num_heads=12,
        batch_size=8,
        eval_batch_size=4,
        max_steps=12000,
        vocab_target_size=9000,
    ),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: ScalingRunConfig, output_dir: Path) -> dict:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_dir = output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs.jsonl"
    summary_path = run_dir / "summary.json"

    if config.dataset_name.startswith("parameter_golf_"):
        train_ids, val_ids, vocab_size = load_pretokenized_ids(
            config.dataset_name,
            max_tokens=config.max_encoded_tokens,
        )
    else:
        text = load_dataset_text(config.dataset_name, split="train")
        text = text[: config.tokenizer_train_chars]
        tokenizer, train_ids, val_ids = prepare_text_data_cached(
            text,
            cache_name=f"{config.dataset_name}_chars{config.tokenizer_train_chars}",
            vocab_size=config.vocab_target_size,
            max_encoded_tokens=config.max_encoded_tokens,
        )
        vocab_size = len(tokenizer.token_to_id)

    model = TinyLanguageModel(
        vocab_size=vocab_size,
        dim=config.dim,
        num_heads=config.num_heads,
        max_seq_len=config.seq_len,
        use_rope=config.use_rope,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        foreach=False,
        fused=False,
    )
    param_count = count_parameters(model)
    tokens_per_param = config.max_encoded_tokens / param_count

    run_config = {
        "type": "run_config",
        **asdict(config),
        "device": str(device),
        "actual_vocab_size": vocab_size,
        "model_parameters": param_count,
        "tokens_per_parameter": tokens_per_param,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(run_config) + "\n")
    print("run config:", run_config)

    best_val_loss = float("inf")
    best_step = 0
    best_train_loss = float("inf")
    last_log_time = time.time()

    for step in range(config.max_steps + 1):
        xb, yb = get_batch(
            "train",
            train_ids,
            val_ids,
            config.batch_size,
            config.seq_len,
            device=device,
        )

        logits = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()

        if step % config.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.time()
            elapsed = now - last_log_time
            logged_steps = 1 if step == 0 else config.log_interval
            tokens_per_sec = (
                config.batch_size * config.seq_len * logged_steps / max(elapsed, 1e-8)
            )
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
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": grad_norm.item(),
                "tokens_per_second": tokens_per_sec,
                "training_tokens_seen": step * config.batch_size * config.seq_len,
                "model_parameters": param_count,
                "tokens_per_parameter": tokens_per_param,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

            print(
                f"{config.run_name} step={step} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"best_val={best_val_loss:.4f}@{best_step} "
                f"grad_norm={grad_norm.item():.4f} "
                f"tokens_sec={tokens_per_sec:.0f} "
                f"params={param_count} "
                f"tokens_per_param={tokens_per_param:.4f}"
            )

    summary = {
        **asdict(config),
        "actual_vocab_size": vocab_size,
        "model_parameters": param_count,
        "tokens_per_parameter": tokens_per_param,
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


def main() -> None:
    output_dir = Path("experiment/scaling_laws_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for config in RUNS:
        summaries.append(run_experiment(config, output_dir))

    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(json.dumps(summary) + "\n")

    print("wrote summary:", summary_path)


if __name__ == "__main__":
    main()
