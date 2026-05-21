import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.pretokenized import load_pretokenized_ids
from data.tokenizer import get_batch
from model.sliding_window import SlidingwindowAttmech
from model.training import count_parameters, estimate_loss


@dataclass
class SlidingWindowRunConfig:
    run_name: str
    window_size: int
    dataset_name: str = "parameter_golf_sp1024"
    max_encoded_tokens: int = 6_400_000
    dim: int = 128
    num_heads: int = 8
    batch_size: int = 16
    eval_batch_size: int = 8
    seq_len: int = 128
    max_steps: int = 12000
    learning_rate: float = 3e-4
    log_interval: int = 10
    eval_iters: int = 50
    seed: int = 1337
    use_rope: bool = True


RUNS = [
    SlidingWindowRunConfig(run_name="sliding_window_32", window_size=32),
    SlidingWindowRunConfig(run_name="sliding_window_64", window_size=64),
    SlidingWindowRunConfig(run_name="sliding_window_128", window_size=128),
]


class SlidingTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        use_rope: bool,
        max_seq_len: int,
    ):
        super().__init__()
        self.window_size = window_size
        self.ln1 = nn.LayerNorm(dim)
        self.attention = SlidingwindowAttmech(
            dim=dim,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        self.ln2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x), window_size=self.window_size)
        x = x + self.feedforward(self.ln2(x))
        return x


class SlidingWindowLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        window_size: int,
        max_seq_len: int,
        use_rope: bool,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, dim)
        self.block1 = SlidingTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        self.block2 = SlidingTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.block1(x)
        x = self.block2(x)
        x = self.ln(x)
        return self.lm_head(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: SlidingWindowRunConfig, output_dir: Path) -> dict:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_dir = output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs.jsonl"
    summary_path = run_dir / "summary.json"

    train_ids, val_ids, vocab_size = load_pretokenized_ids(
        config.dataset_name,
        max_tokens=config.max_encoded_tokens,
    )

    model = SlidingWindowLanguageModel(
        vocab_size=vocab_size,
        dim=config.dim,
        num_heads=config.num_heads,
        window_size=config.window_size,
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

    run_config = {
        "type": "run_config",
        **asdict(config),
        "device": str(device),
        "actual_vocab_size": vocab_size,
        "model_parameters": param_count,
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
                "window_size": config.window_size,
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
                f"window={config.window_size}"
            )

    summary = {
        **asdict(config),
        "actual_vocab_size": vocab_size,
        "model_parameters": param_count,
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
    output_dir = Path("experiment/sliding_window_runs")
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
