import json
import math
import random
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, replace
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
)
from optim.muon import Muon


@dataclass
class OptimizerRunConfig:
    run_name: str
    optimizer_type: str
    dataset_name: str = "parameter_golf_sp1024"
    max_encoded_tokens: int = 50_000_000
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    attention_type: str = "mha"
    num_kv_heads: int | None = None
    ffn_type: str = "gelu"
    ffn_hidden_dim: int = 512
    batch_size: int = 32
    eval_batch_size: int = 32
    seq_len: int = 256
    max_steps: int = 20_000
    adamw_lr: float = 3e-4
    adamw_min_lr: float = 3e-5
    muon_lr: float = 3e-4
    muon_min_lr: float = 3e-5
    weight_decay: float = 0.01
    muon_momentum_beta: float = 0.95
    muon_ns_steps: int = 5
    use_cosine_decay: bool = True
    log_interval: int = 10
    eval_iters: int = 50
    checkpoint_interval: int = 1000
    seed: int = 1337
    use_rope: bool = True
    norm_type: str = "rmsnorm"


RUNS = [
    OptimizerRunConfig(run_name="adamw", optimizer_type="adamw"),
    OptimizerRunConfig(run_name="muon", optimizer_type="muon"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_use_muon(name: str, param: torch.nn.Parameter) -> bool:
    return (
        param.ndim >= 2
        and "embedding" not in name
        and "lm_head" not in name
    )


def split_named_parameters(
    model: torch.nn.Module,
    optimizer_type: str,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    muon_named_params = []
    adamw_named_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if optimizer_type == "muon" and should_use_muon(name, param):
            muon_named_params.append((name, param))
        else:
            adamw_named_params.append((name, param))
    return muon_named_params, adamw_named_params


def make_optimizer_bundle(
    config: OptimizerRunConfig,
    model: torch.nn.Module,
) -> tuple[dict[str, torch.optim.Optimizer], dict]:
    muon_named_params, adamw_named_params = split_named_parameters(
        model,
        config.optimizer_type,
    )
    optimizers: dict[str, torch.optim.Optimizer] = {}

    if muon_named_params:
        optimizers["muon"] = Muon(
            [param for _, param in muon_named_params],
            lr=config.muon_lr,
            momentum_beta=config.muon_momentum_beta,
            weight_decay=config.weight_decay,
            ns_steps=config.muon_ns_steps,
        )
    if adamw_named_params:
        optimizers["adamw"] = torch.optim.AdamW(
            [param for _, param in adamw_named_params],
            lr=config.adamw_lr,
            weight_decay=config.weight_decay,
            foreach=False,
            fused=False,
        )

    metadata = {
        "muon_param_names": [name for name, _ in muon_named_params],
        "adamw_param_names": [name for name, _ in adamw_named_params],
        "muon_parameter_count": sum(param.numel() for _, param in muon_named_params),
        "adamw_parameter_count": sum(param.numel() for _, param in adamw_named_params),
    }
    return optimizers, metadata


def set_lrs(config: OptimizerRunConfig, optimizers: dict[str, torch.optim.Optimizer], step: int) -> dict:
    if config.use_cosine_decay:
        adamw_lr = cosine_lr(step, max(config.max_steps, 1), config.adamw_lr, config.adamw_min_lr)
        muon_lr = cosine_lr(step, max(config.max_steps, 1), config.muon_lr, config.muon_min_lr)
    else:
        adamw_lr = config.adamw_lr
        muon_lr = config.muon_lr

    if "adamw" in optimizers:
        for group in optimizers["adamw"].param_groups:
            group["lr"] = adamw_lr
    if "muon" in optimizers:
        for group in optimizers["muon"].param_groups:
            group["lr"] = muon_lr

    return {
        "adamw_learning_rate": adamw_lr if "adamw" in optimizers else None,
        "muon_learning_rate": muon_lr if "muon" in optimizers else None,
    }


def zero_grad(optimizers: dict[str, torch.optim.Optimizer]) -> None:
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)


def step_optimizers(optimizers: dict[str, torch.optim.Optimizer]) -> None:
    for optimizer in optimizers.values():
        optimizer.step()


def grad_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    for _, param in named_params:
        if param.grad is not None:
            total += param.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def snapshot_params(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in named_params}


def update_metrics(
    named_params: list[tuple[str, torch.nn.Parameter]],
    before: dict[str, torch.Tensor],
) -> dict:
    update_sq = 0.0
    weight_sq = 0.0
    for name, param in named_params:
        current = param.detach()
        update_sq += (current - before[name]).pow(2).sum().item()
        weight_sq += current.pow(2).sum().item()

    update_norm = math.sqrt(update_sq)
    weight_norm = math.sqrt(weight_sq)
    return {
        "update_norm": update_norm,
        "weight_norm": weight_norm,
        "update_to_weight_ratio": update_norm / max(weight_norm, 1e-12),
    }


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    run_config: dict,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "run_config": run_config,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    torch.save(checkpoint, checkpoint_dir / f"step_{step:06d}.pt")


def run_experiment(config: OptimizerRunConfig, output_dir: Path) -> dict:
    if config.optimizer_type not in {"adamw", "muon"}:
        raise ValueError(f"unknown optimizer_type: {config.optimizer_type}")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_dir = output_dir / config.run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
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
        ffn_type=config.ffn_type,
        ffn_hidden_dim=config.ffn_hidden_dim,
    ).to(device)

    optimizers, optimizer_metadata = make_optimizer_bundle(config, model)
    muon_named_params, adamw_named_params = split_named_parameters(
        model,
        config.optimizer_type,
    )
    all_named_params = muon_named_params + adamw_named_params
    param_count = count_parameters(model)
    tokens_per_step = config.batch_size * config.seq_len

    run_config = {
        "type": "run_config",
        **asdict(config),
        **optimizer_metadata,
        "device": str(device),
        "actual_vocab_size": vocab_size,
        "model_parameters": param_count,
        "training_tokens_target": config.max_steps * tokens_per_step,
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
    best_train_loss = float("inf")
    best_step = 0
    last_log_time = time.time()

    for step in range(config.max_steps + 1):
        lr_metadata = set_lrs(config, optimizers, step)
        should_log = step % config.log_interval == 0
        before_params = snapshot_params(all_named_params) if should_log else None

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

        zero_grad(optimizers)
        loss.backward()
        muon_grad_norm = grad_norm(muon_named_params)
        adamw_grad_norm = grad_norm(adamw_named_params)
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        step_optimizers(optimizers)

        if should_log:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.time()
            elapsed = now - last_log_time
            logged_steps = 1 if step == 0 else config.log_interval
            tokens_per_sec = tokens_per_step * logged_steps / max(elapsed, 1e-8)
            last_log_time = now

            total_update_metrics = update_metrics(all_named_params, before_params)
            muon_update_metrics = (
                update_metrics(muon_named_params, before_params)
                if muon_named_params
                else {"update_norm": 0.0, "weight_norm": 0.0, "update_to_weight_ratio": 0.0}
            )
            adamw_update_metrics = (
                update_metrics(adamw_named_params, before_params)
                if adamw_named_params
                else {"update_norm": 0.0, "weight_norm": 0.0, "update_to_weight_ratio": 0.0}
            )

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
                "tokens_per_second": tokens_per_sec,
                "training_tokens_seen": step * tokens_per_step,
                "gradient_norm": float(total_grad_norm.item()),
                "muon_grad_norm": muon_grad_norm,
                "adamw_grad_norm": adamw_grad_norm,
                "update_norm": total_update_metrics["update_norm"],
                "weight_norm": total_update_metrics["weight_norm"],
                "update_to_weight_ratio": total_update_metrics["update_to_weight_ratio"],
                "muon_update_norm": muon_update_metrics["update_norm"],
                "muon_weight_norm": muon_update_metrics["weight_norm"],
                "muon_update_to_weight_ratio": muon_update_metrics["update_to_weight_ratio"],
                "adamw_update_norm": adamw_update_metrics["update_norm"],
                "adamw_weight_norm": adamw_update_metrics["weight_norm"],
                "adamw_update_to_weight_ratio": adamw_update_metrics["update_to_weight_ratio"],
                "newton_schulz_iterations": (
                    config.muon_ns_steps if config.optimizer_type == "muon" else None
                ),
                **lr_metadata,
                "optimizer_type": config.optimizer_type,
                "model_parameters": param_count,
                "muon_parameter_count": optimizer_metadata["muon_parameter_count"],
                "adamw_parameter_count": optimizer_metadata["adamw_parameter_count"],
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

            print(
                f"{config.run_name} step={step} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"best_val={best_val_loss:.4f}@{best_step} "
                f"grad_norm={float(total_grad_norm.item()):.4f} "
                f"update_ratio={total_update_metrics['update_to_weight_ratio']:.3e} "
                f"tokens_sec={tokens_per_sec:.0f}"
            )

        if step > 0 and step % config.checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir, step, model, optimizers, run_config)
            print(f"checkpoint saved: {checkpoint_dir / f'step_{step:06d}.pt'}")

    summary = {
        **asdict(config),
        **optimizer_metadata,
        "actual_vocab_size": vocab_size,
        "model_parameters": param_count,
        "training_tokens_target": config.max_steps * tokens_per_step,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "best_step": best_step,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model
    del optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def parse_args():
    parser = ArgumentParser(description="Run AdamW vs Muon optimizer comparison.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps for every run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment/optimizer_runs"),
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
