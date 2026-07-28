from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pretokenized import load_pretokenized_ids
from data.tokenizer import get_batch
from model.layer import make_feedforward, make_norm
from model.rope import RoPE


MODE_DEFAULTS = {
    "smoke": {
        "dim": 64,
        "num_layers": 3,
        "num_heads": 4,
        "ffn_hidden_dim": 128,
        "seq_len": 24,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "eval_batch_size": 2,
        "max_steps": 2,
        "target_train_tokens": None,
        "max_encoded_tokens": 10_000,
        "eval_batches": 1,
        "eval_interval": 1,
        "warmup_steps": 1,
    },
    "pilot": {
        "dim": 256,
        "num_layers": 8,
        "num_heads": 4,
        "ffn_hidden_dim": 768,
        "seq_len": 256,
        "batch_size": 8,
        "grad_accum_steps": 2,
        "eval_batch_size": 8,
        "max_steps": None,
        "target_train_tokens": 20_000_000,
        "max_encoded_tokens": 100_000_000,
        "eval_batches": 10,
        "eval_interval": 250,
        "warmup_steps": 250,
    },
    "full": {
        "dim": 384,
        "num_layers": 12,
        "num_heads": 6,
        "ffn_hidden_dim": 1024,
        "seq_len": 512,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "eval_batch_size": 4,
        "max_steps": None,
        "target_train_tokens": 100_000_000,
        "max_encoded_tokens": 100_000_000,
        "eval_batches": 20,
        "eval_interval": 500,
        "warmup_steps": 500,
    },
}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1024
    dim: int = 384
    num_layers: int = 12
    num_heads: int = 6
    ffn_hidden_dim: int = 1024
    ffn_type: str = "swiglu"
    norm_type: str = "rmsnorm"
    max_seq_len: int = 512

    def validate(self) -> None:
        if self.dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("dim, num_layers, and num_heads must be positive")
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if (self.dim // self.num_heads) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")


@dataclass(frozen=True)
class TrainConfig:
    architecture: str
    mode: str
    seed: int
    dataset_name: str
    max_encoded_tokens: int
    batch_size: int
    grad_accum_steps: int
    eval_batch_size: int
    seq_len: int
    max_steps: int
    target_train_tokens: int | None
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    warmup_steps: int
    log_interval: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    max_grad_norm: float
    precision: str
    compile_model: bool
    output_dir: Path
    run_name: str
    resume: Path | None

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum_steps * self.seq_len

    @property
    def training_tokens_target(self) -> int:
        return self.max_steps * self.tokens_per_step


class CausalSelfAttention(nn.Module):
    """Shared sequence mixer for both experiments, backed by PyTorch SDPA."""

    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected hidden size {self.dim}, received {dim}")

        def split_heads(projection: nn.Linear) -> torch.Tensor:
            y = projection(x).view(batch, seq_len, self.num_heads, self.head_dim)
            return y.transpose(1, 2)

        q = self.rope(split_heads(self.q_proj))
        k = self.rope(split_heads(self.k_proj))
        v = split_heads(self.v_proj)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(out)


class TransformerFunctions(nn.Module):
    """Attention and FFN residual branches without a hard-coded residual topology."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = make_norm(config.dim, config.norm_type)
        self.attention = CausalSelfAttention(
            dim=config.dim,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
        )
        self.ffn_norm = make_norm(config.dim, config.norm_type)
        self.feedforward = make_feedforward(
            config.dim,
            config.ffn_hidden_dim,
            config.ffn_type,
        )

    def attention_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attention(self.attn_norm(hidden))

    def ffn_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.feedforward(self.ffn_norm(hidden))


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def mean_diagnostics(rows: list[dict[str, torch.Tensor | float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    output: dict[str, float] = {}
    for key in keys:
        values = []
        for row in rows:
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            values.append(float(value))
        if values:
            output[key] = sum(values) / len(values)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_at_step(config: TrainConfig, step: int) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    decay_steps = max(config.max_steps - config.warmup_steps, 1)
    progress = min((step - config.warmup_steps) / decay_steps, 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        if requested != "fp32" and device.type != "cuda":
            raise ValueError(f"{requested} precision requires CUDA")
        if requested == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support BF16")
        return requested
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def _fixed_eval_batches(
    ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    count: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches = []
    for _ in range(count):
        starts = torch.randint(
            0,
            len(ids) - seq_len - 1,
            (batch_size,),
            generator=generator,
        )
        x = torch.stack([ids[index : index + seq_len] for index in starts])
        y = torch.stack([ids[index + 1 : index + seq_len + 1] for index in starts])
        batches.append((x, y))
    return batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    first_diagnostics: dict[str, float] = {}
    for index, (x_cpu, y_cpu) in enumerate(batches):
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        with autocast_context(device, precision):
            if index == 0:
                logits, diagnostics = model(x, return_diagnostics=True)
                first_diagnostics = {
                    key: float(value)
                    for key, value in diagnostics.items()
                }
            else:
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses), first_diagnostics


def _save_checkpoint(
    path: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_config: dict,
    best_val_loss: float,
    best_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_config": run_config,
            "best_val_loss": best_val_loss,
            "best_step": best_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        path,
    )


def smoke_test_model(model: nn.Module, model_config: ModelConfig) -> None:
    model.train()
    token_ids = torch.randint(0, model_config.vocab_size, (2, min(16, model_config.max_seq_len)))
    targets = torch.randint_like(token_ids, 0, model_config.vocab_size)
    logits, diagnostics = model(token_ids, return_diagnostics=True)
    expected = (*token_ids.shape, model_config.vocab_size)
    if tuple(logits.shape) != expected:
        raise AssertionError(f"expected logits shape {expected}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise AssertionError("model produced non-finite logits")
    loss = F.cross_entropy(logits.reshape(-1, model_config.vocab_size), targets.reshape(-1))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("missing or non-finite gradients")

    # A decoder must not let a changed suffix alter logits for an unchanged
    # prefix. This catches accidental cross-token leakage in new memory paths.
    model.eval()
    test_length = max(4, min(model_config.max_seq_len, 16))
    prefix_length = test_length // 2
    first = torch.randint(0, model_config.vocab_size, (2, test_length))
    second = first.clone()
    second[:, prefix_length:] = torch.randint(
        0, model_config.vocab_size, second[:, prefix_length:].shape
    )
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    torch.testing.assert_close(
        first_logits[:, :prefix_length],
        second_logits[:, :prefix_length],
        rtol=1e-5,
        atol=1e-5,
    )
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "loss": loss.item(),
                "parameters": count_parameters(model),
                "diagnostics": diagnostics,
            },
            indent=2,
        )
    )


def run_training(
    model: nn.Module,
    model_config: ModelConfig,
    train_config: TrainConfig,
    architecture_metadata: dict,
) -> dict:
    if train_config.mode == "smoke":
        smoke_test_model(model, model_config)
        return {"smoke_test": "passed", "model_parameters": count_parameters(model)}

    set_seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(train_config.precision, device)
    model = model.to(device)
    if train_config.compile_model:
        model = torch.compile(model)

    train_ids, val_ids, vocab_size = load_pretokenized_ids(
        train_config.dataset_name,
        max_tokens=train_config.max_encoded_tokens,
    )
    if vocab_size != model_config.vocab_size:
        raise ValueError(
            f"dataset vocab size {vocab_size} does not match model vocab size {model_config.vocab_size}"
        )

    run_dir = train_config.output_dir / train_config.architecture / train_config.run_name
    if run_dir.exists() and train_config.resume is None:
        raise FileExistsError(
            f"run directory already exists: {run_dir}; choose --run-name or use --resume"
        )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs.jsonl"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"

    parameter_count = count_parameters(model)
    run_config = {
        "type": "run_config",
        **asdict(train_config),
        "output_dir": str(train_config.output_dir),
        "resume": str(train_config.resume) if train_config.resume else None,
        "model": asdict(model_config),
        "model_parameters": parameter_count,
        "depth_steps": 2 * model_config.num_layers,
        "precision_resolved": precision,
        "device": str(device),
        "training_tokens_target": train_config.training_tokens_target,
        **architecture_metadata,
    }
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if train_config.resume is None:
        log_path.write_text(json.dumps(run_config) + "\n", encoding="utf-8")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=train_config.weight_decay,
        foreach=False,
        fused=False,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
    )

    start_step = 1
    best_val_loss = float("inf")
    best_step = 0
    if train_config.resume is not None:
        checkpoint = torch.load(train_config.resume, map_location=device)
        previous = checkpoint["run_config"]
        for key in ("architecture", "dataset_name", "seq_len"):
            if previous[key] != run_config[key]:
                raise ValueError(f"resume mismatch for {key}: {previous[key]} != {run_config[key]}")
        if previous["model"] != run_config["model"]:
            raise ValueError("resume model configuration does not match")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_step = int(checkpoint.get("best_step", best_step))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "resume",
                        "checkpoint": str(train_config.resume),
                        "start_step": start_step,
                    }
                )
                + "\n"
            )

    train_eval_batches = _fixed_eval_batches(
        train_ids,
        train_config.eval_batch_size,
        train_config.seq_len,
        train_config.eval_batches,
        train_config.seed + 10_000,
    )
    val_eval_batches = _fixed_eval_batches(
        val_ids,
        train_config.eval_batch_size,
        train_config.seq_len,
        train_config.eval_batches,
        train_config.seed + 20_000,
    )

    model.train()
    train_seconds = 0.0
    interval_loss = 0.0
    interval_steps = 0
    wall_start = time.perf_counter()
    print("run config:", json.dumps(run_config, default=str))

    for step in range(start_step, train_config.max_steps + 1):
        learning_rate = learning_rate_at_step(train_config, step)
        set_optimizer_lr(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_start = time.perf_counter()

        for _ in range(train_config.grad_accum_steps):
            x, y = get_batch(
                "train",
                train_ids,
                val_ids,
                train_config.batch_size,
                train_config.seq_len,
                device=device,
            )
            with autocast_context(device, precision):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, vocab_size),
                    y.reshape(-1),
                )
                scaled_loss = loss / train_config.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            step_loss += loss.detach().item() / train_config.grad_accum_steps

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=train_config.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - step_start
        interval_loss += step_loss
        interval_steps += 1

        if step % train_config.log_interval == 0:
            tokens_seen = step * train_config.tokens_per_step
            log_row = {
                "type": "train_log",
                "step": step,
                "train_loss_microbatch_mean": interval_loss / interval_steps,
                "learning_rate": learning_rate,
                "gradient_norm": float(grad_norm),
                "training_tokens_seen": tokens_seen,
                "training_tokens_per_second": tokens_seen / max(train_seconds, 1e-8),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_row) + "\n")
            print(json.dumps(log_row))
            interval_loss = 0.0
            interval_steps = 0

        should_evaluate = step % train_config.eval_interval == 0 or step == train_config.max_steps
        if should_evaluate:
            train_loss, _ = evaluate(model, train_eval_batches, device, precision)
            val_loss, diagnostics = evaluate(model, val_eval_batches, device, precision)
            eval_row = {
                "type": "eval_log",
                "step": step,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "best_validation_loss": min(best_val_loss, val_loss),
                "training_tokens_seen": step * train_config.tokens_per_step,
                **diagnostics,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_row) + "\n")
            print(json.dumps(eval_row))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                _save_checkpoint(
                    checkpoint_dir / "best.pt",
                    step,
                    model,
                    optimizer,
                    run_config,
                    best_val_loss,
                    best_step,
                )

        if step % train_config.checkpoint_interval == 0 or step == train_config.max_steps:
            _save_checkpoint(
                checkpoint_dir / "latest.pt",
                step,
                model,
                optimizer,
                run_config,
                best_val_loss,
                best_step,
            )

    wall_seconds = time.perf_counter() - wall_start
    summary = {
        **run_config,
        "type": "summary",
        "best_validation_loss": best_val_loss,
        "best_step": best_step,
        "wall_seconds": wall_seconds,
        "training_seconds": train_seconds,
        "training_tokens_per_second": train_config.training_tokens_target
        / max(train_seconds, 1e-8),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=tuple(MODE_DEFAULTS), default="smoke")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dataset", default="parameter_golf_sp1024")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--ffn-hidden-dim", type=int)
    parser.add_argument("--ffn-type", choices=("gelu", "swiglu"), default="swiglu")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--target-train-tokens", type=int)
    parser.add_argument("--max-encoded-tokens", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "att-residual-exp" / "runs")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)


def resolve_configs(
    args: argparse.Namespace,
    architecture: str,
    ffn_hidden_dim_override: int | None = None,
) -> tuple[ModelConfig, TrainConfig]:
    defaults = MODE_DEFAULTS[args.mode]

    def selected(name: str):
        value = getattr(args, name)
        return defaults[name] if value is None else value

    dim = selected("dim")
    num_layers = selected("num_layers")
    num_heads = selected("num_heads")
    reference_ffn_hidden_dim = selected("ffn_hidden_dim")
    ffn_hidden_dim = ffn_hidden_dim_override or reference_ffn_hidden_dim
    seq_len = selected("seq_len")
    batch_size = selected("batch_size")
    grad_accum_steps = selected("grad_accum_steps")
    target_train_tokens = (
        args.target_train_tokens
        if args.target_train_tokens is not None
        else defaults["target_train_tokens"]
    )
    explicit_max_steps = args.max_steps
    if explicit_max_steps is not None and args.target_train_tokens is not None:
        raise ValueError("set only one of --max-steps and --target-train-tokens")
    if explicit_max_steps is not None:
        max_steps = explicit_max_steps
        target_train_tokens = None
    elif target_train_tokens is not None:
        tokens_per_step = batch_size * grad_accum_steps * seq_len
        max_steps = math.ceil(target_train_tokens / tokens_per_step)
    else:
        max_steps = defaults["max_steps"]

    model_config = ModelConfig(
        vocab_size=1024,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_type=args.ffn_type,
        norm_type="rmsnorm",
        max_seq_len=seq_len,
    )
    model_config.validate()

    run_name = args.run_name or (
        f"{args.mode}_d{dim}_l{num_layers}_ff{ffn_hidden_dim}_s{seq_len}_seed{args.seed}"
    )
    train_config = TrainConfig(
        architecture=architecture,
        mode=args.mode,
        seed=args.seed,
        dataset_name=args.dataset,
        max_encoded_tokens=selected("max_encoded_tokens"),
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        eval_batch_size=selected("eval_batch_size"),
        seq_len=seq_len,
        max_steps=max_steps,
        target_train_tokens=target_train_tokens,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=selected("warmup_steps"),
        log_interval=args.log_interval,
        eval_interval=selected("eval_interval"),
        eval_batches=selected("eval_batches"),
        checkpoint_interval=args.checkpoint_interval,
        max_grad_norm=args.max_grad_norm,
        precision=args.precision,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
        run_name=run_name,
        resume=args.resume,
    )
    return model_config, train_config


def closest_ffn_hidden_dim(
    target_parameters: int,
    model_factory: Callable[[int], nn.Module],
    reference_hidden_dim: int,
    search_radius: int = 256,
) -> tuple[int, int]:
    """Find a nearby integer FFN width with the closest total parameter count."""
    best_hidden = reference_hidden_dim
    best_count = count_parameters(model_factory(best_hidden))
    best_error = abs(best_count - target_parameters)
    lower = max(1, reference_hidden_dim - search_radius)
    upper = reference_hidden_dim + search_radius
    # Parameter count is affine in the FFN width, so checking every width is cheap
    # algebraically but repeatedly constructing models is not. Estimate the slope
    # from two models and inspect only the nearest candidates.
    next_count = count_parameters(model_factory(reference_hidden_dim + 1))
    slope = next_count - best_count
    if slope <= 0:
        raise RuntimeError("could not infer a positive FFN parameter slope")
    estimate = round(reference_hidden_dim + (target_parameters - best_count) / slope)
    for hidden in range(max(lower, estimate - 2), min(upper, estimate + 2) + 1):
        count = count_parameters(model_factory(hidden))
        error = abs(count - target_parameters)
        if error < best_error:
            best_hidden, best_count, best_error = hidden, count, error
    return best_hidden, best_count
