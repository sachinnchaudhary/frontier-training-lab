import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.datasets import load_dataset_text
from data.pretokenized import SentencePieceTextTokenizer
from data.pretokenized import get_pretokenized_config
from data.pretokenized import load_pretokenized_ids
from data.tokenizer import get_batch, prepare_text_data_cached
from model.layer import TransformerBlocks
from model.layer import make_norm
from model.positional_encoding import TokenPositionalEmbedding
from optim.muon import Muon


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        max_seq_len: int = 128,
        use_rope: bool = False,
        norm_type: str = "layernorm",
        attention_type: str = "mha",
        num_kv_heads: int | None = None,
        ffn_type: str = "gelu",
        ffn_hidden_dim: int = 512,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        if use_rope:
            self.embedding = nn.Embedding(vocab_size, dim)
        else:
            self.embedding = TokenPositionalEmbedding(
                vocab_size=vocab_size,
                d_model=dim,
                max_seq_len=max_seq_len,
            )

        self.blocks = TransformerBlocks(
            dim=dim,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            norm_type=norm_type,
            num_layers=num_layers,
            attention_type=attention_type,
            num_kv_heads=num_kv_heads,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
        )
        self.ln = make_norm(dim, norm_type)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_lr(step: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    progress = min(step / max_steps, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


OptimizerLike = torch.optim.Optimizer | dict[str, torch.optim.Optimizer]


def iter_optimizers(optimizer: OptimizerLike):
    if isinstance(optimizer, dict):
        return optimizer.values()
    return (optimizer,)


def set_optimizer_lr(optimizer: OptimizerLike, lr: float) -> None:
    for opt in iter_optimizers(optimizer):
        for group in opt.param_groups:
            group["lr"] = lr


def zero_grad(optimizer: OptimizerLike) -> None:
    for opt in iter_optimizers(optimizer):
        opt.zero_grad(set_to_none=True)


def step_optimizer(optimizer: OptimizerLike) -> None:
    for opt in iter_optimizers(optimizer):
        opt.step()


def optimizer_state_dict(optimizer: OptimizerLike):
    if isinstance(optimizer, dict):
        return {
            "type": "optimizer_bundle",
            "optimizers": {name: opt.state_dict() for name, opt in optimizer.items()},
        }
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer: OptimizerLike, state_dict) -> None:
    if isinstance(optimizer, dict):
        if not isinstance(state_dict, dict) or state_dict.get("type") != "optimizer_bundle":
            raise ValueError("checkpoint does not contain optimizer bundle state")
        optimizer_states = state_dict["optimizers"]
        for name, opt in optimizer.items():
            opt.load_state_dict(optimizer_states[name])
    else:
        optimizer.load_state_dict(state_dict)


def get_optimizer_lr(optimizer: OptimizerLike) -> float:
    first_optimizer = next(iter(iter_optimizers(optimizer)))
    return first_optimizer.param_groups[0]["lr"]


def make_optimizer(
    model: nn.Module,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    muon_momentum_beta: float,
    muon_ns_steps: int,
) -> tuple[OptimizerLike, dict]:
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            foreach=False,
            fused=False,
        )
        return optimizer, {
            "optimizer_type": optimizer_type,
            "adamw_parameter_count": count_parameters(model),
            "muon_parameter_count": 0,
        }

    if optimizer_type != "muon":
        raise ValueError(f"unknown optimizer_type: {optimizer_type}")

    muon_params = []
    adamw_params = []
    muon_param_count = 0
    adamw_param_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_muon = (
            param.ndim >= 2
            and "embedding" not in name
            and "lm_head" not in name
        )
        if use_muon:
            muon_params.append(param)
            muon_param_count += param.numel()
        else:
            adamw_params.append(param)
            adamw_param_count += param.numel()

    optimizers: dict[str, torch.optim.Optimizer] = {}
    if muon_params:
        optimizers["muon"] = Muon(
            muon_params,
            lr=lr,
            momentum_beta=muon_momentum_beta,
            weight_decay=weight_decay,
            ns_steps=muon_ns_steps,
        )
    if adamw_params:
        optimizers["adamw"] = torch.optim.AdamW(
            adamw_params,
            lr=lr,
            weight_decay=weight_decay,
            foreach=False,
            fused=False,
        )

    return optimizers, {
        "optimizer_type": optimizer_type,
        "muon_parameter_count": muon_param_count,
        "adamw_parameter_count": adamw_param_count,
        "muon_momentum_beta": muon_momentum_beta,
        "muon_ns_steps": muon_ns_steps,
    }


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: OptimizerLike,
    run_config: dict,
    ema_losses: dict[str, float],
    latest_only: bool = False,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict(optimizer),
        "run_config": run_config,
        "ema_losses": ema_losses,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    if not latest_only:
        step_path = checkpoint_dir / f"step_{step:06d}.pt"
        torch.save(checkpoint, step_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: OptimizerLike,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    load_optimizer_state_dict(optimizer, checkpoint["optimizer_state_dict"])

    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        cuda_rng_state_all = [
            state.detach().cpu().to(dtype=torch.uint8)
            for state in checkpoint["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    step = int(checkpoint["step"])
    ema_losses = checkpoint.get("ema_losses", {})
    return step, dict(ema_losses)


def make_eval_batches(
    split: str,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    eval_iters: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    data = train_ids if split == "train" else val_ids
    if len(data) <= seq_len:
        raise ValueError(f"{split} split too small for T={seq_len}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches = []
    for _ in range(eval_iters):
        ix = torch.randint(
            0,
            len(data) - seq_len,
            (batch_size,),
            generator=generator,
        )
        x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        batches.append((x, y))
    return batches


@torch.no_grad()
def estimate_loss(
    model: TinyLanguageModel,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    eval_iters: int = 50,
    eval_batches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
) -> dict[str, float]:
    model.eval()
    losses = {}

    for split in ("train", "val"):
        split_losses = []
        batches = eval_batches[split] if eval_batches is not None else None
        for i in range(eval_iters):
            if batches is None:
                xb, yb = get_batch(
                    split,
                    train_ids,
                    val_ids,
                    batch_size,
                    seq_len,
                    device=device,
                )
            else:
                xb, yb = batches[i]
            logits = model(xb)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                yb.reshape(-1),
            )
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


@torch.no_grad()
def generate(
    model: TinyLanguageModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
) -> str:
    model.eval()

    ids = tokenizer.encode(prompt)
    if not ids:
        if hasattr(tokenizer, "token_to_id"):
            ids = [tokenizer.token_to_id[tokenizer.unk_token]]
        else:
            ids = [tokenizer.unk_id()]

    token_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = token_ids[:, -model.max_seq_len :]
        logits = model(context)
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_id], dim=1)

    return tokenizer.decode(token_ids[0].tolist())


def chat_loop(model: TinyLanguageModel, tokenizer, device: torch.device):
    print("\nEnter a prompt. Type 'quit' to stop.")

    while True:
        prompt = input("you> ").strip()
        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue

        answer = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=100,
            temperature=0.9,
        )
        print("model>", answer)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1337
    batch_size = 32
    seq_len = 256
    dim = 512
    num_heads = 8
    attention_type = "mha"
    num_kv_heads = 2
    num_layers = 4
    max_steps = 20000
    learning_rate = 3e-4
    min_learning_rate = 3e-5
    weight_decay = 0.01
    optimizer_type = "muon"
    muon_momentum_beta = 0.95
    muon_ns_steps = 5
    log_interval = 10
    eval_iters = 50
    dataset_name = "parameter_golf_sp1024"
    vocab_target_size = 9000
    max_encoded_tokens = 50_000_000
    log_path = "training_logs_muon.jsonl"
    checkpoint_dir = Path("checkpoints_muon")
    checkpoint_interval = 1000
    resume_checkpoint_path = None #"""Path("checkpoints/step_020000.pt")"""
    # resume_checkpoint_path = Path("checkpoints/step_004000.pt")
    use_rope = True
    norm_type = "rmsnorm"
    use_cosine_decay = True
    eval_ema_beta = 0.9

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if dataset_name.startswith("parameter_golf_"):
        train_ids, val_ids, vocab_size = load_pretokenized_ids(
            dataset_name,
            max_tokens=max_encoded_tokens,
        )
        tokenizer = SentencePieceTextTokenizer(
            get_pretokenized_config(dataset_name)["tokenizer_path"]
        )
    else:
        text = load_dataset_text(dataset_name, split="train")
        tokenizer, train_ids, val_ids = prepare_text_data_cached(
            text,
            cache_name=dataset_name,
            vocab_size=vocab_target_size,
            max_encoded_tokens=max_encoded_tokens,
        )
        vocab_size = len(tokenizer.token_to_id)

    model = TinyLanguageModel(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
        use_rope=use_rope,
        norm_type=norm_type,
        attention_type=attention_type,
        num_kv_heads=num_kv_heads,
    ).to(device)

    optimizer, optimizer_metadata = make_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=learning_rate,
        weight_decay=weight_decay,
        muon_momentum_beta=muon_momentum_beta,
        muon_ns_steps=muon_ns_steps,
    )
    param_count = count_parameters(model)

    run_config = {
        "type": "run_config",
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "dim": dim,
        "num_heads": num_heads,
        "attention_type": attention_type,
        "num_kv_heads": num_kv_heads if attention_type == "gqa" else None,
        "num_layers": num_layers,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "weight_decay": weight_decay,
        "use_cosine_decay": use_cosine_decay,
        **optimizer_metadata,
        "norm_type": norm_type,
        "dataset": dataset_name,
        "target_vocab_size": vocab_target_size,
        "actual_vocab_size": vocab_size,
        "max_encoded_tokens": max_encoded_tokens,
        "model_parameters": param_count,
        "num_layers": num_layers,
        "attention_type": attention_type,
        "num_kv_heads": num_kv_heads if attention_type == "gqa" else None,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_interval": checkpoint_interval,
        "resume_checkpoint_path": (
            str(resume_checkpoint_path) if resume_checkpoint_path is not None else None
        ),
        "position_encoding": "rope" if use_rope else "learned_absolute",
    }
    print("run config:", run_config)

    eval_batches = {
        "train": make_eval_batches(
            "train",
            train_ids,
            val_ids,
            batch_size,
            seq_len,
            device,
            eval_iters,
            seed=seed + 10_000,
        ),
        "val": make_eval_batches(
            "val",
            train_ids,
            val_ids,
            batch_size,
            seq_len,
            device,
            eval_iters,
            seed=seed + 20_000,
        ),
    }
    ema_losses: dict[str, float] = {}
    start_step = 0
    if resume_checkpoint_path is not None:
        checkpoint_step, ema_losses = load_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_step = checkpoint_step + 1
        print(f"resumed checkpoint: {resume_checkpoint_path} at step {checkpoint_step}")

    log_file = Path(log_path)
    if resume_checkpoint_path is None or not log_file.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(run_config) + "\n")
    else:
        resume_row = {
            "type": "resume",
            "checkpoint_path": str(resume_checkpoint_path),
            "start_step": start_step,
            "max_steps": max_steps,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(resume_row) + "\n")

    last_log_time = time.time()

    logits = None
    for step in range(start_step, max_steps + 1):
        if use_cosine_decay:
            current_lr = cosine_lr(
                step,
                max_steps,
                learning_rate,
                min_learning_rate,
            )
            set_optimizer_lr(optimizer, current_lr)
        else:
            current_lr = get_optimizer_lr(optimizer)

        xb, yb = get_batch(
            "train",
            train_ids,
            val_ids,
            batch_size,
            seq_len,
            device=device,
        )

        logits = model(xb)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1),
        )

        zero_grad(optimizer)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        step_optimizer(optimizer)

        if step % log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.time()
            elapsed = now - last_log_time
            logged_steps = 1 if step == 0 else log_interval
            tokens_per_sec = batch_size * seq_len * logged_steps / max(elapsed, 1e-8)
            last_log_time = now

            losses = estimate_loss(
                model,
                train_ids,
                val_ids,
                batch_size,
                seq_len,
                vocab_size,
                device,
                eval_iters=eval_iters,
                eval_batches=eval_batches,
            )
            for split, value in losses.items():
                if split not in ema_losses:
                    ema_losses[split] = value
                else:
                    ema_losses[split] = (
                        eval_ema_beta * ema_losses[split]
                        + (1.0 - eval_ema_beta) * value
                    )

            log_row = {
                "type": "train_log",
                "step": step,
                "train_loss": losses["train"],
                "validation_loss": losses["val"],
                "train_loss_ema": ema_losses["train"],
                "validation_loss_ema": ema_losses["val"],
                "learning_rate": current_lr,
                **optimizer_metadata,
                "dataset": dataset_name,
                "gradient_norm": grad_norm.item(),
                "tokens_per_second": tokens_per_sec,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "model_parameters": param_count,
                "num_layers": num_layers,
                "attention_type": attention_type,
                "num_kv_heads": num_kv_heads if attention_type == "gqa" else None,
                "seed": seed,
                "position_encoding": "rope" if use_rope else "learned_absolute",
                "norm_type": norm_type,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

            print(
                f"step={step} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"train_ema={ema_losses['train']:.4f} "
                f"val_ema={ema_losses['val']:.4f} "
                f"lr={current_lr:.2e} "
                f"grad_norm={grad_norm.item():.4f} "
                f"tokens_sec={tokens_per_sec:.0f} "
                f"batch_size={batch_size} "
                f"seq_len={seq_len} "
                f"params={param_count} "
                f"optimizer={optimizer_type} "
                f"layers={num_layers} "
                f"attn={attention_type} "
                f"kv_heads={num_kv_heads if attention_type == 'gqa' else 'na'} "
                f"seed={seed} "
                f"pos={'rope' if use_rope else 'learned_absolute'} "
                f"norm={norm_type}"
            )

        if step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                run_config=run_config,
                ema_losses=ema_losses,
            )
            print(f"checkpoint saved: {checkpoint_dir / f'step_{step:06d}.pt'}")

    if logits is not None:
        print("final logits shape:", logits.shape)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=max_steps,
            model=model,
            optimizer=optimizer,
            run_config=run_config,
            ema_losses=ema_losses,
            latest_only=max_steps % checkpoint_interval == 0,
        )
        print(f"final checkpoint saved: {checkpoint_dir / 'latest.pt'}")
    else:
        print(f"checkpoint step is already >= max_steps={max_steps}; nothing to train")
    if tokenizer is not None:
        chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
