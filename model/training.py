import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import get_batch, prepare_data
from model.layer import TwoTransformerBlocks
from model.positional_encoding import TokenPositionalEmbedding


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_heads: int = 8,
        max_seq_len: int = 128,
        use_rope: bool = False,
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

        self.blocks = TwoTransformerBlocks(
            dim=dim,
            num_heads=num_heads,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def estimate_loss(
    model: TinyLanguageModel,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    eval_iters: int = 10,
) -> dict[str, float]:
    model.eval()
    losses = {}

    for split in ("train", "val"):
        split_losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(
                split,
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
        ids = [tokenizer.token_to_id[tokenizer.unk_token]]

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
    seq_len = 128
    dim = 128
    num_heads = 8
    max_steps = 2000
    learning_rate = 3e-4
    log_interval = 10
    eval_iters = 10
    vocab_target_size = 9000
    max_encoded_tokens = 10_000
    log_path = "training_logs.jsonl"
    use_rope = True

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer, train_ids, val_ids = prepare_data(
        "data/synthetic_data.txt",
        vocab_size=vocab_target_size,
        max_encoded_tokens=max_encoded_tokens,
    )
    vocab_size = len(tokenizer.token_to_id)

    model = TinyLanguageModel(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        max_seq_len=seq_len,
        use_rope=use_rope,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    param_count = count_parameters(model)

    run_config = {
        "type": "run_config",
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "dim": dim,
        "num_heads": num_heads,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "target_vocab_size": vocab_target_size,
        "actual_vocab_size": vocab_size,
        "max_encoded_tokens": max_encoded_tokens,
        "model_parameters": param_count,
        "position_encoding": "rope" if use_rope else "learned_absolute",
    }
    print("run config:", run_config)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(run_config) + "\n")

    last_log_time = time.time()

    for step in range(max_steps):
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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
        )
        optimizer.step()

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
            )

            lr = optimizer.param_groups[0]["lr"]
            log_row = {
                "type": "train_log",
                "step": step,
                "train_loss": losses["train"],
                "validation_loss": losses["val"],
                "learning_rate": lr,
                "gradient_norm": grad_norm.item(),
                "tokens_per_second": tokens_per_sec,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "model_parameters": param_count,
                "seed": seed,
                "position_encoding": "rope" if use_rope else "learned_absolute",
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

            print(
                f"step={step} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"lr={lr:.2e} "
                f"grad_norm={grad_norm.item():.4f} "
                f"tokens_sec={tokens_per_sec:.0f} "
                f"batch_size={batch_size} "
                f"seq_len={seq_len} "
                f"params={param_count} "
                f"seed={seed} "
                f"pos={'rope' if use_rope else 'learned_absolute'}"
            )

    print("final logits shape:", logits.shape)
    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
