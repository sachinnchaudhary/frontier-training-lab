from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARAMETER_GOLF_DATASETS = {
    "parameter_golf_sp1024": {
        "data_dir": Path("data/datasets/fineweb10B_sp1024"),
        "tokenizer_path": Path("data/tokenizers/fineweb_1024_bpe.model"),
        "vocab_size": 1024,
    },
}


@dataclass(frozen=True)
class JaxDataset:
    train_ids: np.ndarray
    val_ids: np.ndarray
    vocab_size: int
    tokenizer: object | None


def load_parameter_golf_bin(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(1024)
        magic, version, token_count = struct.unpack("<III", header[:12])
        if magic != 20240520:
            raise ValueError(f"{path} has invalid magic number: {magic}")
        if version != 1:
            raise ValueError(f"{path} has unsupported version: {version}")

    return np.fromfile(
        path,
        dtype=np.int16,
        count=token_count,
        offset=1024,
    ).astype(np.int32)


def load_pretokenized_ids(
    dataset_name: str,
    *,
    max_tokens: int,
    train_ratio: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        config = PARAMETER_GOLF_DATASETS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"unknown pretokenized dataset: {dataset_name}") from exc

    data_dir = config["data_dir"]
    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not train_files:
        raise FileNotFoundError(f"No train shards found in {data_dir}")

    chunks = []
    total = 0
    for path in train_files:
        shard = load_parameter_golf_bin(path)
        remaining = max_tokens - total
        chunks.append(shard[:remaining])
        total += min(shard.size, remaining)
        if total >= max_tokens:
            break

    ids = np.concatenate(chunks)[:max_tokens]
    if ids.size < 2:
        raise ValueError("not enough tokens loaded from pretokenized shards")

    split_idx = int(ids.size * train_ratio)
    train_ids = np.ascontiguousarray(ids[:split_idx])

    if val_files:
        val_ids = load_parameter_golf_bin(val_files[0])[: ids.size - split_idx]
    else:
        val_ids = ids[split_idx:]

    return train_ids, np.ascontiguousarray(val_ids), config["vocab_size"]


def load_cached_lm_dataset(
    dataset_name: str,
    *,
    vocab_target_size: int = 9000,
    max_encoded_tokens: int = 50_000_000,
    train_ratio: float = 0.9,
) -> JaxDataset:
    if dataset_name.startswith("parameter_golf_"):
        train_ids, val_ids, vocab_size = load_pretokenized_ids(
            dataset_name,
            max_tokens=max_encoded_tokens,
            train_ratio=train_ratio,
        )
        tokenizer = None
    else:
        from data.datasets import load_dataset_text
        from data.tokenizer import prepare_text_data_cached

        text = load_dataset_text(dataset_name, split="train")
        tokenizer, train_ids, val_ids = prepare_text_data_cached(
            text,
            cache_name=dataset_name,
            vocab_size=vocab_target_size,
            max_encoded_tokens=max_encoded_tokens,
            train_ratio=train_ratio,
        )
        vocab_size = len(tokenizer.token_to_id)

    return JaxDataset(
        train_ids=np.asarray(train_ids, dtype=np.int32),
        val_ids=np.asarray(val_ids, dtype=np.int32),
        vocab_size=vocab_size,
        tokenizer=tokenizer,
    )


def make_lm_batch(
    ids: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int32)
    if ids.ndim != 1:
        raise ValueError("ids must be a flat 1D token array")
    if ids.size <= seq_len:
        raise ValueError(f"token split too small for seq_len={seq_len}")

    starts = rng.integers(0, ids.size - seq_len, size=(batch_size,))
    x = np.stack([ids[i : i + seq_len] for i in starts])
    y = np.stack([ids[i + 1 : i + seq_len + 1] for i in starts])
    return x.astype(np.int32), y.astype(np.int32)


def get_batch(
    split: str,
    dataset: JaxDataset,
    *,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        ids = dataset.train_ids
    elif split == "val":
        ids = dataset.val_ids
    else:
        raise ValueError(f"unknown split: {split}")
    return make_lm_batch(ids, batch_size=batch_size, seq_len=seq_len, rng=rng)
