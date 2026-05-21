from __future__ import annotations

import struct
from pathlib import Path

import torch


PARAMETER_GOLF_DATASETS = {
    "parameter_golf_sp1024": {
        "data_dir": Path("data/datasets/fineweb10B_sp1024"),
        "tokenizer_path": Path("data/tokenizers/fineweb_1024_bpe.model"),
        "vocab_size": 1024,
    },
}


def get_pretokenized_config(name: str) -> dict:
    try:
        return PARAMETER_GOLF_DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown pretokenized dataset: {name}") from exc


def load_parameter_golf_bin(path: str | Path) -> torch.Tensor:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(1024)
        magic, version, token_count = struct.unpack("<III", header[:12])
        if magic != 20240520:
            raise ValueError(f"{path} has invalid magic number: {magic}")
        if version != 1:
            raise ValueError(f"{path} has unsupported version: {version}")

        tokens = torch.from_file(
            str(path),
            dtype=torch.int16,
            size=token_count + 512,
        )[512:].to(torch.long)

    return tokens


def load_pretokenized_ids(
    dataset_name: str,
    max_tokens: int,
    train_ratio: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    config = get_pretokenized_config(dataset_name)
    data_dir = config["data_dir"]
    vocab_size = config["vocab_size"]

    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not train_files:
        raise FileNotFoundError(
            f"No train shards found in {data_dir}. "
            "Download them with Parameter Golf cached_challenge_fineweb.py first."
        )

    tokens = []
    total = 0
    for path in train_files:
        shard = load_parameter_golf_bin(path)
        remaining = max_tokens - total
        tokens.append(shard[:remaining])
        total += min(len(shard), remaining)
        if total >= max_tokens:
            break

    ids = torch.cat(tokens)[:max_tokens]
    if len(ids) < 2:
        raise ValueError("Not enough tokens loaded from pretokenized shards")

    split_idx = int(len(ids) * train_ratio)
    train_ids = ids[:split_idx].contiguous()

    if val_files:
        val_ids = load_parameter_golf_bin(val_files[0])[: len(ids) - split_idx]
    else:
        val_ids = ids[split_idx:].contiguous()

    return train_ids, val_ids.contiguous(), vocab_size


class SentencePieceTextTokenizer:
    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "SentencePiece decoding requires: pip install sentencepiece"
            ) from exc

        self.processor = spm.SentencePieceProcessor()
        self.processor.load(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)

    def unk_id(self) -> int:
        return self.processor.unk_id()
