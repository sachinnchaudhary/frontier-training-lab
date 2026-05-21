from __future__ import annotations

from pathlib import Path

from data.tokenizer import load_text


DATA_DIR = Path("data")
HF_CACHE_DIR = DATA_DIR / "hf_cache"


def load_dataset_text(dataset_name: str, split: str = "train") -> str:
    if dataset_name == "synthetic":
        return load_text(str(DATA_DIR / "synthetic_data.txt"))

    if dataset_name == "wikitext2":
        return load_wikitext2(split=split)

    if dataset_name == "fineweb_edu":
        return load_fineweb_edu(split=split)

    raise ValueError(f"unknown dataset: {dataset_name}")


def load_wikitext2(split: str = "train") -> str:
    text_cache_path = DATA_DIR / f"wikitext2_{split}.txt"
    if text_cache_path.exists():
        return load_text(str(text_cache_path))

    try:
        from datasets import DownloadConfig
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "WikiText-2 loading requires the Hugging Face datasets package. "
            "Install it with: pip install datasets"
        ) from exc

    download_config = DownloadConfig(
        local_files_only=True,
        max_retries=1,
    )
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=str(HF_CACHE_DIR),
        download_config=download_config,
    )
    lines = [row["text"] for row in dataset if row["text"].strip()]
    text = "\n".join(lines)
    text_cache_path.write_text(text, encoding="utf-8")
    return text


def load_fineweb_edu(
    split: str = "train",
    sample: str = "sample-10BT",
    max_chars: int = 2_500_000,
) -> str:
    text_cache_path = DATA_DIR / f"fineweb_edu_{sample}_{split}.txt"
    if text_cache_path.exists():
        return load_text(str(text_cache_path))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "FineWeb-Edu loading requires the Hugging Face datasets package. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        sample,
        split=split,
        streaming=True,
    )
    lines = []
    total_chars = 0
    for i, row in enumerate(dataset):
        text = row.get("text", "").strip()
        if text:
            lines.append(text)
            total_chars += len(text)
        if total_chars >= max_chars:
            break

    text = "\n".join(lines)
    text_cache_path.write_text(text, encoding="utf-8")
    return text
