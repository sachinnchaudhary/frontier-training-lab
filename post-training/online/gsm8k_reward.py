from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


# Official GSM8K format:
# final reasoning line
# #### 1,250
GSM8K_GOLD_RE = re.compile(
    r"(?m)^####\s+(-?[\d,]+(?:\.\d+)?)\s*$"
)

BOXED_ANSWER_RE = re.compile(
    r"\\boxed\s*\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}",
    flags=re.IGNORECASE,
)

FINAL_ANSWER_FALLBACK_RE = re.compile(
    r"Final\s+answer\s*:\s*(-?[\d,]+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

FINAL_ANSWER_FORMAT_RE = re.compile(
    r"Final\s+answer\s*:\s*"
    r"\\boxed\s*\{\s*-?[\d,]+(?:\.\d+)?\s*\}"
    r"\s*[.!]?\s*$",
    flags=re.IGNORECASE,
)


def completion_to_text(completion: Any) -> str:
    """Convert a TRL completion into plain assistant text."""

    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        if not completion:
            return ""

        last_message = completion[-1]

        if isinstance(last_message, dict):
            return str(last_message.get("content", ""))

        return str(last_message)

    if isinstance(completion, dict):
        return str(completion.get("content", ""))

    return str(completion)


def parse_number(value: str | None) -> Decimal | None:
    """Normalize a GSM8K-style integer or decimal."""

    if value is None:
        return None

    cleaned = (
        value.strip()
        .replace(",", "")
        .replace("$", "")
        .replace("₹", "")
    )

    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def extract_gsm8k_answer(answer: str) -> Decimal | None:
    """Extract the final number from the official GSM8K answer format."""

    matches = GSM8K_GOLD_RE.findall(answer)

    if not matches:
        return None

    return parse_number(matches[-1])


def extract_model_answer(completion: Any) -> Decimal | None:
    """Extract the final boxed model answer."""

    text = completion_to_text(completion)

    boxed_matches = BOXED_ANSWER_RE.findall(text)
    if boxed_matches:
        return parse_number(boxed_matches[-1])

    fallback_matches = FINAL_ANSWER_FALLBACK_RE.findall(text)
    if fallback_matches:
        return parse_number(fallback_matches[-1])

    return None


def correctness_reward(
    completions: list[Any],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward exact numerical agreement with the GSM8K answer."""

    if len(completions) != len(answer):
        raise ValueError(
            "Completion/reference length mismatch: "
            f"{len(completions)} != {len(answer)}"
        )

    rewards: list[float] = []

    for completion, reference in zip(completions, answer):
        prediction = extract_model_answer(completion)
        target = extract_gsm8k_answer(reference)

        correct = (
            prediction is not None
            and target is not None
            and prediction == target
        )

        rewards.append(1.0 if correct else 0.0)

    return rewards


def format_reward(
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """Small reward for following the required final-answer format."""

    return [
        0.05
        if FINAL_ANSWER_FORMAT_RE.search(completion_to_text(completion))
        else 0.0
        for completion in completions
    ]


def validate_gsm8k_answers(dataset_split: Any) -> None:
    """Fail before training if any dataset answer cannot be parsed."""

    invalid_indices = [
        index
        for index, answer in enumerate(dataset_split["answer"])
        if extract_gsm8k_answer(answer) is None
    ]

    if invalid_indices:
        preview = invalid_indices[:10]
        raise ValueError(
            f"Found {len(invalid_indices)} invalid GSM8K answers. "
            f"First indices: {preview}"
        ) 
    