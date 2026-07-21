from __future__ import annotations

from decimal import Decimal
from pathlib import Path  
import sys 

import pytest


POST_TRAINING_DIR = Path(__file__).resolve().parents[1]  
ONLINE_DIR = POST_TRAINING_DIR / "online" 


if str(ONLINE_DIR) not in sys.path: 

    sys.path.insert(0, str(ONLINE_DIR)) 


from gsm8k_reward import (  # noqa: E402 
    completion_to_text,
    correctness_reward,
    extract_gsm8k_answer,
    extract_model_answer,
    format_reward,
    parse_number,
    validate_gsm8k_answers,
)


@pytest.mark.parametrize(
     ("value", "expected"),
    [
        ("42", Decimal("42")),
        ("-12", Decimal("-12")),
        ("1,250", Decimal("1250")),
        ("42.0", Decimal("42.0")),
        ("42.00", Decimal("42.00")),
        ("$1,250.50", Decimal("1250.50")),
        ("  72  ", Decimal("72")),
    ],
)

def test_parse_number_accepts_supported_numbers(value, expected):  
    assert parse_number(value) == expected 


@pytest.mark.parametrize( 
 "value",
    [
        None,
        "",
        "not-a-number",
        "42 dollars",
        "1/2",
        r"\frac{1}{2}",
    ],
)

def test_parse_number_rejects_invalid_values(value):
    assert parse_number(value) is None 


def test_extracts_answer_from_real_gsm8k_format():
    answer = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
        "#### 72"
    )

    assert extract_gsm8k_answer(answer) == Decimal("72")




def test_extracts_comma_separated_gsm8k_answer():
    answer = (
        "The total is 2500 * 4 = <<2500*4=10000>>10,000.\n"
        "#### 10,000"
    )

    assert extract_gsm8k_answer(answer) == Decimal("10000") 


def test_extracts_negative_gsm8k_answer():
    answer = "The final change is negative five.\n#### -5"

    assert extract_gsm8k_answer(answer) == Decimal("-5")


def test_extracts_decimal_gsm8k_answer():
    answer = "The final price is 12.50.\n#### 12.50"

    assert extract_gsm8k_answer(answer) == Decimal("12.50")


def test_rejects_reference_without_gsm8k_marker():
    answer = "The final answer is 72."

    assert extract_gsm8k_answer(answer) is None


def test_does_not_use_intermediate_calculation_as_gold_answer():
    answer = (
        "First calculate 48/2 = <<48/2=24>>24.\n"
        "Then calculate 48+24 = <<48+24=72>>72."
    )

    assert extract_gsm8k_answer(answer) is None


def test_extracts_boxed_model_answer():
    completion = (
        "Half of 48 is 24. Therefore, 48 + 24 = 72.\n"
        r"Final answer: \boxed{72}"
    )

    assert extract_model_answer(completion) == Decimal("72")


def test_extracts_last_boxed_answer():
    completion = (
        r"An early attempt gave \boxed{24}, but that was incomplete. "
        r"Final answer: \boxed{72}"
    )

    assert extract_model_answer(completion) == Decimal("72")


def test_extracts_comma_separated_boxed_answer():
    completion = r"Final answer: \boxed{10,000}"

    assert extract_model_answer(completion) == Decimal("10000")


def test_extracts_decimal_boxed_answer():
    completion = r"Final answer: \boxed{42.00}"

    assert extract_model_answer(completion) == Decimal("42.00")


def test_extracts_negative_boxed_answer():
    completion = r"Final answer: \boxed{-12}"

    assert extract_model_answer(completion) == Decimal("-12")


def test_extracts_answer_from_trl_conversational_completion():
    completion = [
        {
            "role": "assistant",
            "content": (
                "The calculation gives 72.\n"
                r"Final answer: \boxed{72}"
            ),
        }
    ]

    assert extract_model_answer(completion) == Decimal("72")


def test_uses_final_answer_fallback_when_box_is_missing():
    completion = "After calculating everything, Final answer: 72"

    assert extract_model_answer(completion) == Decimal("72")


def test_does_not_extract_arbitrary_number_from_reasoning():
    completion = "I calculated 24 and then 72, but I am unsure."

    assert extract_model_answer(completion) is None


def test_rejects_non_numeric_box_contents():
    completion = r"Final answer: \boxed{42 dollars}"

    assert extract_model_answer(completion) is None


def test_completion_to_text_handles_plain_text():
    assert completion_to_text("hello") == "hello"


def test_completion_to_text_handles_conversational_format():
    completion = [
        {"role": "assistant", "content": "first message"},
        {"role": "assistant", "content": "final message"},
    ]

    assert completion_to_text(completion) == "final message"


def test_completion_to_text_handles_empty_list():
    assert completion_to_text([]) == ""


def test_correctness_reward_for_correct_answer():
    completions = [r"Final answer: \boxed{72}"]
    answers = ["Reasoning goes here.\n#### 72"]

    rewards = correctness_reward(
        completions=completions,
        answer=answers,
    )

    assert rewards == [1.0]


def test_correctness_reward_for_wrong_answer():
    completions = [r"Final answer: \boxed{24}"]
    answers = ["Reasoning goes here.\n#### 72"]

    rewards = correctness_reward(
        completions=completions,
        answer=answers,
    )

    assert rewards == [0.0]


def test_correctness_reward_does_not_use_substring_matching():
    completions = [r"Final answer: \boxed{2}"]
    answers = ["Reasoning goes here.\n#### 42"]

    rewards = correctness_reward(
        completions=completions,
        answer=answers,
    )

    assert rewards == [0.0]


def test_correctness_reward_treats_equivalent_decimals_as_equal():
    completions = [r"Final answer: \boxed{42.00}"]
    answers = ["Reasoning goes here.\n#### 42"]

    rewards = correctness_reward(
        completions=completions,
        answer=answers,
    )

    assert rewards == [1.0]


def test_correctness_reward_handles_multiple_completions():
    completions = [
        r"Final answer: \boxed{72}",
        r"Final answer: \boxed{24}",
        "I could not solve the problem.",
    ]

    answers = [
        "Reasoning.\n#### 72",
        "Reasoning.\n#### 72",
        "Reasoning.\n#### 72",
    ]

    rewards = correctness_reward(
        completions=completions,
        answer=answers,
    )

    assert rewards == [1.0, 0.0, 0.0]


def test_correctness_reward_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        correctness_reward(
            completions=[r"Final answer: \boxed{72}"],
            answer=["Reasoning.\n#### 72", "Reasoning.\n#### 10"],
        )


@pytest.mark.parametrize(
    "completion",
    [
        r"Final answer: \boxed{72}",
        r"Final Answer: \boxed{72}",
        "Some reasoning.\n" r"Final answer: \boxed{1,250}",
        r"Final answer: \boxed{-12}.",
        r"Final answer: \boxed{42.00}",
    ],
)
def test_format_reward_accepts_required_format(completion):
    assert format_reward([completion]) == [0.05]


@pytest.mark.parametrize(
    "completion",
    [
        "The answer is 72.",
        "Final answer: 72",
        r"\boxed{72}",
        r"Final answer: \boxed{72} additional text",
        r"Final answer: \boxed{seventy-two}",
    ],
)
def test_format_reward_rejects_wrong_format(completion):
    assert format_reward([completion]) == [0.0]


def test_format_reward_handles_mixed_batch():
    completions = [
        r"Final answer: \boxed{72}",
        "Final answer: 72",
        r"Final answer: \boxed{24}.",
    ]

    assert format_reward(completions) == [0.05, 0.0, 0.05]


def test_validate_gsm8k_answers_accepts_valid_split():
    dataset_split = {
        "answer": [
            "First solution.\n#### 72",
            "Second solution.\n#### 1,250",
            "Third solution.\n#### -5",
        ]
    }

    validate_gsm8k_answers(dataset_split)


def test_validate_gsm8k_answers_reports_invalid_indices():
    dataset_split = {
        "answer": [
            "Valid solution.\n#### 72",
            "Missing final marker.",
            "Another valid solution.\n#### 10",
            "Malformed marker.\n#### unknown",
        ]
    }

    with pytest.raises(ValueError, match=r"First indices: \[1, 3\]"):
        validate_gsm8k_answers(dataset_split) 

