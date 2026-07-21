from __future__ import annotations 

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
 

ROOT = Path(__file__).resolve().parents[2]  
POST_TRAINING_DIR = ROOT / "post-training"
ONLINE_DIR = POST_TRAINING_DIR / "online" 

if str(ONLINE_DIR) not in sys.path:
    sys.path.insert(0, str(ONLINE_DIR))

from gsm8k_reward import (  # noqa: E402
    extract_gsm8k_answer,
    extract_model_answer,
    format_reward,
    validate_gsm8k_answers,
) 


# This must match the system prompt used by train_grpo_qwen.py.
SYSTEM_PROMPT = """You are solving a grade-school mathematics problem.

Solve the problem carefully and show your reasoning.

End your response with exactly:
Final answer: \\boxed{number}

The content inside \\boxed{} must contain only the final numeric answer.
Do not include units, currency symbols, words, or explanations inside the box.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and GRPO-trained Qwen3 on GSM8K."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Create a separate W&B evaluation run.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

def load_model_and_tokenizer(model_path: str): 

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
    )

    tokenizer.padding_size = "left" 

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
 
    model.eval()  
    return model, tokenizer  


def format_chat_prompt(
    tokenizer,
    messages: list[dict[str, str]],   
) -> str:   
    
    try: 
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    
    except TypeError:  
        # Compatibility fallback for tokenizer versions that do not expose
        # enable_thinking as an explicit argument.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    

@torch.infernce_mode()  
def evaluate_model(
    model_path: str,
    examples,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, tokenizer = load_model_and_tokenizer(model_path)

    results: list[dict[str, Any]] = []

    for start in range(0, len(examples), batch_size):
        end = min(start + batch_size, len(examples))
        batch = examples.select(range(start, end))

        formatted_prompts = [
            format_chat_prompt(
                tokenizer,
                create_prompt(question),
            )
            for question in batch["question"]
        ]

        encoded = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )

        encoded = {
            key: value.to(model.device)
            for key, value in encoded.items()
        }

        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        input_width = encoded["input_ids"].shape[1]
        generated_tokens = generated[:, input_width:]

        completions = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        completion_lengths = (
            generated_tokens != tokenizer.pad_token_id
        ).sum(dim=1)

        for offset, completion in enumerate(completions):
            row_index = start + offset
            question = batch["question"][offset]
            reference = batch["answer"][offset]

            gold = extract_gsm8k_answer(reference)
            prediction = extract_model_answer(completion)

            correct = (
                gold is not None
                and prediction is not None
                and gold == prediction
            )

            valid_format = format_reward([completion])[0] > 0.0

            # GSM8K calculator annotations provide a rough proxy for the
            # number of reasoning/calculation steps in the reference answer.
            calculation_steps = reference.count("<<")

            results.append(
                {
                    "index": row_index,
                    "question": question,
                    "reference_answer": reference,
                    "gold_answer": (
                        None if gold is None else str(gold)
                    ),
                    "prediction": (
                        None if prediction is None else str(prediction)
                    ),
                    "completion": completion,
                    "correct": correct,
                    "answer_extracted": prediction is not None,
                    "valid_format": valid_format,
                    "completion_tokens": int(
                        completion_lengths[offset].item()
                    ),
                    "calculation_steps": calculation_steps,
                }
            )

        completed = min(end, len(examples))
        print(
            f"Evaluated {completed}/{len(examples)} "
            f"examples using {model_path}"
        )

    correct_values = np.array(
        [row["correct"] for row in results],
        dtype=np.float64,
    )

    format_values = np.array(
        [row["valid_format"] for row in results],
        dtype=np.float64,
    )

    extraction_values = np.array(
        [row["answer_extracted"] for row in results],
        dtype=np.float64,
    )

    token_values = np.array(
        [row["completion_tokens"] for row in results],
        dtype=np.float64,
    )

    metrics = {
        "model_path": model_path,
        "examples": len(results),
        "accuracy": float(correct_values.mean()),
        "correct": int(correct_values.sum()),
        "format_rate": float(format_values.mean()),
        "extraction_rate": float(extraction_values.mean()),
        "average_completion_tokens": float(token_values.mean()),
    }

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, metrics


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0

    proportion = successes / total
    denominator = 1.0 + z**2 / total

    center = (
        proportion + z**2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def paired_bootstrap_interval(
    baseline_results: list[dict[str, Any]],
    trained_results: list[dict[str, Any]],
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    baseline = np.array(
        [row["correct"] for row in baseline_results],
        dtype=np.float64,
    )

    trained = np.array(
        [row["correct"] for row in trained_results],
        dtype=np.float64,
    )

    if len(baseline) != len(trained):
        raise ValueError("Baseline and trained result lengths differ.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        low=0,
        high=len(baseline),
        size=(samples, len(baseline)),
    )

    differences = (
        trained[indices].mean(axis=1)
        - baseline[indices].mean(axis=1)
    )

    lower, upper = np.percentile(
        differences,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


def save_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def plot_accuracy_comparison(
    baseline_metrics: dict[str, Any],
    trained_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    labels = ["Baseline", "After GRPO"]
    accuracies = [
        baseline_metrics["accuracy"],
        trained_metrics["accuracy"],
    ]

    intervals = [
        wilson_interval(
            baseline_metrics["correct"],
            baseline_metrics["examples"],
        ),
        wilson_interval(
            trained_metrics["correct"],
            trained_metrics["examples"],
        ),
    ]

    lower_errors = [
        accuracy - interval[0]
        for accuracy, interval in zip(accuracies, intervals)
    ]

    upper_errors = [
        interval[1] - accuracy
        for accuracy, interval in zip(accuracies, intervals)
    ]

    figure, axis = plt.subplots(figsize=(7, 5))

    bars = axis.bar(
        labels,
        accuracies,
        color=["#7A8CA5", "#6B5DD3"],
        yerr=[lower_errors, upper_errors],
        capsize=7,
        width=0.6,
    )

    axis.set_ylabel("Exact-match accuracy")
    axis.set_ylim(0.0, min(1.0, max(accuracies) + 0.25))
    axis.set_title("GSM8K accuracy before and after GRPO")
    axis.grid(axis="y", alpha=0.2)

    for bar, accuracy in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{accuracy:.1%}",
            ha="center",
            fontweight="bold",
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_outcome_transitions(
    baseline_results: list[dict[str, Any]],
    trained_results: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, int]:
    transitions = {
        "Wrong → Wrong": 0,
        "Wrong → Correct": 0,
        "Correct → Wrong": 0,
        "Correct → Correct": 0,
    }

    for baseline, trained in zip(
        baseline_results,
        trained_results,
    ):
        before = baseline["correct"]
        after = trained["correct"]

        if not before and not after:
            transitions["Wrong → Wrong"] += 1
        elif not before and after:
            transitions["Wrong → Correct"] += 1
        elif before and not after:
            transitions["Correct → Wrong"] += 1
        else:
            transitions["Correct → Correct"] += 1

    labels = list(transitions)
    counts = list(transitions.values())

    figure, axis = plt.subplots(figsize=(9, 5))

    bars = axis.bar(
        labels,
        counts,
        color=["#9AA4B2", "#35A36F", "#D95C5C", "#6B5DD3"],
    )

    axis.set_ylabel("Number of evaluation problems")
    axis.set_title("Per-problem outcome transitions after GRPO")
    axis.grid(axis="y", alpha=0.2)

    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            fontweight="bold",
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return transitions


def complexity_bucket(calculation_steps: int) -> str:
    if calculation_steps <= 2:
        return "0–2 calculations"

    if calculation_steps <= 4:
        return "3–4 calculations"

    return "5+ calculations"


def plot_accuracy_by_complexity(
    baseline_results: list[dict[str, Any]],
    trained_results: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    bucket_order = [
        "0–2 calculations",
        "3–4 calculations",
        "5+ calculations",
    ]

    summary: dict[str, Any] = {}

    baseline_accuracies = []
    trained_accuracies = []

    for bucket in bucket_order:
        indices = [
            index
            for index, row in enumerate(baseline_results)
            if complexity_bucket(row["calculation_steps"]) == bucket
        ]

        baseline_values = [
            baseline_results[index]["correct"]
            for index in indices
        ]

        trained_values = [
            trained_results[index]["correct"]
            for index in indices
        ]

        baseline_accuracy = (
            float(np.mean(baseline_values))
            if baseline_values
            else 0.0
        )

        trained_accuracy = (
            float(np.mean(trained_values))
            if trained_values
            else 0.0
        )

        baseline_accuracies.append(baseline_accuracy)
        trained_accuracies.append(trained_accuracy)

        summary[bucket] = {
            "examples": len(indices),
            "baseline_accuracy": baseline_accuracy,
            "trained_accuracy": trained_accuracy,
        }

    positions = np.arange(len(bucket_order))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.bar(
        positions - width / 2,
        baseline_accuracies,
        width,
        label="Baseline",
        color="#7A8CA5",
    )

    axis.bar(
        positions + width / 2,
        trained_accuracies,
        width,
        label="After GRPO",
        color="#6B5DD3",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(bucket_order)
    axis.set_ylabel("Exact-match accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Accuracy by reference calculation count")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return summary


def log_to_wandb(
    config: dict[str, Any],
    summary: dict[str, Any],
    plot_paths: list[Path],
) -> None:
    import wandb

    run = wandb.init(
        project=os.environ.get(
            "WANDB_PROJECT",
            "qwen3-1.7b-gsm8k-grpo",
        ),
        entity=os.environ.get("WANDB_ENTITY"),
        name=f"{config['run_name']}-evaluation",
        job_type="evaluation",
        config=config,
    )

    wandb.log(
        {
            "eval/baseline_accuracy":
                summary["baseline"]["accuracy"],
            "eval/trained_accuracy":
                summary["trained"]["accuracy"],
            "eval/absolute_improvement":
                summary["absolute_improvement"],
            "eval/baseline_format_rate":
                summary["baseline"]["format_rate"],
            "eval/trained_format_rate":
                summary["trained"]["format_rate"],
            "eval/baseline_extraction_rate":
                summary["baseline"]["extraction_rate"],
            "eval/trained_extraction_rate":
                summary["trained"]["extraction_rate"],
        }
    )

    wandb.log(
        {
            f"plots/{path.stem}": wandb.Image(str(path))
            for path in plot_paths
        }
    )

    run.finish()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])

    dataset = load_dataset(
        config["dataset_name"],
        config["dataset_config"],
    )

    if "test" not in dataset:
        raise ValueError("GSM8K does not contain a test split.")

    eval_samples = config["eval_samples"]

    if eval_samples > len(dataset["test"]):
        raise ValueError(
            f"Requested {eval_samples} evaluation rows, but GSM8K test "
            f"only contains {len(dataset['test'])} rows."
        )

    examples = (
        dataset["test"]
        .shuffle(seed=config["seed"])
        .select(range(eval_samples))
    )

    validate_gsm8k_answers(examples)

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config.get("eval_batch_size", 16)
    )

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else config["max_completion_length"]
    )

    evaluation_dir = (
        Path(config["output_dir"]) / "evaluation"
    )

    plot_dir = evaluation_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    trained_model_path = (
        Path(config["output_dir"]) / "final"
    )

    if not trained_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {trained_model_path}"
        )

    print("Evaluating baseline model...")

    baseline_results, baseline_metrics = evaluate_model(
        model_path=config["model_name"],
        examples=examples,
        batch_size=batch_size,
        max_prompt_length=config["max_prompt_length"],
        max_new_tokens=max_new_tokens,
    )

    print("Evaluating GRPO-trained model...")

    trained_results, trained_metrics = evaluate_model(
        model_path=str(trained_model_path),
        examples=examples,
        batch_size=batch_size,
        max_prompt_length=config["max_prompt_length"],
        max_new_tokens=max_new_tokens,
    )

    improvement = (
        trained_metrics["accuracy"]
        - baseline_metrics["accuracy"]
    )

    bootstrap_lower, bootstrap_upper = paired_bootstrap_interval(
        baseline_results=baseline_results,
        trained_results=trained_results,
        seed=config["seed"],
    )

    accuracy_plot = plot_dir / "accuracy_comparison.png"
    transition_plot = plot_dir / "outcome_transitions.png"
    complexity_plot = plot_dir / "accuracy_by_complexity.png"

    plot_accuracy_comparison(
        baseline_metrics,
        trained_metrics,
        accuracy_plot,
    )

    transitions = plot_outcome_transitions(
        baseline_results,
        trained_results,
        transition_plot,
    )

    complexity_summary = plot_accuracy_by_complexity(
        baseline_results,
        trained_results,
        complexity_plot,
    )

    summary = {
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "absolute_improvement": improvement,
        "paired_bootstrap_95_percent_interval": {
            "lower": bootstrap_lower,
            "upper": bootstrap_upper,
        },
        "transitions": transitions,
        "accuracy_by_complexity": complexity_summary,
    }

    save_jsonl(
        evaluation_dir / "baseline_predictions.jsonl",
        baseline_results,
    )

    save_jsonl(
        evaluation_dir / "trained_predictions.jsonl",
        trained_results,
    )

    with (
        evaluation_dir / "evaluation_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print()
    print("Evaluation complete")
    print(
        f"  baseline accuracy: "
        f"{baseline_metrics['accuracy']:.2%}"
    )
    print(
        f"  trained accuracy:  "
        f"{trained_metrics['accuracy']:.2%}"
    )
    print(f"  improvement:        {improvement:+.2%}")
    print(
        "  improvement 95% CI: "
        f"[{bootstrap_lower:+.2%}, {bootstrap_upper:+.2%}]"
    )
    print(f"  results: {evaluation_dir}")

    plot_paths = [
        accuracy_plot,
        transition_plot,
        complexity_plot,
    ]

    if args.log_to_wandb:
        log_to_wandb(
            config=config,
            summary=summary,
            plot_paths=plot_paths,
        )


if __name__ == "__main__":
    main()  
