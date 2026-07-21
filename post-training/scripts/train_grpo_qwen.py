from __future__ import annotations

import argparse
import os  
from pathlib import Path 
import random 
import sys 
from typing import Any 


import numpy as np  
import torch 

import yaml
from datasets import Dataset, load_dataset 
from transformers import AutoTokenizer 
from trl import GRPOConfig, GRPOTrainer 

ROOT = Path(__file__).resolve().parents[2]  
POST_TRAINING_DIR = ROOT / "post-training"  
ONLINE_DIR = POST_TRAINING_DIR / "online"  

if str(ONLINE_DIR) not in sys.path:  
    sys.path.insert(0, str(ONLINE_DIR)) 

from gsm8k_reward import (  # noqa: E402
    correctness_reward,
    format_reward,
    validate_gsm8k_answers,
) 


SYSTEM_PROMPT = """You are solving a grade-school mathematics problem.

Solve the problem carefully and show your reasoning.

End your response with exactly:
Final answer: \\boxed{number}

The content inside \\boxed{} must contain only the final numeric answer.
Do not include units, currency symbols, words, or explanations inside the box.
"""
 


def parse_args() -> argparse.Namespace:  
    parser = argparse.ArgumentParser( 
       description="Full-parameter GRPO training for Qwen3 on GSM8K."
    )

    parser.add_argument(
         "--config",
        type=Path,
        required=True,
        help="Path to the GRPO YAML configuration.",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint directory from which to resume.",
    ) 

    return parser.parse_args()  

def load_config(path: Path) -> dict[str, Any]:

    if not path.exists():  
        raise FileNotFoundError(f"Configuration file not found: {path}")  

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    return config


def require_keys(config: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]

    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}") 


def validate_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        [
            "model_name",
            "dataset_name",
            "dataset_config",
            "output_dir",
            "train_samples",
            "seed",
            "max_steps",
            "learning_rate",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_generations",
            "max_completion_length",
            "temperature",
            "bf16",
            "gradient_checkpointing",
            "use_vllm",
            "vllm_mode",
            "vllm_gpu_memory_utilization",
            "use_lora",
            "logging_steps",
            "save_steps",
            "report_to",
            "run_name",
        ],
    )

    if config["use_lora"]: 
        raise ValueError(
            "This runner is for full-parameter GRPO. "
            "Set `use_lora: false`."
        )

    if config["max_steps"] <= 0:
        raise ValueError("max_steps must be positive.")

    if config["train_samples"] <= 0:
        raise ValueError("train_samples must be positive.")

    if config["num_generations"] < 2:
        raise ValueError("GRPO requires at least two generations per prompt.")

    effective_batch_size = (
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
    )

    if effective_batch_size % config["num_generations"] != 0:
        raise ValueError(
            "The effective batch size must be divisible by num_generations. "
            f"Received effective_batch_size={effective_batch_size} and "
            f"num_generations={config['num_generations']}."
        )
    
    if config["vllm_mode"] not in {"colocate", "server"}:
        raise ValueError("vllm_mode must be 'colocate' or 'server'.")

    if not 0.0 < config["vllm_gpu_memory_utilization"] < 1.0:
        raise ValueError(
            "vllm_gpu_memory_utilization must be between 0 and 1."
        )

    report_to = config["report_to"]
    uses_wandb = (
        report_to == "wandb"
        or isinstance(report_to, list)
        and "wandb" in report_to
    ) 

    if uses_wandb and not os.environ.get("WANDB_API_KEY"):
        print(
            "WARNING: WANDB_API_KEY is not set. W&B must already be "
            "authenticated on this Pod or logging will fail."
        ) 
    

def set_seed(seed: int) -> None:  

     random.seed(seed)  
     np.random.seed(seed) 
     torch.manual_seed(seed)  

     if torch.cuda.is_available():  
         torch.cuda.manual_seed_all(seed)  


def prepare_example(example: dict[str, Any]) -> dict[str, Any]:  

    return {  

         "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": example["question"],
            },
        ],
        # This remains in the dataset so TRL forwards it to the reward
        # functions as the `answer` argument.
        "answer": example["answer"],      

    }  


def prepare_split(
    split: Dataset, 
    sample_count: int, 
    seed: int,  
) -> Dataset: 
    
    if sample_count > len(split):  
        raise ValueError(
            f"Requested {sample_count} rows, but the split only contains "
            f"{len(split)} rows."
        )
    
    selected = (  
        split
        .shuffle(seed=seed)
        .select(range(sample_count))
    )    

    validate_gsm8k_answers(selected)

    selected = selected.map(
        prepare_example,
        remove_columns=["question"],
        desc="Formatting GSM8K prompts",
    )

    expected_columns = {"prompt", "answer"}
    missing = expected_columns.difference(selected.column_names)

    if missing:
        raise ValueError(
            f"Prepared dataset is missing required columns: {missing}"
        )
    
    return selected   

def load_training_data(
        config: dict[str, Any], 
) ->  tuple[Dataset, Dataset | None]:

   dataset = load_dataset(  
        config["dataset_name"],
        config["dataset_config"],
   )

   if "train" not in dataset:  
       raise ValueError("GSM8K dataset does not contain a train split.")
   
   train_dataset = prepare_split(
       split=dataset["train"], 
       sample_count=config["train_samples"], 
       seed=config["seed"], 
   )

   eval_strategy = config.get("eval_strategy", "no") 
   eval_dataset = None 

   if eval_strategy != "no":  
       if "test" not in dataset:  
           raise ValueError("GSM8K dataset does not contain a test split.") 
   
       eval_dataset = prepare_split(
            split=dataset["test"],
            sample_count=config["eval_samples"],
            seed=config["seed"],
        )

   return train_dataset, eval_dataset  


def build_tokenizer(config: dict[str, Any]):  

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=False,
    )

    tokenizer.padding_side = "left"  
    
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token  

    return tokenizer  


def build_training_args(config: dict[str, Any]) -> GRPOConfig:
    eval_strategy = config.get("eval_strategy", "no")

    arguments: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "seed": config["seed"],
        "data_seed": config["seed"],

        # Explicitly load model weights in BF16 instead of allowing the
        # trainer to load them in FP32.
        "model_init_kwargs": {
            "dtype": "bfloat16" if config["bf16"] else "float32",
        },

        "max_steps": config["max_steps"],
        "learning_rate": config["learning_rate"],
        "warmup_ratio": config["warmup_ratio"],
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "optim": config.get("optim", "adamw_torch_fused"),

        "per_device_train_batch_size":
            config["per_device_train_batch_size"],
        "gradient_accumulation_steps":
            config["gradient_accumulation_steps"],

        "num_generations": config["num_generations"],
        "max_prompt_length": config["max_prompt_length"],
        "max_completion_length": config["max_completion_length"],
        "temperature": config["temperature"],

        # beta=0 means no explicit reference-model KL penalty.
        "beta": config.get("beta", 0.0),

        "bf16": config["bf16"],
        "gradient_checkpointing":
            config["gradient_checkpointing"],
        "use_cache": False,

        "use_vllm": config["use_vllm"],
        "vllm_mode": config["vllm_mode"],
        "vllm_gpu_memory_utilization":
            config["vllm_gpu_memory_utilization"],

        # Keep the answer column so it reaches correctness_reward().
        "remove_unused_columns": False,

        "logging_strategy":
            config.get("logging_strategy", "steps"),
        "logging_steps": config["logging_steps"],
        "logging_first_step":
            config.get("logging_first_step", True),

        "save_strategy": "steps",
        "save_steps": config["save_steps"],
        "save_total_limit":
            config.get("save_total_limit", 3),

        "eval_strategy": eval_strategy,

        "report_to": config["report_to"],
        "run_name": config["run_name"],

        "log_completions":
            config.get("log_completions", True),
        "num_completions_to_print":
            config.get("num_completions_to_print", 2),
    }

    if eval_strategy != "no":
        arguments["eval_steps"] = config["eval_steps"]
        arguments["per_device_eval_batch_size"] = config.get(
            "per_device_eval_batch_size",
            config["per_device_train_batch_size"],
        )
        arguments["num_generations_eval"] = config.get(
            "num_generations_eval",
            config["num_generations"],
        )

    return GRPOConfig(**arguments)


def print_run_summary(
    config: dict[str, Any], 
    train_dataset: Dataset,  
    eval_dataset: Dataset | None, 
) -> None:  
    
    effective_batch_size = ( 
        config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
    )

    print("GRPO run configuration")
    print(f"  model: {config['model_name']}")
    print(f"  full-parameter training: {not config['use_lora']}")
    print(f"  train examples: {len(train_dataset)}")
    print(
        "  eval examples: "
        f"{0 if eval_dataset is None else len(eval_dataset)}"
    )
    print(f"  max steps: {config['max_steps']}")
    print(f"  effective batch size: {effective_batch_size}")
    print(f"  generations per prompt: {config['num_generations']}")
    print(f"  output directory: {config['output_dir']}")
    print(f"  report to: {config['report_to']}")
    print(f"  run name: {config['run_name']}")


def main() -> None:  

    args = parse_args()  
    config = load_config(args.config)  

    validate_config(config)  
    set_seed(config["seed"])  

    train_dataset, eval_dataset = load_training_data(config)
    tokenizer = build_tokenizer(config)
    training_args = build_training_args(config)

    print_run_summary(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer = GRPOTrainer(
        model = config["model_name"], 
        reward_funcs = [
             correctness_reward,
            format_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,

        # No PEFT configuration: all model parameters are trainable.
        peft_config=None,
    )

    trainer.train(
        resume_from_checkpoint= args.resume_from_checkpoint, 
    )

    final_output_dir = Path(config["output_dir"]) / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(final_output_dir)

    print(f"Saved final model to: {final_output_dir}")


if __name__ == "__main__":
    main()

