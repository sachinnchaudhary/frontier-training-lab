"""Common-manifest evaluation for AttnRes and depth-reader experiments.

Run from the repository root after the paired 100M-token runs finish:

    python -u att-residual-exp/evaluate_checkpoints.py --resume
    python -u att-residual-exp/evaluate_checkpoints.py --comparison reader --resume

The training harness repeatedly evaluated the first 10M validation tokens.
The original AttnRes comparison uses [10M, 20M). The later reader ablation
uses a separately frozen [20M, 30M) slice that was not inspected by the prior
common evaluation. Every checkpoint in one comparison sees the exact same
blocks in the exact same order.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t

from _common import ModelConfig, autocast_context, count_parameters, resolve_precision
from associative_read_depth_kda import AssociativeReadDepthKDALM
from attention_residual_baseline import FullAttentionResidualLM
from softmax_read_depth_kda import SoftmaxReadGatedDeltaDepthMemoryLM


ROOT = Path(__file__).resolve().parents[1]

BASELINE_ARCHITECTURE = "full_attention_residual"
PROPOSAL_ARCHITECTURE = "softmax_read_gated_delta_depth_memory"
ASSOCIATIVE_ARCHITECTURE = "associative_read_depth_kda"
DEFAULT_ARCHITECTURES = (BASELINE_ARCHITECTURE, PROPOSAL_ARCHITECTURE)
DEFAULT_SEEDS = (1337, 2027, 3407)
COMPARISON_PLANS = {
    "attnres": {
        "control": BASELINE_ARCHITECTURE,
        "candidate": PROPOSAL_ARCHITECTURE,
        "output_dir": "common_eval_4m_seed424242",
        "memory_off_intervention": False,
        "pool_start_token": 10_000_000,
        "expected_pool_sha256": (
            "503a67ddea82a04bebf57cdfb3ce88dd002693e134eac745553f0190b15fea34"
        ),
        "expected_block_ids_sha256": (
            "6084f8386ab45dc508b3b9405e98b3b45dadd79c3bdc258816521ec3f059d22c"
        ),
        "pool_status": "not used by the training harness validation sampler",
    },
    "reader": {
        "control": PROPOSAL_ARCHITECTURE,
        "candidate": ASSOCIATIVE_ARCHITECTURE,
        "output_dir": "common_eval_reader_4m_seed424242",
        "memory_off_intervention": True,
        "pool_start_token": 20_000_000,
        "expected_pool_sha256": (
            "6295b2db17d16427efbeb6355ffcdbac17a84d7289cd886c3c17227cce39cea6"
        ),
        "expected_block_ids_sha256": (
            "61a2e237a7f43fef93dc630d6f5f866f5632baf2f557d67cb3c834d8a8fca3f7"
        ),
        "pool_status": (
            "not used by training or the prior AttnRes common evaluation"
        ),
    },
}

DATASET_NAME = "parameter_golf_sp1024"
VALIDATION_SHARD = ROOT / "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
TOKENIZER_FILE = ROOT / "data/tokenizers/fineweb_1024_bpe.model"
EXPECTED_SHARD_SHA256 = "4860ad80c0516a150b6917ee9ed17be5d27a0726a0b44ba9fe28c323b527c86f"
EXPECTED_TOKENIZER_SHA256 = "4f5e8adb109c66b4886963bc75a7befd73bda36d27fd7102df8e9e66503b0e2a"

POOL_START_TOKEN = 10_000_000
POOL_TOKEN_COUNT = 10_000_000
EXPECTED_POOL_SHA256 = "503a67ddea82a04bebf57cdfb3ce88dd002693e134eac745553f0190b15fea34"
SEQUENCE_LENGTH = 512
BLOCK_TARGET_TOKENS = 16_384
DEFAULT_SELECTED_BLOCKS = 256
DEFAULT_EVAL_SEED = 424_242
SELECTION_PREFIX = bytes.fromhex(
    "6174747265732d636f6d6d6f6e2d6576616c2d763100"
)
EXPECTED_DEFAULT_BLOCK_IDS_SHA256 = (
    "6084f8386ab45dc508b3b9405e98b3b45dadd79c3bdc258816521ec3f059d22c"
)
POOL_STATUS = "not used by the training harness validation sampler"

MANIFEST_SCHEMA = "attres-common-eval-v1"
RESULT_SCHEMA = "attres-common-eval-result-v1"
COMPARISON_SCHEMA = "attres-common-eval-comparison-v1"


def configure_validation_pool(plan: Mapping[str, Any]) -> None:
    """Select the frozen validation slice belonging to one comparison plan."""
    global POOL_START_TOKEN
    global EXPECTED_POOL_SHA256
    global EXPECTED_DEFAULT_BLOCK_IDS_SHA256
    global POOL_STATUS

    POOL_START_TOKEN = int(plan["pool_start_token"])
    EXPECTED_POOL_SHA256 = str(plan["expected_pool_sha256"])
    EXPECTED_DEFAULT_BLOCK_IDS_SHA256 = str(
        plan["expected_block_ids_sha256"]
    )
    POOL_STATUS = str(plan["pool_status"])


@dataclass(frozen=True)
class CheckpointSpec:
    architecture: str
    seed: int
    run_name: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    run_config: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path, *, offset: int = 0, length: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining is None or remaining > 0:
            read_size = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            chunk = handle.read(read_size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in (None, 0):
        raise ValueError(f"{path} ended before the requested hash range")
    return digest.hexdigest()


def read_parameter_golf_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(1024)
    if len(header) != 1024:
        raise ValueError(f"{path} has a truncated Parameter Golf header")
    magic, version, token_count = struct.unpack("<III", header[:12])
    if magic != 20240520:
        raise ValueError(f"{path} has invalid magic number {magic}")
    if version != 1:
        raise ValueError(f"{path} has unsupported version {version}")
    expected_size = 1024 + 2 * token_count
    actual_size = path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(
            f"{path} is truncated: expected at least {expected_size} bytes, got {actual_size}"
        )
    return {"magic": magic, "version": version, "token_count": token_count}


def map_parameter_golf_tokens(path: Path, token_count: int) -> torch.Tensor:
    mapped = torch.from_file(
        str(path.resolve()),
        shared=False,
        size=token_count + 512,
        dtype=torch.int16,
    )
    return mapped[512:]


def select_block_ids(
    candidate_count: int,
    selected_count: int,
    seed: int,
    pool_sha256: str,
) -> list[int]:
    if not 0 < selected_count <= candidate_count:
        raise ValueError(
            f"selected block count must be in [1, {candidate_count}], got {selected_count}"
        )
    seed_bytes = struct.pack("<Q", seed)
    pool_bytes = bytes.fromhex(pool_sha256)

    def selection_digest(block_id: int) -> bytes:
        message = SELECTION_PREFIX + seed_bytes + pool_bytes + struct.pack("<Q", block_id)
        return hashlib.sha256(message).digest()

    ranked = sorted(range(candidate_count), key=selection_digest)
    return sorted(ranked[:selected_count])


def block_ids_sha256(block_ids: Sequence[int]) -> str:
    packed = b"".join(struct.pack("<I", block_id) for block_id in block_ids)
    return hashlib.sha256(packed).hexdigest()


def build_manifest(
    *,
    shard_path: Path,
    tokenizer_path: Path,
    eval_seed: int,
    selected_block_count: int,
    batch_size: int,
) -> dict[str, Any]:
    if BLOCK_TARGET_TOKENS % SEQUENCE_LENGTH != 0:
        raise AssertionError("block target count must be divisible by sequence length")
    sequences_per_block = BLOCK_TARGET_TOKENS // SEQUENCE_LENGTH
    if sequences_per_block % batch_size != 0:
        raise ValueError(
            f"batch size {batch_size} must divide {sequences_per_block} sequences per block"
        )
    if not shard_path.is_file():
        raise FileNotFoundError(f"validation shard not found: {shard_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    header = read_parameter_golf_header(shard_path)
    pool_end = POOL_START_TOKEN + POOL_TOKEN_COUNT
    if pool_end > header["token_count"]:
        raise ValueError(
            f"validation pool ends at {pool_end:,}, beyond shard length {header['token_count']:,}"
        )

    shard_sha256 = sha256_file(shard_path)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    pool_sha256 = sha256_file(
        shard_path,
        offset=1024 + 2 * POOL_START_TOKEN,
        length=2 * POOL_TOKEN_COUNT,
    )
    expected_hashes = {
        "validation shard": (shard_sha256, EXPECTED_SHARD_SHA256),
        "validation pool": (pool_sha256, EXPECTED_POOL_SHA256),
        "tokenizer": (tokenizer_sha256, EXPECTED_TOKENIZER_SHA256),
    }
    for label, (actual, expected) in expected_hashes.items():
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")

    candidate_count = (POOL_TOKEN_COUNT - 1) // BLOCK_TARGET_TOKENS
    selected_block_ids = select_block_ids(
        candidate_count,
        selected_block_count,
        eval_seed,
        pool_sha256,
    )
    ids_sha256 = block_ids_sha256(selected_block_ids)
    if (
        eval_seed == DEFAULT_EVAL_SEED
        and selected_block_count == DEFAULT_SELECTED_BLOCKS
        and ids_sha256 != EXPECTED_DEFAULT_BLOCK_IDS_SHA256
    ):
        raise AssertionError(
            "default block selection changed: "
            f"expected {EXPECTED_DEFAULT_BLOCK_IDS_SHA256}, got {ids_sha256}"
        )

    core = {
        "schema": MANIFEST_SCHEMA,
        "dataset": {
            "name": DATASET_NAME,
            "split": "validation",
            "shard_name": shard_path.name,
            "shard_size_bytes": shard_path.stat().st_size,
            "shard_sha256": shard_sha256,
            "header": header,
            "tokenizer_name": tokenizer_path.name,
            "tokenizer_sha256": tokenizer_sha256,
        },
        "pool": {
            "start_token_in_shard": POOL_START_TOKEN,
            "token_count": POOL_TOKEN_COUNT,
            "end_token_exclusive": pool_end,
            "raw_uint16_le_sha256": pool_sha256,
            "status_before_this_evaluation": POOL_STATUS,
        },
        "sampling": {
            "algorithm": "sha256_rank_without_replacement",
            "selection_prefix_hex": SELECTION_PREFIX.hex(),
            "eval_seed": eval_seed,
            "candidate_block_count": candidate_count,
            "selected_block_count": selected_block_count,
            "selected_block_ids": selected_block_ids,
            "selected_block_ids_uint32_le_sha256": ids_sha256,
            "block_target_tokens": BLOCK_TARGET_TOKENS,
            "sequence_length": SEQUENCE_LENGTH,
            "sequences_per_block": sequences_per_block,
            "batch_size": batch_size,
            "batches_per_block": sequences_per_block // batch_size,
            "total_target_tokens": selected_block_count * BLOCK_TARGET_TOKENS,
        },
    }
    manifest = {**core, "manifest_sha256": object_sha256(core), "created_utc": utc_now()}
    return manifest


def manifest_core(manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "dataset",
        "pool",
        "sampling",
        "manifest_sha256",
        "created_utc",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            f"manifest fields changed: expected {sorted(expected_keys)}, got {sorted(manifest)}"
        )
    return {
        "schema": manifest["schema"],
        "dataset": manifest["dataset"],
        "pool": manifest["pool"],
        "sampling": manifest["sampling"],
    }


def validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    core = manifest_core(manifest)
    expected_manifest_hash = object_sha256(core)
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError(
            "manifest content does not match its SHA-256: "
            f"expected {expected_manifest_hash}, got {manifest.get('manifest_sha256')}"
        )
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema: {manifest.get('schema')!r}")
    dataset = manifest.get("dataset")
    pool = manifest.get("pool")
    sampling = manifest.get("sampling")
    if not all(isinstance(value, dict) for value in (dataset, pool, sampling)):
        raise TypeError("manifest dataset, pool, and sampling fields must be objects")
    if dataset.get("shard_sha256") != EXPECTED_SHARD_SHA256:
        raise ValueError("manifest validation-shard hash is not the frozen experiment hash")
    if dataset.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("manifest tokenizer hash is not the frozen experiment hash")
    if pool.get("raw_uint16_le_sha256") != EXPECTED_POOL_SHA256:
        raise ValueError("manifest validation-pool hash is not the frozen experiment hash")
    block_ids = sampling.get("selected_block_ids")
    if not isinstance(block_ids, list) or not all(isinstance(value, int) for value in block_ids):
        raise TypeError("manifest selected_block_ids must be a list of integers")
    if block_ids != sorted(set(block_ids)):
        raise ValueError("manifest selected block IDs must be unique and sorted")
    if len(block_ids) != int(sampling.get("selected_block_count", -1)):
        raise ValueError("manifest selected block count does not match its block-ID list")
    if block_ids_sha256(block_ids) != sampling.get("selected_block_ids_uint32_le_sha256"):
        raise ValueError("manifest selected block-ID list does not match its SHA-256")
    expected_tokens = len(block_ids) * BLOCK_TARGET_TOKENS
    if int(sampling.get("total_target_tokens", -1)) != expected_tokens:
        raise ValueError("manifest total target-token count is inconsistent")


def source_code_identity() -> dict[str, Any]:
    source_paths = (
        Path(__file__).resolve(),
        ROOT / "att-residual-exp/_common.py",
        ROOT / "att-residual-exp/attention_residual_baseline.py",
        ROOT / "att-residual-exp/softmax_read_depth_kda.py",
        ROOT / "att-residual-exp/associative_read_depth_kda.py",
        ROOT / "model/layer.py",
        ROOT / "model/rope.py",
        ROOT / "data/pretokenized.py",
    )
    files: dict[str, str] = {}
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"evaluation source dependency not found: {path}")
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return {"files": files, "combined_sha256": object_sha256(files)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def discover_checkpoints(
    *,
    runs_dir: Path,
    architectures: Sequence[str],
    seeds: Sequence[int],
    run_pattern: str,
    checkpoint_name: str,
    expected_step: int,
) -> list[CheckpointSpec]:
    expected_pairs = {(architecture, seed) for architecture in architectures for seed in seeds}
    found: dict[tuple[str, int], CheckpointSpec] = {}
    for architecture in architectures:
        architecture_dir = runs_dir / architecture
        if not architecture_dir.is_dir():
            raise FileNotFoundError(f"architecture run directory not found: {architecture_dir}")
        for run_dir in sorted(architecture_dir.glob(run_pattern)):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "config.json"
            if not config_path.is_file():
                continue
            run_config = load_json(config_path)
            seed = int(run_config.get("seed", -1))
            pair = (architecture, seed)
            if pair not in expected_pairs:
                continue
            if pair in found:
                raise ValueError(
                    f"duplicate run for architecture={architecture}, seed={seed}: "
                    f"{found[pair].run_dir} and {run_dir}"
                )
            if run_config.get("architecture") != architecture:
                raise ValueError(f"architecture mismatch in {config_path}")
            if run_config.get("run_name") != run_dir.name:
                raise ValueError(f"run-name mismatch in {config_path}")
            if int(run_config.get("max_steps", -1)) != expected_step:
                raise ValueError(
                    f"expected max_steps={expected_step} in {config_path}, "
                    f"got {run_config.get('max_steps')}"
                )
            checkpoint_path = run_dir / "checkpoints" / checkpoint_name
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            found[pair] = CheckpointSpec(
                architecture=architecture,
                seed=seed,
                run_name=run_dir.name,
                run_dir=run_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                run_config=run_config,
            )

    missing = sorted(expected_pairs - set(found))
    if missing:
        raise FileNotFoundError(f"missing checkpoint pairs: {missing}")
    if set(found) != expected_pairs:
        raise AssertionError("checkpoint discovery returned an unexpected matrix")
    specs = [found[pair] for pair in sorted(found)]
    validate_run_matrix(specs, architectures, seeds, expected_step)
    return specs


def validate_run_matrix(
    specs: Sequence[CheckpointSpec],
    architectures: Sequence[str],
    seeds: Sequence[int],
    expected_step: int,
) -> None:
    shared_training_keys = (
        "dataset_name",
        "max_encoded_tokens",
        "batch_size",
        "grad_accum_steps",
        "seq_len",
        "max_steps",
        "target_train_tokens",
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_grad_norm",
        "precision_resolved",
        "compile_model",
        "training_tokens_target",
    )
    reference = specs[0].run_config
    reference_model = reference.get("model")
    if not isinstance(reference_model, dict):
        raise ValueError(f"model config is missing from {specs[0].config_path}")
    shared_model_keys = (
        "vocab_size",
        "dim",
        "num_layers",
        "num_heads",
        "ffn_type",
        "norm_type",
        "max_seq_len",
    )
    for spec in specs:
        config = spec.run_config
        if config.get("dataset_name") != DATASET_NAME:
            raise ValueError(f"unexpected dataset in {spec.config_path}")
        if int(config.get("seq_len", -1)) != SEQUENCE_LENGTH:
            raise ValueError(f"expected seq_len={SEQUENCE_LENGTH} in {spec.config_path}")
        model = config.get("model")
        if not isinstance(model, dict) or int(model.get("max_seq_len", -1)) != SEQUENCE_LENGTH:
            raise ValueError(f"model max_seq_len mismatch in {spec.config_path}")
        for key in shared_model_keys:
            if model.get(key) != reference_model.get(key):
                raise ValueError(
                    f"unmatched model setting {key}: {spec.config_path} has "
                    f"{model.get(key)!r}, expected {reference_model.get(key)!r}"
                )
        if int(config.get("max_steps", -1)) != expected_step:
            raise ValueError(f"checkpoint-step plan mismatch in {spec.config_path}")
        for key in shared_training_keys:
            if config.get(key) != reference.get(key):
                raise ValueError(
                    f"unmatched training setting {key}: {spec.config_path} has "
                    f"{config.get(key)!r}, expected {reference.get(key)!r}"
                )

    for seed in seeds:
        paired = [spec for spec in specs if spec.seed == seed]
        if {spec.architecture for spec in paired} != set(architectures):
            raise ValueError(f"seed {seed} does not have one checkpoint per architecture")


def validate_reader_ablation_matrix(
    specs: Sequence[CheckpointSpec],
    seeds: Sequence[int],
) -> None:
    """Fail closed unless the reader is the intended causal difference."""
    by_pair = {(spec.architecture, spec.seed): spec.run_config for spec in specs}
    shared_writer_keys = (
        "num_slots",
        "memory_dim",
        "alpha_bias",
        "beta_bias",
        "gamma_init",
        "depth_memory_granularity",
        "depth_memory_state_scope",
        "depth_memory_write",
    )
    for seed in seeds:
        softmax = by_pair[(PROPOSAL_ARCHITECTURE, seed)]
        associative = by_pair[(ASSOCIATIVE_ARCHITECTURE, seed)]
        softmax_model = softmax.get("model")
        associative_model = associative.get("model")
        if not isinstance(softmax_model, dict) or not isinstance(associative_model, dict):
            raise ValueError(f"seed {seed} is missing a model configuration")
        if softmax_model != associative_model:
            raise ValueError(
                f"reader ablation seed {seed} has unequal backbone/model configurations"
            )
        for key in shared_writer_keys:
            if softmax.get(key) != associative.get(key):
                raise ValueError(
                    f"reader ablation seed {seed} has unequal writer setting {key}: "
                    f"{softmax.get(key)!r} != {associative.get(key)!r}"
                )
        if int(softmax.get("read_key_dim", -1)) != int(
            associative.get("softmax_control_read_key_dim", -2)
        ) or int(softmax.get("read_value_dim", -1)) != int(
            associative.get("softmax_control_read_value_dim", -2)
        ):
            raise ValueError("associative run was initialized from the wrong softmax reader")
        if associative.get("parameter_matching") != "softmax_reader_control_backbone":
            raise ValueError(
                "primary reader ablation must pin the softmax control backbone width"
            )
        if associative.get("shared_initialization_from_softmax_control") is not True:
            raise ValueError(
                "associative checkpoint did not use paired shared-parameter initialization"
            )
        if int(associative.get("softmax_control_ffn_hidden_dim", -1)) != int(
            associative_model.get("ffn_hidden_dim", -2)
        ):
            raise ValueError("associative checkpoint did not retain the control FFN width")
        if int(associative.get("model_parameters", -1)) >= int(
            softmax.get("model_parameters", -1)
        ):
            raise ValueError("associative reader was expected to use fewer parameters")
        if int(associative.get("softmax_control_parameters", -1)) != int(
            softmax.get("model_parameters", -2)
        ):
            raise ValueError("associative run recorded the wrong softmax control size")


def normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state_dict)
    if not keys:
        raise ValueError("checkpoint model_state_dict is empty")
    prefixed = [key.startswith("_orig_mod.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise ValueError("checkpoint has a mixture of compiled and uncompiled state-dict keys")
    if all(prefixed):
        return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return dict(state_dict)


def construct_model(run_config: Mapping[str, Any]) -> torch.nn.Module:
    architecture = run_config.get("architecture")
    model_data = run_config.get("model")
    if not isinstance(model_data, dict):
        raise ValueError("checkpoint run_config.model must be an object")
    model_config = ModelConfig(**model_data)
    if model_config.max_seq_len != int(run_config.get("seq_len", -1)):
        raise ValueError("checkpoint model max_seq_len does not match training seq_len")

    if architecture == BASELINE_ARCHITECTURE:
        model: torch.nn.Module = FullAttentionResidualLM(model_config)
    elif architecture == PROPOSAL_ARCHITECTURE:
        required = (
            "num_slots",
            "memory_dim",
            "read_key_dim",
            "read_value_dim",
            "alpha_bias",
            "beta_bias",
            "gamma_init",
        )
        missing = [key for key in required if key not in run_config]
        if missing:
            raise ValueError(f"proposal checkpoint is missing reconstruction fields: {missing}")
        model = SoftmaxReadGatedDeltaDepthMemoryLM(
            model_config,
            num_slots=int(run_config["num_slots"]),
            memory_dim=int(run_config["memory_dim"]),
            read_key_dim=int(run_config["read_key_dim"]),
            read_value_dim=int(run_config["read_value_dim"]),
            alpha_bias=float(run_config["alpha_bias"]),
            beta_bias=float(run_config["beta_bias"]),
            gamma_init=float(run_config["gamma_init"]),
        )
    elif architecture == ASSOCIATIVE_ARCHITECTURE:
        required = (
            "num_slots",
            "memory_dim",
            "alpha_bias",
            "beta_bias",
            "gamma_init",
        )
        missing = [key for key in required if key not in run_config]
        if missing:
            raise ValueError(
                f"associative checkpoint is missing reconstruction fields: {missing}"
            )
        model = AssociativeReadDepthKDALM(
            model_config,
            num_slots=int(run_config["num_slots"]),
            memory_dim=int(run_config["memory_dim"]),
            alpha_bias=float(run_config["alpha_bias"]),
            beta_bias=float(run_config["beta_bias"]),
            gamma_init=float(run_config["gamma_init"]),
        )
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture!r}")

    expected_parameters = int(run_config.get("model_parameters", -1))
    actual_parameters = count_parameters(model)
    if actual_parameters != expected_parameters:
        raise ValueError(
            f"parameter-count mismatch: reconstructed {actual_parameters:,}, "
            f"checkpoint config says {expected_parameters:,}"
        )
    return model


def configs_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    normalized_left = json.loads(json.dumps(left, default=str))
    normalized_right = json.loads(json.dumps(right, default=str))
    return normalized_left == normalized_right


def load_checkpoint_model(
    spec: CheckpointSpec,
    *,
    expected_step: int,
    known_checkpoint_sha256: str | None = None,
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    checkpoint_sha256 = known_checkpoint_sha256 or sha256_file(spec.checkpoint_path)
    try:
        checkpoint = torch.load(
            spec.checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"safe weights-only loading failed for trusted checkpoint {spec.checkpoint_path}; "
            "the evaluator will not fall back to unrestricted pickle loading"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint is not an object: {spec.checkpoint_path}")
    if int(checkpoint.get("step", -1)) != expected_step:
        raise ValueError(
            f"expected checkpoint step {expected_step}, got {checkpoint.get('step')} in "
            f"{spec.checkpoint_path}"
        )
    embedded_config = checkpoint.get("run_config")
    if not isinstance(embedded_config, dict):
        raise ValueError(f"checkpoint has no embedded run_config: {spec.checkpoint_path}")
    if not configs_match(embedded_config, spec.run_config):
        raise ValueError(
            f"embedded run_config disagrees with sibling config.json: {spec.checkpoint_path}"
        )
    if embedded_config.get("architecture") != spec.architecture:
        raise ValueError(f"embedded architecture mismatch: {spec.checkpoint_path}")
    if int(embedded_config.get("seed", -1)) != spec.seed:
        raise ValueError(f"embedded seed mismatch: {spec.checkpoint_path}")

    raw_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(raw_state_dict, Mapping):
        raise ValueError(f"checkpoint has no model_state_dict: {spec.checkpoint_path}")
    state_dict = normalize_state_dict(raw_state_dict)
    model = construct_model(embedded_config)
    model.load_state_dict(state_dict, strict=True)
    del state_dict, raw_state_dict, checkpoint
    gc.collect()
    metadata = {
        "step": expected_step,
        "training_tokens_seen": expected_step
        * int(spec.run_config["batch_size"])
        * int(spec.run_config["grad_accum_steps"])
        * int(spec.run_config["seq_len"]),
        "model_parameters": count_parameters(model),
    }
    return model, checkpoint_sha256, metadata


def command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            list(command), cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=10
        ).strip()
    except Exception:
        return None


def environment_info(device: torch.device, precision: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "precision": precision,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": command_output(["git", "status", "--porcelain"]),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "")) or None,
        }
    return info


def environment_fingerprint(info: Mapping[str, Any]) -> str:
    comparable = {
        "python": info.get("python"),
        "platform": info.get("platform"),
        "torch": info.get("torch"),
        "numpy": info.get("numpy"),
        "cuda_runtime": info.get("cuda_runtime"),
        "cudnn": info.get("cudnn"),
        "device": info.get("device"),
        "precision": info.get("precision"),
        "git_commit": info.get("git_commit"),
        "git_status_porcelain": info.get("git_status_porcelain"),
        "gpu": {
            key: info.get("gpu", {}).get(key)
            for key in ("name", "compute_capability", "total_memory_bytes")
        }
        if isinstance(info.get("gpu"), dict)
        else None,
    }
    return object_sha256(comparable)


def make_cpu_batch(
    token_ids: torch.Tensor,
    sequence_starts: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack(
        [token_ids[start : start + SEQUENCE_LENGTH].to(torch.long) for start in sequence_starts]
    )
    y = torch.stack(
        [
            token_ids[start + 1 : start + SEQUENCE_LENGTH + 1].to(torch.long)
            for start in sequence_starts
        ]
    )
    return x, y


def cuda_cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def preflight_checkpoint_models(
    *,
    specs: Sequence[CheckpointSpec],
    expected_step: int,
    token_ids: torch.Tensor,
    first_block_id: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> list[dict[str, Any]]:
    source_start = POOL_START_TOKEN + first_block_id * BLOCK_TARGET_TOKENS
    starts = [source_start + index * SEQUENCE_LENGTH for index in range(batch_size)]
    x_cpu, _ = make_cpu_batch(token_ids, starts)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        model: torch.nn.Module | None = None
        cuda_cleanup(device)
        try:
            model, checkpoint_sha256, metadata = load_checkpoint_model(
                spec,
                expected_step=expected_step,
            )
            model = model.to(device)
            model.eval()
            x = x_cpu.to(device)
            with torch.inference_mode(), autocast_context(device, precision):
                logits = model(x)
            if tuple(logits.shape[:2]) != (batch_size, SEQUENCE_LENGTH):
                raise ValueError(f"preflight produced an unexpected logit shape for {spec.run_name}")
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError(f"preflight produced non-finite logits for {spec.run_name}")
            rows.append(
                {
                    "architecture": spec.architecture,
                    "seed": spec.seed,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_step": metadata["step"],
                    "model_parameters": metadata["model_parameters"],
                    "forward_shape": list(logits.shape),
                    "status": "strict_load_and_forward_passed",
                }
            )
            del x, logits
        finally:
            if model is not None:
                del model
            cuda_cleanup(device)
    return rows


def evaluate_model(
    *,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    block_ids: Sequence[int],
    batch_size: int,
    device: torch.device,
    precision: str,
    progress_every_blocks: int,
    label: str,
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()

    first_block_start = POOL_START_TOKEN + block_ids[0] * BLOCK_TARGET_TOKENS
    warmup_starts = [first_block_start + index * SEQUENCE_LENGTH for index in range(batch_size)]
    warmup_x_cpu, _ = make_cpu_batch(token_ids, warmup_starts)
    warmup_x = warmup_x_cpu.to(device)
    with torch.inference_mode(), autocast_context(device, precision):
        warmup_logits, diagnostics = model(warmup_x, return_diagnostics=True)
    del warmup_logits, warmup_x, warmup_x_cpu
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    blocks: list[dict[str, Any]] = []
    total_nll_parts: list[float] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for block_index, block_id in enumerate(block_ids, start=1):
            source_start = POOL_START_TOKEN + block_id * BLOCK_TARGET_TOKENS
            sequence_starts = [
                source_start + index * SEQUENCE_LENGTH
                for index in range(BLOCK_TARGET_TOKENS // SEQUENCE_LENGTH)
            ]
            sequence_nll_sums: list[float] = []
            for offset in range(0, len(sequence_starts), batch_size):
                batch_starts = sequence_starts[offset : offset + batch_size]
                x_cpu, y_cpu = make_cpu_batch(token_ids, batch_starts)
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                with autocast_context(device, precision):
                    logits = model(x)
                token_nll = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="none",
                ).reshape(len(batch_starts), SEQUENCE_LENGTH)
                nonfinite_count = int((~torch.isfinite(token_nll)).sum().item())
                if nonfinite_count:
                    raise FloatingPointError(
                        f"{label}: found {nonfinite_count} non-finite token losses in block {block_id}"
                    )
                sequence_nll_sums.extend(token_nll.double().sum(dim=1).cpu().tolist())
                del x_cpu, y_cpu, x, y, logits, token_nll

            if len(sequence_nll_sums) != BLOCK_TARGET_TOKENS // SEQUENCE_LENGTH:
                raise AssertionError("wrong sequence count while evaluating a block")
            block_nll_sum = math.fsum(sequence_nll_sums)
            block_record = {
                "block_id": block_id,
                "source_start_token": source_start,
                "sequence_starts": sequence_starts,
                "sequence_nll_sums": sequence_nll_sums,
                "token_count": BLOCK_TARGET_TOKENS,
                "nll_sum": block_nll_sum,
                "mean_nll": block_nll_sum / BLOCK_TARGET_TOKENS,
                "nonfinite_count": 0,
            }
            blocks.append(block_record)
            total_nll_parts.append(block_nll_sum)

            if progress_every_blocks > 0 and (
                block_index % progress_every_blocks == 0 or block_index == len(block_ids)
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                tokens_done = block_index * BLOCK_TARGET_TOKENS
                print(
                    json.dumps(
                        {
                            "type": "eval_progress",
                            "checkpoint": label,
                            "blocks_complete": block_index,
                            "blocks_total": len(block_ids),
                            "target_tokens_complete": tokens_done,
                            "tokens_per_second": tokens_done / max(elapsed, 1e-9),
                        }
                    ),
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_seconds = time.perf_counter() - started
    total_tokens = len(block_ids) * BLOCK_TARGET_TOKENS
    total_nll_sum = math.fsum(total_nll_parts)
    mean_nll = total_nll_sum / total_tokens
    memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    } if device.type == "cuda" else {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    return {
        "target_token_count": total_tokens,
        "nll_sum": total_nll_sum,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "eval_seconds": eval_seconds,
        "target_tokens_per_second": total_tokens / max(eval_seconds, 1e-9),
        "timing_scope": (
            "end-to-end quality-evaluation loop including CPU slicing, host-to-device transfer, "
            "per-batch FP64 loss transfer, finite checks, and progress logging; not a standalone "
            "architecture throughput benchmark"
        ),
        "cuda_memory": memory,
        "diagnostics_first_batch": diagnostics,
        "blocks": blocks,
    }


@torch.no_grad()
def disable_memory_reads(model: torch.nn.Module) -> int:
    """Set every depth-memory output gamma to zero for a causal intervention."""
    transitions = getattr(model, "transitions", None)
    if transitions is None:
        raise TypeError("memory-off intervention requires a model.transitions collection")
    count = 0
    for transition in transitions:
        gamma = getattr(transition, "gamma", None)
        if not isinstance(gamma, torch.nn.Parameter) or gamma.numel() != 1:
            raise TypeError("memory transition is missing its scalar gamma parameter")
        gamma.zero_()
        count += 1
    if count == 0:
        raise ValueError("memory-off intervention found no live transitions")
    config = getattr(model, "config", None)
    num_layers = getattr(config, "num_layers", None)
    if not isinstance(num_layers, int):
        raise TypeError("memory-off intervention could not determine model depth")
    expected_count = 2 * num_layers - 1
    if count != expected_count:
        raise ValueError(
            f"memory-off intervention found {count} transitions, "
            f"expected {expected_count}"
        )
    return count


def validate_metric_payload(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Apply the full result-integrity checks to a nested intervention payload."""
    validate_completed_result(
        {
            "schema": RESULT_SCHEMA,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            **metrics,
        },
        manifest,
    )


def require_close(label: str, actual: Any, expected: float) -> None:
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{label} is inconsistent: expected {expected!r}, got {actual!r}")


def validate_completed_result(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    architecture: str | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    checkpoint_sha256: str | None = None,
    evaluation_sha256: str | None = None,
) -> None:
    validate_manifest_integrity(manifest)
    if record.get("schema") != RESULT_SCHEMA or record.get("status") != "complete":
        raise ValueError("result is not a complete record with the expected schema")
    expected_top_level = {
        "architecture": architecture,
        "seed": seed,
        "run_name": run_name,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_sha256": evaluation_sha256,
    }
    for key, expected in expected_top_level.items():
        if expected is not None and record.get(key) != expected:
            raise ValueError(
                f"result {key} mismatch: expected {expected!r}, got {record.get(key)!r}"
            )
    if record.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("result manifest hash does not match the active manifest")

    expected_block_ids = [int(value) for value in manifest["sampling"]["selected_block_ids"]]
    blocks = record.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(expected_block_ids):
        raise ValueError("result block list is missing or has the wrong length")
    actual_block_ids = [int(block.get("block_id", -1)) for block in blocks]
    if actual_block_ids != expected_block_ids:
        raise ValueError("result block IDs/order do not match the frozen manifest")

    block_nll_sums: list[float] = []
    sequences_per_block = BLOCK_TARGET_TOKENS // SEQUENCE_LENGTH
    for block, block_id in zip(blocks, expected_block_ids):
        source_start = POOL_START_TOKEN + block_id * BLOCK_TARGET_TOKENS
        expected_starts = [
            source_start + index * SEQUENCE_LENGTH for index in range(sequences_per_block)
        ]
        if int(block.get("source_start_token", -1)) != source_start:
            raise ValueError(f"result block {block_id} has the wrong source start")
        if block.get("sequence_starts") != expected_starts:
            raise ValueError(f"result block {block_id} has inconsistent sequence starts")
        if int(block.get("token_count", -1)) != BLOCK_TARGET_TOKENS:
            raise ValueError(f"result block {block_id} has the wrong token count")
        if int(block.get("nonfinite_count", -1)) != 0:
            raise ValueError(f"result block {block_id} reports non-finite losses")
        sequence_sums = block.get("sequence_nll_sums")
        if not isinstance(sequence_sums, list) or len(sequence_sums) != sequences_per_block:
            raise ValueError(f"result block {block_id} has the wrong sequence-loss count")
        numeric_sequence_sums = [float(value) for value in sequence_sums]
        if not all(math.isfinite(value) and value >= 0.0 for value in numeric_sequence_sums):
            raise ValueError(f"result block {block_id} has invalid sequence NLL sums")
        expected_block_sum = math.fsum(numeric_sequence_sums)
        require_close(f"result block {block_id} NLL sum", block.get("nll_sum"), expected_block_sum)
        require_close(
            f"result block {block_id} mean NLL",
            block.get("mean_nll"),
            expected_block_sum / BLOCK_TARGET_TOKENS,
        )
        block_nll_sums.append(expected_block_sum)

    expected_total_tokens = len(expected_block_ids) * BLOCK_TARGET_TOKENS
    if int(record.get("target_token_count", -1)) != expected_total_tokens:
        raise ValueError("result total target-token count is inconsistent")
    expected_total_nll = math.fsum(block_nll_sums)
    expected_mean_nll = expected_total_nll / expected_total_tokens
    require_close("result total NLL sum", record.get("nll_sum"), expected_total_nll)
    require_close("result mean NLL", record.get("mean_nll"), expected_mean_nll)
    require_close("result perplexity", record.get("perplexity"), math.exp(expected_mean_nll))
    eval_seconds = float(record.get("eval_seconds", float("nan")))
    throughput = float(record.get("target_tokens_per_second", float("nan")))
    if not math.isfinite(eval_seconds) or eval_seconds <= 0.0:
        raise ValueError("result evaluation duration is invalid")
    if not math.isfinite(throughput) or throughput <= 0.0:
        raise ValueError("result evaluation throughput is invalid")
    diagnostics = record.get("diagnostics_first_batch")
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise ValueError("result is missing first-batch architecture diagnostics")
    for key, value in diagnostics.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"result diagnostic {key} is non-finite")


def bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> list[float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("bootstrap requires a one-dimensional array with at least two values")
    generator = np.random.Generator(np.random.PCG64(seed))
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        chunk = min(1_000, remaining)
        indices = generator.integers(0, len(values), size=(chunk, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= chunk
    distribution = np.concatenate(means)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return [float(low), float(high)]


def summarize_comparison(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    manifest: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
    practical_margin: float,
    architectures: Sequence[str] = DEFAULT_ARCHITECTURES,
    control_architecture: str = BASELINE_ARCHITECTURE,
    candidate_architecture: str = PROPOSAL_ARCHITECTURE,
) -> dict[str, Any]:
    architectures = tuple(architectures)
    if set(architectures) != {control_architecture, candidate_architecture}:
        raise ValueError("comparison architectures must match control and candidate")
    completed = [record for record in records if record.get("status") == "complete"]
    for record in completed:
        validate_completed_result(record, manifest)
    by_pair = {(str(record["architecture"]), int(record["seed"])): record for record in completed}
    expected = {(architecture, seed) for architecture in architectures for seed in seeds}
    if set(by_pair) != expected:
        raise ValueError(f"cannot compare incomplete result matrix: {sorted(set(expected) - set(by_pair))}")

    architecture_summaries: dict[str, Any] = {}
    for architecture in architectures:
        losses = [float(by_pair[(architecture, seed)]["mean_nll"]) for seed in seeds]
        architecture_summaries[architecture] = {
            "mean_nll_by_seed": losses,
            "mean_nll": statistics.fmean(losses),
            "sample_sd_across_training_seeds": statistics.stdev(losses),
            "mean_perplexity_from_mean_nll": math.exp(statistics.fmean(losses)),
        }

    paired_rows: list[dict[str, Any]] = []
    paired_block_differences: list[list[float]] = []
    for seed in seeds:
        control = by_pair[(control_architecture, seed)]
        candidate = by_pair[(candidate_architecture, seed)]
        control_blocks = {int(row["block_id"]): row for row in control["blocks"]}
        candidate_blocks = {int(row["block_id"]): row for row in candidate["blocks"]}
        if list(sorted(control_blocks)) != list(sorted(candidate_blocks)):
            raise ValueError(f"paired block IDs differ for seed {seed}")
        block_differences = [
            (
                float(candidate_blocks[block_id]["nll_sum"])
                - float(control_blocks[block_id]["nll_sum"])
            )
            / BLOCK_TARGET_TOKENS
            for block_id in sorted(control_blocks)
        ]
        paired_block_differences.append(block_differences)
        delta = float(candidate["mean_nll"]) - float(control["mean_nll"])
        paired_rows.append(
            {
                "seed": seed,
                "control_architecture": control_architecture,
                "candidate_architecture": candidate_architecture,
                "control_mean_nll": float(control["mean_nll"]),
                "candidate_mean_nll": float(candidate["mean_nll"]),
                "candidate_minus_control": delta,
                "conditional_block_bootstrap_95_ci": bootstrap_interval(
                    np.asarray(block_differences, dtype=np.float64),
                    bootstrap_samples,
                    bootstrap_seed + seed,
                ),
            }
        )

    seed_deltas = np.asarray(
        [row["candidate_minus_control"] for row in paired_rows], dtype=np.float64
    )
    seed_count = len(seed_deltas)
    mean_delta = float(seed_deltas.mean())
    seed_sd = float(seed_deltas.std(ddof=1))
    seed_se = seed_sd / math.sqrt(seed_count)
    two_sided_critical = float(student_t.ppf(0.975, df=seed_count - 1))
    one_sided_critical = float(student_t.ppf(0.95, df=seed_count - 1))
    training_seed_ci = [
        mean_delta - two_sided_critical * seed_se,
        mean_delta + two_sided_critical * seed_se,
    ]
    noninferiority_upper = mean_delta + one_sided_critical * seed_se

    block_matrix = np.asarray(paired_block_differences, dtype=np.float64)
    common_block_means = block_matrix.mean(axis=0)
    conditional_block_ci = bootstrap_interval(
        common_block_means,
        bootstrap_samples,
        bootstrap_seed,
    )
    selected_blocks = int(manifest["sampling"]["selected_block_count"])
    candidate_blocks = int(manifest["sampling"]["candidate_block_count"])
    finite_population_se = math.sqrt(
        (1.0 - selected_blocks / candidate_blocks)
        * float(common_block_means.var(ddof=1))
        / selected_blocks
    )
    fixed_pool_critical = float(student_t.ppf(0.975, df=selected_blocks - 1))
    fixed_pool_ci = [
        mean_delta - fixed_pool_critical * finite_population_se,
        mean_delta + fixed_pool_critical * finite_population_se,
    ]

    memory_off_summaries: dict[str, Any] = {}
    for architecture in architectures:
        architecture_records = [by_pair[(architecture, seed)] for seed in seeds]
        interventions = [record.get("memory_off") for record in architecture_records]
        if any(intervention is not None for intervention in interventions):
            if not all(isinstance(intervention, dict) for intervention in interventions):
                raise ValueError(
                    f"memory-off results are incomplete for architecture {architecture}"
                )
            penalties = []
            for record, intervention in zip(architecture_records, interventions):
                assert isinstance(intervention, dict)
                validate_metric_payload(intervention, manifest)
                penalties.append(
                    float(intervention["mean_nll"]) - float(record["mean_nll"])
                )
            memory_off_summaries[architecture] = {
                "definition": "memory-off mean NLL minus normal mean NLL; positive means memory helps",
                "penalty_by_seed": penalties,
                "mean_penalty": statistics.fmean(penalties),
                "sample_sd_across_training_seeds": statistics.stdev(penalties),
                "positive_seed_count": sum(value > 0.0 for value in penalties),
                "all_seeds_positive": all(value > 0.0 for value in penalties),
            }

    legacy_names = (
        control_architecture == BASELINE_ARCHITECTURE
        and candidate_architecture == PROPOSAL_ARCHITECTURE
    )
    if legacy_names:
        for row in paired_rows:
            row.update(
                {
                    "baseline_mean_nll": row["control_mean_nll"],
                    "proposal_mean_nll": row["candidate_mean_nll"],
                    "proposal_minus_baseline": row["candidate_minus_control"],
                }
            )
    if training_seed_ci[1] < -practical_margin:
        verdict = "proposal_practically_superior" if legacy_names else "candidate_practically_superior"
    elif training_seed_ci[0] > practical_margin:
        verdict = "baseline_practically_superior" if legacy_names else "control_practically_superior"
    elif noninferiority_upper < practical_margin:
        verdict = (
            "proposal_noninferior_at_declared_margin"
            if legacy_names
            else "candidate_noninferior_at_declared_margin"
        )
    else:
        verdict = "inconclusive_add_paired_training_seeds"

    delta_definition = (
        "proposal mean NLL minus baseline mean NLL; negative favors proposal"
        if legacy_names
        else (
            f"{candidate_architecture} mean NLL minus {control_architecture} mean NLL; "
            f"negative favors {candidate_architecture}"
        )
    )

    return {
        "schema": COMPARISON_SCHEMA,
        "created_utc": utc_now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "control_architecture": control_architecture,
        "candidate_architecture": candidate_architecture,
        "delta_definition": delta_definition,
        "architecture_summaries": architecture_summaries,
        "paired_by_training_seed": paired_rows,
        "memory_off_causal_intervention": memory_off_summaries,
        "paired_training_seed_inference_primary": {
            "training_seed_count": seed_count,
            "mean_delta": mean_delta,
            "sample_sd": seed_sd,
            "standard_error": seed_se,
            "two_sided_95_t_ci": training_seed_ci,
            "two_sided_t_critical": two_sided_critical,
            "one_sided_95_upper_bound": noninferiority_upper,
            "one_sided_t_critical": one_sided_critical,
        },
        "evaluation_sample_uncertainty_conditional_on_checkpoints": {
            "paired_common_block_bootstrap_95_ci": conditional_block_ci,
            "bootstrap_interval_estimand": (
                "superpopulation-style validation-block uncertainty; no finite-population correction"
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_generator": f"numpy.PCG64({bootstrap_seed})",
            "finite_population_standard_error_for_fixed_10m_pool": finite_population_se,
            "fixed_10m_pool_approximate_95_t_ci": fixed_pool_ci,
            "fixed_pool_t_critical": fixed_pool_critical,
            "selected_blocks": selected_blocks,
            "candidate_blocks": candidate_blocks,
        },
        "decision": {
            "practical_margin_nll": practical_margin,
            "rule": (
                "two-sided seed CI entirely below -margin: candidate superior; entirely above "
                "+margin: control superior; one-sided 95% upper bound below +margin: candidate "
                "noninferior; otherwise inconclusive"
            ),
            "verdict": verdict,
        },
        "warning": (
            "Training seeds, not validation blocks, are the independent architecture replicates. "
            "Do not treat seed-block rows as independent observations."
        ),
    }


def write_results_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = [
        "architecture",
        "seed",
        "run_name",
        "checkpoint_step",
        "training_tokens_seen",
        "model_parameters",
        "target_token_count",
        "nll_sum",
        "mean_nll",
        "perplexity",
        "memory_off_mean_nll",
        "memory_off_penalty_nll",
        "eval_seconds",
        "target_tokens_per_second",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "checkpoint_sha256",
        "manifest_sha256",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda row: (str(row["architecture"]), int(row["seed"]))):
            writer.writerow(
                {
                    "architecture": record["architecture"],
                    "seed": record["seed"],
                    "run_name": record["run_name"],
                    "checkpoint_step": record["checkpoint_step"],
                    "training_tokens_seen": record["training_tokens_seen"],
                    "model_parameters": record["model_parameters"],
                    "target_token_count": record["target_token_count"],
                    "nll_sum": record["nll_sum"],
                    "mean_nll": record["mean_nll"],
                    "perplexity": record["perplexity"],
                    "memory_off_mean_nll": (
                        record["memory_off"]["mean_nll"]
                        if isinstance(record.get("memory_off"), dict)
                        else None
                    ),
                    "memory_off_penalty_nll": (
                        float(record["memory_off"]["mean_nll"])
                        - float(record["mean_nll"])
                        if isinstance(record.get("memory_off"), dict)
                        else None
                    ),
                    "eval_seconds": record["eval_seconds"],
                    "target_tokens_per_second": record["target_tokens_per_second"],
                    "peak_allocated_bytes": record["cuda_memory"]["peak_allocated_bytes"],
                    "peak_reserved_bytes": record["cuda_memory"]["peak_reserved_bytes"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "manifest_sha256": record["manifest_sha256"],
                }
            )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paired depth-memory checkpoints on one common validation manifest."
    )
    parser.add_argument(
        "--comparison",
        choices=tuple(COMPARISON_PLANS),
        default="attnres",
        help="Use 'reader' for softmax-read versus associative-read Depth KDA.",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("att-residual-exp/runs"))
    parser.add_argument("--run-pattern", default="full_100m_seed*")
    parser.add_argument("--checkpoint-name", choices=("latest.pt", "best.pt"), default="latest.pt")
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--expected-step", type=int, default=12_208)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--selected-blocks", type=int, default=DEFAULT_SELECTED_BLOCKS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--progress-every-blocks", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_728)
    parser.add_argument("--practical-margin", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = COMPARISON_PLANS[args.comparison]
    configure_validation_pool(plan)
    control_architecture = str(plan["control"])
    candidate_architecture = str(plan["candidate"])
    architectures = (control_architecture, candidate_architecture)
    if args.architectures is not None and tuple(args.architectures) != architectures:
        raise ValueError(
            f"comparison {args.comparison!r} requires architectures {architectures}"
        )
    if tuple(args.seeds) != DEFAULT_SEEDS:
        raise ValueError(
            f"the frozen six-checkpoint analysis requires seeds {DEFAULT_SEEDS}, got {tuple(args.seeds)}"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.selected_blocks < 2:
        raise ValueError("--selected-blocks must be at least 2 for uncertainty estimates")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.practical_margin != 0.01:
        raise ValueError("the frozen analysis plan requires --practical-margin 0.01")

    runs_dir = args.runs_dir.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("att-residual-exp/runs") / str(plan["output_dir"])
    ).resolve()
    specs = discover_checkpoints(
        runs_dir=runs_dir,
        architectures=architectures,
        seeds=args.seeds,
        run_pattern=args.run_pattern,
        checkpoint_name=args.checkpoint_name,
        expected_step=args.expected_step,
    )
    if args.comparison == "reader":
        validate_reader_ablation_matrix(specs, args.seeds)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    precision = resolve_precision(args.precision, requested_device)
    environment = environment_info(requested_device, precision)
    environment_hash = environment_fingerprint(environment)
    source_identity = source_code_identity()

    manifest = build_manifest(
        shard_path=VALIDATION_SHARD,
        tokenizer_path=TOKENIZER_FILE,
        eval_seed=args.eval_seed,
        selected_block_count=args.selected_blocks,
        batch_size=args.batch_size,
    )
    manifest_path = output_dir / "eval_manifest.json"
    validate_manifest_integrity(manifest)
    if manifest_path.exists():
        existing_manifest = load_json(manifest_path)
        validate_manifest_integrity(existing_manifest)
        if manifest_core(existing_manifest) != manifest_core(manifest):
            raise ValueError(
                f"existing manifest disagrees with requested evaluation: {manifest_path}; "
                "use a new --output-dir"
            )
        manifest = existing_manifest
    else:
        atomic_write_json(manifest_path, manifest)

    analysis_plan = {
        "comparison": args.comparison,
        "architectures": list(architectures),
        "control_architecture": control_architecture,
        "candidate_architecture": candidate_architecture,
        "paired_training_seeds": list(DEFAULT_SEEDS),
        "delta_definition": "candidate mean NLL minus control mean NLL",
        "practical_margin_nll": 0.01,
        "memory_off_intervention": bool(plan["memory_off_intervention"]),
        "validation_pool_start_token": POOL_START_TOKEN,
        "validation_pool_token_count": POOL_TOKEN_COUNT,
        "validation_pool_sha256": EXPECTED_POOL_SHA256,
        "primary_interval": "two-sided 95% paired Student-t interval across training seeds",
        "noninferiority_bound": "one-sided 95% paired Student-t upper bound",
        "conditional_block_interval": "paired 95% percentile bootstrap with common block resamples",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    analysis_plan_sha256 = object_sha256(analysis_plan)
    evaluation_identity = {
        "manifest_sha256": manifest["manifest_sha256"],
        "environment_fingerprint": environment_hash,
        "source_code_combined_sha256": source_identity["combined_sha256"],
        "analysis_plan_sha256": analysis_plan_sha256,
        "precision": precision,
        "device_type": requested_device.type,
        "checkpoint_name": args.checkpoint_name,
        "expected_step": args.expected_step,
        "comparison": args.comparison,
        "memory_off_intervention": bool(plan["memory_off_intervention"]),
    }
    evaluation_sha256 = object_sha256(evaluation_identity)
    token_ids = map_parameter_golf_tokens(
        VALIDATION_SHARD,
        int(manifest["dataset"]["header"]["token_count"]),
    )
    block_ids = [int(value) for value in manifest["sampling"]["selected_block_ids"]]
    checkpoint_preflight = preflight_checkpoint_models(
        specs=specs,
        expected_step=args.expected_step,
        token_ids=token_ids,
        first_block_id=block_ids[0],
        batch_size=args.batch_size,
        device=requested_device,
        precision=precision,
    )
    preflight = {
        "type": "common_eval_preflight",
        "status": "strict_checkpoint_load_and_forward_passed",
        "checkpoint_count": len(specs),
        "checkpoint_matrix": [
            {
                "architecture": spec.architecture,
                "seed": spec.seed,
                "run_name": spec.run_name,
                "path": str(spec.checkpoint_path),
                "size_bytes": spec.checkpoint_path.stat().st_size,
            }
            for spec in specs
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "target_tokens_per_checkpoint": manifest["sampling"]["total_target_tokens"],
        "total_target_tokens_all_checkpoints": (
            manifest["sampling"]["total_target_tokens"] * len(specs)
        ),
        "total_target_tokens_all_evaluations": (
            manifest["sampling"]["total_target_tokens"]
            * len(specs)
            * (2 if bool(plan["memory_off_intervention"]) else 1)
        ),
        "environment": environment,
        "environment_fingerprint": environment_hash,
        "source_code_identity": source_identity,
        "analysis_plan": analysis_plan,
        "analysis_plan_sha256": analysis_plan_sha256,
        "checkpoint_validation": checkpoint_preflight,
        "evaluation_sha256": evaluation_sha256,
        "output_dir": str(output_dir),
    }
    atomic_write_json(output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return

    results_dir = output_dir / "checkpoint_results"
    records: list[dict[str, Any]] = []
    failures = 0

    for spec in specs:
        result_path = results_dir / f"{spec.architecture}_seed{spec.seed}.json"
        label = f"{spec.architecture}/seed{spec.seed}"
        checkpoint_sha256 = sha256_file(spec.checkpoint_path)
        if result_path.exists():
            existing = load_json(result_path)
            if args.resume and existing.get("status") == "complete":
                validate_completed_result(
                    existing,
                    manifest,
                    architecture=spec.architecture,
                    seed=spec.seed,
                    run_name=spec.run_name,
                    checkpoint_sha256=checkpoint_sha256,
                    evaluation_sha256=evaluation_sha256,
                )
                print(json.dumps({"type": "eval_skip", "checkpoint": label}), flush=True)
                records.append(existing)
                continue
            if existing.get("status") == "complete":
                raise FileExistsError(
                    f"non-resumable completed result exists at {result_path}; use a new --output-dir"
                )

        cuda_cleanup(requested_device)
        started_utc = utc_now()
        model: torch.nn.Module | None = None
        try:
            model, loaded_sha256, checkpoint_metadata = load_checkpoint_model(
                spec,
                expected_step=args.expected_step,
                known_checkpoint_sha256=checkpoint_sha256,
            )
            if loaded_sha256 != checkpoint_sha256:
                raise RuntimeError(f"checkpoint changed while being read: {spec.checkpoint_path}")
            metrics = evaluate_model(
                model=model,
                token_ids=token_ids,
                block_ids=block_ids,
                batch_size=args.batch_size,
                device=requested_device,
                precision=precision,
                progress_every_blocks=args.progress_every_blocks,
                label=label,
            )
            memory_off_metrics: dict[str, Any] | None = None
            memory_off_transition_count: int | None = None
            if bool(plan["memory_off_intervention"]):
                memory_off_transition_count = disable_memory_reads(model)
                memory_off_metrics = evaluate_model(
                    model=model,
                    token_ids=token_ids,
                    block_ids=block_ids,
                    batch_size=args.batch_size,
                    device=requested_device,
                    precision=precision,
                    progress_every_blocks=args.progress_every_blocks,
                    label=f"{label}/memory_off",
                )
                validate_metric_payload(memory_off_metrics, manifest)
            record = {
                "schema": RESULT_SCHEMA,
                "status": "complete",
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "architecture": spec.architecture,
                "seed": spec.seed,
                "run_name": spec.run_name,
                "checkpoint_path": str(spec.checkpoint_path),
                "checkpoint_name": args.checkpoint_name,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": checkpoint_metadata["step"],
                "training_tokens_seen": checkpoint_metadata["training_tokens_seen"],
                "model_parameters": checkpoint_metadata["model_parameters"],
                "manifest_sha256": manifest["manifest_sha256"],
                "evaluation_sha256": evaluation_sha256,
                "environment_fingerprint": environment_hash,
                "environment": environment,
                "precision": precision,
                **metrics,
            }
            if memory_off_metrics is not None:
                record["memory_off"] = {
                    "intervention": "all_depth_memory_output_gammas_set_to_zero",
                    "transition_count": memory_off_transition_count,
                    "normal_mean_nll": metrics["mean_nll"],
                    "penalty_nll": (
                        float(memory_off_metrics["mean_nll"])
                        - float(metrics["mean_nll"])
                    ),
                    **memory_off_metrics,
                }
            validate_completed_result(
                record,
                manifest,
                architecture=spec.architecture,
                seed=spec.seed,
                run_name=spec.run_name,
                checkpoint_sha256=checkpoint_sha256,
                evaluation_sha256=evaluation_sha256,
            )
            atomic_write_json(result_path, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "type": "eval_complete",
                        "checkpoint": label,
                        "mean_nll": record["mean_nll"],
                        "perplexity": record["perplexity"],
                        "tokens_per_second": record["target_tokens_per_second"],
                        "memory_off_penalty_nll": (
                            record["memory_off"]["penalty_nll"]
                            if isinstance(record.get("memory_off"), dict)
                            else None
                        ),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failures += 1
            failure = {
                "schema": RESULT_SCHEMA,
                "status": "failed",
                "started_utc": started_utc,
                "failed_utc": utc_now(),
                "architecture": spec.architecture,
                "seed": spec.seed,
                "run_name": spec.run_name,
                "checkpoint_path": str(spec.checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_sha256": manifest["manifest_sha256"],
                "evaluation_sha256": evaluation_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(result_path, failure)
            print(json.dumps(failure, indent=2), flush=True)
        finally:
            if model is not None:
                del model
            cuda_cleanup(requested_device)

    if failures:
        raise RuntimeError(f"{failures} checkpoint evaluation(s) failed; inspect {results_dir}")

    comparison = summarize_comparison(
        records,
        seeds=args.seeds,
        manifest=manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        practical_margin=args.practical_margin,
        architectures=architectures,
        control_architecture=control_architecture,
        candidate_architecture=candidate_architecture,
    )
    comparison["evaluation_sha256"] = evaluation_sha256
    comparison["environment_fingerprint"] = environment_hash
    comparison["source_code_combined_sha256"] = source_identity["combined_sha256"]
    comparison["analysis_plan_sha256"] = analysis_plan_sha256
    comparison["checkpoint_sha256_by_architecture_and_seed"] = {
        f"{record['architecture']}/seed{record['seed']}": record["checkpoint_sha256"]
        for record in records
    }
    atomic_write_json(output_dir / "comparison.json", comparison)
    write_results_csv(output_dir / "results.csv", records)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
