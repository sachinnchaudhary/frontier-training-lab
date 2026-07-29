"""Integrity checks for reusing archived common-evaluation references."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate_checkpoints import (
    ASSOCIATIVE_ARCHITECTURE,
    COMPARISON_SCHEMA,
    DEFAULT_SEEDS,
    PROPOSAL_ARCHITECTURE,
    ROOT,
    environment_fingerprint,
    load_json,
    object_sha256,
    sha256_file,
)


def runtime_environment_core(info: Mapping[str, Any]) -> dict[str, Any]:
    """Runtime/hardware identity without Git fields or physical GPU UUID."""

    gpu = info.get("gpu")
    return {
        "python": info.get("python"),
        "platform": info.get("platform"),
        "torch": info.get("torch"),
        "numpy": info.get("numpy"),
        "cuda_runtime": info.get("cuda_runtime"),
        "cudnn": info.get("cudnn"),
        "device": info.get("device"),
        "precision": info.get("precision"),
        "gpu": {
            key: gpu.get(key)
            for key in ("name", "compute_capability", "total_memory_bytes")
        }
        if isinstance(gpu, Mapping)
        else None,
    }


def validate_confirmatory_bundle(
    *,
    confirmatory_dir: Path,
    comparison: Mapping[str, Any],
    manifest: Mapping[str, Any],
    current_environment: Mapping[str, Any],
    current_source_identity: Mapping[str, Any],
    new_source_paths: Sequence[str],
) -> dict[str, Any]:
    """Link comparison, preflight, sources, runtime, and checkpoint identities."""

    preflight_path = confirmatory_dir / "preflight.json"
    preflight = load_json(preflight_path)
    if (
        preflight.get("type") != "common_eval_preflight"
        or preflight.get("status") != "strict_checkpoint_load_and_forward_passed"
    ):
        raise ValueError("confirmatory preflight is not a completed strict preflight")
    if comparison.get("schema") != COMPARISON_SCHEMA:
        raise ValueError("confirmatory comparison schema mismatch")
    if (
        comparison.get("control_architecture") != PROPOSAL_ARCHITECTURE
        or comparison.get("candidate_architecture") != ASSOCIATIVE_ARCHITECTURE
    ):
        raise ValueError("confirmatory reader architecture labels mismatch")
    if preflight.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("confirmatory preflight manifest mismatch")

    archived_plan = preflight.get("analysis_plan")
    if not isinstance(archived_plan, Mapping) or preflight.get(
        "analysis_plan_sha256"
    ) != object_sha256(archived_plan):
        raise ValueError("confirmatory analysis-plan hash mismatch")

    archived_source = preflight.get("source_code_identity")
    if not isinstance(archived_source, Mapping) or not isinstance(
        archived_source.get("files"), Mapping
    ):
        raise ValueError("confirmatory source identity is missing")
    archived_files = archived_source["files"]
    if archived_source.get("combined_sha256") != object_sha256(archived_files):
        raise ValueError("confirmatory source identity hash mismatch")

    archived_environment = preflight.get("environment")
    if not isinstance(archived_environment, Mapping) or preflight.get(
        "environment_fingerprint"
    ) != environment_fingerprint(archived_environment):
        raise ValueError("confirmatory environment fingerprint mismatch")

    linked_values = {
        "manifest_sha256": preflight.get("manifest_sha256"),
        "evaluation_sha256": preflight.get("evaluation_sha256"),
        "environment_fingerprint": preflight.get("environment_fingerprint"),
        "source_code_combined_sha256": archived_source.get("combined_sha256"),
        "analysis_plan_sha256": preflight.get("analysis_plan_sha256"),
    }
    for key, archived_value in linked_values.items():
        if comparison.get(key) != archived_value:
            raise ValueError(f"comparison/preflight {key} mismatch")

    current_files = current_source_identity.get("files")
    if not isinstance(current_files, Mapping):
        raise ValueError("current evaluator source identity is missing")
    expected_archived_paths = set(current_files) - set(new_source_paths)
    if set(archived_files) != expected_archived_paths:
        raise ValueError("archived shared-source file set is inconsistent")
    root = ROOT.resolve()
    for relative, expected_hash in archived_files.items():
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("archived source path escapes repository root")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"archived shared source changed: {relative}")
        if current_files.get(relative) != expected_hash:
            raise ValueError(f"current source identity changed: {relative}")

    if runtime_environment_core(current_environment) != runtime_environment_core(
        archived_environment
    ):
        raise ValueError("current runtime/hardware differs from confirmatory evaluation")

    checkpoint_rows = preflight.get("checkpoint_validation")
    expected_pairs = {
        (architecture, seed)
        for architecture in (PROPOSAL_ARCHITECTURE, ASSOCIATIVE_ARCHITECTURE)
        for seed in DEFAULT_SEEDS
    }
    if (
        not isinstance(checkpoint_rows, list)
        or len(checkpoint_rows) != len(expected_pairs)
        or {
            (row.get("architecture"), row.get("seed"))
            for row in checkpoint_rows
            if isinstance(row, Mapping)
        }
        != expected_pairs
    ):
        raise ValueError("confirmatory preflight checkpoint matrix mismatch")

    checkpoint_hashes = comparison.get(
        "checkpoint_sha256_by_architecture_and_seed"
    )
    expected_hash_keys = {
        f"{architecture}/seed{seed}"
        for architecture, seed in expected_pairs
    }
    if not isinstance(checkpoint_hashes, Mapping) or set(checkpoint_hashes) != (
        expected_hash_keys
    ):
        raise ValueError("confirmatory comparison checkpoint hashes are inconsistent")
    for row in checkpoint_rows:
        key = f'{row["architecture"]}/seed{row["seed"]}'
        if row.get("checkpoint_sha256") != checkpoint_hashes.get(key):
            raise ValueError(f"preflight/comparison checkpoint hash mismatch: {key}")

    return {
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_evaluation_sha256": preflight["evaluation_sha256"],
        "preflight_environment_fingerprint": preflight[
            "environment_fingerprint"
        ],
        "preflight_source_code_combined_sha256": archived_source[
            "combined_sha256"
        ],
        "preflight_analysis_plan_sha256": preflight["analysis_plan_sha256"],
        "runtime_environment_core": runtime_environment_core(archived_environment),
    }
