"""Exploratory read-component interventions for trained associative readers.

This evaluator deliberately does not alter the archived confirmatory reader
comparison. It always reuses that comparison's frozen manifest and checkpoint
identity. Depending on the declared reference policy, it either strictly reuses
the archived normal and gamma-zero records or fully recomputes those references
in the current CUDA environment before evaluating three new forward
interventions on the completed direct-reader checkpoints:

* ``history_only`` keeps the recurrent write but hides its same-step read;
* ``current_correction_only`` reads only the same-step delta correction (whose
  innovation remains history-conditioned);
* ``first_current_off`` hides the unambiguously non-historical read at the
  first transition while retaining that write for later transitions.

The interventions are evaluator-only views over the frozen checkpoints.  No
parameter, buffer, state-dict key, or recurrent state update is changed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t

from _common import autocast_context, resolve_precision
from _associative_intervention_integrity import (
    validate_confirmatory_bundle,
    validate_recompute_frozen_metadata,
)
from associative_read_depth_kda import AssociativeReadDepthKDALM
from evaluate_checkpoints import (
    ASSOCIATIVE_ARCHITECTURE,
    BLOCK_TARGET_TOKENS,
    COMPARISON_SCHEMA,
    COMPARISON_PLANS,
    DEFAULT_EVAL_SEED,
    DEFAULT_SEEDS,
    DEFAULT_SELECTED_BLOCKS,
    PROPOSAL_ARCHITECTURE,
    ROOT,
    SEQUENCE_LENGTH,
    VALIDATION_SHARD,
    TOKENIZER_FILE,
    atomic_write_json,
    block_ids_sha256,
    bootstrap_interval,
    build_manifest,
    configs_match,
    configure_validation_pool,
    cuda_cleanup,
    discover_checkpoints,
    disable_memory_reads,
    environment_fingerprint,
    environment_info,
    evaluate_model,
    load_checkpoint_model,
    load_json,
    make_cpu_batch,
    manifest_core,
    map_parameter_golf_tokens,
    object_sha256,
    sha256_file,
    utc_now,
    validate_completed_result,
    validate_manifest_integrity,
    validate_metric_payload,
)


RESULT_SCHEMA = "attres-associative-intervention-result-v2"
REFERENCE_RESULT_SCHEMA = "attres-associative-reference-result-v1"
SUMMARY_SCHEMA = "attres-associative-intervention-summary-v2"
ANALYSIS_KIND = "exploratory_posthoc_checkpoint_intervention"
REFERENCE_POLICIES = ("archived_strict", "recompute_current_environment")
DEFAULT_CONFIRMATORY_DIR = (
    ROOT / "att-residual-exp/runs/common_eval_reader_4m_seed424242"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "att-residual-exp/runs/associative_interventions_4m_seed424242"
)
DEFAULT_RECOMPUTE_OUTPUT_DIR = (
    ROOT
    / "att-residual-exp/runs/associative_interventions_recomputed_4m_seed424242"
)

NOVEL_MODES: dict[str, dict[str, Any]] = {
    "history_only": {
        "description": (
            "read decayed historical state only; compute and retain every current write "
            "for later transitions"
        ),
        "history_scale": 1.0,
        "current_scale": 0.0,
        "scope": "all_transitions",
        "augmented_component_diagnostics_scope": "all_transitions",
    },
    "current_correction_only": {
        "description": (
            "read only the same-step correction; its innovation remains conditioned on "
            "the decayed historical state"
        ),
        "history_scale": 0.0,
        "current_scale": 1.0,
        "scope": "all_transitions",
        "augmented_component_diagnostics_scope": "all_transitions",
    },
    "first_current_off": {
        "description": (
            "disable the current correction read only at transition zero; retain its "
            "write and use normal reads thereafter"
        ),
        "history_scale": 1.0,
        "current_scale": 0.0,
        "scope": "transition_zero_only",
        "augmented_component_diagnostics_scope": "transition_zero_only",
    },
}

REFERENCE_MODES = {
    "normal": {
        "description": "unmodified trained write-then-read checkpoint",
        "intervention": "none",
    },
    "memory_output_off": {
        "description": "every learned memory-output gamma set to zero",
        "intervention": "all_depth_memory_output_gammas_set_to_zero",
    },
}


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _quantile(value: torch.Tensor, probability: float) -> torch.Tensor:
    return torch.quantile(value.float().reshape(-1), probability)


def source_code_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "att-residual-exp/_associative_intervention_integrity.py",
        ROOT / "att-residual-exp/evaluate_checkpoints.py",
        ROOT / "att-residual-exp/_common.py",
        ROOT / "att-residual-exp/associative_read_depth_kda.py",
        ROOT / "att-residual-exp/softmax_read_depth_kda.py",
        ROOT / "att-residual-exp/attention_residual_baseline.py",
        ROOT / "model/layer.py",
        ROOT / "model/rope.py",
        ROOT / "data/pretokenized.py",
    )
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
        if path.is_file()
    }
    if len(files) != len(paths):
        raise FileNotFoundError("an intervention evaluator source dependency is missing")
    return {"files": files, "combined_sha256": object_sha256(files)}


def mode_scales(mode: str, transition_index: int) -> tuple[float, float]:
    if mode == "normal":
        return 1.0, 1.0
    if mode == "memory_output_off":
        return 0.0, 0.0
    if mode not in NOVEL_MODES:
        raise ValueError(f"unsupported reader intervention: {mode!r}")
    definition = NOVEL_MODES[mode]
    if mode == "first_current_off" and transition_index != 0:
        return 1.0, 1.0
    return float(definition["history_scale"]), float(definition["current_scale"])


def _intervened_transition_forward(
    transition: torch.nn.Module,
    original_forward: Callable[..., Any],
    history_scale: float,
    current_scale: float,
) -> Callable[..., Any]:
    """Wrap one transition while delegating its state update to frozen code."""

    def forward(
        hidden: torch.Tensor,
        delta: torch.Tensor,
        state: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Any:
        original_output = original_forward(
            hidden,
            delta,
            state,
            return_diagnostics,
        )
        if return_diagnostics:
            _, next_state, diagnostics = original_output
            diagnostics = dict(diagnostics)
        else:
            _, next_state = original_output
            diagnostics = None

        # Recompute only the read decomposition.  ``next_state`` comes from
        # the unmodified checkpoint transition, so every recurrent write is
        # bit-for-bit delegated to the frozen architecture implementation.
        write_key, write_value, alpha, beta = transition.write_parameters(delta)
        decayed_state = alpha.unsqueeze(-1) * state
        predicted_value = torch.einsum(
            "btsr,bts->btr",
            decayed_state,
            write_key,
        )
        innovation = write_value - predicted_value
        correction = (
            beta.unsqueeze(-1)
            * write_key.unsqueeze(-1)
            * innovation.unsqueeze(-2)
        )
        query = transition.read_query(hidden)
        historical = transition.read_scale * torch.einsum(
            "btsr,bts->btr",
            decayed_state,
            query,
        )
        current = transition.read_scale * torch.einsum(
            "btsr,bts->btr",
            correction,
            query,
        )
        if history_scale == 1.0 and current_scale == 0.0:
            retrieval = historical
        elif history_scale == 0.0 and current_scale == 1.0:
            retrieval = current
        elif history_scale == 0.0 and current_scale == 0.0:
            retrieval = torch.zeros_like(historical)
        else:
            retrieval = history_scale * historical + current_scale * current
        projected_retrieval = transition.out_proj(retrieval)
        injected_update = transition.gamma.to(projected_retrieval.dtype) * projected_retrieval
        intervened_hidden = hidden + injected_update

        if diagnostics is None:
            return intervened_hidden, next_state

        eps = 1e-8
        projected_history = transition.out_proj(historical)
        projected_current = transition.out_proj(current)
        active_history = history_scale * projected_history
        active_current = current_scale * projected_current
        active_history_norm = active_history.float().norm(dim=-1)
        active_current_norm = active_current.float().norm(dim=-1)
        available_history_norm = projected_history.float().norm(dim=-1)
        available_current_norm = projected_current.float().norm(dim=-1)
        query_key_alignment = (query * write_key).sum(dim=-1)
        normal_combined = projected_history + projected_current
        normal_energy = normal_combined.float().square().sum().clamp_min(eps)
        history_attribution = (
            projected_history.float() * normal_combined.float()
        ).sum() / normal_energy
        current_attribution = (
            projected_current.float() * normal_combined.float()
        ).sum() / normal_energy
        active_rows = history_scale * decayed_state + current_scale * correction
        row_contributions = (
            transition.read_scale * query.unsqueeze(-1) * active_rows
        )
        cancellation_denominator = row_contributions.norm(dim=-1).sum(dim=-1)
        hidden_rms = _rms(hidden)
        delta_rms = _rms(delta)
        injected_rms = _rms(injected_update)

        diagnostics.update(
            {
                "memory_history_read_scale": history_scale,
                "memory_current_read_scale": current_scale,
                "memory_history_read_rms": _rms(active_history),
                "memory_current_write_read_rms": _rms(active_current),
                "memory_active_history_projected_read_rms": _rms(active_history),
                "memory_active_current_correction_projected_read_rms": _rms(
                    active_current
                ),
                "memory_available_history_projected_read_rms": _rms(
                    projected_history
                ),
                "memory_available_current_correction_projected_read_rms": _rms(
                    projected_current
                ),
                "memory_available_normal_raw_read_rms": _rms(historical + current),
                "memory_available_normal_projected_read_rms": _rms(normal_combined),
                "memory_available_current_correction_fraction": (
                    available_current_norm
                    / (available_history_norm + available_current_norm + eps)
                ).mean(),
                "memory_current_write_fraction": (
                    active_current_norm
                    / (active_history_norm + active_current_norm + eps)
                ).mean(),
                "memory_raw_read_rms": _rms(retrieval),
                "memory_projected_read_rms": _rms(projected_retrieval),
                "memory_injected_update_rms": injected_rms,
                "memory_injected_to_hidden_rms_ratio": injected_rms
                / hidden_rms.clamp_min(eps),
                "memory_injected_to_delta_rms_ratio": injected_rms
                / delta_rms.clamp_min(eps),
                "memory_injected_hidden_cosine": F.cosine_similarity(
                    injected_update.float(), hidden.float(), dim=-1, eps=eps
                ).mean(),
                "memory_injected_delta_cosine": F.cosine_similarity(
                    injected_update.float(), delta.float(), dim=-1, eps=eps
                ).mean(),
                "memory_available_history_current_cosine": F.cosine_similarity(
                    projected_history.float(),
                    projected_current.float(),
                    dim=-1,
                    eps=eps,
                ).mean(),
                "memory_available_history_signed_energy_attribution": history_attribution,
                "memory_available_current_signed_energy_attribution": current_attribution,
                "memory_available_signed_energy_attribution_sum": (
                    history_attribution + current_attribution
                ),
                "memory_query_write_key_alignment": query_key_alignment.mean(),
                "memory_query_write_key_abs_alignment": (
                    query_key_alignment.abs().mean()
                ),
                "memory_query_write_key_alignment_rms": _rms(
                    query_key_alignment
                ),
                "memory_query_write_key_alignment_p05": _quantile(
                    query_key_alignment, 0.05
                ),
                "memory_query_write_key_alignment_p50": _quantile(
                    query_key_alignment, 0.50
                ),
                "memory_query_write_key_alignment_p95": _quantile(
                    query_key_alignment, 0.95
                ),
                "memory_query_write_key_positive_fraction": (
                    query_key_alignment > 0
                ).float().mean(),
                "memory_innovation_to_write_value_rms_ratio": diagnostics[
                    "memory_innovation_rms"
                ]
                / _rms(write_value).clamp_min(eps),
                "memory_signed_cancellation_ratio": (
                    retrieval.norm(dim=-1)
                    / cancellation_denominator.clamp_min(eps)
                ).mean(),
                "depth_hidden_rms": _rms(intervened_hidden),
            }
        )
        return intervened_hidden, next_state, diagnostics

    return forward


def install_intervention(
    model: torch.nn.Module,
    mode: str,
) -> list[dict[str, float | int]]:
    if not isinstance(model, AssociativeReadDepthKDALM):
        raise TypeError("read-component interventions require AssociativeReadDepthKDALM")
    if mode not in {*NOVEL_MODES, "normal", "memory_output_off"}:
        raise ValueError(f"unsupported reader intervention: {mode!r}")
    if any(hasattr(transition, "_intervention_original_forward") for transition in model.transitions):
        raise RuntimeError("an intervention is already installed on this model")

    rows: list[dict[str, float | int]] = []
    for index, transition in enumerate(model.transitions):
        history_scale, current_scale = mode_scales(mode, index)
        rows.append(
            {
                "transition_index": index,
                "history_scale": history_scale,
                "current_scale": current_scale,
            }
        )
        if history_scale == 1.0 and current_scale == 1.0:
            continue
        original_forward = transition.forward
        transition._intervention_original_forward = original_forward
        transition._intervention_mode = mode
        transition._intervention_history_scale = history_scale
        transition._intervention_current_scale = current_scale
        transition.forward = _intervened_transition_forward(
            transition,
            original_forward,
            history_scale,
            current_scale,
        )
    return rows


def _safe_load_checkpoint_model(*args: Any, **kwargs: Any) -> Any:
    # Legacy checkpoints serialized ``torch.__version__`` as TorchVersion.
    # Keep weights-only loading and allowlist only that known benign subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return load_checkpoint_model(*args, **kwargs)


def _metrics_view(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "target_token_count",
        "nll_sum",
        "mean_nll",
        "perplexity",
        "eval_seconds",
        "target_tokens_per_second",
        "timing_scope",
        "cuda_memory",
        "diagnostics_first_batch",
        "blocks",
    )
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"confirmatory metric payload is missing fields: {missing}")
    return {key: record[key] for key in keys}


def validate_reference_anchor(
    *,
    current: Mapping[str, Any],
    archived: Mapping[str, Any],
    block_id: int,
    label: str,
    sequence_sum_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare one newly evaluated block with its archived per-sequence losses."""

    current_blocks = current.get("blocks")
    archived_blocks = archived.get("blocks")
    if not isinstance(current_blocks, list) or len(current_blocks) != 1:
        raise ValueError(f"{label}: current anchor must contain exactly one block")
    archived_by_id = {
        int(row["block_id"]): row
        for row in archived_blocks
        if isinstance(row, Mapping) and "block_id" in row
    } if isinstance(archived_blocks, list) else {}
    if block_id not in archived_by_id:
        raise ValueError(f"{label}: archived anchor block is missing")
    current_block = current_blocks[0]
    archived_block = archived_by_id[block_id]
    if int(current_block.get("block_id", -1)) != block_id:
        raise ValueError(f"{label}: current anchor used the wrong block")
    current_sums = current_block.get("sequence_nll_sums")
    archived_sums = archived_block.get("sequence_nll_sums")
    if not isinstance(current_sums, list) or not isinstance(archived_sums, list) or (
        len(current_sums) != len(archived_sums)
    ):
        raise ValueError(f"{label}: anchor sequence-loss vectors differ in length")
    differences = [
        abs(float(actual) - float(expected))
        for actual, expected in zip(current_sums, archived_sums)
    ]
    maximum = max(differences, default=0.0)
    if maximum > sequence_sum_tolerance:
        raise ValueError(
            f"{label}: archived numerical anchor drifted by {maximum:.9g}, "
            f"above tolerance {sequence_sum_tolerance:.9g}"
        )
    return {
        "block_id": block_id,
        "sequence_count": len(current_sums),
        "sequence_nll_sum_absolute_tolerance": sequence_sum_tolerance,
        "maximum_absolute_sequence_nll_sum_difference": maximum,
        "status": "archived_reference_numerically_reproduced",
    }


def load_confirmatory_references(
    *,
    confirmatory_dir: Path,
    specs: Sequence[Any],
    manifest: Mapping[str, Any],
    current_environment: Mapping[str, Any],
    current_source_identity: Mapping[str, Any],
    require_runtime_match: bool,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    comparison_path = confirmatory_dir / "comparison.json"
    comparison = load_json(comparison_path)
    evaluator_path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    integrity_path = (
        ROOT / "att-residual-exp/_associative_intervention_integrity.py"
    ).relative_to(ROOT).as_posix()
    bundle_identity = validate_confirmatory_bundle(
        confirmatory_dir=confirmatory_dir,
        comparison=comparison,
        manifest=manifest,
        current_environment=current_environment,
        current_source_identity=current_source_identity,
        new_source_paths=(evaluator_path, integrity_path),
        require_runtime_match=require_runtime_match,
    )
    if comparison.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("confirmatory comparison uses a different manifest")
    if comparison.get("candidate_architecture") != ASSOCIATIVE_ARCHITECTURE:
        raise ValueError("confirmatory comparison has the wrong candidate architecture")
    evaluation_sha256 = comparison.get("evaluation_sha256")
    checkpoint_hashes = comparison.get("checkpoint_sha256_by_architecture_and_seed")
    if not isinstance(evaluation_sha256, str) or not isinstance(checkpoint_hashes, dict):
        raise ValueError("confirmatory comparison is missing integrity hashes")

    references: dict[int, dict[str, dict[str, Any]]] = {}
    result_hashes: dict[str, str] = {}
    for spec in specs:
        result_path = (
            confirmatory_dir
            / "checkpoint_results"
            / f"{ASSOCIATIVE_ARCHITECTURE}_seed{spec.seed}.json"
        )
        record = load_json(result_path)
        checkpoint_sha256 = sha256_file(spec.checkpoint_path)
        expected_hash = checkpoint_hashes.get(
            f"{ASSOCIATIVE_ARCHITECTURE}/seed{spec.seed}"
        )
        if checkpoint_sha256 != expected_hash:
            raise ValueError(f"checkpoint hash changed for seed {spec.seed}")
        validate_completed_result(
            record,
            manifest,
            architecture=ASSOCIATIVE_ARCHITECTURE,
            seed=spec.seed,
            run_name=spec.run_name,
            checkpoint_sha256=checkpoint_sha256,
            evaluation_sha256=evaluation_sha256,
        )
        memory_off = record.get("memory_off")
        if not isinstance(memory_off, dict):
            raise ValueError(f"seed {spec.seed} lacks a memory-off reference")
        validate_metric_payload(memory_off, manifest)
        if memory_off.get("intervention") != "all_depth_memory_output_gammas_set_to_zero":
            raise ValueError("unexpected confirmatory memory-off intervention")
        expected_transitions = 2 * int(spec.run_config["model"]["num_layers"]) - 1
        if int(memory_off.get("transition_count", -1)) != expected_transitions:
            raise ValueError("confirmatory memory-off transition count is inconsistent")
        if not math.isclose(
            float(memory_off["normal_mean_nll"]),
            float(record["mean_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("confirmatory memory-off normal NLL is inconsistent")
        expected_penalty = float(memory_off["mean_nll"]) - float(record["mean_nll"])
        if not math.isclose(
            float(memory_off.get("penalty_nll", float("nan"))),
            expected_penalty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("confirmatory memory-off penalty arithmetic is inconsistent")
        references[spec.seed] = {
            "normal": _metrics_view(record),
            "memory_output_off": _metrics_view(memory_off),
        }
        result_hashes[str(spec.seed)] = sha256_file(result_path)

    identity = {
        **bundle_identity,
        "comparison_path": str(comparison_path),
        "comparison_sha256": sha256_file(comparison_path),
        "comparison_evaluation_sha256": evaluation_sha256,
        "checkpoint_result_sha256_by_seed": result_hashes,
    }
    return references, identity


def expected_transition_scales(
    mode: str,
    transition_count: int,
) -> list[dict[str, float | int]]:
    if transition_count <= 0:
        raise ValueError("transition count must be positive")
    return [
        {
            "transition_index": index,
            "history_scale": mode_scales(mode, index)[0],
            "current_scale": mode_scales(mode, index)[1],
        }
        for index in range(transition_count)
    ]


def expected_diagnostic_scope(mode: str) -> dict[str, str]:
    if mode not in NOVEL_MODES:
        raise ValueError(f"diagnostic scope requires a novel mode, got {mode!r}")
    return {
        "shared_architecture_diagnostics": "all_transitions",
        "wrapper_added_component_diagnostics": str(
            NOVEL_MODES[mode]["augmented_component_diagnostics_scope"]
        ),
    }


def validate_intervention_result(
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    evaluation_sha256: str,
    checkpoint_sha256: str,
    seed: int,
    mode: str,
    run_name: str,
    checkpoint_step: int,
    environment_hash: str,
    transition_count: int,
    reference_policy: str,
    gpu_uuid: str | None,
) -> None:
    if record.get("schema") != RESULT_SCHEMA or record.get("status") != "complete":
        raise ValueError("intervention result is not complete")
    expected = {
        "analysis_kind": ANALYSIS_KIND,
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_sha256": evaluation_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": ASSOCIATIVE_ARCHITECTURE,
        "seed": seed,
        "mode": mode,
        "run_name": run_name,
        "checkpoint_step": checkpoint_step,
        "environment_fingerprint": environment_hash,
        "reference_policy": reference_policy,
        "gpu_uuid": gpu_uuid,
        "mode_definition": NOVEL_MODES[mode],
        "transition_scales": expected_transition_scales(mode, transition_count),
        "diagnostic_scope": expected_diagnostic_scope(mode),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"intervention result {key} mismatch: {record.get(key)!r} != {value!r}"
            )
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("intervention result has no metric payload")
    validate_metric_payload(metrics, manifest)


def validate_local_reference_result(
    record: Mapping[str, Any],
    *,
    mode: str,
    spec: Any,
    manifest: Mapping[str, Any],
    evaluation_sha256: str,
    environment_hash: str,
    gpu_uuid: str | None,
    checkpoint_sha256: str,
    expected_step: int,
) -> None:
    if mode not in REFERENCE_MODES:
        raise ValueError(f"unsupported local reference mode: {mode!r}")
    expected = {
        "schema": REFERENCE_RESULT_SCHEMA,
        "status": "complete",
        "analysis_kind": ANALYSIS_KIND,
        "reference_policy": "recompute_current_environment",
        "reference_source": "fully_recomputed_current_environment",
        "architecture": ASSOCIATIVE_ARCHITECTURE,
        "seed": spec.seed,
        "run_name": spec.run_name,
        "mode": mode,
        "mode_definition": REFERENCE_MODES[mode],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": expected_step,
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_sha256": evaluation_sha256,
        "environment_fingerprint": environment_hash,
        "gpu_uuid": gpu_uuid,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"local reference {key} mismatch: {record.get(key)!r} != {value!r}"
            )

    expected_transitions = 2 * int(spec.run_config["model"]["num_layers"]) - 1
    if mode == "normal":
        if record.get("intervention") != "none" or record.get(
            "transition_count"
        ) is not None:
            raise ValueError("normal local reference has intervention metadata")
    else:
        if record.get("intervention") != (
            "all_depth_memory_output_gammas_set_to_zero"
        ) or int(record.get("transition_count", -1)) != expected_transitions:
            raise ValueError("memory-off local reference intervention is inconsistent")

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("local reference has no metric payload")
    validate_metric_payload(metrics, manifest)
    if mode == "memory_output_off":
        diagnostics = metrics["diagnostics_first_batch"]
        for key in ("memory_gamma", "memory_gamma_abs"):
            if not math.isclose(
                float(diagnostics.get(key, float("nan"))),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"memory-off local reference has nonzero {key}")


def evaluate_local_references(
    *,
    specs: Sequence[Any],
    manifest: Mapping[str, Any],
    evaluation_sha256: str,
    environment_hash: str,
    gpu_uuid: str | None,
    checkpoint_hashes: Mapping[str, str],
    expected_step: int,
    token_ids: torch.Tensor,
    block_ids: Sequence[int],
    batch_size: int,
    device: torch.device,
    precision: str,
    progress_every_blocks: int,
    output_dir: Path,
    resume: bool,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    result_dir = output_dir / "reference_results"
    records: list[tuple[Path, dict[str, Any]]] = []
    failures = 0
    for spec in specs:
        for mode in REFERENCE_MODES:
            result_path = result_dir / f"seed{spec.seed}" / f"{mode}.json"
            checkpoint_sha256 = checkpoint_hashes[str(spec.seed)]
            if result_path.exists():
                existing = load_json(result_path)
                if resume and existing.get("status") == "complete":
                    validate_local_reference_result(
                        existing,
                        mode=mode,
                        spec=spec,
                        manifest=manifest,
                        evaluation_sha256=evaluation_sha256,
                        environment_hash=environment_hash,
                        gpu_uuid=gpu_uuid,
                        checkpoint_sha256=checkpoint_sha256,
                        expected_step=expected_step,
                    )
                    print(
                        json.dumps(
                            {
                                "type": "reference_skip",
                                "seed": spec.seed,
                                "mode": mode,
                            }
                        ),
                        flush=True,
                    )
                    records.append((result_path, existing))
                    continue
                if existing.get("status") == "complete":
                    raise FileExistsError(
                        f"completed local reference exists at {result_path}; "
                        "use --resume or a new output directory"
                    )

            model = None
            started_utc = utc_now()
            cuda_cleanup(device)
            try:
                model, loaded_sha256, metadata = _safe_load_checkpoint_model(
                    spec,
                    expected_step=expected_step,
                    known_checkpoint_sha256=checkpoint_sha256,
                )
                if loaded_sha256 != checkpoint_sha256:
                    raise RuntimeError("checkpoint changed while loading local reference")
                if any(
                    hasattr(transition, "_intervention_original_forward")
                    for transition in model.transitions
                ):
                    raise RuntimeError("fresh local reference model contains a wrapper")
                intervention = "none"
                transition_count: int | None = None
                if mode == "memory_output_off":
                    intervention = "all_depth_memory_output_gammas_set_to_zero"
                    transition_count = disable_memory_reads(model)
                metrics = evaluate_model(
                    model=model,
                    token_ids=token_ids,
                    block_ids=block_ids,
                    batch_size=batch_size,
                    device=device,
                    precision=precision,
                    progress_every_blocks=progress_every_blocks,
                    label=f"{ASSOCIATIVE_ARCHITECTURE}/seed{spec.seed}/{mode}",
                )
                record = {
                    "schema": REFERENCE_RESULT_SCHEMA,
                    "status": "complete",
                    "analysis_kind": ANALYSIS_KIND,
                    "reference_policy": "recompute_current_environment",
                    "reference_source": "fully_recomputed_current_environment",
                    "started_utc": started_utc,
                    "completed_utc": utc_now(),
                    "architecture": ASSOCIATIVE_ARCHITECTURE,
                    "seed": spec.seed,
                    "run_name": spec.run_name,
                    "mode": mode,
                    "mode_definition": REFERENCE_MODES[mode],
                    "intervention": intervention,
                    "transition_count": transition_count,
                    "checkpoint_path": str(spec.checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_step": metadata["step"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "evaluation_sha256": evaluation_sha256,
                    "environment_fingerprint": environment_hash,
                    "gpu_uuid": gpu_uuid,
                    "metrics": metrics,
                }
                validate_local_reference_result(
                    record,
                    mode=mode,
                    spec=spec,
                    manifest=manifest,
                    evaluation_sha256=evaluation_sha256,
                    environment_hash=environment_hash,
                    gpu_uuid=gpu_uuid,
                    checkpoint_sha256=checkpoint_sha256,
                    expected_step=expected_step,
                )
                atomic_write_json(result_path, record)
                records.append((result_path, record))
                print(
                    json.dumps(
                        {
                            "type": "reference_complete",
                            "seed": spec.seed,
                            "mode": mode,
                            "mean_nll": metrics["mean_nll"],
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                failure = {
                    "schema": REFERENCE_RESULT_SCHEMA,
                    "status": "failed",
                    "analysis_kind": ANALYSIS_KIND,
                    "reference_policy": "recompute_current_environment",
                    "seed": spec.seed,
                    "mode": mode,
                    "failed_utc": utc_now(),
                    "checkpoint_sha256": checkpoint_sha256,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "evaluation_sha256": evaluation_sha256,
                    "environment_fingerprint": environment_hash,
                    "gpu_uuid": gpu_uuid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(result_path, failure)
                print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
            finally:
                if model is not None:
                    del model
                cuda_cleanup(device)

    if failures:
        raise RuntimeError(f"{failures} local reference evaluation(s) failed")
    expected_count = len(specs) * len(REFERENCE_MODES)
    if len(records) != expected_count:
        raise RuntimeError("local reference result matrix is incomplete")
    references: dict[int, dict[str, dict[str, Any]]] = {
        spec.seed: {} for spec in specs
    }
    result_hashes: dict[str, str] = {}
    for result_path, record in records:
        seed = int(record["seed"])
        mode = str(record["mode"])
        if mode in references[seed]:
            raise RuntimeError("duplicate local reference result")
        references[seed][mode] = record["metrics"]
        result_hashes[f"seed{seed}/{mode}"] = sha256_file(result_path)
    return references, {
        "source": "fully_recomputed_current_environment",
        "reference_policy": "recompute_current_environment",
        "environment_fingerprint": environment_hash,
        "gpu_uuid": gpu_uuid,
        "result_sha256_by_seed_and_mode": result_hashes,
    }


def _seed_inference(values: Sequence[float]) -> dict[str, Any]:
    if len(values) < 2:
        raise ValueError("training-seed inference requires at least two values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    return {
        "training_seed_count": len(values),
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "two_sided_95_t_ci": [mean - critical * se, mean + critical * se],
        "two_sided_t_critical": critical,
    }


def summarize(
    *,
    records: Sequence[Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Mapping[str, Any]]],
    manifest: Mapping[str, Any],
    evaluation_sha256: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    source_identity: Mapping[str, Any],
    confirmatory_identity: Mapping[str, Any],
    reference_policy: str,
    reference_identity: Mapping[str, Any],
    environment: Mapping[str, Any],
    gpu_uuid: str | None,
) -> dict[str, Any]:
    by_pair = {(int(row["seed"]), str(row["mode"])): row for row in records}
    expected = {(seed, mode) for seed in DEFAULT_SEEDS for mode in NOVEL_MODES}
    if len(records) != len(expected) or len(by_pair) != len(records):
        raise ValueError("duplicate or wrong-sized intervention result matrix")
    if set(by_pair) != expected:
        raise ValueError(f"incomplete intervention matrix: {sorted(expected - set(by_pair))}")

    mode_summaries: dict[str, Any] = {}
    for mode in (*REFERENCE_MODES, *NOVEL_MODES):
        paired_rows: list[dict[str, Any]] = []
        mode_nlls: list[float] = []
        delta_vs_normal: list[float] = []
        benefit_vs_off: list[float] = []
        for seed in DEFAULT_SEEDS:
            normal = references[seed]["normal"]
            memory_off = references[seed]["memory_output_off"]
            if mode in REFERENCE_MODES:
                metrics = references[seed][mode]
            else:
                metrics = by_pair[(seed, mode)]["metrics"]
            normal_blocks = {int(row["block_id"]): row for row in normal["blocks"]}
            mode_blocks = {int(row["block_id"]): row for row in metrics["blocks"]}
            if sorted(normal_blocks) != sorted(mode_blocks):
                raise ValueError(f"block IDs differ for seed {seed}, mode {mode}")
            block_deltas = [
                (
                    float(mode_blocks[block_id]["nll_sum"])
                    - float(normal_blocks[block_id]["nll_sum"])
                )
                / BLOCK_TARGET_TOKENS
                for block_id in sorted(normal_blocks)
            ]
            delta = float(metrics["mean_nll"]) - float(normal["mean_nll"])
            benefit = float(memory_off["mean_nll"]) - float(metrics["mean_nll"])
            mode_nlls.append(float(metrics["mean_nll"]))
            delta_vs_normal.append(delta)
            benefit_vs_off.append(benefit)
            paired_rows.append(
                {
                    "seed": seed,
                    "normal_mean_nll": float(normal["mean_nll"]),
                    "mode_mean_nll": float(metrics["mean_nll"]),
                    "mode_minus_normal_nll": delta,
                    "memory_off_minus_mode_nll": benefit,
                    "conditional_block_bootstrap_95_ci_for_mode_minus_normal": (
                        bootstrap_interval(
                            np.asarray(block_deltas, dtype=np.float64),
                            bootstrap_samples,
                            bootstrap_seed + seed + 10_000 * list(
                                (*REFERENCE_MODES, *NOVEL_MODES)
                            ).index(mode),
                        )
                    ),
                }
            )
        mode_summaries[mode] = {
            "definition": (
                REFERENCE_MODES.get(mode) or NOVEL_MODES.get(mode)
            ),
            "paired_by_training_seed": paired_rows,
            "mean_nll_seed_inference": _seed_inference(mode_nlls),
            "perplexity_from_mean_seed_nll": math.exp(statistics.fmean(mode_nlls)),
            "mode_minus_normal_seed_inference": _seed_inference(delta_vs_normal),
            "memory_off_minus_mode_seed_inference": _seed_inference(benefit_vs_off),
        }

    return {
        "schema": SUMMARY_SCHEMA,
        "created_utc": utc_now(),
        "analysis_kind": ANALYSIS_KIND,
        "warning": (
            "Exploratory dependency tests on write-then-read-trained checkpoints. "
            "Effects are not additive causal component attributions and do not predict "
            "how a read-then-write model would train. Wrapped-mode throughput includes "
            "duplicated evaluator computations and is not an architecture-efficiency "
            "measurement."
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_sha256": evaluation_sha256,
        "reference_policy": reference_policy,
        "reference_identity": reference_identity,
        "current_environment": environment,
        "gpu_uuid": gpu_uuid,
        "source_code_identity": source_identity,
        "frozen_artifact_identity": confirmatory_identity,
        "mode_summaries": mode_summaries,
    }


def write_csv(path: Path, summary: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = (
        "mode",
        "seed",
        "normal_mean_nll",
        "mode_mean_nll",
        "mode_minus_normal_nll",
        "memory_off_minus_mode_nll",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, mode_summary in summary["mode_summaries"].items():
            for row in mode_summary["paired_by_training_seed"]:
                writer.writerow({"mode": mode, **{key: row[key] for key in fields[1:]}})
    os.replace(temporary, path)


def _check_nonoverlap(output_dir: Path, confirmatory_dir: Path) -> None:
    output_dir = output_dir.resolve()
    confirmatory_dir = confirmatory_dir.resolve()
    if (
        output_dir == confirmatory_dir
        or output_dir in confirmatory_dir.parents
        or confirmatory_dir in output_dir.parents
    ):
        raise ValueError("intervention output and confirmatory directories must not overlap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exploratory history/current read interventions."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("att-residual-exp/runs"))
    parser.add_argument("--confirmatory-dir", type=Path, default=DEFAULT_CONFIRMATORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-policy",
        choices=REFERENCE_POLICIES,
        default="archived_strict",
    )
    parser.add_argument("--run-pattern", default="full_100m_seed*")
    parser.add_argument("--checkpoint-name", choices=("latest.pt",), default="latest.pt")
    parser.add_argument("--expected-step", type=int, default=12_208)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--selected-blocks", type=int, default=DEFAULT_SELECTED_BLOCKS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--progress-every-blocks", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_729)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if tuple(args.seeds) != DEFAULT_SEEDS:
        raise ValueError(f"the frozen intervention plan requires seeds {DEFAULT_SEEDS}")
    if args.eval_seed != DEFAULT_EVAL_SEED or args.selected_blocks != DEFAULT_SELECTED_BLOCKS:
        raise ValueError("the intervention plan must reuse the confirmatory frozen manifest")
    if args.batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("batch size and bootstrap sample count must be positive")
    if args.reference_policy == "recompute_current_environment" and args.device != "cuda":
        raise ValueError("current-environment reference recomputation requires CUDA")

    configure_validation_pool(COMPARISON_PLANS["reader"])
    runs_dir = args.runs_dir.resolve()
    confirmatory_dir = args.confirmatory_dir.resolve()
    selected_output_dir = args.output_dir
    if selected_output_dir is None:
        selected_output_dir = (
            DEFAULT_RECOMPUTE_OUTPUT_DIR
            if args.reference_policy == "recompute_current_environment"
            else DEFAULT_OUTPUT_DIR
        )
    output_dir = selected_output_dir.resolve()
    _check_nonoverlap(output_dir, confirmatory_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_checkpoints(
        runs_dir=runs_dir,
        architectures=(ASSOCIATIVE_ARCHITECTURE,),
        seeds=args.seeds,
        run_pattern=args.run_pattern,
        checkpoint_name=args.checkpoint_name,
        expected_step=args.expected_step,
    )
    manifest = load_json(confirmatory_dir / "eval_manifest.json")
    validate_manifest_integrity(manifest)
    rebuilt_manifest = build_manifest(
        shard_path=VALIDATION_SHARD,
        tokenizer_path=TOKENIZER_FILE,
        eval_seed=args.eval_seed,
        selected_block_count=args.selected_blocks,
        batch_size=args.batch_size,
    )
    if manifest_core(manifest) != manifest_core(rebuilt_manifest):
        raise ValueError("current data no longer reproduces the confirmatory manifest")
    manifest_path = output_dir / "eval_manifest.json"
    if manifest_path.exists():
        if not configs_match(load_json(manifest_path), manifest):
            raise ValueError("existing intervention manifest is inconsistent")
    else:
        atomic_write_json(manifest_path, manifest)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = resolve_precision(args.precision, requested_device)
    environment = environment_info(requested_device, precision)
    environment_hash = environment_fingerprint(environment)
    gpu = environment.get("gpu")
    gpu_uuid = gpu.get("uuid") if isinstance(gpu, dict) else None
    source_identity = source_code_identity()
    checkpoint_hashes = {
        str(spec.seed): sha256_file(spec.checkpoint_path) for spec in specs
    }
    evaluator_path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    integrity_path = (
        ROOT / "att-residual-exp/_associative_intervention_integrity.py"
    ).relative_to(ROOT).as_posix()
    if args.reference_policy == "archived_strict":
        references, confirmatory_identity = load_confirmatory_references(
            confirmatory_dir=confirmatory_dir,
            specs=specs,
            manifest=manifest,
            current_environment=environment,
            current_source_identity=source_identity,
            require_runtime_match=True,
        )
    else:
        references = {}
        confirmatory_identity = validate_recompute_frozen_metadata(
            confirmatory_dir=confirmatory_dir,
            manifest=manifest,
            current_environment=environment,
            current_source_identity=source_identity,
            new_source_paths=(evaluator_path, integrity_path),
            current_checkpoint_sha256_by_seed=checkpoint_hashes,
        )
    analysis_plan = {
        "analysis_kind": ANALYSIS_KIND,
        "status": "exploratory_not_confirmatory",
        "reference_policy": args.reference_policy,
        "reference_source": (
            "fully_recomputed_current_environment"
            if args.reference_policy == "recompute_current_environment"
            else "archived_confirmatory_records"
        ),
        "seeds": list(DEFAULT_SEEDS),
        "novel_modes_in_order": list(NOVEL_MODES),
        "novel_mode_definitions": NOVEL_MODES,
        "reference_modes": REFERENCE_MODES,
        "delta_definition": "intervention mean NLL minus own normal mean NLL",
        "state_update_policy": "unchanged_full_write_in_every_mode",
        "diagnostic_scope_by_mode": {
            mode: expected_diagnostic_scope(mode) for mode in NOVEL_MODES
        },
        "reference_environment_policy": (
            "recompute all normal and gamma-zero references in the current CUDA environment"
            if args.reference_policy == "recompute_current_environment"
            else "require the archived runtime/hardware core and all shared-source hashes"
        ),
        "timing_policy": (
            "wrapped-mode throughput includes the frozen forward plus duplicated writer/read "
            "instrumentation and must not be used for architecture-efficiency comparisons"
        ),
        "reference_anchor_policy": (
            "not used; full normal and gamma-zero references are recomputed"
            if args.reference_policy == "recompute_current_environment"
            else "re-evaluate the first frozen block for normal and gamma-zero in every "
            "seed and require each 512-token sequence NLL sum to match within 1e-5"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    evaluation_identity = {
        "analysis_plan_sha256": object_sha256(analysis_plan),
        "reference_policy": args.reference_policy,
        "gpu_uuid": gpu_uuid,
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256_by_seed": checkpoint_hashes,
        "confirmatory_identity": confirmatory_identity,
        "source_code_combined_sha256": source_identity["combined_sha256"],
        "environment_fingerprint": environment_hash,
        "precision": precision,
        "device": str(requested_device),
        "checkpoint_name": args.checkpoint_name,
        "expected_step": args.expected_step,
    }
    evaluation_sha256 = object_sha256(evaluation_identity)
    run_identity = {
        "schema": "attres-associative-intervention-run-identity-v1",
        "evaluation_identity": evaluation_identity,
        "evaluation_sha256": evaluation_sha256,
    }
    run_identity_path = output_dir / "run_identity.json"
    if run_identity_path.exists():
        if load_json(run_identity_path) != run_identity:
            raise ValueError("existing output directory has a different run identity")
    else:
        atomic_write_json(run_identity_path, run_identity)

    token_ids = map_parameter_golf_tokens(
        VALIDATION_SHARD,
        int(manifest["dataset"]["header"]["token_count"]),
    )
    block_ids = [int(value) for value in manifest["sampling"]["selected_block_ids"]]
    first_source = (
        int(manifest["pool"]["start_token_in_shard"])
        + block_ids[0] * BLOCK_TARGET_TOKENS
    )
    first_starts = [first_source + index * SEQUENCE_LENGTH for index in range(args.batch_size)]
    preflight_x, _ = make_cpu_batch(token_ids, first_starts)

    reference_anchor_rows: list[dict[str, Any]] = []
    reference_preflight_rows: list[dict[str, Any]] = []
    for spec in specs:
        if args.reference_policy == "recompute_current_environment":
            for reference_mode in REFERENCE_MODES:
                reference_model = None
                cuda_cleanup(requested_device)
                try:
                    reference_model, checkpoint_sha256, metadata = (
                        _safe_load_checkpoint_model(
                            spec,
                            expected_step=args.expected_step,
                            known_checkpoint_sha256=checkpoint_hashes[str(spec.seed)],
                        )
                    )
                    if any(
                        hasattr(transition, "_intervention_original_forward")
                        for transition in reference_model.transitions
                    ):
                        raise RuntimeError("fresh reference preflight contains a wrapper")
                    transition_count: int | None = None
                    if reference_mode == "memory_output_off":
                        transition_count = disable_memory_reads(reference_model)
                    reference_model = reference_model.to(requested_device).eval()
                    x = preflight_x.to(requested_device)
                    with torch.inference_mode(), autocast_context(
                        requested_device, precision
                    ):
                        logits, diagnostics = reference_model(
                            x, return_diagnostics=True
                        )
                    expected_shape = (
                        len(first_starts),
                        SEQUENCE_LENGTH,
                        reference_model.config.vocab_size,
                    )
                    if tuple(logits.shape) != expected_shape:
                        raise ValueError(
                            "unexpected local-reference preflight logit shape"
                        )
                    if not bool(torch.isfinite(logits).all().item()) or any(
                        not math.isfinite(float(value))
                        for value in diagnostics.values()
                    ):
                        raise FloatingPointError(
                            "local-reference preflight is non-finite"
                        )
                    if reference_mode == "memory_output_off":
                        for key in ("memory_gamma", "memory_gamma_abs"):
                            if not math.isclose(
                                float(diagnostics.get(key, float("nan"))),
                                0.0,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            ):
                                raise ValueError(
                                    f"local-reference preflight has nonzero {key}"
                                )
                    reference_preflight_rows.append(
                        {
                            "seed": spec.seed,
                            "mode": reference_mode,
                            "checkpoint_sha256": checkpoint_sha256,
                            "checkpoint_step": metadata["step"],
                            "transition_count": transition_count,
                            "forward_shape": list(logits.shape),
                            "status": "strict_load_and_reference_forward_passed",
                        }
                    )
                    del x, logits, diagnostics
                finally:
                    if reference_model is not None:
                        del reference_model
                    cuda_cleanup(requested_device)
            continue

        model = None
        cuda_cleanup(requested_device)
        try:
            model, checkpoint_sha256, metadata = _safe_load_checkpoint_model(
                spec,
                expected_step=args.expected_step,
                known_checkpoint_sha256=checkpoint_hashes[str(spec.seed)],
            )
            normal_anchor = evaluate_model(
                model=model,
                token_ids=token_ids,
                block_ids=(block_ids[0],),
                batch_size=args.batch_size,
                device=requested_device,
                precision=precision,
                progress_every_blocks=0,
                label=f"{ASSOCIATIVE_ARCHITECTURE}/seed{spec.seed}/normal_anchor",
            )
            normal_check = validate_reference_anchor(
                current=normal_anchor,
                archived=references[spec.seed]["normal"],
                block_id=block_ids[0],
                label=f"seed{spec.seed}/normal",
            )
            transition_count = disable_memory_reads(model)
            memory_off_anchor = evaluate_model(
                model=model,
                token_ids=token_ids,
                block_ids=(block_ids[0],),
                batch_size=args.batch_size,
                device=requested_device,
                precision=precision,
                progress_every_blocks=0,
                label=f"{ASSOCIATIVE_ARCHITECTURE}/seed{spec.seed}/memory_off_anchor",
            )
            memory_off_check = validate_reference_anchor(
                current=memory_off_anchor,
                archived=references[spec.seed]["memory_output_off"],
                block_id=block_ids[0],
                label=f"seed{spec.seed}/memory_output_off",
            )
            reference_anchor_rows.append(
                {
                    "seed": spec.seed,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_step": metadata["step"],
                    "memory_off_transition_count": transition_count,
                    "normal": normal_check,
                    "memory_output_off": memory_off_check,
                    "status": "normal_and_memory_off_archived_references_reproduced",
                }
            )
            del normal_anchor, memory_off_anchor
        finally:
            if model is not None:
                del model
            cuda_cleanup(requested_device)

    preflight_rows: list[dict[str, Any]] = []
    for spec in specs:
        for mode in NOVEL_MODES:
            model = None
            cuda_cleanup(requested_device)
            try:
                model, checkpoint_sha256, metadata = _safe_load_checkpoint_model(
                    spec,
                    expected_step=args.expected_step,
                    known_checkpoint_sha256=checkpoint_hashes[str(spec.seed)],
                )
                before_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                scales = install_intervention(model, mode)
                after_state = model.state_dict()
                if tuple(after_state) != tuple(before_state) or any(
                    not torch.equal(after_state[name], value)
                    for name, value in before_state.items()
                ):
                    raise RuntimeError("installing intervention changed checkpoint state")
                if scales != expected_transition_scales(mode, len(model.transitions)):
                    raise RuntimeError("installed transition scales differ from the plan")
                del before_state, after_state
                model = model.to(requested_device).eval()
                x = preflight_x.to(requested_device)
                with torch.inference_mode(), autocast_context(requested_device, precision):
                    logits, diagnostics = model(x, return_diagnostics=True)
                expected_shape = (
                    len(first_starts),
                    SEQUENCE_LENGTH,
                    model.config.vocab_size,
                )
                if tuple(logits.shape) != expected_shape:
                    raise ValueError(
                        f"unexpected intervention logit shape: {tuple(logits.shape)}"
                    )
                if not bool(torch.isfinite(logits).all().item()):
                    raise FloatingPointError("intervention preflight produced non-finite logits")
                if not isinstance(diagnostics, dict) or not diagnostics or any(
                    not math.isfinite(float(value)) for value in diagnostics.values()
                ):
                    raise FloatingPointError(
                        "intervention diagnostics are missing or non-finite"
                    )
                preflight_rows.append(
                    {
                        "seed": spec.seed,
                        "mode": mode,
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_step": metadata["step"],
                        "transition_scales": scales,
                        "diagnostic_scope": expected_diagnostic_scope(mode),
                        "forward_shape": list(logits.shape),
                        "status": "strict_load_and_intervened_forward_passed",
                    }
                )
                del x, logits, diagnostics
            finally:
                if model is not None:
                    del model
                cuda_cleanup(requested_device)

    full_reference_evaluations = (
        len(specs) * len(REFERENCE_MODES)
        if args.reference_policy == "recompute_current_environment"
        else 0
    )
    reference_anchor_target_tokens = (
        0
        if args.reference_policy == "recompute_current_environment"
        else BLOCK_TARGET_TOKENS * len(specs) * len(REFERENCE_MODES)
    )
    targets_per_evaluation = int(manifest["sampling"]["total_target_tokens"])
    preflight = {
        "type": "associative_intervention_preflight",
        "status": "passed",
        "analysis_plan": analysis_plan,
        "evaluation_identity": evaluation_identity,
        "evaluation_sha256": evaluation_sha256,
        "environment": environment,
        "source_code_identity": source_identity,
        "confirmatory_identity": confirmatory_identity,
        "target_tokens_per_novel_evaluation": manifest["sampling"]["total_target_tokens"],
        "total_novel_evaluations": len(specs) * len(NOVEL_MODES),
        "total_full_reference_evaluations": full_reference_evaluations,
        "total_full_reference_target_tokens": (
            targets_per_evaluation * full_reference_evaluations
        ),
        "total_full_evaluation_target_tokens": (
            targets_per_evaluation
            * (len(specs) * len(NOVEL_MODES) + full_reference_evaluations)
        ),
        "reference_anchor_target_tokens": reference_anchor_target_tokens,
        "total_scored_target_tokens_including_anchors": (
            targets_per_evaluation
            * (len(specs) * len(NOVEL_MODES) + full_reference_evaluations)
            + reference_anchor_target_tokens
        ),
        "reference_preflight_rows": reference_preflight_rows,
        "reference_anchor_rows": reference_anchor_rows,
        "rows": preflight_rows,
    }
    atomic_write_json(output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return

    if args.reference_policy == "recompute_current_environment":
        references, reference_identity = evaluate_local_references(
            specs=specs,
            manifest=manifest,
            evaluation_sha256=evaluation_sha256,
            environment_hash=environment_hash,
            gpu_uuid=gpu_uuid,
            checkpoint_hashes=checkpoint_hashes,
            expected_step=args.expected_step,
            token_ids=token_ids,
            block_ids=block_ids,
            batch_size=args.batch_size,
            device=requested_device,
            precision=precision,
            progress_every_blocks=args.progress_every_blocks,
            output_dir=output_dir,
            resume=args.resume,
        )
    else:
        reference_identity = {
            "source": "archived_confirmatory_records",
            "reference_policy": "archived_strict",
            "environment_fingerprint": confirmatory_identity[
                "preflight_environment_fingerprint"
            ],
            "checkpoint_result_sha256_by_seed": confirmatory_identity[
                "checkpoint_result_sha256_by_seed"
            ],
        }

    results_dir = output_dir / "mode_results"
    records: list[dict[str, Any]] = []
    failures = 0
    for spec in specs:
        for mode, definition in NOVEL_MODES.items():
            result_path = results_dir / f"seed{spec.seed}" / f"{mode}.json"
            checkpoint_sha256 = checkpoint_hashes[str(spec.seed)]
            transition_count = 2 * int(spec.run_config["model"]["num_layers"]) - 1
            if result_path.exists():
                existing = load_json(result_path)
                if args.resume and existing.get("status") == "complete":
                    validate_intervention_result(
                        existing,
                        manifest=manifest,
                        evaluation_sha256=evaluation_sha256,
                        checkpoint_sha256=checkpoint_sha256,
                        seed=spec.seed,
                        mode=mode,
                        run_name=spec.run_name,
                        checkpoint_step=args.expected_step,
                        environment_hash=environment_hash,
                        transition_count=transition_count,
                        reference_policy=args.reference_policy,
                        gpu_uuid=gpu_uuid,
                    )
                    print(
                        json.dumps({"type": "eval_skip", "seed": spec.seed, "mode": mode}),
                        flush=True,
                    )
                    records.append(existing)
                    continue
                if existing.get("status") == "complete":
                    raise FileExistsError(
                        f"completed result exists at {result_path}; use --resume or a new output"
                    )

            model = None
            started_utc = utc_now()
            cuda_cleanup(requested_device)
            try:
                model, loaded_sha256, metadata = _safe_load_checkpoint_model(
                    spec,
                    expected_step=args.expected_step,
                    known_checkpoint_sha256=checkpoint_sha256,
                )
                if loaded_sha256 != checkpoint_sha256:
                    raise RuntimeError("checkpoint changed while being loaded")
                transition_scales = install_intervention(model, mode)
                metrics = evaluate_model(
                    model=model,
                    token_ids=token_ids,
                    block_ids=block_ids,
                    batch_size=args.batch_size,
                    device=requested_device,
                    precision=precision,
                    progress_every_blocks=args.progress_every_blocks,
                    label=f"{ASSOCIATIVE_ARCHITECTURE}/seed{spec.seed}/{mode}",
                )
                metrics["timing_scope"] = (
                    str(metrics["timing_scope"])
                    + "; intervention wrapper duplicates writer/query/read computations for "
                    "measurement and is not an architecture-efficiency benchmark"
                )
                validate_metric_payload(metrics, manifest)
                record = {
                    "schema": RESULT_SCHEMA,
                    "status": "complete",
                    "analysis_kind": ANALYSIS_KIND,
                    "started_utc": started_utc,
                    "completed_utc": utc_now(),
                    "architecture": ASSOCIATIVE_ARCHITECTURE,
                    "seed": spec.seed,
                    "run_name": spec.run_name,
                    "mode": mode,
                    "mode_definition": definition,
                    "transition_scales": transition_scales,
                    "diagnostic_scope": expected_diagnostic_scope(mode),
                    "checkpoint_path": str(spec.checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_step": metadata["step"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "evaluation_sha256": evaluation_sha256,
                    "environment_fingerprint": environment_hash,
                    "reference_policy": args.reference_policy,
                    "gpu_uuid": gpu_uuid,
                    "metrics": metrics,
                }
                validate_intervention_result(
                    record,
                    manifest=manifest,
                    evaluation_sha256=evaluation_sha256,
                    checkpoint_sha256=checkpoint_sha256,
                    seed=spec.seed,
                    mode=mode,
                    run_name=spec.run_name,
                    checkpoint_step=args.expected_step,
                    environment_hash=environment_hash,
                    transition_count=transition_count,
                    reference_policy=args.reference_policy,
                    gpu_uuid=gpu_uuid,
                )
                atomic_write_json(result_path, record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "type": "eval_complete",
                            "seed": spec.seed,
                            "mode": mode,
                            "mean_nll": metrics["mean_nll"],
                            "mode_minus_normal_nll": (
                                float(metrics["mean_nll"])
                                - float(references[spec.seed]["normal"]["mean_nll"])
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
                    "analysis_kind": ANALYSIS_KIND,
                    "seed": spec.seed,
                    "mode": mode,
                    "failed_utc": utc_now(),
                    "checkpoint_sha256": checkpoint_sha256,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "evaluation_sha256": evaluation_sha256,
                    "reference_policy": args.reference_policy,
                    "gpu_uuid": gpu_uuid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(result_path, failure)
                print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
            finally:
                if model is not None:
                    del model
                cuda_cleanup(requested_device)

    if failures:
        raise RuntimeError(f"{failures} intervention evaluations failed")
    summary = summarize(
        records=records,
        references=references,
        manifest=manifest,
        evaluation_sha256=evaluation_sha256,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        source_identity=source_identity,
        confirmatory_identity=confirmatory_identity,
        reference_policy=args.reference_policy,
        reference_identity=reference_identity,
        environment=environment,
        gpu_uuid=gpu_uuid,
    )
    atomic_write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "results.csv", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
