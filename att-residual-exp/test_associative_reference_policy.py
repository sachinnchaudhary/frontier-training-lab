"""CPU-only identity checks for same-environment reference recomputation."""

from __future__ import annotations

import copy
from types import SimpleNamespace

from evaluate_associative_interventions import (
    ANALYSIS_KIND,
    REFERENCE_MODES,
    REFERENCE_RESULT_SCHEMA,
    validate_local_reference_result,
)
from evaluate_checkpoints import ASSOCIATIVE_ARCHITECTURE
from test_evaluator import fake_manifest, fake_record


def make_metrics(manifest: dict, *, memory_off: bool) -> dict:
    record = fake_record(ASSOCIATIVE_ARCHITECTURE, 1337, 2.7, manifest)
    keys = (
        "target_token_count",
        "nll_sum",
        "mean_nll",
        "perplexity",
        "eval_seconds",
        "target_tokens_per_second",
        "diagnostics_first_batch",
        "blocks",
    )
    metrics = {key: copy.deepcopy(record[key]) for key in keys}
    metrics["timing_scope"] = "synthetic same-environment reference"
    metrics["cuda_memory"] = {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if memory_off:
        metrics["diagnostics_first_batch"].update(
            {"memory_gamma": 0.0, "memory_gamma_abs": 0.0}
        )
    return metrics


def make_record(mode: str, manifest: dict) -> tuple[dict, SimpleNamespace]:
    spec = SimpleNamespace(
        seed=1337,
        run_name="full_100m_seed1337",
        run_config={"model": {"num_layers": 12}},
    )
    is_off = mode == "memory_output_off"
    record = {
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
        "intervention": (
            "all_depth_memory_output_gammas_set_to_zero" if is_off else "none"
        ),
        "transition_count": 23 if is_off else None,
        "checkpoint_sha256": "c" * 64,
        "checkpoint_step": 12_208,
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_sha256": "e" * 64,
        "environment_fingerprint": "f" * 64,
        "gpu_uuid": "GPU-synthetic-4090",
        "metrics": make_metrics(manifest, memory_off=is_off),
    }
    return record, spec


def validate(record: dict, spec: SimpleNamespace, manifest: dict, mode: str) -> None:
    validate_local_reference_result(
        record,
        mode=mode,
        spec=spec,
        manifest=manifest,
        evaluation_sha256="e" * 64,
        environment_hash="f" * 64,
        gpu_uuid="GPU-synthetic-4090",
        checkpoint_sha256="c" * 64,
        expected_step=12_208,
    )


def check_valid_reference_modes() -> None:
    manifest = fake_manifest()
    for mode in REFERENCE_MODES:
        record, spec = make_record(mode, manifest)
        validate(record, spec, manifest, mode)


def check_identity_and_gamma_tampering_rejected() -> None:
    manifest = fake_manifest()
    record, spec = make_record("memory_output_off", manifest)
    mutations = (
        ("reference_policy", "archived_strict"),
        ("gpu_uuid", "GPU-other"),
        ("transition_count", 22),
    )
    for key, value in mutations:
        tampered = copy.deepcopy(record)
        tampered[key] = value
        try:
            validate(tampered, spec, manifest, "memory_output_off")
        except ValueError:
            pass
        else:
            raise AssertionError(f"tampered local reference {key} was accepted")

    tampered = copy.deepcopy(record)
    tampered["metrics"]["diagnostics_first_batch"]["memory_gamma"] = 0.01
    try:
        validate(tampered, spec, manifest, "memory_output_off")
    except ValueError:
        pass
    else:
        raise AssertionError("nonzero gamma in memory-off reference was accepted")


def main() -> None:
    check_valid_reference_modes()
    check_identity_and_gamma_tampering_rejected()
    print(
        {
            "test": "associative_reference_policy",
            "status": "passed",
            "checks": [
                "normal_and_gamma_zero_reference_validation",
                "policy_gpu_transition_and_gamma_tamper_rejection",
            ],
        }
    )


if __name__ == "__main__":
    main()
