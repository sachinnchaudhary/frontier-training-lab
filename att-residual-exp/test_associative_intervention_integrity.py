"""Equation and archive-integrity checks for the intervention evaluator."""

from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path

import torch

from _associative_intervention_integrity import validate_confirmatory_bundle
from evaluate_associative_interventions import (
    install_intervention,
    source_code_identity,
    validate_reference_anchor,
)
from evaluate_checkpoints import (
    ASSOCIATIVE_ARCHITECTURE,
    COMPARISON_SCHEMA,
    DEFAULT_SEEDS,
    PROPOSAL_ARCHITECTURE,
    environment_fingerprint,
    object_sha256,
)
from test_associative_interventions import make_model


@torch.no_grad()
def check_exact_gated_delta_read_equation() -> None:
    torch.manual_seed(117)
    base = make_model()
    transition = base.transitions[0]
    hidden = torch.randn(2, 5, base.config.dim)
    delta = torch.randn_like(hidden)
    state = torch.randn(2, 5, base.num_slots, base.memory_dim)

    key, value, alpha, beta = transition.write_parameters(delta)
    decayed = alpha.unsqueeze(-1) * state
    prediction = torch.einsum("btsr,bts->btr", decayed, key)
    innovation = value - prediction
    correction = (
        beta.unsqueeze(-1) * key.unsqueeze(-1) * innovation.unsqueeze(-2)
    )
    expected_state = decayed + correction
    query = transition.read_query(hidden)
    historical = transition.read_scale * torch.einsum(
        "btsr,bts->btr", decayed, query
    )
    current = transition.read_scale * torch.einsum(
        "btsr,bts->btr", correction, query
    )
    gamma = transition.gamma.to(hidden.dtype)
    expected_history = hidden + gamma * transition.out_proj(historical)
    expected_current = hidden + gamma * transition.out_proj(current)

    for mode, expected_hidden in (
        ("history_only", expected_history),
        ("current_correction_only", expected_current),
    ):
        model = copy.deepcopy(base)
        install_intervention(model, mode)
        actual_hidden, actual_state, diagnostics = model.transitions[0](
            hidden, delta, state, True
        )
        torch.testing.assert_close(actual_state, expected_state, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actual_hidden, expected_hidden, rtol=1e-6, atol=1e-7
        )
        if float(diagnostics["memory_history_read_scale"]) != (
            1.0 if mode == "history_only" else 0.0
        ):
            raise AssertionError("history diagnostic scale is wrong")
        if float(diagnostics["memory_current_read_scale"]) != (
            0.0 if mode == "history_only" else 1.0
        ):
            raise AssertionError("current diagnostic scale is wrong")
        if any(
            not math.isfinite(float(value)) for value in diagnostics.values()
        ):
            raise AssertionError("transition diagnostics contain a non-finite value")


@torch.no_grad()
def check_zero_state_closed_form() -> None:
    torch.manual_seed(119)
    base = make_model()
    transition = base.transitions[0]
    hidden = torch.randn(2, 5, base.config.dim)
    delta = torch.randn_like(hidden)
    state = torch.zeros(2, 5, base.num_slots, base.memory_dim)
    key, value, _, beta = transition.write_parameters(delta)
    query = transition.read_query(hidden)
    key_query = (key * query).sum(dim=-1)
    closed_form = (
        transition.read_scale
        * (beta.squeeze(-1) * key_query).unsqueeze(-1)
        * value
    )
    expected = hidden + transition.gamma.to(hidden.dtype) * transition.out_proj(
        closed_form
    )
    model = copy.deepcopy(base)
    install_intervention(model, "current_correction_only")
    actual, _ = model.transitions[0](hidden, delta, state)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def check_reference_anchor_validation() -> None:
    archived = {
        "blocks": [
            {"block_id": 7, "sequence_nll_sums": [101.0, 102.0, 103.0]}
        ]
    }
    current = {
        "blocks": [
            {
                "block_id": 7,
                "sequence_nll_sums": [101.0, 102.0 + 1e-6, 103.0],
            }
        ]
    }
    result = validate_reference_anchor(
        current=current,
        archived=archived,
        block_id=7,
        label="synthetic",
    )
    if result["status"] != "archived_reference_numerically_reproduced":
        raise AssertionError("valid numerical anchor was rejected")
    tampered = copy.deepcopy(current)
    tampered["blocks"][0]["sequence_nll_sums"][1] += 1e-3
    try:
        validate_reference_anchor(
            current=tampered,
            archived=archived,
            block_id=7,
            label="tampered",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("drifted numerical anchor was accepted")


def check_confirmatory_bundle_links() -> None:
    current_source = source_code_identity()
    evaluator_path = "att-residual-exp/evaluate_associative_interventions.py"
    integrity_path = "att-residual-exp/_associative_intervention_integrity.py"
    archived_files = {
        name: digest
        for name, digest in current_source["files"].items()
        if name not in {evaluator_path, integrity_path}
    }
    archived_source = {
        "files": archived_files,
        "combined_sha256": object_sha256(archived_files),
    }
    current_environment = {
        "python": "3.12.0",
        "platform": "synthetic-linux",
        "torch": "2.7.0",
        "numpy": "2.0.0",
        "cuda_runtime": "12.8",
        "cudnn": 90100,
        "device": "cuda",
        "precision": "bf16",
        "git_commit": "new-evaluator-commit",
        "git_status_porcelain": "",
        "gpu": {
            "name": "Synthetic GPU",
            "compute_capability": [10, 0],
            "total_memory_bytes": 96 * 2**30,
            "uuid": "new-physical-device",
        },
    }
    archived_environment = copy.deepcopy(current_environment)
    archived_environment["git_commit"] = "confirmatory-commit"
    archived_environment["gpu"]["uuid"] = "old-physical-device"
    analysis_plan = {"comparison": "reader", "memory_off_intervention": True}
    checkpoint_hashes = {
        f"{architecture}/seed{seed}": f"hash-{architecture}-{seed}"
        for architecture in (PROPOSAL_ARCHITECTURE, ASSOCIATIVE_ARCHITECTURE)
        for seed in DEFAULT_SEEDS
    }
    checkpoint_rows = [
        {
            "architecture": architecture,
            "seed": seed,
            "checkpoint_sha256": checkpoint_hashes[
                f"{architecture}/seed{seed}"
            ],
        }
        for architecture in (PROPOSAL_ARCHITECTURE, ASSOCIATIVE_ARCHITECTURE)
        for seed in DEFAULT_SEEDS
    ]
    preflight = {
        "type": "common_eval_preflight",
        "status": "strict_checkpoint_load_and_forward_passed",
        "manifest_sha256": "manifest-hash",
        "evaluation_sha256": "evaluation-hash",
        "environment": archived_environment,
        "environment_fingerprint": environment_fingerprint(archived_environment),
        "source_code_identity": archived_source,
        "analysis_plan": analysis_plan,
        "analysis_plan_sha256": object_sha256(analysis_plan),
        "checkpoint_validation": checkpoint_rows,
    }
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "manifest_sha256": "manifest-hash",
        "evaluation_sha256": "evaluation-hash",
        "environment_fingerprint": preflight["environment_fingerprint"],
        "source_code_combined_sha256": archived_source["combined_sha256"],
        "analysis_plan_sha256": preflight["analysis_plan_sha256"],
        "control_architecture": PROPOSAL_ARCHITECTURE,
        "candidate_architecture": ASSOCIATIVE_ARCHITECTURE,
        "checkpoint_sha256_by_architecture_and_seed": checkpoint_hashes,
    }
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "preflight.json").write_text(
            json.dumps(preflight), encoding="utf-8"
        )
        identity = validate_confirmatory_bundle(
            confirmatory_dir=directory,
            comparison=comparison,
            manifest={"manifest_sha256": "manifest-hash"},
            current_environment=current_environment,
            current_source_identity=current_source,
            new_source_paths=(evaluator_path, integrity_path),
        )
        if not identity.get("preflight_sha256"):
            raise AssertionError("confirmatory preflight hash was not retained")

        tampered_comparison = copy.deepcopy(comparison)
        tampered_comparison["evaluation_sha256"] = "tampered"
        try:
            validate_confirmatory_bundle(
                confirmatory_dir=directory,
                comparison=tampered_comparison,
                manifest={"manifest_sha256": "manifest-hash"},
                current_environment=current_environment,
                current_source_identity=current_source,
                new_source_paths=(evaluator_path, integrity_path),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("comparison/preflight hash mismatch was accepted")

        changed_runtime = copy.deepcopy(current_environment)
        changed_runtime["precision"] = "fp32"
        try:
            validate_confirmatory_bundle(
                confirmatory_dir=directory,
                comparison=comparison,
                manifest={"manifest_sha256": "manifest-hash"},
                current_environment=changed_runtime,
                current_source_identity=current_source,
                new_source_paths=(evaluator_path, integrity_path),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("incompatible runtime/precision was accepted")


def main() -> None:
    check_exact_gated_delta_read_equation()
    check_zero_state_closed_form()
    check_reference_anchor_validation()
    check_confirmatory_bundle_links()
    print(
        {
            "test": "associative_intervention_integrity",
            "status": "passed",
            "checks": [
                "exact_gated_delta_history_current_equations",
                "zero_state_current_read_closed_form",
                "archived_numerical_anchor_validation",
                "confirmatory_preflight_source_runtime_checkpoint_linkage",
                "tamper_and_incompatible_precision_rejection",
            ],
        }
    )


if __name__ == "__main__":
    main()
