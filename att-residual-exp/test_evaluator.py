"""Focused CPU checks for ``evaluate_checkpoints.py``.

Run from the repository root:

    python att-residual-exp/test_evaluator.py

The checks use tiny synthetic checkpoints and never touch real run artifacts.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from _common import ModelConfig, count_parameters
from associative_read_depth_kda import AssociativeReadDepthKDALM
from attention_residual_baseline import FullAttentionResidualLM
from evaluate_checkpoints import (
    ASSOCIATIVE_ARCHITECTURE,
    BASELINE_ARCHITECTURE,
    BLOCK_TARGET_TOKENS,
    EXPECTED_DEFAULT_BLOCK_IDS_SHA256,
    EXPECTED_POOL_SHA256,
    EXPECTED_SHARD_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    MANIFEST_SCHEMA,
    POOL_START_TOKEN,
    PROPOSAL_ARCHITECTURE,
    RESULT_SCHEMA,
    SEQUENCE_LENGTH,
    CheckpointSpec,
    block_ids_sha256,
    construct_model,
    disable_memory_reads,
    evaluate_model,
    load_checkpoint_model,
    normalize_state_dict,
    object_sha256,
    select_block_ids,
    summarize_comparison,
    validate_completed_result,
    validate_manifest_integrity,
)
from softmax_read_depth_kda import SoftmaxReadGatedDeltaDepthMemoryLM


def baseline_config(model_config: ModelConfig, model: torch.nn.Module, seed: int) -> dict:
    return {
        "type": "run_config",
        "architecture": BASELINE_ARCHITECTURE,
        "mode": "full",
        "seed": seed,
        "dataset_name": "parameter_golf_sp1024",
        "batch_size": 2,
        "grad_accum_steps": 1,
        "seq_len": model_config.max_seq_len,
        "max_steps": 2,
        "run_name": f"synthetic_seed{seed}",
        "model": asdict(model_config),
        "model_parameters": count_parameters(model),
    }


def proposal_config(model_config: ModelConfig, model: torch.nn.Module, seed: int) -> dict:
    config = baseline_config(model_config, model, seed)
    config.update(
        {
            "architecture": PROPOSAL_ARCHITECTURE,
            "num_slots": 3,
            "memory_dim": 7,
            "read_key_dim": 5,
            "read_value_dim": 6,
            "alpha_bias": -3.7,
            "beta_bias": -1.4,
            "gamma_init": 0.002,
            "reference_attnres_ffn_hidden_dim": 999,
            "matched_ffn_hidden_dim": model_config.ffn_hidden_dim,
        }
    )
    return config


def associative_config(
    model_config: ModelConfig,
    model: torch.nn.Module,
    seed: int,
) -> dict:
    config = baseline_config(model_config, model, seed)
    config.update(
        {
            "architecture": ASSOCIATIVE_ARCHITECTURE,
            "num_slots": 3,
            "memory_dim": 7,
            "alpha_bias": -3.7,
            "beta_bias": -1.4,
            "gamma_init": 0.002,
        }
    )
    return config


def write_checkpoint(
    root: Path,
    run_config: dict,
    state_dict: dict[str, torch.Tensor],
) -> CheckpointSpec:
    architecture = str(run_config["architecture"])
    seed = int(run_config["seed"])
    run_name = str(run_config["run_name"])
    run_dir = root / architecture / run_name
    checkpoint_path = run_dir / "checkpoints/latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    torch.save(
        {
            "step": 2,
            "model_state_dict": state_dict,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "run_config": run_config,
            "best_val_loss": 1.0,
            "best_step": 2,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": None,
        },
        checkpoint_path,
    )
    return CheckpointSpec(
        architecture=architecture,
        seed=seed,
        run_name=run_name,
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        run_config=run_config,
    )


def assert_roundtrip(
    root: Path,
    model: torch.nn.Module,
    run_config: dict,
    token_ids: torch.Tensor,
    *,
    compiled_prefix: bool = False,
) -> None:
    model.eval()
    with torch.no_grad():
        expected_logits, expected_diagnostics = model(token_ids, return_diagnostics=True)
    state_dict = dict(model.state_dict())
    if compiled_prefix:
        state_dict = {f"_orig_mod.{key}": value for key, value in state_dict.items()}
    spec = write_checkpoint(root, run_config, state_dict)
    restored, checkpoint_hash, metadata = load_checkpoint_model(spec, expected_step=2)
    restored.eval()
    with torch.no_grad():
        actual_logits, actual_diagnostics = restored(token_ids, return_diagnostics=True)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0.0, atol=0.0)
    if actual_diagnostics.keys() != expected_diagnostics.keys():
        raise AssertionError("diagnostic keys changed during checkpoint reconstruction")
    for key in actual_diagnostics:
        if actual_diagnostics[key] != expected_diagnostics[key]:
            raise AssertionError(f"diagnostic {key} changed during checkpoint reconstruction")
    if len(checkpoint_hash) != 64:
        raise AssertionError("checkpoint SHA-256 was not recorded")
    if metadata["model_parameters"] != count_parameters(model):
        raise AssertionError("checkpoint reconstruction changed parameter count")
    if any(parameter.device.type != "cpu" for parameter in restored.parameters()):
        raise AssertionError("checkpoint reconstruction unexpectedly moved parameters off CPU")


def check_checkpoint_roundtrips() -> None:
    torch.manual_seed(91)
    token_ids = torch.randint(0, 64, (2, 16))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        baseline_model_config = ModelConfig(
            vocab_size=64,
            dim=32,
            num_layers=2,
            num_heads=4,
            ffn_hidden_dim=48,
            max_seq_len=16,
        )
        baseline = FullAttentionResidualLM(baseline_model_config)
        baseline_run_config = baseline_config(baseline_model_config, baseline, seed=11)
        assert_roundtrip(root, baseline, baseline_run_config, token_ids)

        proposal_model_config = ModelConfig(
            vocab_size=64,
            dim=32,
            num_layers=2,
            num_heads=4,
            ffn_hidden_dim=37,
            max_seq_len=16,
        )
        proposal = SoftmaxReadGatedDeltaDepthMemoryLM(
            proposal_model_config,
            num_slots=3,
            memory_dim=7,
            read_key_dim=5,
            read_value_dim=6,
            alpha_bias=-3.7,
            beta_bias=-1.4,
            gamma_init=0.002,
        )
        proposal_run_config = proposal_config(proposal_model_config, proposal, seed=12)
        assert_roundtrip(
            root,
            proposal,
            proposal_run_config,
            token_ids,
            compiled_prefix=True,
        )

        associative = AssociativeReadDepthKDALM(
            proposal_model_config,
            num_slots=3,
            memory_dim=7,
            alpha_bias=-3.7,
            beta_bias=-1.4,
            gamma_init=0.002,
        )
        associative_run_config = associative_config(
            proposal_model_config,
            associative,
            seed=13,
        )
        assert_roundtrip(
            root,
            associative,
            associative_run_config,
            token_ids,
        )
        transition_count = disable_memory_reads(associative)
        if transition_count != 2 * proposal_model_config.num_layers - 1:
            raise AssertionError("memory-off intervention changed the wrong transition count")
        if any(transition.gamma.item() != 0.0 for transition in associative.transitions):
            raise AssertionError("memory-off intervention left a nonzero gamma")


def check_fail_closed_behavior() -> None:
    model_config = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_dim=24,
        max_seq_len=8,
    )
    baseline = FullAttentionResidualLM(model_config)
    config = baseline_config(model_config, baseline, seed=1)

    unknown = copy.deepcopy(config)
    unknown["architecture"] = "unknown_architecture"
    try:
        construct_model(unknown)
    except ValueError as exc:
        if "unsupported" not in str(exc):
            raise
    else:
        raise AssertionError("unknown architecture was accepted")

    incomplete_proposal = copy.deepcopy(config)
    incomplete_proposal["architecture"] = PROPOSAL_ARCHITECTURE
    try:
        construct_model(incomplete_proposal)
    except ValueError as exc:
        if "missing reconstruction fields" not in str(exc):
            raise
    else:
        raise AssertionError("proposal with missing reconstruction fields was accepted")

    state_dict = dict(baseline.state_dict())
    first_key = next(iter(state_dict))
    mixed = {f"_orig_mod.{first_key}": state_dict[first_key], **state_dict}
    try:
        normalize_state_dict(mixed)
    except ValueError as exc:
        if "mixture" not in str(exc):
            raise
    else:
        raise AssertionError("mixed compiled/uncompiled state dict was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        broken_state = dict(state_dict)
        del broken_state[first_key]
        spec = write_checkpoint(Path(temporary), config, broken_state)
        try:
            load_checkpoint_model(spec, expected_step=2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("strict loading accepted a missing state-dict key")


def check_manifest_selection() -> None:
    candidate_count = (10_000_000 - 1) // BLOCK_TARGET_TOKENS
    first = select_block_ids(candidate_count, 256, 424_242, EXPECTED_POOL_SHA256)
    second = select_block_ids(candidate_count, 256, 424_242, EXPECTED_POOL_SHA256)
    if first != second:
        raise AssertionError("manifest block selection is not deterministic")
    if first != sorted(set(first)):
        raise AssertionError("manifest block IDs are duplicated or unsorted")
    if block_ids_sha256(first) != EXPECTED_DEFAULT_BLOCK_IDS_SHA256:
        raise AssertionError("manifest block selection no longer matches its frozen hash")
    target_ranges = [
        (block_id * BLOCK_TARGET_TOKENS + 1, (block_id + 1) * BLOCK_TARGET_TOKENS)
        for block_id in first
    ]
    for previous, current in zip(target_ranges, target_ranges[1:]):
        if previous[1] >= current[0]:
            raise AssertionError("selected scoring blocks overlap in target positions")


def fake_manifest(block_count: int = 256) -> dict:
    block_ids = list(range(block_count))
    core = {
        "schema": MANIFEST_SCHEMA,
        "dataset": {
            "shard_sha256": EXPECTED_SHARD_SHA256,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        },
        "pool": {"raw_uint16_le_sha256": EXPECTED_POOL_SHA256},
        "sampling": {
            "selected_block_ids": block_ids,
            "selected_block_count": len(block_ids),
            "selected_block_ids_uint32_le_sha256": block_ids_sha256(block_ids),
            "total_target_tokens": len(block_ids) * BLOCK_TARGET_TOKENS,
            "candidate_block_count": 610,
        },
    }
    return {
        **core,
        "manifest_sha256": object_sha256(core),
        "created_utc": "2026-07-28T00:00:00+00:00",
    }


def fake_record(architecture: str, seed: int, loss: float, manifest: dict) -> dict:
    blocks = []
    for block_id in range(256):
        block_loss = loss + (block_id - 127.5) * 1e-6
        source_start = POOL_START_TOKEN + block_id * BLOCK_TARGET_TOKENS
        sequence_sums = [block_loss * SEQUENCE_LENGTH] * (
            BLOCK_TARGET_TOKENS // SEQUENCE_LENGTH
        )
        block_nll_sum = sum(sequence_sums)
        blocks.append(
            {
                "block_id": block_id,
                "source_start_token": source_start,
                "sequence_starts": [
                    source_start + index * SEQUENCE_LENGTH
                    for index in range(BLOCK_TARGET_TOKENS // SEQUENCE_LENGTH)
                ],
                "sequence_nll_sums": sequence_sums,
                "token_count": BLOCK_TARGET_TOKENS,
                "nll_sum": block_nll_sum,
                "mean_nll": block_nll_sum / BLOCK_TARGET_TOKENS,
                "nonfinite_count": 0,
            }
        )
    total_nll = sum(float(block["nll_sum"]) for block in blocks)
    total_tokens = len(blocks) * BLOCK_TARGET_TOKENS
    mean_nll = total_nll / total_tokens
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "architecture": architecture,
        "seed": seed,
        "run_name": f"fake_seed{seed}",
        "checkpoint_sha256": f"{seed:064x}",
        "evaluation_sha256": "e" * 64,
        "manifest_sha256": manifest["manifest_sha256"],
        "target_token_count": total_tokens,
        "nll_sum": total_nll,
        "mean_nll": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll, dtype=torch.float64))),
        "eval_seconds": 1.0,
        "target_tokens_per_second": float(total_tokens),
        "diagnostics_first_batch": {"fake_diagnostic": 0.0},
        "blocks": blocks,
    }


def check_comparison_summary() -> None:
    seeds = [1, 2, 3]
    records = []
    manifest = fake_manifest()
    expected_deltas = [-0.02, -0.01, -0.03]
    for seed, delta in zip(seeds, expected_deltas):
        baseline_loss = 2.7 + seed * 0.01
        records.append(fake_record(BASELINE_ARCHITECTURE, seed, baseline_loss, manifest))
        records.append(fake_record(PROPOSAL_ARCHITECTURE, seed, baseline_loss + delta, manifest))
    summary = summarize_comparison(
        records,
        seeds=seeds,
        manifest=manifest,
        bootstrap_samples=100,
        bootstrap_seed=7,
        practical_margin=0.01,
    )
    actual = summary["paired_training_seed_inference_primary"]["mean_delta"]
    if abs(actual - sum(expected_deltas) / len(expected_deltas)) > 1e-12:
        raise AssertionError("paired comparison mean is incorrect")
    if summary["delta_definition"].split(";")[1].strip() != "negative favors proposal":
        raise AssertionError("comparison delta direction is unclear")

    reader_records = []
    for seed in seeds:
        softmax = fake_record(PROPOSAL_ARCHITECTURE, seed, 2.70, manifest)
        associative = fake_record(ASSOCIATIVE_ARCHITECTURE, seed, 2.705, manifest)
        softmax["memory_off"] = fake_record(
            PROPOSAL_ARCHITECTURE,
            seed,
            2.80,
            manifest,
        )
        associative["memory_off"] = fake_record(
            ASSOCIATIVE_ARCHITECTURE,
            seed,
            2.755,
            manifest,
        )
        reader_records.extend((softmax, associative))
    reader_summary = summarize_comparison(
        reader_records,
        seeds=seeds,
        manifest=manifest,
        bootstrap_samples=100,
        bootstrap_seed=9,
        practical_margin=0.01,
        architectures=(PROPOSAL_ARCHITECTURE, ASSOCIATIVE_ARCHITECTURE),
        control_architecture=PROPOSAL_ARCHITECTURE,
        candidate_architecture=ASSOCIATIVE_ARCHITECTURE,
    )
    if reader_summary["candidate_architecture"] != ASSOCIATIVE_ARCHITECTURE:
        raise AssertionError("reader comparison candidate was mislabeled")
    intervention = reader_summary["memory_off_causal_intervention"]
    if not intervention[ASSOCIATIVE_ARCHITECTURE]["all_seeds_positive"]:
        raise AssertionError("memory-off penalties were not summarized")


class UniformLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self, token_ids: torch.Tensor, return_diagnostics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        logits = torch.zeros(
            *token_ids.shape,
            self.vocab_size,
            device=token_ids.device,
            dtype=torch.float32,
        )
        if return_diagnostics:
            return logits, {"uniform_logit": 0.0}
        return logits


def check_evaluation_numerics_and_integrity() -> None:
    vocab_size = 8
    token_ids = torch.zeros(
        POOL_START_TOKEN + BLOCK_TARGET_TOKENS + 1,
        dtype=torch.int16,
    )
    metrics = evaluate_model(
        model=UniformLanguageModel(vocab_size),
        token_ids=token_ids,
        block_ids=[0],
        batch_size=4,
        device=torch.device("cpu"),
        precision="fp32",
        progress_every_blocks=0,
        label="uniform_test_model",
    )
    if metrics["target_token_count"] != BLOCK_TARGET_TOKENS:
        raise AssertionError("evaluate_model scored the wrong number of tokens")
    if abs(metrics["mean_nll"] - math.log(vocab_size)) > 1e-6:
        raise AssertionError("evaluate_model did not recover the known uniform-model NLL")
    if len(metrics["blocks"]) != 1 or len(metrics["blocks"][0]["sequence_nll_sums"]) != 32:
        raise AssertionError("evaluate_model did not preserve the block/sequence decomposition")

    manifest = fake_manifest(block_count=1)
    record = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "architecture": BASELINE_ARCHITECTURE,
        "seed": 1,
        "run_name": "uniform",
        "checkpoint_sha256": "c" * 64,
        "evaluation_sha256": "e" * 64,
        "manifest_sha256": manifest["manifest_sha256"],
        **metrics,
    }
    validate_completed_result(
        record,
        manifest,
        architecture=BASELINE_ARCHITECTURE,
        seed=1,
        run_name="uniform",
        checkpoint_sha256="c" * 64,
        evaluation_sha256="e" * 64,
    )

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["sampling"]["selected_block_ids"] = [1]
    try:
        validate_manifest_integrity(tampered_manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered manifest was accepted")

    tampered_result = copy.deepcopy(record)
    tampered_result["blocks"][0]["sequence_nll_sums"].pop()
    try:
        validate_completed_result(tampered_result, manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered completed result was accepted")


def main() -> None:
    check_checkpoint_roundtrips()
    check_fail_closed_behavior()
    check_manifest_selection()
    check_comparison_summary()
    check_evaluation_numerics_and_integrity()
    print(
        json.dumps(
            {
                "test": "evaluate_checkpoints",
                "status": "passed",
                "checks": [
                    "baseline_checkpoint_roundtrip",
                    "proposal_nondefault_compiled_checkpoint_roundtrip",
                    "associative_checkpoint_roundtrip_and_memory_off_intervention",
                    "fail_closed_dispatch_and_strict_state_loading",
                    "frozen_untouched_pool_manifest_selection",
                    "paired_comparison_summary",
                    "known_nll_evaluation_and_tamper_rejection",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
