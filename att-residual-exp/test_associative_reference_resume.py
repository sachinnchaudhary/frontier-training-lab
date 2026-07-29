"""Fast resumability checks for recomputed same-GPU reference records."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

import evaluate_associative_interventions as evaluator


class FakeModel:
    def __init__(self) -> None:
        self.transitions = [SimpleNamespace() for _ in range(3)]
        self.memory_off = False


def fake_metrics(memory_off: bool) -> dict:
    return {
        "target_token_count": 1,
        "nll_sum": 2.8 if memory_off else 2.7,
        "mean_nll": 2.8 if memory_off else 2.7,
        "perplexity": 16.0,
        "eval_seconds": 1.0,
        "target_tokens_per_second": 1.0,
        "timing_scope": "synthetic",
        "cuda_memory": {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        },
        "diagnostics_first_batch": {
            "memory_gamma": 0.0 if memory_off else 0.1,
            "memory_gamma_abs": 0.0 if memory_off else 0.1,
        },
        "blocks": [],
    }


def check_atomic_resume_and_identity_rejection() -> None:
    spec = SimpleNamespace(
        seed=1337,
        run_name="full_100m_seed1337",
        run_config={"model": {"num_layers": 2}},
        checkpoint_path=Path("synthetic/latest.pt"),
    )
    calls = {"load": 0, "eval": 0}

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        return FakeModel(), "c" * 64, {"step": 12_208}

    def fake_disable(model):
        model.memory_off = True
        return 3

    def fake_evaluate(*, model, **kwargs):
        calls["eval"] += 1
        return fake_metrics(model.memory_off)

    originals = {
        "_safe_load_checkpoint_model": evaluator._safe_load_checkpoint_model,
        "disable_memory_reads": evaluator.disable_memory_reads,
        "evaluate_model": evaluator.evaluate_model,
        "validate_metric_payload": evaluator.validate_metric_payload,
    }
    evaluator._safe_load_checkpoint_model = fake_load
    evaluator.disable_memory_reads = fake_disable
    evaluator.evaluate_model = fake_evaluate
    evaluator.validate_metric_payload = lambda metrics, manifest: None
    try:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            arguments = {
                "specs": (spec,),
                "manifest": {"manifest_sha256": "m" * 64},
                "evaluation_sha256": "e" * 64,
                "environment_hash": "f" * 64,
                "gpu_uuid": "GPU-4090-test",
                "checkpoint_hashes": {"1337": "c" * 64},
                "expected_step": 12_208,
                "token_ids": torch.zeros(1, dtype=torch.long),
                "block_ids": (0,),
                "batch_size": 1,
                "device": torch.device("cpu"),
                "precision": "fp32",
                "progress_every_blocks": 0,
                "output_dir": output,
            }
            references, identity = evaluator.evaluate_local_references(
                **arguments, resume=False
            )
            if set(references[1337]) != {"normal", "memory_output_off"}:
                raise AssertionError("local reference matrix is incomplete")
            if len(identity["result_sha256_by_seed_and_mode"]) != 2:
                raise AssertionError("local reference hashes are incomplete")
            if calls != {"load": 2, "eval": 2}:
                raise AssertionError(f"fresh reference calls are wrong: {calls}")

            calls.update(load=0, eval=0)
            evaluator.evaluate_local_references(**arguments, resume=True)
            if calls != {"load": 0, "eval": 0}:
                raise AssertionError("resume recomputed completed references")

            path = output / "reference_results/seed1337/normal.json"
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["gpu_uuid"] = "GPU-other"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            try:
                evaluator.evaluate_local_references(**arguments, resume=True)
            except ValueError:
                pass
            else:
                raise AssertionError("resume accepted a different physical GPU identity")
    finally:
        for name, value in originals.items():
            setattr(evaluator, name, value)


def main() -> None:
    check_atomic_resume_and_identity_rejection()
    print(
        {
            "test": "associative_reference_resume",
            "status": "passed",
            "checks": [
                "fresh_normal_and_gamma_zero_records",
                "completed_record_resume_skip",
                "physical_gpu_identity_tamper_rejection",
            ],
        }
    )


if __name__ == "__main__":
    main()
