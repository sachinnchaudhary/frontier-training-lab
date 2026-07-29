"""Compatibility checks for strict TorchVersion checkpoint allowlisting."""

from __future__ import annotations

from typing import Any

import torch

import evaluate_associative_interventions as evaluator


def check_context_manager_api() -> None:
    calls: list[tuple[str, Any]] = []

    class FakeContext:
        def __enter__(self) -> None:
            calls.append(("enter", None))

        def __exit__(self, *exc: Any) -> None:
            calls.append(("exit", exc[0]))

    def fake_context(allowed: list[type]) -> FakeContext:
        calls.append(("context", allowed))
        return FakeContext()

    originals = (
        evaluator.load_checkpoint_model,
        getattr(torch.serialization, "safe_globals", None),
        getattr(torch.serialization, "add_safe_globals", None),
    )
    try:
        evaluator.load_checkpoint_model = lambda *args, **kwargs: "loaded"
        torch.serialization.safe_globals = fake_context
        torch.serialization.add_safe_globals = lambda allowed: (_ for _ in ()).throw(
            AssertionError("new API path unexpectedly used global registration")
        )
        if evaluator._safe_load_checkpoint_model() != "loaded":
            raise AssertionError("context-manager API did not return the loader result")
        if [name for name, _ in calls] != ["context", "enter", "exit"]:
            raise AssertionError(f"context-manager API calls are wrong: {calls}")
    finally:
        evaluator.load_checkpoint_model = originals[0]
        torch.serialization.safe_globals = originals[1]
        torch.serialization.add_safe_globals = originals[2]


def check_legacy_registration_api() -> None:
    originals = (
        evaluator.load_checkpoint_model,
        getattr(torch.serialization, "safe_globals", None),
        getattr(torch.serialization, "add_safe_globals", None),
        getattr(torch.serialization, "get_safe_globals", None),
        getattr(torch.serialization, "clear_safe_globals", None),
    )
    try:
        torch.serialization.safe_globals = None

        class SyntheticLoadError(Exception):
            pass

        for should_fail in (False, True):
            registry: list[type] = [str]
            events: list[tuple[str, Any]] = []

            def fake_add(allowed: list[type]) -> None:
                values = list(allowed)
                events.append(("add", values))
                registry.extend(values)

            def fake_get() -> list[type]:
                events.append(("get", None))
                return list(registry)

            def fake_clear() -> None:
                events.append(("clear", None))
                registry.clear()

            torch.serialization.add_safe_globals = fake_add
            torch.serialization.get_safe_globals = fake_get
            torch.serialization.clear_safe_globals = fake_clear
            if should_fail:
                evaluator.load_checkpoint_model = (
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        SyntheticLoadError("synthetic load failure")
                    )
                )
                try:
                    evaluator._safe_load_checkpoint_model()
                except SyntheticLoadError:
                    pass
                else:
                    raise AssertionError("synthetic loader failure was swallowed")
            else:
                evaluator.load_checkpoint_model = lambda *args, **kwargs: "loaded"
                if evaluator._safe_load_checkpoint_model() != "loaded":
                    raise AssertionError("legacy API did not return the loader result")

            if registry != [str]:
                raise AssertionError(f"legacy allowlist was not restored: {registry}")
            expected_names = ["get", "add", "clear", "add"]
            if [name for name, _ in events] != expected_names:
                raise AssertionError(f"legacy API calls are wrong: {events}")
            if events[1][1] != [torch.torch_version.TorchVersion]:
                raise AssertionError(f"legacy allowlist is wrong: {events}")
    finally:
        evaluator.load_checkpoint_model = originals[0]
        torch.serialization.safe_globals = originals[1]
        torch.serialization.add_safe_globals = originals[2]
        torch.serialization.get_safe_globals = originals[3]
        torch.serialization.clear_safe_globals = originals[4]


def check_missing_safe_api_fails_closed() -> None:
    originals = (
        evaluator.load_checkpoint_model,
        getattr(torch.serialization, "safe_globals", None),
        getattr(torch.serialization, "add_safe_globals", None),
    )
    try:
        evaluator.load_checkpoint_model = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint loader ran without a safe allowlisting API")
        )
        torch.serialization.safe_globals = None
        torch.serialization.add_safe_globals = None
        try:
            evaluator._safe_load_checkpoint_model()
        except RuntimeError as exc:
            if "cannot safely scope" not in str(exc):
                raise
        else:
            raise AssertionError("missing safe serialization APIs were accepted")
    finally:
        evaluator.load_checkpoint_model = originals[0]
        torch.serialization.safe_globals = originals[1]
        torch.serialization.add_safe_globals = originals[2]


def main() -> None:
    check_context_manager_api()
    check_legacy_registration_api()
    check_missing_safe_api_fails_closed()
    print(
        {
            "test": "torch_serialization_compat",
            "status": "passed",
            "checks": [
                "new_safe_globals_context_api",
                "legacy_add_get_clear_restored_on_success_and_failure",
                "missing_safe_api_fails_closed",
            ],
        }
    )


if __name__ == "__main__":
    main()
