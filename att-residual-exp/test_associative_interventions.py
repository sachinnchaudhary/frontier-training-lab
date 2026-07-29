"""Focused CPU checks for associative-reader checkpoint interventions.

Run from the repository root:

    python att-residual-exp/test_associative_interventions.py

The checks use tiny randomly initialized models and never touch run artifacts.
"""

from __future__ import annotations

import copy

import torch

from _common import ModelConfig
from associative_read_depth_kda import AssociativeReadDepthKDALM
from evaluate_associative_interventions import install_intervention, mode_scales


def make_model() -> AssociativeReadDepthKDALM:
    config = ModelConfig(
        vocab_size=64,
        dim=32,
        num_layers=3,
        num_heads=4,
        ffn_hidden_dim=48,
        max_seq_len=16,
    )
    return AssociativeReadDepthKDALM(
        config,
        num_slots=4,
        memory_dim=7,
        alpha_bias=-2.7,
        beta_bias=-0.8,
        gamma_init=0.13,
    ).eval()


def assert_same_state(
    left: torch.Tensor,
    right: torch.Tensor,
    label: str,
) -> None:
    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0, msg=label)


def check_mode_dispatch() -> None:
    assert mode_scales("normal", 0) == (1.0, 1.0)
    assert mode_scales("memory_output_off", 4) == (0.0, 0.0)
    assert mode_scales("history_only", 2) == (1.0, 0.0)
    assert mode_scales("current_correction_only", 2) == (0.0, 1.0)
    assert mode_scales("first_current_off", 0) == (1.0, 0.0)
    assert mode_scales("first_current_off", 1) == (1.0, 1.0)
    try:
        mode_scales("unknown", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown intervention mode was accepted")


def check_installation_preserves_parameters_and_is_fail_closed() -> None:
    model = make_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rows = install_intervention(model, "first_current_off")
    after = model.state_dict()
    if tuple(before) != tuple(after):
        raise AssertionError("intervention installation changed state-dict keys")
    for name, expected in before.items():
        assert_same_state(after[name], expected, f"parameter changed during install: {name}")
    if rows[0]["current_scale"] != 0.0:
        raise AssertionError("first transition retained its current read")
    if any(row["current_scale"] != 1.0 for row in rows[1:]):
        raise AssertionError("first-current-off changed a later transition")
    if not hasattr(model.transitions[0], "_intervention_original_forward"):
        raise AssertionError("first transition was not wrapped")
    if any(
        hasattr(transition, "_intervention_original_forward")
        for transition in model.transitions[1:]
    ):
        raise AssertionError("first-current-off wrapped a later transition")
    try:
        install_intervention(model, "history_only")
    except RuntimeError:
        pass
    else:
        raise AssertionError("a second intervention was installed over an existing one")


@torch.no_grad()
def check_normal_and_memory_off_equivalence() -> None:
    torch.manual_seed(20260729)
    base = make_model()
    tokens = torch.randint(0, base.config.vocab_size, (2, 11))

    normal = copy.deepcopy(base)
    rows = install_intervention(normal, "normal")
    if len(rows) != len(normal.transitions):
        raise AssertionError("normal intervention plan omitted transitions")
    base_logits = base(tokens)
    normal_logits = normal(tokens)
    assert_same_state(normal_logits, base_logits, "normal view changed logits")

    wrapped_off = copy.deepcopy(base)
    gamma_off = copy.deepcopy(base)
    install_intervention(wrapped_off, "memory_output_off")
    for transition in gamma_off.transitions:
        transition.gamma.zero_()
    wrapped_logits = wrapped_off(tokens)
    gamma_logits = gamma_off(tokens)
    assert_same_state(
        wrapped_logits,
        gamma_logits,
        "read-component zeroing differs from the archived gamma-zero intervention",
    )


def run_transition(
    model: AssociativeReadDepthKDALM,
    mode: str,
    hidden: torch.Tensor,
    delta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    install_intervention(model, mode)
    return model.transitions[0](hidden, delta, state)


@torch.no_grad()
def check_zero_state_first_transition() -> None:
    torch.manual_seed(81)
    base = make_model()
    hidden = torch.randn(2, 5, base.config.dim)
    delta = torch.randn_like(hidden)
    state = torch.zeros(2, 5, base.num_slots, base.memory_dim)

    outputs = {
        mode: run_transition(copy.deepcopy(base), mode, hidden, delta, state)
        for mode in (
            "normal",
            "history_only",
            "current_correction_only",
            "first_current_off",
            "memory_output_off",
        )
    }
    reference_state = outputs["normal"][1]
    if not bool((reference_state != 0).any().item()):
        raise AssertionError("first transition did not retain its write")
    for mode, (_, next_state) in outputs.items():
        assert_same_state(next_state, reference_state, f"{mode} changed the write")

    assert_same_state(
        outputs["history_only"][0],
        hidden,
        "history-only injected a read from a zero incoming state",
    )
    assert_same_state(
        outputs["first_current_off"][0],
        hidden,
        "first-current-off injected a read at transition zero",
    )
    assert_same_state(
        outputs["memory_output_off"][0],
        hidden,
        "memory-output-off changed hidden state",
    )
    torch.testing.assert_close(
        outputs["current_correction_only"][0],
        outputs["normal"][0],
        rtol=1e-6,
        atol=1e-6,
        msg="normal first read was not entirely the current correction",
    )


@torch.no_grad()
def check_component_decomposition_and_state_invariance() -> None:
    torch.manual_seed(94)
    base = make_model()
    hidden = torch.randn(2, 5, base.config.dim)
    delta = torch.randn_like(hidden)
    state = torch.randn(2, 5, base.num_slots, base.memory_dim)
    outputs = {
        mode: run_transition(copy.deepcopy(base), mode, hidden, delta, state)
        for mode in (
            "normal",
            "history_only",
            "current_correction_only",
            "memory_output_off",
        )
    }
    reference_state = outputs["normal"][1]
    for mode, (_, next_state) in outputs.items():
        assert_same_state(next_state, reference_state, f"{mode} changed recurrent state")

    normal_update = outputs["normal"][0] - hidden
    history_update = outputs["history_only"][0] - hidden
    current_update = outputs["current_correction_only"][0] - hidden
    torch.testing.assert_close(
        normal_update,
        history_update + current_update,
        rtol=2e-5,
        atol=2e-6,
        msg="normal read did not decompose into history plus current correction",
    )
    assert_same_state(
        outputs["memory_output_off"][0],
        hidden,
        "memory-output-off changed hidden state",
    )


@torch.no_grad()
def check_autoregressive_causality() -> None:
    torch.manual_seed(106)
    model = make_model()
    install_intervention(model, "history_only")
    tokens = torch.randint(0, model.config.vocab_size, (1, 12))
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(0, model.config.vocab_size, (1, 4))
    first = model(tokens)
    second = model(changed)
    assert_same_state(
        first[:, :8],
        second[:, :8],
        "future-token mutation changed intervention-prefix logits",
    )


def main() -> None:
    check_mode_dispatch()
    check_installation_preserves_parameters_and_is_fail_closed()
    check_normal_and_memory_off_equivalence()
    check_zero_state_first_transition()
    check_component_decomposition_and_state_invariance()
    check_autoregressive_causality()
    print(
        {
            "test": "associative_interventions",
            "status": "passed",
            "checks": [
                "mode_dispatch_and_fail_closed_installation",
                "state_dict_and_parameter_invariance",
                "normal_and_gamma_zero_equivalence",
                "zero_state_first_transition_semantics",
                "history_current_decomposition_and_write_invariance",
                "autoregressive_prefix_causality",
            ],
        }
    )


if __name__ == "__main__":
    main()
