"""Fast correctness checks for both depth-residual experiments.

Run from the repository root:

    python att-residual-exp/test_architectures.py

This script uses synthetic tokens only. It does not load the dataset, create a
checkpoint, or start a training run.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from _common import ModelConfig, count_parameters
from associative_read_depth_kda import (
    AssociativeReadDepthKDALM,
    assert_associative_read_equation,
    initialize_from_softmax_control,
)
from attention_residual_baseline import FullAttentionResidualLM
from softmax_read_depth_kda import (
    SoftmaxReadGatedDeltaDepthMemoryLM,
    assert_vectorized_gated_delta_equation,
)


def check_model(name: str, model: torch.nn.Module, config: ModelConfig) -> dict:
    """Check output shape, finite gradients, and autoregressive causality."""
    model.train()
    token_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))
    targets = torch.randint_like(token_ids, 0, config.vocab_size)
    logits, diagnostics = model(token_ids, return_diagnostics=True)

    expected_shape = (*token_ids.shape, config.vocab_size)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(
            f"{name}: expected logits shape {expected_shape}, got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise AssertionError(f"{name}: non-finite logits")

    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError(f"{name}: no gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError(f"{name}: non-finite gradients")

    prefix_length = config.max_seq_len // 2
    alternate_token_ids = token_ids.clone()
    alternate_token_ids[:, prefix_length:] = (
        alternate_token_ids[:, prefix_length:] + 1
    ) % config.vocab_size
    model.eval()
    with torch.no_grad():
        prefix_logits = model(token_ids)[:, :prefix_length]
        alternate_prefix_logits = model(alternate_token_ids)[:, :prefix_length]
    torch.testing.assert_close(
        prefix_logits,
        alternate_prefix_logits,
        rtol=1e-5,
        atol=1e-5,
        msg=lambda message: f"{name}: future-token mutation changed prefix logits\n{message}",
    )

    return {
        "architecture": name,
        "status": "passed",
        "loss": loss.item(),
        "parameters": count_parameters(model),
        "diagnostics": diagnostics,
    }


def main() -> None:
    torch.manual_seed(1337)
    config = ModelConfig(
        vocab_size=1024,
        dim=64,
        num_layers=3,
        num_heads=4,
        ffn_hidden_dim=128,
        max_seq_len=16,
    )

    baseline = FullAttentionResidualLM(config)
    baseline.assert_zero_query_uniformity()
    baseline_result = check_model("full_attention_residual", baseline, config)

    proposal = SoftmaxReadGatedDeltaDepthMemoryLM(
        config,
        num_slots=8,
        memory_dim=32,
        read_key_dim=32,
        read_value_dim=32,
    )
    proposal.assert_gated_delta_initialization()
    assert_vectorized_gated_delta_equation()
    proposal_result = check_model("softmax_read_gated_delta_depth_memory", proposal, config)

    associative = AssociativeReadDepthKDALM(
        config,
        num_slots=8,
        memory_dim=32,
    )
    shared_keys, reader_specific_keys = initialize_from_softmax_control(
        associative,
        proposal,
    )
    if not shared_keys or not reader_specific_keys:
        raise AssertionError("paired reader initialization did not separate shared tensors")
    if not all("query_proj" in key for key in reader_specific_keys):
        raise AssertionError(
            f"unexpected reader-specific initialization keys: {reader_specific_keys}"
        )

    # With both memory-output gates disabled, the two readers must reduce to
    # exactly the same shared ordinary-residual backbone. This is a direct
    # causal check of paired initialization, not just a key-name audit.
    paired_tokens = torch.randint(
        0,
        config.vocab_size,
        (2, config.max_seq_len),
    )
    with torch.no_grad():
        for transition in proposal.transitions:
            transition.gamma.zero_()
        for transition in associative.transitions:
            transition.gamma.zero_()
        proposal_without_memory = proposal(paired_tokens)
        associative_without_memory = associative(paired_tokens)
        torch.testing.assert_close(
            proposal_without_memory,
            associative_without_memory,
            rtol=0.0,
            atol=0.0,
            msg="paired readers disagree when both memory paths are disabled",
        )
        for transition in proposal.transitions:
            transition.gamma.fill_(proposal.gamma_init)
        for transition in associative.transitions:
            transition.gamma.fill_(associative.gamma_init)
    associative.assert_gated_delta_initialization()
    assert_associative_read_equation()
    associative_result = check_model(
        "associative_read_depth_kda",
        associative,
        config,
    )
    if associative_result["diagnostics"]["memory_query_l2_norm"] < 0.999:
        raise AssertionError("associative reader query was not L2 normalized")
    if not any(
        key.startswith("depth_00_")
        for key in associative_result["diagnostics"]
    ):
        raise AssertionError("associative reader did not emit per-depth diagnostics")

    print(json.dumps([baseline_result, proposal_result, associative_result], indent=2))


if __name__ == "__main__":
    main()
