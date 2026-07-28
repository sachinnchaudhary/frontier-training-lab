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

    print(json.dumps([baseline_result, proposal_result], indent=2))


if __name__ == "__main__":
    main()
