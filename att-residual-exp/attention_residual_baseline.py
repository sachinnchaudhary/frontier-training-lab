"""Full Attention Residuals baseline for the depth-memory experiment.

Run from the repository root:

    python att-residual-exp/attention_residual_baseline.py --mode smoke
    python att-residual-exp/attention_residual_baseline.py --mode pilot
    python att-residual-exp/attention_residual_baseline.py --mode full

Every self-attention branch and every FFN branch is one depth step. The stored
values are the raw branch outputs, not cumulative hidden states.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn

from _common import (
    ModelConfig,
    TransformerFunctions,
    add_shared_arguments,
    count_parameters,
    make_norm,
    mean_diagnostics,
    resolve_configs,
    run_training,
    set_seed,
)


class FullAttentionResidual(nn.Module):
    """Paper-faithful softmax attention over raw outputs along depth."""

    def __init__(self, dim: int, norm_type: str = "rmsnorm"):
        super().__init__()
        self.route_norm = make_norm(dim, norm_type)
        self.pseudo_query = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        history: list[torch.Tensor],
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not history:
            raise ValueError("AttnRes history must contain at least the token embedding")
        values = torch.stack(history, dim=2)  # [B, T, depth, D]
        keys = self.route_norm(values)
        logits = torch.einsum("d,btnd->btn", self.pseudo_query, keys)
        weights = torch.softmax(logits.float(), dim=-1).to(values.dtype)
        hidden = torch.einsum("btn,btnd->btd", weights, values)
        if not return_diagnostics:
            return hidden

        probabilities = weights.float()
        entropy = -(
            probabilities * probabilities.clamp_min(1e-9).log()
        ).sum(dim=-1)
        diagnostics = {
            "depth_attention_entropy": entropy.mean(),
            "depth_attention_entropy_fraction": (
                entropy.mean() / max(math.log(len(history)), 1e-9)
                if len(history) > 1
                else entropy.new_zeros(())
            ),
            "depth_attention_max_weight": probabilities.max(dim=-1).values.mean(),
            "depth_attention_newest_weight": probabilities[..., -1].mean(),
            "attnres_hidden_rms": hidden.float().pow(2).mean().sqrt(),
        }
        return hidden, diagnostics


class FullAttnResBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.functions = TransformerFunctions(config)
        self.after_attention = FullAttentionResidual(config.dim, config.norm_type)
        self.after_ffn = FullAttentionResidual(config.dim, config.norm_type)


class FullAttentionResidualLM(nn.Module):
    architecture_name = "full_attention_residual"

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            FullAttnResBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = make_norm(config.dim, config.norm_type)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        hidden = self.embedding(token_ids)
        history = [hidden]
        diagnostic_rows: list[dict[str, torch.Tensor | float]] = []

        for block in self.blocks:
            attention_delta = block.functions.attention_delta(hidden)
            history.append(attention_delta)
            if return_diagnostics:
                hidden, diagnostics = block.after_attention(history, True)
                diagnostics["sublayer_delta_rms"] = (
                    attention_delta.float().pow(2).mean().sqrt()
                )
                diagnostic_rows.append(diagnostics)
            else:
                hidden = block.after_attention(history)

            ffn_delta = block.functions.ffn_delta(hidden)
            history.append(ffn_delta)
            if return_diagnostics:
                hidden, diagnostics = block.after_ffn(history, True)
                diagnostics["sublayer_delta_rms"] = (
                    ffn_delta.float().pow(2).mean().sqrt()
                )
                diagnostic_rows.append(diagnostics)
            else:
                hidden = block.after_ffn(history)

        logits = self.lm_head(self.final_norm(hidden))
        if not return_diagnostics:
            return logits

        diagnostics = mean_diagnostics(diagnostic_rows)
        diagnostics["attnres_history_size"] = float(len(history))
        return logits, diagnostics

    @torch.no_grad()
    def assert_zero_query_uniformity(self) -> None:
        for block in self.blocks:
            for mixer in (block.after_attention, block.after_ffn):
                if not torch.equal(
                    mixer.pseudo_query,
                    torch.zeros_like(mixer.pseudo_query),
                ):
                    raise AssertionError("AttnRes pseudo-query is not zero-initialized")
        sample = torch.randn(2, 3, self.config.dim)
        history = [sample, sample * 2.0, sample * -0.5]
        mixer = self.blocks[0].after_attention
        values = torch.stack(history, dim=2)
        logits = torch.einsum(
            "d,btnd->btn",
            mixer.pseudo_query,
            mixer.route_norm(values),
        )
        weights = logits.softmax(dim=-1)
        expected = torch.full_like(weights, 1.0 / len(history))
        torch.testing.assert_close(weights, expected, rtol=0.0, atol=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Full Attention Residuals language-model baseline."
    )
    add_shared_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config, train_config = resolve_configs(
        args,
        architecture=FullAttentionResidualLM.architecture_name,
    )
    set_seed(train_config.seed)
    model = FullAttentionResidualLM(model_config)
    model.assert_zero_query_uniformity()
    metadata = {
        "residual_type": "full_attnres",
        "attnres_source": "raw_sublayer_outputs",
        "attnres_query_initialization": "zeros",
        "attnres_logit_scale": "none",
        "architecture_parameters": sum(
            count_parameters(block.after_attention) + count_parameters(block.after_ffn)
            for block in model.blocks
        ),
    }
    run_training(model, model_config, train_config, metadata)


if __name__ == "__main__":
    main()
