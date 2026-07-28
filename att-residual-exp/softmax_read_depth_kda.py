"""Softmax-read gated-delta memory recurrent over Transformer depth.

Run from the repository root:

    python att-residual-exp/softmax_read_depth_kda.py --mode smoke
    python att-residual-exp/softmax_read_depth_kda.py --mode pilot
    python att-residual-exp/softmax_read_depth_kda.py --mode full

The KDA-inspired gated-delta recurrence advances after every self-attention or
FFN sublayer. It is per token and never scans or pools the sequence axis.
Writes are dense; top-k content filtering is intentionally outside this first
experiment. This reference implementation does not claim KDA's full structured
gating or chunkwise training algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import (
    ModelConfig,
    TransformerFunctions,
    add_shared_arguments,
    closest_ffn_hidden_dim,
    count_parameters,
    make_norm,
    mean_diagnostics,
    resolve_configs,
    run_training,
    set_seed,
)
from attention_residual_baseline import FullAttentionResidualLM


def gated_delta_depth_update(
    state: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    write_key: torch.Tensor,
    write_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta-rule recurrence with state [..., S, r].

    alpha and write_key have shape [..., S], beta [..., 1], and write_value
    [..., r]. The returned innovation has shape [..., r].
    """
    decayed_state = alpha.unsqueeze(-1) * state
    predicted_value = torch.einsum("...sr,...s->...r", decayed_state, write_key)
    innovation = write_value - predicted_value
    correction = (
        beta.unsqueeze(-1)
        * write_key.unsqueeze(-1)
        * innovation.unsqueeze(-2)
    )
    return decayed_state + correction, innovation


class GatedDeltaDepthTransition(nn.Module):
    """Write one raw sublayer delta, then prepare the next sublayer input."""

    def __init__(
        self,
        dim: int,
        num_slots: int,
        memory_dim: int,
        read_key_dim: int,
        read_value_dim: int,
        norm_type: str,
        alpha_bias: float,
        beta_bias: float,
        gamma_init: float,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.read_key_dim = read_key_dim

        self.write_norm = make_norm(dim, norm_type)
        self.write_key_proj = nn.Linear(dim, num_slots, bias=False)
        self.write_value_proj = nn.Linear(dim, memory_dim, bias=False)
        self.alpha_proj = nn.Linear(dim, num_slots, bias=True)
        self.beta_proj = nn.Linear(dim, 1, bias=True)

        self.read_norm = make_norm(dim, norm_type)
        self.query_proj = nn.Linear(dim, read_key_dim, bias=False)
        self.memory_key_proj = nn.Linear(memory_dim, read_key_dim, bias=False)
        self.memory_value_proj = nn.Linear(memory_dim, read_value_dim, bias=False)
        self.out_proj = nn.Linear(read_value_dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, alpha_bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta_bias)

    def write_parameters(
        self,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.write_norm(delta)
        write_key = F.normalize(
            self.write_key_proj(x).float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        write_value = self.write_value_proj(x).float()
        alpha = torch.exp(-F.softplus(self.alpha_proj(x).float()))
        beta = torch.sigmoid(self.beta_proj(x).float())
        return write_key, write_value, alpha, beta

    def forward(
        self,
        hidden: torch.Tensor,
        delta: torch.Tensor,
        state: torch.Tensor,
        slot_key_embedding: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        write_key, write_value, alpha, beta = self.write_parameters(delta)
        state, innovation = gated_delta_depth_update(
            state=state,
            alpha=alpha,
            beta=beta,
            write_key=write_key,
            write_value=write_value,
        )

        query = self.query_proj(self.read_norm(hidden))
        memory_key = self.memory_key_proj(state)
        memory_key = memory_key + slot_key_embedding.to(memory_key.dtype)
        memory_value = self.memory_value_proj(state)
        scores = torch.einsum("btd,btsd->bts", query, memory_key)
        scores = scores / self.read_key_dim**0.5
        read_weights = torch.softmax(scores.float(), dim=-1).to(memory_value.dtype)
        retrieval = torch.einsum("bts,btsv->btv", read_weights, memory_value)
        retrieval = self.out_proj(retrieval)
        hidden = hidden + self.gamma.to(hidden.dtype) * retrieval

        if not return_diagnostics:
            return hidden, state

        probabilities = read_weights.float()
        entropy = -(
            probabilities * probabilities.clamp_min(1e-9).log()
        ).sum(dim=-1)
        normalized_state = F.normalize(state, p=2, dim=-1, eps=1e-6)
        gram = torch.einsum("btsr,btur->btsu", normalized_state, normalized_state)
        if self.num_slots > 1:
            identity = torch.eye(
                self.num_slots,
                device=gram.device,
                dtype=torch.bool,
            )
            slot_cosine = gram[..., ~identity].mean()
        else:
            slot_cosine = gram.new_zeros(())

        diagnostics = {
            "memory_read_entropy": entropy.mean(),
            "memory_read_max_weight": probabilities.max(dim=-1).values.mean(),
            "memory_alpha_mean": alpha.mean(),
            "memory_alpha_min": alpha.min(),
            "memory_beta_mean": beta.mean(),
            "memory_innovation_rms": innovation.pow(2).mean().sqrt(),
            "memory_state_rms": state.pow(2).mean().sqrt(),
            "memory_slot_cosine": slot_cosine,
            "memory_write_key_abs_mean": write_key.abs().mean(),
            "memory_write_key_max_abs": write_key.abs().max(dim=-1).values.mean(),
            "memory_gamma": self.gamma,
            "sublayer_delta_rms": delta.float().pow(2).mean().sqrt(),
            "depth_hidden_rms": hidden.float().pow(2).mean().sqrt(),
        }
        return hidden, state, diagnostics


class DepthMemoryBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.functions = TransformerFunctions(config)


class SoftmaxReadGatedDeltaDepthMemoryLM(nn.Module):
    architecture_name = "softmax_read_gated_delta_depth_memory"

    def __init__(
        self,
        config: ModelConfig,
        num_slots: int = 8,
        memory_dim: int = 32,
        read_key_dim: int = 32,
        read_value_dim: int = 32,
        alpha_bias: float = -4.6,
        beta_bias: float = -2.0,
        gamma_init: float = 1e-3,
    ):
        super().__init__()
        config.validate()
        if min(num_slots, memory_dim, read_key_dim, read_value_dim) <= 0:
            raise ValueError("memory dimensions and num_slots must be positive")
        self.config = config
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.alpha_bias = alpha_bias
        self.beta_bias = beta_bias
        self.gamma_init = gamma_init

        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(DepthMemoryBlock(config) for _ in range(config.num_layers))

        # There are 2L sublayers. The first sublayer has no history to read and
        # the final sublayer has no successor, so only 2L-1 transitions are live.
        self.transitions = nn.ModuleList(
            GatedDeltaDepthTransition(
                dim=config.dim,
                num_slots=num_slots,
                memory_dim=memory_dim,
                read_key_dim=read_key_dim,
                read_value_dim=read_value_dim,
                norm_type=config.norm_type,
                alpha_bias=alpha_bias,
                beta_bias=beta_bias,
                gamma_init=gamma_init,
            )
            for _ in range(2 * config.num_layers - 1)
        )
        self.slot_key_embedding = nn.Parameter(
            torch.empty(num_slots, read_key_dim).normal_(mean=0.0, std=0.02)
        )
        self.final_norm = make_norm(config.dim, config.norm_type)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def _transition(
        self,
        index: int,
        hidden: torch.Tensor,
        delta: torch.Tensor,
        state: torch.Tensor,
        return_diagnostics: bool,
    ):
        return self.transitions[index](
            hidden,
            delta,
            state,
            self.slot_key_embedding,
            return_diagnostics,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        hidden = self.embedding(token_ids)
        batch, seq_len, _ = hidden.shape
        # Keep the recurrent fast-weight state in FP32 for the first reference
        # implementation; projections may still run under BF16 autocast.
        state = torch.zeros(
            batch,
            seq_len,
            self.num_slots,
            self.memory_dim,
            device=hidden.device,
            dtype=torch.float32,
        )
        diagnostic_rows: list[dict[str, torch.Tensor | float]] = []
        transition_index = 0

        for block_index, block in enumerate(self.blocks):
            attention_delta = block.functions.attention_delta(hidden)
            hidden = hidden + attention_delta
            if return_diagnostics:
                hidden, state, diagnostics = self._transition(
                    transition_index,
                    hidden,
                    attention_delta,
                    state,
                    True,
                )
                diagnostic_rows.append(diagnostics)
            else:
                hidden, state = self._transition(
                    transition_index,
                    hidden,
                    attention_delta,
                    state,
                    False,
                )
            transition_index += 1

            ffn_delta = block.functions.ffn_delta(hidden)
            hidden = hidden + ffn_delta
            if block_index < len(self.blocks) - 1:
                if return_diagnostics:
                    hidden, state, diagnostics = self._transition(
                        transition_index,
                        hidden,
                        ffn_delta,
                        state,
                        True,
                    )
                    diagnostic_rows.append(diagnostics)
                else:
                    hidden, state = self._transition(
                        transition_index,
                        hidden,
                        ffn_delta,
                        state,
                        False,
                    )
                transition_index += 1
            elif return_diagnostics:
                diagnostic_rows.append(
                    {
                        "sublayer_delta_rms": ffn_delta.float().pow(2).mean().sqrt(),
                        "depth_hidden_rms": hidden.float().pow(2).mean().sqrt(),
                    }
                )

        if transition_index != len(self.transitions):
            raise RuntimeError(
                f"used {transition_index} transitions, expected {len(self.transitions)}"
            )
        logits = self.lm_head(self.final_norm(hidden))
        if not return_diagnostics:
            return logits
        diagnostics = mean_diagnostics(diagnostic_rows)
        diagnostics["depth_memory_transitions"] = float(len(self.transitions))
        return logits, diagnostics

    @torch.no_grad()
    def assert_gated_delta_initialization(self) -> None:
        expected_alpha = torch.exp(-F.softplus(torch.tensor(self.alpha_bias)))
        expected_beta = torch.sigmoid(torch.tensor(self.beta_bias))
        sample = torch.randn(2, 3, self.config.dim)
        for transition in self.transitions:
            _, _, alpha, beta = transition.write_parameters(sample)
            torch.testing.assert_close(
                alpha,
                torch.full_like(alpha, expected_alpha),
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                beta,
                torch.full_like(beta, expected_beta),
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                transition.gamma,
                torch.tensor(self.gamma_init),
                rtol=0.0,
                atol=0.0,
            )


@torch.no_grad()
def assert_vectorized_gated_delta_equation() -> None:
    torch.manual_seed(7)
    batch, tokens, slots, width = 2, 3, 4, 5
    state = torch.randn(batch, tokens, slots, width)
    alpha = torch.sigmoid(torch.randn(batch, tokens, slots))
    beta = torch.sigmoid(torch.randn(batch, tokens, 1))
    key = F.normalize(torch.randn(batch, tokens, slots), dim=-1)
    value = torch.randn(batch, tokens, width)
    actual, _ = gated_delta_depth_update(state, alpha, beta, key, value)

    identity = torch.eye(slots).expand(batch, tokens, slots, slots)
    transition = identity - beta.unsqueeze(-1) * torch.einsum(
        "bts,btu->btsu", key, key
    )
    decayed = alpha.unsqueeze(-1) * state
    expected = torch.einsum("btsu,btur->btsr", transition, decayed)
    expected = expected + beta.unsqueeze(-1) * key.unsqueeze(-1) * value.unsqueeze(-2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the proposed softmax-read gated-delta depth-memory model."
    )
    add_shared_arguments(parser)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--memory-dim", type=int, default=32)
    parser.add_argument("--read-key-dim", type=int, default=32)
    parser.add_argument("--read-value-dim", type=int, default=32)
    parser.add_argument("--alpha-bias", type=float, default=-4.6)
    parser.add_argument("--beta-bias", type=float, default=-2.0)
    parser.add_argument("--gamma-init", type=float, default=1e-3)
    parser.add_argument(
        "--no-param-match",
        action="store_true",
        help="Keep the baseline FFN width instead of shrinking it to match total parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_config, _ = resolve_configs(
        args,
        architecture=SoftmaxReadGatedDeltaDepthMemoryLM.architecture_name,
    )

    def make_depth_model(ffn_hidden_dim: int) -> SoftmaxReadGatedDeltaDepthMemoryLM:
        config = replace(reference_config, ffn_hidden_dim=ffn_hidden_dim)
        return SoftmaxReadGatedDeltaDepthMemoryLM(
            config,
            num_slots=args.num_slots,
            memory_dim=args.memory_dim,
            read_key_dim=args.read_key_dim,
            read_value_dim=args.read_value_dim,
            alpha_bias=args.alpha_bias,
            beta_bias=args.beta_bias,
            gamma_init=args.gamma_init,
        )

    set_seed(args.seed)
    reference_model = FullAttentionResidualLM(reference_config)
    target_parameter_count = count_parameters(reference_model)
    del reference_model

    if args.no_param_match:
        matched_hidden_dim = reference_config.ffn_hidden_dim
    else:
        matched_hidden_dim, _ = closest_ffn_hidden_dim(
            target_parameters=target_parameter_count,
            model_factory=make_depth_model,
            reference_hidden_dim=reference_config.ffn_hidden_dim,
        )

    model_config, train_config = resolve_configs(
        args,
        architecture=SoftmaxReadGatedDeltaDepthMemoryLM.architecture_name,
        ffn_hidden_dim_override=matched_hidden_dim,
    )
    set_seed(train_config.seed)
    model = make_depth_model(matched_hidden_dim)
    model.assert_gated_delta_initialization()
    assert_vectorized_gated_delta_equation()
    parameter_count = count_parameters(model)
    metadata = {
        "residual_type": "ordinary_plus_softmax_read_gated_delta_depth_memory",
        "recurrence_family": "kda_inspired_gated_delta_rule",
        "depth_memory_granularity": "attention_and_ffn_sublayers",
        "depth_memory_state_scope": "per_token",
        "depth_memory_write": "dense_raw_sublayer_delta",
        "num_slots": args.num_slots,
        "memory_dim": args.memory_dim,
        "read_key_dim": args.read_key_dim,
        "read_value_dim": args.read_value_dim,
        "alpha_bias": args.alpha_bias,
        "beta_bias": args.beta_bias,
        "gamma_init": args.gamma_init,
        "parameter_matching": not args.no_param_match,
        "reference_attnres_ffn_hidden_dim": reference_config.ffn_hidden_dim,
        "matched_ffn_hidden_dim": matched_hidden_dim,
        "reference_attnres_parameters": target_parameter_count,
        "parameter_difference_from_reference": parameter_count - target_parameter_count,
        "architecture_parameters": count_parameters(model.transitions)
        + model.slot_key_embedding.numel(),
    }
    run_training(model, model_config, train_config, metadata)


if __name__ == "__main__":
    main()
