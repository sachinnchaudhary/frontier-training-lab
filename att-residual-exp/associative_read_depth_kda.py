"""Direct associative-read KDA memory recurrent over Transformer depth.

This is a reader-only ablation of ``softmax_read_depth_kda.py``. The writer,
ordinary residual path, write-before-read ordering, and initial gates are kept
the same.  The state is read with the native KDA associative operation

    retrieval = Z.transpose(-1, -2) @ q

instead of softmax attention over projected state rows.

Run from the repository root:

    python att-residual-exp/associative_read_depth_kda.py --mode smoke
    python att-residual-exp/associative_read_depth_kda.py --mode pilot
    python att-residual-exp/associative_read_depth_kda.py --mode full
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import (
    ModelConfig,
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
from softmax_read_depth_kda import (
    DepthMemoryBlock,
    SoftmaxReadGatedDeltaDepthMemoryLM,
    assert_vectorized_gated_delta_equation,
)


DEPTH_PROFILE_KEYS = (
    "depth_hidden_rms",
    "sublayer_delta_rms",
    "memory_alpha_mean",
    "memory_beta_mean",
    "memory_gamma",
    "memory_state_rms",
    "memory_state_effective_rank",
    "memory_raw_read_rms",
    "memory_injected_update_rms",
    "memory_injected_to_hidden_rms_ratio",
    "memory_injected_to_delta_rms_ratio",
    "memory_current_write_fraction",
    "memory_query_effective_channels",
    "memory_query_write_key_abs_alignment",
)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().pow(2).mean().sqrt()


def _quantile(value: torch.Tensor, probability: float) -> torch.Tensor:
    return torch.quantile(value.float().reshape(-1), probability)


class AssociativeReadDepthTransition(nn.Module):
    """Write one sublayer delta and read the updated state with ``Z^T q``."""

    def __init__(
        self,
        dim: int,
        num_slots: int,
        memory_dim: int,
        norm_type: str,
        alpha_bias: float,
        beta_bias: float,
        gamma_init: float,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        # KDA's implementation uses the default score scale 1 / sqrt(d_k)
        # after q/k L2 normalization.  It also keeps the initial signed read
        # magnitude comparable to a convex eight-row mixture.
        self.read_scale = num_slots**-0.5

        # This writer intentionally matches the softmax-reader control.
        self.write_norm = make_norm(dim, norm_type)
        self.write_key_proj = nn.Linear(dim, num_slots, bias=False)
        self.write_value_proj = nn.Linear(dim, memory_dim, bias=False)
        self.alpha_proj = nn.Linear(dim, num_slots, bias=True)
        self.beta_proj = nn.Linear(dim, 1, bias=True)

        self.read_norm = make_norm(dim, norm_type)
        self.query_proj = nn.Linear(dim, num_slots, bias=False)
        self.out_proj = nn.Linear(memory_dim, dim, bias=False)
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

    def read_query(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            self.query_proj(self.read_norm(hidden)).float(),
            p=2,
            dim=-1,
            eps=1e-6,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        delta: torch.Tensor,
        state: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        write_key, write_value, alpha, beta = self.write_parameters(delta)

        decayed_state = alpha.unsqueeze(-1) * state
        predicted_value = torch.einsum(
            "btsr,bts->btr",
            decayed_state,
            write_key,
        )
        innovation = write_value - predicted_value
        correction = (
            beta.unsqueeze(-1)
            * write_key.unsqueeze(-1)
            * innovation.unsqueeze(-2)
        )
        state = decayed_state + correction

        query = self.read_query(hidden)
        history_read = self.read_scale * torch.einsum(
            "btsr,bts->btr",
            decayed_state,
            query,
        )
        current_write_read = self.read_scale * torch.einsum(
            "btsr,bts->btr",
            correction,
            query,
        )
        retrieval = history_read + current_write_read
        projected_retrieval = self.out_proj(retrieval)
        injected_update = self.gamma.to(projected_retrieval.dtype) * projected_retrieval
        hidden_before_read = hidden
        hidden = hidden + injected_update

        if not return_diagnostics:
            return hidden, state

        eps = 1e-8
        alpha_float = alpha.float()
        beta_float = beta.float()
        query_energy = query.square()
        query_energy = query_energy / query_energy.sum(dim=-1, keepdim=True).clamp_min(eps)
        query_entropy = -(
            query_energy * query_energy.clamp_min(eps).log()
        ).sum(dim=-1)
        query_effective_channels = query_entropy.exp()
        query_key_alignment = (query * write_key).sum(dim=-1)

        normalized_state = F.normalize(state, p=2, dim=-1, eps=1e-6)
        cosine_gram = torch.einsum(
            "btsr,btur->btsu",
            normalized_state,
            normalized_state,
        )
        identity = torch.eye(
            self.num_slots,
            device=state.device,
            dtype=torch.bool,
        )
        off_diagonal_cosine = cosine_gram[..., ~identity]

        state_gram = torch.einsum("btsr,btur->btsu", state, state)
        state_trace = state_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        state_effective_rank = state_trace.square() / state_gram.square().sum(
            dim=(-2, -1)
        ).clamp_min(eps)

        row_rms = state.float().pow(2).mean(dim=-1).sqrt()
        row_rms_cv = row_rms.std(dim=-1, unbiased=False) / row_rms.mean(
            dim=-1
        ).clamp_min(eps)

        # The cancellation ratio is one for perfectly aligned signed row
        # contributions and approaches zero when they mostly cancel.
        row_contributions = (
            self.read_scale
            * query.unsqueeze(-1)
            * state
        )
        cancellation_denominator = row_contributions.norm(dim=-1).sum(dim=-1)
        cancellation_ratio = retrieval.norm(dim=-1) / cancellation_denominator.clamp_min(eps)

        history_norm = history_read.norm(dim=-1)
        current_norm = current_write_read.norm(dim=-1)
        current_write_raw_fraction = current_norm / (
            history_norm + current_norm + eps
        )

        # Keep the shared history/current metrics in the same post-projection
        # space as the softmax-reader control. Raw-space metrics are retained
        # under explicit names because they are useful for the signed reader.
        projected_history_read = self.out_proj(history_read)
        projected_current_write_read = self.out_proj(current_write_read)
        projected_history_norm = projected_history_read.float().norm(dim=-1)
        projected_current_norm = projected_current_write_read.float().norm(dim=-1)
        current_write_fraction = projected_current_norm / (
            projected_history_norm + projected_current_norm + eps
        )

        hidden_rms = _rms(hidden_before_read)
        delta_rms = _rms(delta)
        injected_rms = _rms(injected_update)
        injected_hidden_cosine = F.cosine_similarity(
            injected_update.float(),
            hidden_before_read.float(),
            dim=-1,
            eps=1e-8,
        )
        injected_delta_cosine = F.cosine_similarity(
            injected_update.float(),
            delta.float(),
            dim=-1,
            eps=1e-8,
        )

        # ``alpha`` is retention.  This diagnostic is the implied half-life if
        # the instantaneous gate were held constant across later depth steps.
        half_life = -math.log(2.0) / alpha_float.clamp(
            min=1e-7,
            max=1.0 - 1e-7,
        ).log()

        diagnostics = {
            "memory_alpha_mean": alpha_float.mean(),
            "memory_alpha_std": alpha_float.std(unbiased=False),
            "memory_alpha_p05": _quantile(alpha_float, 0.05),
            "memory_alpha_p50": _quantile(alpha_float, 0.50),
            "memory_alpha_p95": _quantile(alpha_float, 0.95),
            "memory_alpha_below_0_1_fraction": (alpha_float < 0.1).float().mean(),
            "memory_alpha_above_0_99_fraction": (alpha_float > 0.99).float().mean(),
            "memory_alpha_half_life_p50": _quantile(half_life, 0.50),
            "memory_beta_mean": beta_float.mean(),
            "memory_beta_p05": _quantile(beta_float, 0.05),
            "memory_beta_p50": _quantile(beta_float, 0.50),
            "memory_beta_p95": _quantile(beta_float, 0.95),
            "memory_decayed_state_rms": _rms(decayed_state),
            "memory_predicted_value_rms": _rms(predicted_value),
            "memory_write_value_rms": _rms(write_value),
            "memory_innovation_rms": _rms(innovation),
            "memory_correction_rms": _rms(correction),
            "memory_state_rms": _rms(state),
            "memory_state_effective_rank": state_effective_rank.mean(),
            "memory_state_row_rms_cv": row_rms_cv.mean(),
            "memory_state_row_cosine": off_diagonal_cosine.mean(),
            "memory_state_row_abs_cosine": off_diagonal_cosine.abs().mean(),
            "memory_write_key_abs_mean": write_key.abs().mean(),
            "memory_write_key_max_abs": write_key.abs().max(dim=-1).values.mean(),
            "memory_query_l2_norm": query.norm(dim=-1).mean(),
            "memory_query_max_abs": query.abs().max(dim=-1).values.mean(),
            "memory_query_positive_fraction": (query > 0).float().mean(),
            "memory_query_energy_entropy_fraction": (
                query_entropy / math.log(self.num_slots)
            ).mean(),
            "memory_query_effective_channels": query_effective_channels.mean(),
            "memory_query_write_key_alignment": query_key_alignment.mean(),
            "memory_query_write_key_abs_alignment": query_key_alignment.abs().mean(),
            "memory_history_raw_read_rms": _rms(history_read),
            "memory_current_write_raw_read_rms": _rms(current_write_read),
            "memory_current_write_raw_fraction": current_write_raw_fraction.mean(),
            "memory_history_read_rms": _rms(projected_history_read),
            "memory_current_write_read_rms": _rms(projected_current_write_read),
            "memory_current_write_fraction": current_write_fraction.mean(),
            "memory_raw_read_rms": _rms(retrieval),
            "memory_signed_cancellation_ratio": cancellation_ratio.mean(),
            "memory_projected_read_rms": _rms(projected_retrieval),
            "memory_injected_update_rms": injected_rms,
            "memory_injected_to_hidden_rms_ratio": injected_rms
            / hidden_rms.clamp_min(eps),
            "memory_injected_to_delta_rms_ratio": injected_rms
            / delta_rms.clamp_min(eps),
            "memory_injected_hidden_cosine": injected_hidden_cosine.mean(),
            "memory_injected_delta_cosine": injected_delta_cosine.mean(),
            "memory_gamma": self.gamma,
            "memory_gamma_abs": self.gamma.abs(),
            "sublayer_delta_rms": delta_rms,
            "depth_hidden_rms": _rms(hidden),
        }
        return hidden, state, diagnostics


class AssociativeReadDepthKDALM(nn.Module):
    architecture_name = "associative_read_depth_kda"

    def __init__(
        self,
        config: ModelConfig,
        num_slots: int = 8,
        memory_dim: int = 32,
        alpha_bias: float = -4.6,
        beta_bias: float = -2.0,
        gamma_init: float = 1e-3,
    ):
        super().__init__()
        config.validate()
        if min(num_slots, memory_dim) <= 0:
            raise ValueError("num_slots and memory_dim must be positive")
        if num_slots < 2:
            raise ValueError("num_slots must be at least two for reader diagnostics")
        self.config = config
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.alpha_bias = alpha_bias
        self.beta_bias = beta_bias
        self.gamma_init = gamma_init

        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            DepthMemoryBlock(config) for _ in range(config.num_layers)
        )
        self.transitions = nn.ModuleList(
            AssociativeReadDepthTransition(
                dim=config.dim,
                num_slots=num_slots,
                memory_dim=memory_dim,
                norm_type=config.norm_type,
                alpha_bias=alpha_bias,
                beta_bias=beta_bias,
                gamma_init=gamma_init,
            )
            for _ in range(2 * config.num_layers - 1)
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
            return_diagnostics,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        hidden = self.embedding(token_ids)
        batch, seq_len, _ = hidden.shape
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
                        "sublayer_delta_rms": _rms(ffn_delta),
                        "depth_hidden_rms": _rms(hidden),
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
        for depth_index, row in enumerate(diagnostic_rows):
            for key in DEPTH_PROFILE_KEYS:
                if key not in row:
                    continue
                value = row[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().mean().item()
                diagnostics[f"depth_{depth_index:02d}_{key}"] = float(value)
        diagnostics["depth_memory_transitions"] = float(len(self.transitions))
        diagnostics["depth_diagnostic_rows"] = float(len(diagnostic_rows))
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
    def gradient_diagnostics(self) -> dict[str, float]:
        """Return pre-clipping gradient RMS values for architecture groups."""
        bucket_squares: dict[str, torch.Tensor] = {}
        bucket_elements: dict[str, int] = {}

        def add(bucket: str, parameter: nn.Parameter) -> None:
            if parameter.grad is None:
                return
            squared = parameter.grad.detach().float().square().sum()
            bucket_squares[bucket] = bucket_squares.get(bucket, squared.new_zeros(())) + squared
            bucket_elements[bucket] = bucket_elements.get(bucket, 0) + parameter.numel()

        writer_names = (
            "write_norm",
            "write_key_proj",
            "write_value_proj",
            "alpha_proj",
            "beta_proj",
        )
        reader_names = ("read_norm", "query_proj", "out_proj", "gamma")
        for name, parameter in self.named_parameters():
            if name.startswith("transitions.") and any(part in name for part in writer_names):
                add("writer", parameter)
            elif name.startswith("transitions.") and any(part in name for part in reader_names):
                add("reader", parameter)
            else:
                add("backbone", parameter)
            if ".alpha_proj." in name:
                add("alpha", parameter)
            elif ".beta_proj." in name:
                add("beta", parameter)
            elif ".query_proj." in name:
                add("query", parameter)
            elif name.endswith(".gamma"):
                add("gamma", parameter)

        output: dict[str, float] = {}
        for bucket, squared in bucket_squares.items():
            output[f"gradient_rms_{bucket}"] = float(
                (squared / max(bucket_elements[bucket], 1)).sqrt().item()
            )
        gamma = torch.stack([transition.gamma.detach().float() for transition in self.transitions])
        output.update(
            {
                "memory_gamma_parameter_mean": float(gamma.mean().item()),
                "memory_gamma_parameter_abs_mean": float(gamma.abs().mean().item()),
                "memory_gamma_parameter_min": float(gamma.min().item()),
                "memory_gamma_parameter_max": float(gamma.max().item()),
            }
        )
        return output


@torch.no_grad()
def assert_associative_read_equation() -> None:
    torch.manual_seed(11)
    batch, tokens, slots, width = 2, 3, 4, 5
    state = torch.randn(batch, tokens, slots, width)
    query = F.normalize(torch.randn(batch, tokens, slots), dim=-1)
    scale = slots**-0.5
    actual = scale * torch.einsum("btsr,bts->btr", state, query)
    expected = scale * torch.matmul(
        state.transpose(-1, -2),
        query.unsqueeze(-1),
    ).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def initialize_from_softmax_control(
    model: AssociativeReadDepthKDALM,
    control: SoftmaxReadGatedDeltaDepthMemoryLM,
) -> tuple[list[str], list[str]]:
    """Copy every semantically shared tensor from the paired control init.

    Different module sizes otherwise shift the RNG stream between transitions,
    so equal seeds alone would not give equal writer and LM-head initialization.
    The direct query projection is intentionally left reader-specific because
    its output dimension differs from the softmax query projection.
    """
    model_state = model.state_dict()
    control_state = control.state_dict()
    compatible = {
        name: control_state[name]
        for name, value in model_state.items()
        if name in control_state and control_state[name].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    if result.unexpected_keys:
        raise AssertionError(
            f"unexpected shared initialization tensors: {result.unexpected_keys}"
        )
    return sorted(compatible), sorted(result.missing_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the direct associative-read depth-KDA reader ablation."
    )
    add_shared_arguments(parser)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--memory-dim", type=int, default=32)
    parser.add_argument("--alpha-bias", type=float, default=-4.6)
    parser.add_argument("--beta-bias", type=float, default=-2.0)
    parser.add_argument("--gamma-init", type=float, default=1e-3)
    parser.add_argument(
        "--parameter-match-attnres",
        action="store_true",
        help=(
            "Widen the FFN to match Full AttnRes parameters. Do not use this for "
            "the primary reader-only ablation, which pins the softmax control FFN width."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_config, _ = resolve_configs(
        args,
        architecture=AssociativeReadDepthKDALM.architecture_name,
    )

    def make_softmax_control(ffn_hidden_dim: int) -> SoftmaxReadGatedDeltaDepthMemoryLM:
        config = replace(reference_config, ffn_hidden_dim=ffn_hidden_dim)
        return SoftmaxReadGatedDeltaDepthMemoryLM(
            config,
            num_slots=args.num_slots,
            memory_dim=args.memory_dim,
            read_key_dim=32,
            read_value_dim=args.memory_dim,
            alpha_bias=args.alpha_bias,
            beta_bias=args.beta_bias,
            gamma_init=args.gamma_init,
        )

    def make_direct_model(ffn_hidden_dim: int) -> AssociativeReadDepthKDALM:
        config = replace(reference_config, ffn_hidden_dim=ffn_hidden_dim)
        return AssociativeReadDepthKDALM(
            config,
            num_slots=args.num_slots,
            memory_dim=args.memory_dim,
            alpha_bias=args.alpha_bias,
            beta_bias=args.beta_bias,
            gamma_init=args.gamma_init,
        )

    set_seed(args.seed)
    reference_model = FullAttentionResidualLM(reference_config)
    target_parameter_count = count_parameters(reference_model)
    del reference_model

    softmax_control_hidden_dim, _ = closest_ffn_hidden_dim(
        target_parameters=target_parameter_count,
        model_factory=make_softmax_control,
        reference_hidden_dim=reference_config.ffn_hidden_dim,
    )
    if args.parameter_match_attnres:
        selected_hidden_dim, _ = closest_ffn_hidden_dim(
            target_parameters=target_parameter_count,
            model_factory=make_direct_model,
            reference_hidden_dim=reference_config.ffn_hidden_dim,
        )
        parameter_matching = "full_attnres_total_parameters"
    else:
        selected_hidden_dim = softmax_control_hidden_dim
        parameter_matching = "softmax_reader_control_backbone"

    model_config, train_config = resolve_configs(
        args,
        architecture=AssociativeReadDepthKDALM.architecture_name,
        ffn_hidden_dim_override=selected_hidden_dim,
    )
    set_seed(train_config.seed)
    initialization_control = make_softmax_control(selected_hidden_dim)
    set_seed(train_config.seed)
    model = make_direct_model(selected_hidden_dim)
    shared_initialization_keys, reader_specific_initialization_keys = (
        initialize_from_softmax_control(model, initialization_control)
    )
    del initialization_control
    model.assert_gated_delta_initialization()
    assert_vectorized_gated_delta_equation()
    assert_associative_read_equation()

    parameter_count = count_parameters(model)
    softmax_control_parameters = count_parameters(
        make_softmax_control(softmax_control_hidden_dim)
    )
    metadata = {
        "residual_type": "ordinary_plus_associative_read_depth_kda",
        "recurrence_family": "kda_diagonal_decay_delta_rule",
        "reader_type": "l2_normalized_signed_associative_state_read",
        "reader_equation": "retrieval=(state^T@query)/sqrt(num_slots)",
        "reader_state_timing": "write_then_read_updated_state",
        "reader_output_norm": "none_for_isolated_ablation",
        "reader_output_gate": "learned_scalar_gamma_per_transition",
        "depth_memory_granularity": "attention_and_ffn_sublayers",
        "depth_memory_state_scope": "per_token",
        "depth_memory_write": "dense_raw_sublayer_delta",
        "state_row_semantics": "kda_key_coordinates_not_independent_slots",
        "num_slots": args.num_slots,
        "memory_dim": args.memory_dim,
        "read_scale": args.num_slots**-0.5,
        "alpha_bias": args.alpha_bias,
        "beta_bias": args.beta_bias,
        "gamma_init": args.gamma_init,
        "parameter_matching": parameter_matching,
        "softmax_control_ffn_hidden_dim": softmax_control_hidden_dim,
        "matched_ffn_hidden_dim": selected_hidden_dim,
        "reference_attnres_ffn_hidden_dim": reference_config.ffn_hidden_dim,
        "reference_attnres_parameters": target_parameter_count,
        "softmax_control_parameters": softmax_control_parameters,
        "softmax_control_read_key_dim": 32,
        "softmax_control_read_value_dim": args.memory_dim,
        "parameter_difference_from_softmax_control": (
            parameter_count - softmax_control_parameters
        ),
        "parameter_difference_from_reference": parameter_count - target_parameter_count,
        "architecture_parameters": count_parameters(model.transitions),
        "shared_initialization_from_softmax_control": True,
        "shared_initialization_tensor_count": len(shared_initialization_keys),
        "reader_specific_initialization_keys": reader_specific_initialization_keys,
        "diagnostics_schema": "associative-depth-kda-v1",
        "depth_profile_keys": list(DEPTH_PROFILE_KEYS),
    }
    run_training(model, model_config, train_config, metadata)


if __name__ == "__main__":
    main()
