import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_seq_len).float()
        angles = positions[:, None] * inv_freq[None, :]

        cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(2)
        cos = self.cos[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin[:seq_len].to(dtype=x.dtype, device=x.device)

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        return (x * cos) + (self.rotate_half(x) * sin)
