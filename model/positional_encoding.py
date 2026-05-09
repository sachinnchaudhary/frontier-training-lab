import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, max_seq_len: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must have shape (batch_size, seq_len)")

        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} is greater than max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)
        return token_emb + pos_emb


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 64, max_seq_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape (batch_size, seq_len, d_model)")

        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

 
