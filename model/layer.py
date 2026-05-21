import torch 
import torch.nn as nn   
from model.group_query_att import GroupQueryAtt
from model.standard_attention import Attentionmech


class RMSNorm(nn.Module):
     def __init__(self, dim: int, eps: float = 1e-6):
          super().__init__()
          self.eps = eps
          self.weight = nn.Parameter(torch.ones(dim))

     def forward(self, x: torch.Tensor) -> torch.Tensor:
          rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
          return self.weight * x * rms


def make_norm(dim: int, norm_type: str):
     if norm_type == "layernorm":
          return nn.LayerNorm(dim)
     if norm_type == "rmsnorm":
          return RMSNorm(dim)
     raise ValueError(f"unknown norm_type: {norm_type}")


class SwiGLU(nn.Module):
     def __init__(self, dim: int, hidden_dim: int):
          super().__init__()
          self.gate_proj = nn.Linear(dim, hidden_dim)
          self.up_proj = nn.Linear(dim, hidden_dim)
          self.down_proj = nn.Linear(hidden_dim, dim)

     def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def make_feedforward(dim: int, hidden_dim: int, ffn_type: str):
     if ffn_type == "gelu":
          return nn.Sequential(
              nn.Linear(dim, hidden_dim),
              nn.GELU(),
              nn.Linear(hidden_dim, dim),
          )
     if ffn_type == "swiglu":
          return SwiGLU(dim, hidden_dim)
     raise ValueError(f"unknown ffn_type: {ffn_type}")


class Transformerblock(nn.Module):

     def __init__(
          self,
          dim=64,
          num_heads=8,
          use_rope=False,
          max_seq_len=128,
          norm_type="layernorm",
          attention_type="mha",
          num_kv_heads=None,
          ffn_type="gelu",
          ffn_hidden_dim=512,
     ):
          super().__init__()
          self.ln1 = make_norm(dim, norm_type)
          if attention_type == "mha":
               self.attention = Attentionmech(
                    dim=dim,
                    num_heads=num_heads,
                    use_rope=use_rope,
                    max_seq_len=max_seq_len,
               )
          elif attention_type == "gqa":
               if num_kv_heads is None:
                    raise ValueError("num_kv_heads must be set for gqa")
               self.attention = GroupQueryAtt(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    use_rope=use_rope,
                    max_seq_len=max_seq_len,
               )
          else:
               raise ValueError(f"unknown attention_type: {attention_type}")

          self.ln2 = make_norm(dim, norm_type)
          self.feedforward = make_feedforward(dim, ffn_hidden_dim, ffn_type)

     def forward(self, x):  
          att_out = self.attention(self.ln1(x)) 
          x = x + att_out 

          feedforward_out = self.feedforward(self.ln2(x)) 
          x = x + feedforward_out

          return x  


class TransformerBlocks(nn.Module):
     
     def __init__(
          self,
          dim=64,
          num_heads=8,
          use_rope=False,
          max_seq_len=128,
          norm_type="layernorm",
          num_layers=2,
          attention_type="mha",
          num_kv_heads=None,
          ffn_type="gelu",
          ffn_hidden_dim=512,
     ):
          super().__init__() 
          if num_layers < 1:
               raise ValueError("num_layers must be at least 1")

          self.blocks = nn.ModuleList(
              [
                  Transformerblock(
                      dim=dim,
                      num_heads=num_heads,
                      use_rope=use_rope,
                      max_seq_len=max_seq_len,
                      norm_type=norm_type,
                      attention_type=attention_type,
                      num_kv_heads=num_kv_heads,
                      ffn_type=ffn_type,
                      ffn_hidden_dim=ffn_hidden_dim,
                  )
                  for _ in range(num_layers)
              ]
          )

     def forward(self, x): 
          for block in self.blocks:
               x = block(x)

          return x  


TwoTransformerBlocks = TransformerBlocks

     
