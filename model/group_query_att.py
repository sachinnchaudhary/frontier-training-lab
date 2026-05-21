import math

import torch
import torch.nn as nn

from model.rope import RoPE

class GroupQueryAtt(nn.Module):  

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        use_rope: bool,
    ):
        super().__init__()   
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.dim = dim
        self.num_heads = num_heads  
        self.num_kv_heads = num_kv_heads 
        self.head_dim = dim // num_heads 
        self.group_size = num_heads // num_kv_heads 
        self.use_rope = use_rope

        self.q_proj = nn.Linear(dim, dim, bias=None)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=None)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=None)
        self.out_proj = nn.Linear(dim, dim, bias=None)
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len) if use_rope else None

    def forward(self, x):  
        B, T, D = x.shape  
        if D != self.dim:
            raise ValueError(f"expected embedding dim {self.dim}, got {D}")

        q = self.q_proj(x) 
        k = self.k_proj(x)
        v = self.v_proj(x) 

        q = q.view(B, T, self.num_heads, self.head_dim) 
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)    

        if self.rope is not None:
             q = self.rope(q) 
             k = self.rope(k)  

        k = torch.repeat_interleave(k, repeats=self.group_size, dim=1)
        v = torch.repeat_interleave(v, repeats=self.group_size, dim=1)     
        
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        
        out = weights @ v

        out = out.transpose(1, 2) 
        out = out.reshape(B, T, D)  

        return self.out_proj(out) 



        
