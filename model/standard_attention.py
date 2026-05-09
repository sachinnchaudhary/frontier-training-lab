import torch.nn as nn 
import torch 
import math  
from model.rope import RoPE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

class Attentionmech(nn.Module):

     def __init__(
         self,
         dim: int = 64,
         num_heads: int = 8,
         use_rope: bool = False,
         max_seq_len: int = 128,
     ):
         super().__init__()

         if dim % num_heads != 0:
             raise ValueError("dim must be divisible by num_heads")
                   
         self.head_dim = dim // num_heads
         self.num_heads = num_heads
         self.dim = dim
         self.use_rope = use_rope
         self.query = nn.Linear(dim, dim, bias=None)  
         self.key = nn.Linear(dim, dim, bias= None)  
         self.value = nn.Linear(dim, dim, bias= None)  
         self.sftmx = torch.nn.Softmax(dim=-1)
         self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len) if use_rope else None


     def forward(self, x):  
            
            B, T, D = x.shape   
            if D != self.dim:
                raise ValueError(f"expected embedding dim {self.dim}, got {D}")

            q = self.query(x).reshape(B, T, self.num_heads, self.head_dim)  
            k = self.key(x).reshape(B, T, self.num_heads, self.head_dim) 
            v = self.value(x).reshape(B, T, self.num_heads, self.head_dim)    

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.rope is not None:
                q = self.rope(q)
                k = self.rope(k)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = self.sftmx(scores)
            logits = weights @ v
            logits = logits.transpose(1, 2).contiguous().reshape(B, T, D)
          
            return logits  

    
          
        

    



       

             


             






