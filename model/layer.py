import torch 
import torch.nn as nn   
from model.standard_attention import Attentionmech

class Transformerblock(nn.Module):

     def __init__(
          self,
          dim=64,
          num_heads=8,
          use_rope=False,
          max_seq_len=128,
     ):
          super().__init__()
          self.ln1 = nn.LayerNorm(dim)  
          self.attention = Attentionmech(
              dim=dim,
              num_heads=num_heads,
              use_rope=use_rope,
              max_seq_len=max_seq_len,
          )

          self.ln2 = nn.LayerNorm(dim)  
          self.feedforward = nn.Sequential(
              nn.Linear(dim, 512), 
              nn.GELU(), 
              nn.Linear(512, dim)
          )

     def forward(self, x):  
          att_out = self.attention(self.ln1(x)) 
          x = x + att_out 

          feedforward_out = self.feedforward(self.ln2(x)) 
          x = x + feedforward_out

          return x  


class TwoTransformerBlocks(nn.Module):
     
     def __init__(
          self,
          dim=64,
          num_heads=8,
          use_rope=False,
          max_seq_len=128,
     ):
          super().__init__() 

          self.block1 = Transformerblock(
              dim=dim,
              num_heads=num_heads,
              use_rope=use_rope,
              max_seq_len=max_seq_len,
          )
          self.block2 = Transformerblock(
              dim=dim,
              num_heads=num_heads,
              use_rope=use_rope,
              max_seq_len=max_seq_len,
          )

     def forward(self, x): 
          x= self.block1(x)
          x= self.block2(x)     

          return x  

     
