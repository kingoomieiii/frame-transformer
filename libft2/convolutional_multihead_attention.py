import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from libft2.rotary_embedding_torch import RotaryEmbedding

class ConvolutionalMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(channels // num_heads)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.o_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape

        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)).transpose(2,3)
        v = self.v_proj(x if mem is None else mem).reshape(b,c,h*w).transpose(1,2).reshape(b,h*w,self.num_heads,-1).permute(0,2,1,3)
        qk = torch.matmul(q,k) / math.sqrt(c)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(1,2).reshape(b,h,w,c).permute(0,3,1,2)
        x = self.o_proj(a)

        return x, qk