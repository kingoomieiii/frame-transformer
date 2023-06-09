import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from libft2gan.causal_conv2d import CausalConv2d
from libft2gan.rotary_embedding_torch import RotaryEmbedding
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.multichannel_layernorm import MultichannelLayerNorm

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1, dtype=torch.float, causal=False, mem_causal=None):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads, dtype=dtype)

        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, dtype=dtype),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), dtype=dtype))
        
        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, dtype=dtype),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), dtype=dtype))
        
        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, dtype=dtype),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), dtype=dtype))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features, dtype=dtype)
        
    def forward(self, x, mem=None, prev_qk=None, mask=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        if mask is not None:
            qk = qk + mask

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)
        x = self.o_proj(a)

        return x, qk