import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from libft.rotary_embedding_torch import RotaryEmbedding
from libft.multichannel_linear import MultichannelLinear

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, depthwise=False):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(features  // num_heads)
        
        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=depthwise),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1), groups=channels))
        
        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=depthwise),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1), groups=channels))
        
        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=depthwise),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1), groups=channels))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features, depthwise=depthwise)

    def __call__(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        k = self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,4,2)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)
        x = self.o_proj(a)

        return x, qk