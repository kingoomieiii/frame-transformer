import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from libft.rotary_embedding_torch import RotaryEmbedding
from libft.multichannel_linear import MultichannelLinear

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, depthwise=True, include_conv=True, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=1, bias=False) if include_conv else nn.Identity(),
            MultichannelLinear(channels, channels, features, features))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=1, bias=False) if include_conv else nn.Identity(),
            MultichannelLinear(channels, channels, features, features))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=1, bias=False) if include_conv else nn.Identity(),
            MultichannelLinear(channels, channels, features, features))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features, depthwise=depthwise)

    def __call__(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        k = self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)

            if prev_qk is not None:
                qk = qk + prev_qk

            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.o_proj(a)

        return x, qk