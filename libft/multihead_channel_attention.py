import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadChannelAttention(nn.Module):
    def __init__(self, channels, num_heads, depthwise=True, include_conv=True, kernel_size=9, padding=4):
        super().__init__()

        self.num_heads = num_heads
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.o_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x, mem=None):
        b,c,h,w = x.shape
        q = self.q_proj(x).reshape(b,c,self.num_heads,-1).permute(0,2,1,3)
        k = self.k_proj(x).reshape(b,c,self.num_heads,-1).permute(0,2,3,1)
        v = self.v_proj(x).reshape(b,c,self.num_heads,-1).permute(0,2,1,3)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h * w)
            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(1,2).reshape(b,c,h,-1)

        x = self.o_proj(a)

        return x