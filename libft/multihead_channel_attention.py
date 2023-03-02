import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadChannelAttention(nn.Module):
    def __init__(self, channels, num_heads, depthwise=True, include_conv=True, kernel_size=9, padding=4):
        super().__init__()

        self.num_heads = num_heads

        self.pos = None        
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.o_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)

    def __call__(self, x, mem=None):
        b,c,h,w = x.shape

        if self.pos is None:        
            position = torch.arange(c).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, (h * w), 2) * (-math.log(10000.0) / (h * w)))
            pe = torch.zeros(c, 1, (h * w))
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.pos = pe.squeeze(1).reshape((c,h,w)).unsqueeze(0).expand((b, -1, -1, -1)).to(x.device)

        x = x + self.pos
        q = self.q_proj(x).reshape(b,c,self.num_heads,-1).permute(0,2,1,3)
        k = self.k_proj(x).reshape(b,c,self.num_heads,-1).permute(0,2,3,1)
        v = self.v_proj(x).reshape(b,c,self.num_heads,-1).permute(0,2,1,3)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h * w)

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(1,2).reshape(b,c,h,-1)
        x = self.o_proj(a)

        return x