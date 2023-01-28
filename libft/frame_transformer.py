import math
import torch
from torch import nn
import torch.nn.functional as F

from libft.res_block import ResBlock
from libft.positional_embedding import PositionalEmbedding
from libft.multichannel_layernorm import MultichannelLayerNorm
from libft.multichannel_linear import MultichannelLinear

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=15, repeats=1):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.channels = channels
        self.out_channels = out_channels
        self.repeats = repeats

        self.positional_embedding = PositionalEmbedding(channels, self.max_bin)
        self.transformer = nn.Sequential(*[FrameTransformerEncoder(channels + 1, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads) for _ in range(num_layers)])

    def __call__(self, x):
        if x.shape[1] != self.channels:
            x = torch.cat([x for _ in range(self.channels // x.shape[1])], dim=1)

        h = torch.cat((self.positional_embedding(x), x), dim=1)
        return self.transformer(h)[:, -self.out_channels:]

class MultibandFrameAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)    
        self.o_proj = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

    def __call__(self, x, mem=None):
        b,c,h,w = x.shape
        q = self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        k = self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,4,2)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)
            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.o_proj(a)

        return x
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultibandFrameAttention(channels, num_heads, features)
        self.alpha1 = nn.Parameter(torch.zeros(channels, features, 1))

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        self.alpha2 = nn.Parameter(torch.zeros(channels, features, 1))
        
    def __call__(self, x):       
        z = self.attn(self.norm1(x))
        h = x + self.dropout(z) * self.alpha1

        z = self.conv2(torch.relu(self.conv1(self.norm2(h))) ** 2)
        h = h + self.dropout(z) * self.alpha2

        return h