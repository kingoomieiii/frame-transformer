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

        self.embed = ResBlock(in_channels, channels - in_channels * repeats, self.max_bin) if in_channels != channels else nn.Identity()
        self.positional_embedding = PositionalEmbedding(channels, self.max_bin)
        self.transformer = nn.Sequential(*[FrameTransformerEncoder(in_channels * repeats + 1, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, out_channels=out_channels) for _ in range(num_layers)])

    def __call__(self, x):
        h = torch.cat((self.embed(x), *[x for _ in range(self.repeats)]), dim=1)
        h = torch.cat((self.positional_embedding(h), h), dim=1)
        h = self.transformer(h)
        return h[:, -self.out_channels:]

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, num_blocks=3):
        super(FrameEncoder, self).__init__()

        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, features, downsample=True if i == num_blocks - 1 and downsample else False) for i in range(num_blocks)])

    def __call__(self, x):
        x = self.body(x)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features):
        super().__init__()

        self.num_heads = num_heads

        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, bias=True),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)))

        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, bias=True),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)))
            
        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, bias=True),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)))
            
        self.o_proj = MultichannelLinear(channels, channels, features, features, depthwise=True)

    def __call__(self, x, mem=None):
        b,c,h,w = x.shape
        q = self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        k = self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,4,2)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q, k) / math.sqrt(h)
        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)
        x = self.o_proj(a)

        return x
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8, out_channels=2):
        super(FrameTransformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)
        self.alpha1 = nn.Parameter(torch.ones(channels, features, 1))
        self.alpha1.data[-out_channels:] = 0

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        self.alpha2 = nn.Parameter(torch.ones(channels, features, 1))
        self.alpha2.data[-out_channels:] = 0
        
    def __call__(self, x):       
        z = self.attn(self.norm1(x))
        h = x + self.dropout(z) * self.alpha1

        z = self.conv2(torch.relu(self.conv1(self.norm2(h))) ** 2)
        h = h + self.dropout(z) * self.alpha2

        return h