import math
import torch
from torch import nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding
from multichannel_layernorm import MultichannelLayerNorm, FrameNorm
from multichannel_linear import MultichannelLinear

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=15, unlock_first=4, unlock_last=4):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.lock_first = unlock_first
        self.lock_last = unlock_last
        self.channels = channels

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin)
        self.transformer = nn.Sequential(*[FrameTransformerEncoder(channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads) for _ in range(num_layers)])
        self.resnet = nn.Sequential(*[ResBlock(channels, channels, self.max_bin, expansion=2) for _ in range(num_layers)])
        self.out = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0)
        
    def __call__(self, x):
        h = self.enc1(x)

        for i in range(len(self.transformer)):
            h = self.transformer[i](h)
            h = self.resnet[i](h)

        out = self.out(h)

        return out

    def lock(self):
        for idx in range(len(self.transformer)):
            self.transformer[idx].lock()

class SquaredReLU(nn.Module):
    def __call__(self, x):
        return torch.relu(x) ** 2

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=False, expansion=1):
        super(ResBlock, self).__init__()

        self.activate = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1, bias=True)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True) if in_channels != out_channels or downsample else nn.Identity()
        self.norm = MultichannelLayerNorm(out_channels, features)

    def __call__(self, x):
        h = self.conv2(self.activate(self.conv1(x)))
        x = self.norm(self.identity(x) + h)

        return x

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, num_blocks=6):
        super(FrameEncoder, self).__init__()

        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, features, expansion=2) for i in range(num_blocks)])

    def __call__(self, x):
        x = self.body(x)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads, learned_freq=True)
        self.q_proj = MultichannelLinear(channels, channels, features, features, bias=True)
        self.k_proj = MultichannelLinear(channels, channels, features, features, bias=True)
        self.v_proj = MultichannelLinear(channels, channels, features, features, bias=True)
        self.out_proj = MultichannelLinear(channels, channels, features, features, bias=True)

    def __call__(self, x, mem=None):
        b,c,h,w = x.shape
        q = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.rotary_embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)
            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.out_proj(a)

        return x
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)
        self.norm1 = MultichannelLayerNorm(channels, features)

        self.conv1 = MultichannelLinear(channels, channels, features, features * expansion, bias=True)
        self.conv2 = MultichannelLinear(channels, channels, features * expansion, features, bias=True)
        self.norm2 = MultichannelLayerNorm(channels, features)

    def __call__(self, x):
        z = self.attn(x)
        h = self.norm1(x + self.dropout(z))

        z = self.conv2(self.activate(self.conv1(h)))
        h = self.norm2(h + self.dropout(z))

        return h

    def lock(self):
        self.norm1.requires_grad_(False)
        self.conv1.requires_grad_(False)
        self.conv2.requires_grad_(False)
        self.norm2.requires_grad_(False)