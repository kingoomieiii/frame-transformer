from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import math

from frame_primer.rotary_embedding_torch import RotaryEmbedding

class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activate=nn.GELU, norm=True, downsamples=0, n_fft=2048):
        super(FrameConv, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.norm = nn.LayerNorm(bins * in_channels) if norm else None
        self.activate = activate() if activate is not None else None

        self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                stride=(stride, 1),
                groups=groups,
                bias=False)

    def __call__(self, x):
        b,c,h,w = x.shape

        if self.norm is not None:
            x = self.norm(x.transpose(1,3).reshape(b,w,h*c)).reshape(b,w,h,c).transpose(1,3)

        if self.activate is not None:
            x = self.activate(x)

        x = x.permute(0,3,1,2).reshape(b*w,c,h).unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze(-1).reshape(b,w,x.shape[1],x.shape[2]).permute(0,2,3,1)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activate=nn.GELU, downsamples=0, n_fft=2048, expansion=1):
        super(ResBlock, self).__init__()

        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(stride, 1)) if in_channels != out_channels or stride > 1 else nn.Identity()
        self.conv1 = FrameConv(in_channels, out_channels * expansion, kernel_size, 1, padding, activate=activate, downsamples=downsamples, n_fft=n_fft)
        self.conv2 = FrameConv(out_channels * expansion, out_channels, kernel_size, stride, padding, activate=activate, downsamples=downsamples, n_fft=n_fft)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = h + self.identity(x)

        return h
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsamples=0, n_fft=2048, num_res_blocks=1):
        super(FrameEncoder, self).__init__()

        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride if i == num_res_blocks - 1 else 1, downsamples=downsamples, n_fft=n_fft) for i in range(0, num_res_blocks)])
        
    def __call__(self, x):
        h = self.body(x)
        
        return h

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, downsamples=0, n_fft=2048, num_res_blocks=1, upsample=True):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True) if upsample else nn.Identity()
        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding=padding, downsamples=downsamples, n_fft=n_fft) for i in range(0, num_res_blocks)])

    def __call__(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
            
        h = self.body(x)

        return h

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, num_heads, bins, kernel_size=3, padding=1, bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = bins // num_heads // 2, learned_freq=True)
        self.q_proj = nn.Linear(bins, bins, bias=bias)
        self.q_conv = FrameConv(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, norm=False, activate=None)
        self.k_proj = nn.Linear(bins, bins, bias=bias)
        self.k_conv = FrameConv(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, norm=False, activate=None)
        self.v_proj = nn.Linear(bins, bins, bias=bias)
        self.v_conv = FrameConv(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, norm=False, activate=None)
        self.out_proj = nn.Linear(bins, bins, bias=bias)

    def forward(self, x, mem=None):
        b,c,w,h = x.shape
        q = self.rotary_embedding.rotate_queries_or_keys(self.q_conv(self.q_proj(x).transpose(1,3)).transpose(1,3).reshape(b, c, w, self.num_heads, -1).permute(0,1,3,2,4)).contiguous()
        k = self.rotary_embedding.rotate_queries_or_keys(self.k_conv(self.k_proj(x if mem is None else mem).transpose(1,3)).transpose(1,3).reshape(b, c, w, self.num_heads, -1).permute(0,1,3,2,4)).transpose(3,4).contiguous()
        v = self.v_conv(self.v_proj(x if mem is None else mem).transpose(1,3)).transpose(1,3).reshape(b, c, w, self.num_heads, -1).permute(0,1,3,2,4).contiguous()

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q,k) / math.sqrt(h)
            a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).contiguous()
                
        x = self.out_proj(a)

        return x

class FramePrimerEncoder(nn.Module):
    def __init__(self, num_heads=4, bias=False, dropout=0.1, downsamples=0, n_fft=2048, expansion=4):
        super(FramePrimerEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.num_bands = num_heads

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(bins)
        self.attn = MultichannelMultiheadAttention(num_heads, bins, kernel_size=9, padding=4)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.linear1 = nn.Linear(bins, bins * expansion, bias=bias)
        self.linear2 = nn.Linear(bins * expansion, bins, bias=bias)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def __call__(self, x):
        x = x.transpose(2,3)

        z = self.norm1(x)
        z = self.attn(z)
        x = x + self.dropout1(z)

        z = self.norm2(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout2(z)

        return x.transpose(2,3)

class FramePrimerDecoder(nn.Module):
    def __init__(self, num_heads=4, bias=False, dropout=0.1, downsamples=0, n_fft=2048, expansion=4):
        super(FramePrimerDecoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.num_bands = num_heads

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(bins)
        self.attn1 = MultichannelMultiheadAttention(num_heads, bins, kernel_size=9, padding=4)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.attn2 = MultichannelMultiheadAttention(num_heads, bins, kernel_size=9, padding=4)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm3 = nn.LayerNorm(bins) 
        self.linear1 = nn.Linear(bins, bins * expansion, bias=bias)
        self.linear2 = nn.Linear(bins * expansion, bins, bias=bias)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip=None):
        x = x.transpose(2,3)
        skip = skip.transpose(2,3)

        z = self.norm1(x)
        z = self.attn1(z)
        x = x + self.dropout1(z)

        z = self.norm2(x)
        z = self.attn2(z, mem=skip)
        x = x + self.dropout2(z)

        z = self.norm3(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout3(z)

        return x.transpose(2,3)