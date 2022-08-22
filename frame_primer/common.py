from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import math

from frame_primer.rotary_embedding_torch import RotaryEmbedding

class MultichannelLinear(nn.Module):
    def __init__(self, channels, in_features, out_features, separable=True):
        super(MultichannelLinear, self).__init__()

        self.separable = separable
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(channels, out_features, in_features)) if separable else nn.Parameter(torch.empty(out_features * channels, in_features * channels))
        nn.init.uniform_(self.weight, a=-1/math.sqrt(in_features if separable else in_features * channels), b=1/math.sqrt(in_features if separable else in_features * channels))

    def __call__(self, x):
        b,c,h,w = x.shape

        if self.separable:
            return torch.matmul(x.transpose(2,3), self.weight.transpose(1,2)).transpose(2,3)
        else:
            return torch.matmul(x.permute(0,3,2,1).reshape(b,w,h*c), self.weight.t()).reshape(b,w,self.out_features,c).permute(0,3,2,1)

class MultichannelLinear2(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, skip_redundant=False):
        super(MultichannelLinear2, self).__init__()

        self.weight_frame = None
        if in_features != out_features or not skip_redundant:
            self.weight_frame = nn.Parameter(torch.empty(out_channels, out_features, in_features))
            nn.init.uniform_(self.weight_frame, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features))

        self.weight_depthwise = None
        if in_channels != out_channels or not skip_redundant:
            self.weight_depthwise = nn.Parameter(torch.empty(out_channels, in_channels))
            nn.init.uniform_(self.weight_depthwise, a=-1/math.sqrt(in_channels), b=1/math.sqrt(out_channels))

    def __call__(self, x):
        if self.weight_depthwise is not None:
            x = torch.matmul(x.transpose(1,3), self.weight_depthwise.t()).transpose(1,3)
        
        if self.weight_frame is not None:
            x = torch.matmul(x.transpose(2,3), self.weight_frame.transpose(1,2)).transpose(2,3)
        
        return x

class FrameNorm(nn.Module):
    def __init__(self, bins, channels=None):
        super(FrameNorm, self).__init__()

        self.norm = nn.LayerNorm((bins, channels)) if channels is not None else nn.LayerNorm(bins)

    def __call__(self, x):
        if len(self.norm.weight.shape) == 1:
            return self.norm(x.transpose(2,3)).transpose(2,3)
        else:
            return self.norm(x.transpose(1,3)).transpose(1,3)


class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activate=nn.GELU, norm=True, downsamples=0, n_fft=2048):
        super(FrameConv, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.norm = nn.LayerNorm(bins * in_channels) if norm else None # FrameNorm(bins, in_channels)
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
            x = self.norm(x.transpose(1,3).reshape(b,w,h*c)).reshape(b,w,h,c).transpose(1,3) # self.norm(x)

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

class FrameEncoder2(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, kernel_size=3, stride=1, padding=1, downsamples=0, n_fft=2048, num_res_blocks=1):
        super(FrameEncoder2, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.norm = FrameNorm(bins)
        self.gelu = nn.GELU()
        self.linear1 = MultichannelLinear2(in_channels, out_channels, bins, bins * 2)
        self.linear2 = MultichannelLinear2(out_channels, out_channels, bins * 2, bins // 2 if downsample else bins)
        self.identity = MultichannelLinear2(in_channels, out_channels, bins, bins // 2 if downsample else bins, skip_redundant=True)

    def __call__(self, x):
        h = self.norm(x)
        h = self.gelu(h)
        h = self.linear2(self.gelu(self.linear1(h)))
        h = h + self.identity(x)
        
        return h

class FrameDecoder2(nn.Module):
    def __init__(self, in_channels, out_channels, downsamples=0, n_fft=2048, upsample=True, expansion=2):
        super(FrameDecoder2, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.upsample = MultichannelLinear2(in_channels, in_channels, bins, bins * 2) if upsample else nn.Identity()

        self.norm = FrameNorm(bins)
        self.gelu = nn.GELU()
        self.linear1 = MultichannelLinear2(in_channels + out_channels, out_channels, bins, bins * expansion)
        self.linear2 = MultichannelLinear2(out_channels, out_channels, bins * expansion, bins)
        self.identity = MultichannelLinear2(in_channels + out_channels, out_channels, bins, bins, skip_redundant=True)
        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True) if upsample else nn.Identity()

    def __call__(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)

        h = self.norm(x)
        h = self.gelu(h)
        h = self.linear2(self.gelu(self.linear1(h)))
        h = h + self.identity(x)
            
        return h

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, bins, kernel_size=3, padding=1, separable=True):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = bins // num_heads // 2, freqs_for='lang')

        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, bins, bins, separable=separable),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding)))
        
        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, bins, bins, separable=separable),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding)))

        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, bins, bins, separable=separable),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding)))

        self.out_proj = MultichannelLinear(channels, bins, bins, separable=separable)

    def forward(self, x, mem=None):
        b,c,h,w = x.shape

        q = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).contiguous()
        k = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4).contiguous()
        v = self.q_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4).contiguous()

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q,k) / math.sqrt(h)
            a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3).contiguous()
                
        x = self.out_proj(a)

        return x

class FramePrimerEncoder(nn.Module):
    def __init__(self, channels, num_heads=4, bias=False, dropout=0.1, downsamples=0, n_fft=2048, expansion=4, feedforward_dim=4096, kernel_size=3, padding=1):
        super(FramePrimerEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = FrameNorm(bins)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, bins, kernel_size=kernel_size, padding=padding)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(bins)
        self.linear1 = MultichannelLinear(channels, bins, bins * expansion)
        self.linear2 = MultichannelLinear(channels, bins * expansion, bins)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def __call__(self, x):
        z = self.norm1(x)
        z = self.attn(z)
        x = x + self.dropout1(z.transpose(2,3)).transpose(2,3)

        z = self.norm2(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout2(z.transpose(2,3)).transpose(2,3)

        return x

class FramePrimerDecoder(nn.Module):
    def __init__(self, channels, num_heads=4, bias=False, dropout=0.1, downsamples=0, n_fft=2048, expansion=4, feedforward_dim=4096, kernel_size=3, padding=1):
        super(FramePrimerDecoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = FrameNorm(bins)
        self.attn1 = MultichannelMultiheadAttention(channels, num_heads, bins, kernel_size=kernel_size, padding=padding)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(bins)
        self.attn2 = MultichannelMultiheadAttention(channels, num_heads, bins, kernel_size=kernel_size, padding=padding)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm3 = FrameNorm(bins)
        self.linear1 = MultichannelLinear(channels, bins, bins * expansion)
        self.linear2 = MultichannelLinear(channels, bins * expansion, bins)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip):
        z = self.norm1(x)
        z = self.attn1(z)
        x = x + self.dropout1(z.transpose(2,3)).transpose(2,3)

        z = self.norm2(x)
        z = self.attn2(z, mem=skip)
        x = x + self.dropout2(z.transpose(2,3)).transpose(2,3)

        z = self.norm3(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout3(z.transpose(2,3)).transpose(2,3)

        return x

class FramePrimerEncoder2(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1, downsamples=0, n_fft=2048, expansion=4, kernel_size=3, padding=1):
        super(FramePrimerEncoder2, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = FrameNorm(bins)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, bins, kernel_size=kernel_size, padding=padding)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(bins)
        self.linear1 = MultichannelLinear2(channels, channels, bins, bins * expansion, skip_redundant=True)
        self.linear2 = MultichannelLinear2(channels, channels, bins * expansion, bins, skip_redundant=True)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def __call__(self, x):
        z = self.norm1(x)
        z = self.attn(z)
        x = x + self.dropout1(z.transpose(2,3)).transpose(2,3)

        z = self.norm2(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout2(z.transpose(2,3)).transpose(2,3)

        return x

class FramePrimerDecoder2(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1, downsamples=0, n_fft=2048, expansion=4, kernel_size=3, padding=1):
        super(FramePrimerDecoder2, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = FrameNorm(bins)
        self.attn1 = MultichannelMultiheadAttention(channels, num_heads, bins, kernel_size=kernel_size, padding=padding)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(bins)
        self.attn2 = MultichannelMultiheadAttention(channels, num_heads, bins, kernel_size=kernel_size, padding=padding)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm3 = FrameNorm(bins)
        self.linear1 = MultichannelLinear2(channels, channels, bins, bins * expansion, skip_redundant=True)
        self.linear2 = MultichannelLinear2(channels, channels, bins * expansion, bins, skip_redundant=True)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip):
        z = self.norm1(x)
        z = self.attn1(z)
        x = x + self.dropout1(z.transpose(2,3)).transpose(2,3)

        z = self.norm2(x)
        z = self.attn2(z, mem=skip)
        x = x + self.dropout2(z.transpose(2,3)).transpose(2,3)

        z = self.norm3(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout3(z.transpose(2,3)).transpose(2,3)

        return x