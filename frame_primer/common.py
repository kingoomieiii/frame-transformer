from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import math

class MultibandFrameAttention(nn.Module):
    def __init__(self, num_bands, bins, cropsize, kernel_size=3, padding=1, bias=False):
        super().__init__()

        self.num_bands = num_bands
        self.q_proj = nn.Linear(bins, bins, bias=bias)
        self.q_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, bias=bias)
        self.k_proj = nn.Linear(bins, bins, bias=bias)
        self.k_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, bias=bias)
        self.v_proj = nn.Linear(bins, bins, bias=bias)
        self.v_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, bias=bias)
        self.o_proj = nn.Linear(bins, bins, bias=bias)

        self.weight = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.weight)

    def forward(self, x, mem=None, prev_qk=None):
        b,w,c = x.shape

        q = self.q_conv(self.q_proj(x).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3).contiguous()
        k = self.k_conv(self.k_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,3,1).contiguous()
        v = self.v_conv(self.v_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3).contiguous()
        p = F.pad(torch.matmul(q,self.weight[:, :w]), (1,0)).transpose(2,3)[:,:,1:,:]

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = (torch.matmul(q,k)+p) / math.sqrt(c)

            if prev_qk is not None:
                qk = qk + prev_qk

            a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(1,2).reshape(b,w,-1).contiguous()
                
        o = self.o_proj(a)

        return o, qk

class FramePrimerEncoder2(nn.Module):
    def __init__(self, channels, num_bands=4, cropsize=1024, bias=False, dropout=0.1, downsamples=0, n_fft=2048, expansion=4):
        super(FramePrimerEncoder2, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)
        
        self.norm1 = nn.LayerNorm(bins * channels)
        self.attn = MultibandFrameAttention(channels, bins * channels, cropsize)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins * channels)
        self.linear1 = nn.Linear(bins * channels, bins * channels * expansion, bias=bias)
        self.linear2 = nn.Linear(bins * channels * expansion, bins * channels, bias=bias)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, prev_qk=None):
        z = self.norm1(x)
        z, prev_qk = self.attn(z, prev_qk=prev_qk)
        x = x + self.dropout1(z)
        
        z = self.norm2(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout2(z)

        return x, prev_qk

class FramePrimerEncoder(nn.Module):
    def __init__(self, channels, bins=0, num_bands=4, cropsize=1024, feedforward_dim=2048, bias=False, dropout=0.1, downsamples=0, n_fft=2048, bottleneck=1, expansion=4):
        super(FramePrimerEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)
        
        self.bottleneck = bottleneck
        self.in_project = nn.Conv2d(channels, 1, kernel_size=1, padding=0) if channels != self.bottleneck else nn.Identity()

        self.norm1 = nn.LayerNorm(bins * self.bottleneck)
        self.attn = MultibandFrameAttention(num_bands, bins * self.bottleneck, cropsize)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins * self.bottleneck)
        self.linear1 = nn.Linear(bins * self.bottleneck, bins * self.bottleneck * expansion, bias=bias)
        self.linear2 = nn.Linear(bins * self.bottleneck * expansion, bins * self.bottleneck, bias=bias)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, prev_qk=None):
        b,c,h,w = x.shape
        x = self.in_project(x).transpose(1,3).reshape(b,w,h*self.bottleneck)

        z = self.norm1(x)
        z, prev_qk = self.attn(z, prev_qk=prev_qk)
        x = x + self.dropout1(z)
        
        z = self.norm2(x)
        z = self.linear2(torch.square(self.relu(self.linear1(z))))
        x = x + self.dropout2(z)

        return x.transpose(1,2).reshape(b,self.bottleneck,h,w), prev_qk

class FramePrimerDecoder(nn.Module):
    def __init__(self, channels, mem_channels, bins=0, num_bands=4, cropsize=1024, feedforward_dim=2048, bias=False, dropout=0.1, downsamples=0, n_fft=2048):
        super(FramePrimerDecoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)

        self.in_project = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        self.mem_project = nn.Conv2d(mem_channels, 1, kernel_size=1, padding=0)

        self.norm1 = nn.LayerNorm(bins)
        self.attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm3 = nn.LayerNorm(bins)
        self.linear1 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.linear2 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, mem=None, prev_qk1=None, prev_qk2=None):
        x = self.in_project(x).transpose(1,3).squeeze(-1)
        mem = self.mem_project(mem).transpose(1,3).squeeze(-1)

        h = self.norm1(x)
        h, prev_qk1 = self.attn1(h, prev_qk=prev_qk1)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h, prev_qk2 = self.attn2(h, mem=mem, prev_qk=prev_qk2)
        x = x + self.dropout2(h)
        
        h = self.norm3(x)
        h = self.linear2(torch.square(self.relu(self.linear1(h))))
        x = x + self.dropout3(h)

        return x.transpose(1,2).unsqueeze(1), prev_qk1, prev_qk2

class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, activate=nn.SiLU, norm=True, cropsize=1024, downsamples=0, n_fft=2048, dropout=None):
        super(FrameConv, self).__init__()

        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.norm = nn.LayerNorm(bins * in_channels) if norm else None
        self.activate = activate(inplace=True) if activate is not None else None

        self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                stride=(stride, 1),
                dilation=(dilation, 1),
                groups=groups,
                bias=False)

    def __call__(self, x):
        b,c,h,w = x.shape

        if self.norm is not None:
            x = self.norm(x.transpose(1,3).reshape(b,w,h*c)).reshape(b,w,h,c).transpose(1,3)

        if self.activate is not None:
            x = self.activate(x)

        x = self.conv(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.SiLU, cropsize=1024, downsamples=0, n_fft=2048):
        super(ResBlock, self).__init__()

        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(stride, 1)) if in_channels != out_channels or stride > 1 else nn.Identity()
        self.conv1 = FrameConv(in_channels, out_channels, kernel_size, 1, padding, activate=activ, cropsize=cropsize, downsamples=downsamples, n_fft=n_fft)
        self.conv2 = FrameConv(out_channels, out_channels, kernel_size, stride, padding, activate=activ, cropsize=cropsize, downsamples=downsamples, n_fft=n_fft)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = h + self.identity(x)

        return h
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.SiLU, cropsize=1024, downsamples=0, n_fft=2048, num_res_blocks=1):
        super(FrameEncoder, self).__init__()

        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride if i == num_res_blocks - 1 else 1, cropsize=cropsize, downsamples=downsamples, n_fft=n_fft) for i in range(0, num_res_blocks)])
        
    def __call__(self, x):
        h = self.body(x)
        
        return h

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, upsample=True, activ=nn.SiLU, norm=True, dropout=False, cropsize=1024, downsamples=0, n_fft=2048, num_res_blocks=1):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True) if upsample else nn.Identity()
        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding=padding, cropsize=cropsize, downsamples=downsamples, n_fft=n_fft) for i in range(0, num_res_blocks)])

    def __call__(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
            
        h = self.body(x)

        return h