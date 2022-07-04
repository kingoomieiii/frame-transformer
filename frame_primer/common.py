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
        p = F.pad(torch.matmul(q,self.weight), (1,0)).transpose(2,3)[:,:,1:,:]

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = (torch.matmul(q,k)+p) / math.sqrt(c)

            if prev_qk is not None:
                qk = qk + prev_qk

            a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(1,2).reshape(b,w,-1).contiguous()
                
        o = self.o_proj(a)

        return o, qk

class FramePrimerEncoder(nn.Module):
    def __init__(self, channels, bins=0, num_bands=4, cropsize=1024, feedforward_dim=2048, bias=False, dropout=0.1, downsamples=0, n_fft=2048):
        super(FramePrimerEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)

        self.in_norm = nn.InstanceNorm2d(cropsize, affine=True)
        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.norm1 = nn.InstanceNorm2d(cropsize, affine=True)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.InstanceNorm2d(cropsize, affine=True)
        self.linear1 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.linear2 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, prev_qk=None):
        x = self.in_project(self.relu(self.in_norm(x.transpose(1,3)))).squeeze(-1)

        h = self.norm1(x.unsqueeze(-1)).squeeze(-1)
        h, prev_qk = self.attn(h, prev_qk=prev_qk)
        x = x + self.dropout1(h)
        
        h = self.norm2(x.unsqueeze(-1)).squeeze(-1)
        h = self.linear2(torch.square(self.relu(self.linear1(h))))
        x = x + self.dropout2(h)

        return x.transpose(1,2).unsqueeze(1), prev_qk

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

        self.in_norm = nn.InstanceNorm2d(cropsize, affine=True)
        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.mem_norm = nn.InstanceNorm2d(cropsize, affine=True)
        self.mem_project = nn.Linear(mem_channels, 1, bias=bias)

        self.norm1 = nn.InstanceNorm2d(cropsize, affine=True)
        self.attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.InstanceNorm2d(cropsize, affine=True)
        self.attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm3 = nn.InstanceNorm2d(cropsize, affine=True)
        self.linear1 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.linear2 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, mem=None, prev_qk1=None, prev_qk2=None):
        x = self.in_project(self.relu(self.in_norm(x.transpose(1,3)))).squeeze(-1)
        mem = self.mem_project(self.relu(self.mem_norm(mem.transpose(1,3)))).squeeze(-1)

        h = self.norm1(x.unsqueeze(-1)).squeeze(-1)
        h, prev_qk1 = self.attn1(h, prev_qk=prev_qk1)
        x = x + self.dropout1(h)

        h = self.norm2(x.unsqueeze(-1)).squeeze(-1)
        h, prev_qk2 = self.attn2(h, mem=mem, prev_qk=prev_qk2)
        x = x + self.dropout2(h)
        
        h = self.norm3(x.unsqueeze(-1)).squeeze(-1)
        h = self.linear2(torch.square(self.relu(self.linear1(h))))
        x = x + self.dropout3(h)

        return x.transpose(1,2).unsqueeze(1), prev_qk1, prev_qk2

class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, activate=nn.ReLU, norm=True, cropsize=1024, downsamples=0, n_fft=2048, dropout=None):
        super(FrameConv, self).__init__()

        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.norm = nn.InstanceNorm2d(cropsize, affine=True) if norm else None
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
        if self.norm is not None:
            x = self.norm(x.transpose(1,3)).transpose(1,3)

        if self.activate is not None:
            x = self.activate(x)

        x = self.conv(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU, cropsize=1024, downsamples=0, n_fft=2048):
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU, cropsize=1024, downsamples=0, n_fft=2048, num_res_blocks=1):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, cropsize=cropsize)
        
    def __call__(self, x):
        h = self.body(x)
        
        return h

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, upsample=True, activ=nn.LeakyReLU, norm=True, dropout=False, cropsize=1024, downsamples=0, n_fft=2048, num_res_blocks=1):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True) if upsample else nn.Identity()
        self.body = ResBlock(in_channels, out_channels, cropsize=cropsize)

    def __call__(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)
            
        h = self.body(x)

        return h