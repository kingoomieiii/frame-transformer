import torch
from torch import log_softmax, nn
import torch.nn.functional as F
import math
import lib.spec_utils as spec_utils

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

        self.er = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.er)

    def forward(self, x, mem=None, mask=None):
        b,w,c = x.shape

        q = self.q_conv(self.q_proj(x).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        k = self.q_conv(self.k_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,3,1)
        v = self.q_conv(self.v_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        p = F.pad(torch.matmul(q,self.er), (1,0)).transpose(2,3)[:,:,1:,:]
        qk = (torch.matmul(q,k)+p) / math.sqrt(c)

        if mask is not None:
            qk = qk + mask

        a = F.softmax(qk, dim=-1)
        a = torch.matmul(a,v).transpose(1,2).reshape(b,w,-1)
        o = self.o_proj(a)

        return o

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, bins=0, num_bands=4, cropsize=1024, feedforward_dim=2048, bias=False, dropout=0.1, downsamples=0, n_fft=2048):
        super(FrameTransformerEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.in_norm = nn.BatchNorm2d(channels)
        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(bins)
        self.glu = nn.Sequential(
            nn.Linear(bins, bins * 2, bias=bias),
            nn.GLU())
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Conv1d(bins, feedforward_dim, kernel_size=1, padding=0, bias=bias)
        self.conv1R =  nn.Conv1d(bins, bins // 2, kernel_size=3, padding=1, bias=bias)
        self.norm3 = nn.LayerNorm(feedforward_dim)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim, feedforward_dim, kernel_size=9, padding=4, groups=feedforward_dim // 2, bias=bias),
            nn.Conv1d(feedforward_dim, bins, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm4 = nn.LayerNorm(bins)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize, bias=bias)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm5 = nn.LayerNorm(bins)
        self.conv3 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.conv4 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x):
        x = self.in_project(self.in_norm(x).transpose(1,3)).squeeze(-1)

        h = self.norm1(x)
        h = self.glu(h)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h.transpose(1,2)).transpose(1,2))
        hR = self.conv1R(h.transpose(1,2)).transpose(1,2)
        h = self.norm3(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))
        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = x + h

        h = self.norm4(x)
        h = self.attn(h)
        x = x + self.dropout3(h)

        h = self.norm5(x)
        h = self.conv4(torch.square(self.relu(self.conv3(h))))
        x = x + self.dropout4(h)

        return x.transpose(1,2).unsqueeze(1)

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, skip_channels, num_bands=4, cropsize=1024, n_fft=2048, feedforward_dim=2048, downsamples=0, bias=False, dropout=0.1):
        super(FrameTransformerDecoder, self).__init__()

        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.in_project = nn.Linear(channels, 1, bias=bias)
        self.skip_project = nn.Linear(skip_channels, 1, bias=bias)

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(bins)
        self.self_attn1 = MultibandFrameAttention(num_bands, bins, cropsize, bias=bias)
        self.skip_attn1 = MultibandFrameAttention(num_bands, bins, cropsize, bias=bias)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=11, padding=5, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 2, kernel_size=1, padding=0, bias=bias))
        self.conv1R = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=7, padding=3, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 4, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(feedforward_dim // 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim // 2, feedforward_dim // 2, kernel_size=7, padding=3, groups=feedforward_dim // 2, bias=bias),
            nn.Conv1d(feedforward_dim // 2, bins, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm4 = nn.LayerNorm(bins)
        self.self_attn2 = MultibandFrameAttention(num_bands, bins, cropsize, bias=bias)
        self.norm4 = nn.LayerNorm(bins)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm5 = nn.LayerNorm(bins)
        self.skip_attn2 = MultibandFrameAttention(num_bands, bins, cropsize, bias=bias)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm6 = nn.LayerNorm(bins)
        self.conv3 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.silu = nn.SiLU(inplace=True)
        self.conv4 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout5 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip, mask=None):
        x = self.in_project(x.transpose(1,3)).squeeze(-1)
        skip = self.skip_project(skip.transpose(1,3)).squeeze(-1)

        h = self.norm1(x)
        hs = self.self_attn1(h, mask=mask)
        hm = self.skip_attn1(h, mem=skip, mask=mask)
        x = x + self.dropout1(hs + hm)

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h.transpose(1,2)).transpose(1,2))
        hR = self.conv1R(h.transpose(1,2)).transpose(1,2)
        h = self.norm3(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))
        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = x + h

        h = self.norm4(x)
        h = self.dropout3(self.self_attn2(h, mask=mask))
        x = x + h

        h = self.norm5(x)
        h = self.dropout4(self.skip_attn2(h, mem=skip, mask=mask))
        x = x + h

        h = self.norm6(x)
        h = self.conv4(self.silu(self.conv3(h)))
        x = x + self.dropout5(h)

        return x.transpose(1,2).unsqueeze(1)

class FrameNorm(nn.Module):
    def __init__(self, bins, channels):
        super(FrameNorm, self).__init__()

        self.norm = nn.LayerNorm(bins * channels)

    def __call__(self, x):
        h = x.reshape(x.shape[0], 1, x.shape[1] * x.shape[2], x.shape[3])
        return self.norm(h.transpose(2,3)).transpose(2,3).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, activate=nn.ReLU, norm=True, cropsize=1024, dropout=None):
        super(FrameConv, self).__init__()

        self.weight = nn.Parameter(torch.empty(kernel_size))

        self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                dilation=(dilation, 1),
                groups=groups,
                bias=False)

        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.activate = activate(inplace=True) if activate is not None else None

    def __call__(self, x):
        h = self.conv(x)
        
        if self.norm is not None:
            h = self.norm(h)

        if self.activate is not None:
            h = self.activate(h)

        return h
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU, cropsize=1024):
        super(FrameEncoder, self).__init__()

        self.conv1 = FrameConv(in_channels, out_channels, kernel_size, 1, padding, activate=activ, cropsize=cropsize)
        self.conv2 = FrameConv(out_channels, out_channels, kernel_size, stride, padding, activate=activ, cropsize=cropsize)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        return h

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activ=nn.LeakyReLU, norm=True, dropout=False, cropsize=1024):
        super(FrameDecoder, self).__init__()

        self.conv1 = FrameConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, activate=activ, norm=norm, cropsize=cropsize)
        self.conv2 = FrameConv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, activate=activ, norm=norm, cropsize=cropsize)
        self.dropout = nn.Dropout2d(dropout) if dropout is not None else None

    def __call__(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=(skip.shape[2],skip.shape[3]), mode='bilinear', align_corners=True)
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)

        h = self.conv1(x)
        h = self.conv2(h)

        if self.dropout is not None:
            h = self.dropout(h)

        return h