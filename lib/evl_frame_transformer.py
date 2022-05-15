import torch
from torch import nn
import torch.nn.functional as F
import math

class ReLU1(nn.Module):
    def __init__(self, inplace=True):
        super(ReLU1, self).__init__()

        self.relu = nn.ReLU6(inplace=inplace)

    def __call__(self, x):
        return self.relu(x) / 6.0

class FrameTransformer(nn.Module):
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=4, num_encoders=1, cropsize=1024, bias=False, out_activate=ReLU1(), dropout=0.1):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_encoders)])
        self.out = nn.Linear(channels + num_encoders, 2, bias=bias)
        self.activate = out_activate if out_activate is not None else nn.Identity()

    def __call__(self, src, mask=None):
        src = src[:, :, :self.max_bin]

        for module in self.encoder:
            t = module(src, mask=mask)
            src = torch.cat((src, t), dim=1)

        return F.pad(
            input=self.activate(self.out(src.transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )

class MultibandFrameAttention(nn.Module):
    def __init__(self, num_bands, bins, cropsize, kernel_size=3, padding=1):
        super().__init__()

        self.num_bands = num_bands

        self.q_proj = nn.Linear(bins, bins)
        self.q_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins)

        self.k_proj = nn.Linear(bins, bins)
        self.k_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins)

        self.v_proj = nn.Linear(bins, bins)
        self.v_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins)

        self.o_proj = nn.Linear(bins, bins)

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
            qk = qk + mask.unsqueeze(1)

        a = F.softmax(qk, dim=-1)
        a = torch.matmul(a,v).transpose(1,2).reshape(b,w,-1)
        o = self.o_proj(a)
        return o

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, bins, num_bands=4, cropsize=1024, feedforward_dim=2048, bias=False, dropout=0.1):
        super(FrameTransformerEncoder, self).__init__()

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(bins)
        self.glu = nn.Sequential(
            nn.Linear(bins, bins * 2, bias=bias),
            nn.GLU())
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=11, padding=5, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 2, kernel_size=1, padding=0, bias=bias))
        self.conv1R = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=7, padding=3, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 4, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(feedforward_dim // 2)
        self.conv1M = nn.Sequential(
            nn.Conv1d(feedforward_dim // 2, feedforward_dim // 2, kernel_size=7, padding=3, groups=feedforward_dim // 2, bias=bias),
            nn.Conv1d(feedforward_dim // 2, bins, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm4 = nn.LayerNorm(bins)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize, kernel_size=7, padding=3)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm5 = nn.LayerNorm(bins)
        self.conv2 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.conv3 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, mask=None):
        x = self.in_project(x.transpose(1,3)).squeeze(3)

        h = self.norm1(x)
        h = self.glu(h)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h))
        hR = self.relu(self.conv1R(h.transpose(1,2))).transpose(1,2)
        h = self.norm3(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))
        h = self.conv1M(h.transpose(1,2)).transpose(1,2)
        x = x + self.dropout2(h)

        h = self.norm4(x)
        h = self.attn(h, mask=mask)
        x = x + self.dropout3(h)

        h = self.norm5(x)
        h = self.conv3(torch.square(self.relu(self.conv2(h))))
        x = x + self.dropout4(h)

        return x.transpose(1,2).unsqueeze(1)