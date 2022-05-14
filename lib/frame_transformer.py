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
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=4, num_encoders=1, num_decoders=1, cropsize=1024, bias=False, out_activate=ReLU1()):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])
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
    def __init__(self, num_bands, bins, cropsize, kernel_size=3):
        super().__init__()

        self.num_bands = num_bands

        self.q_proj = nn.Linear(bins, bins)
        self.k_proj = nn.Linear(bins, bins)
        self.v_proj = nn.Linear(bins, bins)
        self.o_proj = nn.Linear(bins, bins)

        self.er = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.er)

    def forward(self, x, mem=None, mask=None):
        b,w,c = x.shape

        q = self.q_proj(x).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        k = self.k_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,3,1)
        v = self.v_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        p = F.pad(torch.matmul(q,self.er), (1,0)).transpose(2,3)[:,:,1:,:]
        qk = (torch.matmul(q,k)+p) / math.sqrt(c)

        if mask is not None:
            qk = qk + mask

        a = F.softmax(qk, dim=-1)
        a = torch.matmul(a,v).transpose(1,2).reshape(b,w,-1)
        o = self.o_proj(a)
        return o

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, bins, num_bands=4, cropsize=1024, feedforward_dim=2048, bias=False, dropout=0.1, autoregressive=False):
        super(FrameTransformerEncoder, self).__init__()

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands
        self.autoregressive = autoregressive

        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.norm1 = nn.LayerNorm(bins)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize, kernel_size=3)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(bins)
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout3 = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        x = self.in_project(x.transpose(1,3)).squeeze(3)

        h = self.norm1(x)
        h = self.attn(h, mask=mask)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.linear2(self.dropout2(torch.square(self.relu(self.linear1(h)))))
        x = x + self.dropout3(h)

        return x.transpose(1,2).unsqueeze(1)