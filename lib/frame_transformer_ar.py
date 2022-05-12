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
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=4, num_encoders=1, num_decoders=1, cropsize=1024, bias=False, autoregressive=False, out_activate=ReLU1(), encoder_only=False):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        
        self.encoder_only = encoder_only
        self.register_buffer('mask', torch.triu(torch.ones(cropsize, cropsize) * float('-inf'), diagonal=1))
        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])
        self.decoder = nn.ModuleList([FrameTransformerDecoder(channels + i, channels + num_encoders, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)]) if not encoder_only else None
        self.out = nn.Linear(channels + (num_decoders if not encoder_only else num_encoders), 2, bias=bias)
        self.activate = out_activate if out_activate is not None else nn.Identity()

        # self.register_buffer('indices', torch.arange(cropsize))
        # self.embedding = nn.Embedding(cropsize, self.max_bin)

    def embed(self, x):
        e = self.embedding(self.indices).t()
        return x + e

    def __call__(self, src, tgt=None):
        if self.encoder_only:
            out = self.encode(src)
        else:
            mem = self.encode(src)
            out = self.decode(tgt, mem=mem)

        return F.pad(
            input=self.activate(self.out(out.transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )

    def encode(self, src):
        src = src[:, :, :self.max_bin]

        for module in self.encoder:
            t = module(src, mask=self.mask)
            src = torch.cat((src, t), dim=1)

        return src

    def decode(self, tgt, mem):
        tgt = tgt[:, :, :self.max_bin]

        for module in self.decoder:
            t = module(tgt, mem=mem, mask=self.mask)
            tgt = torch.cat((tgt, t), dim=1)

        return F.pad(
            input=self.activate(self.out(tgt.transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )
        
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.groups = groups
        self.stride = stride
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.conv1d(F.pad(x, (self.kernel_size - 1, 0)), weight=self.weight, bias=self.bias, stride=self.stride, groups=self.groups)

class MultibandFrameAttention(nn.Module):
    def __init__(self, num_bands, bins, cropsize, kernel_size=3):
        super().__init__()

        self.num_bands = num_bands

        self.q_proj = nn.Linear(bins, bins)
        self.q_conv = CausalConv1d(bins, bins, kernel_size=kernel_size, groups=bins)

        self.k_proj = nn.Linear(bins, bins)
        self.k_conv = CausalConv1d(bins, bins, kernel_size=kernel_size, groups=bins)

        self.v_proj = nn.Linear(bins, bins)
        self.v_conv = CausalConv1d(bins, bins, kernel_size=kernel_size, groups=bins)

        self.o_proj = nn.Linear(bins, bins)

        self.er = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.er)

    def forward(self, x, mem=None, mask=None):
        b,w,c = x.shape

        q = self.q_conv(self.q_proj(x).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        k = self.k_conv(self.k_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,3,1)
        v = self.v_conv(self.v_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
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

        self.in_norm = nn.LayerNorm(bins)
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
        x = self.in_norm(x.transpose(2,3)).transpose(2,3)        
        x = self.in_project(x.transpose(1,3)).squeeze(3)

        h = self.norm1(x)
        h = self.attn(h, mask=mask)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.linear2(self.dropout2(torch.square(self.relu(self.linear1(h)))))
        x = x + self.dropout3(h)

        return x.transpose(1,2).unsqueeze(1)

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, skip_channels, num_bands=4, cropsize=1024, bins=2048, feedforward_dim=2048, downsamples=0, bias=False, dropout=0.1):
        super(FrameTransformerDecoder, self).__init__()

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands
        self.in_project = nn.Linear(channels, 1, bias=bias)
        self.mem_project = nn.Linear(skip_channels, 1, bias=bias)
        self.decoder = nn.TransformerDecoderLayer(bins, num_bands, feedforward_dim, batch_first=True, norm_first=True, dropout=dropout)

    def __call__(self, x, mem, mask=None):
        x = self.in_project(x.transpose(1,3)).squeeze(3)
        mem = self.mem_project(mem.transpose(1,3)).squeeze(3)
        return self.decoder(tgt=x, memory=mem, tgt_mask=mask).transpose(1,2).unsqueeze(1)

class FrameNorm(nn.Module):
    def __init__(self, channels, cropsize=1024, n_fft=2048, downsamples=0):
        super(FrameNorm, self).__init__()

        bins = get_bins(n_fft, downsamples)
        self.norm = nn.LayerNorm((bins, channels))

    def __call__(self, x):
        return self.norm(x.transpose(1,3)).transpose(1,3)

def get_bins(n_fft, downsamples):    
    bins = (n_fft // 2)
    if downsamples > 0:
        for _ in range(downsamples):
            bins = ((bins - 1) // 2) + 1

    return bins