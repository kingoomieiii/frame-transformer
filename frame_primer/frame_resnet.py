import torch
from torch import nn
import torch.nn.functional as F
import math

class RelativePositionalEncoding(nn.Module):
    def __init__(self, bins, cropsize, num_bands):
        super().__init__()

        self.bins = bins
        self.num_bands = num_bands
        self.weight = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.weight)

    def __call__(self, q, w):
        return F.pad(torch.matmul(q,self.weight[:, :w]), (1,0)).transpose(2,3)[:,:,1:,:]

class MultibandFrameAttention(nn.Module):
    def __init__(self, num_bands, bins, cropsize, kernel_size=3, padding=1, bias=False):
        super().__init__()

        self.bins = bins

        self.num_bands = num_bands
        self.q_proj = nn.Linear(bins, bins, bias=bias)
        self.q_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, bias=bias)
        self.k_proj = nn.Linear(bins, bins, bias=bias)
        self.k_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, bias=bias)
        self.v_proj = nn.Linear(bins, bins, bias=bias)
        self.v_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins, bias=bias)
        self.o_proj = nn.Linear(bins, bins, bias=bias)
        self.positional_encoding = RelativePositionalEncoding(bins, cropsize, num_bands)

        self.weight = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.weight)

    def forward(self, x, mem=None, prev_qk=None):
        b,w,c = x.shape

        q = self.q_conv(self.q_proj(x).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3).contiguous()
        k = self.k_conv(self.k_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,3,1).contiguous()
        v = self.v_conv(self.v_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3).contiguous()
        p = self.positional_encoding(q, w)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = (torch.matmul(q,k)+p) / math.sqrt(c)

            if prev_qk is not None:
                qk = qk + prev_qk

            a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(1,2).reshape(b,w,-1).contiguous()
                
        o = self.o_proj(a)

        return o, qk

class FrameAttention(nn.Module):
    def __init__(self, channels, skip_channels=None, num_bands=4, cropsize=1024, bias=False, downsamples=0, n_fft=2048):
        super(FrameAttention, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.in_project = nn.Conv2d(channels, 1, kernel_size=(3,1), padding=(1,0), bias=bias)
        self.attn = MultibandFrameAttention(num_bands=num_bands, bins=bins, cropsize=cropsize)

    def __call__(self, x):
        h = self.in_project(x).transpose(1,3).squeeze(-1)
        h, _ = self.attn(h)

        h = h.transpose(1,2).unsqueeze(1)

        return h

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU, cropsize=1024, downsamples=0, n_fft=2048, attention=False, num_bands=8):
        super(ResBlock, self).__init__()

        bins1 = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples-1 if stride > 1 else downsamples):
                bins1 = ((bins1 - 1) // 2) + 1

        bins2 = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins2 = ((bins2 - 1) // 2) + 1

        self.attention = FrameAttention(in_channels, num_bands=num_bands, cropsize=cropsize, downsamples=downsamples-1 if stride > 1 else downsamples, n_fft=n_fft) if attention else None
        self.norm1 = nn.LayerNorm(bins1)

        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(stride, 1), bias=False) if in_channels != out_channels or stride > 1 else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=(kernel_size, 1), padding=(padding,0), stride=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=(kernel_size,1), padding=(padding,0), stride=(stride, 1), bias=False)
        self.norm2 = nn.LayerNorm(bins2)

    def __call__(self, x):
        h = self.attention(x)
        x = self.norm1((x + h).transpose(2,3)).transpose(2,3)

        h = self.conv2(self.relu(self.conv1(x)))
        x = self.norm2((self.identity(x) + h).transpose(2,3)).transpose(2,3)

        return h
        
class FrameResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU, cropsize=1024, downsamples=0, n_fft=2048, num_res_blocks=1, num_bands=8):
        super(FrameResEncoder, self).__init__()

        self.cropsize = cropsize
        self.dowsnsamples = downsamples
        self.body = nn.ModuleList([ResBlock(in_channels if i == 0 else out_channels, out_channels, stride=stride if i == 0 else 1, kernel_size=kernel_size, padding=padding, cropsize=cropsize, downsamples=downsamples if stride == 1 else downsamples+1, n_fft=n_fft, attention=True, num_bands=num_bands) for i in range(num_res_blocks)])

    def __call__(self, x):
        for module in self.body:
            x = module(x)
        
        return x

class FrameResDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, upsample=True, activ=nn.LeakyReLU, norm=True, dropout=False, cropsize=1024, downsamples=0, n_fft=2048, num_res_blocks=1, num_bands=8):
        super(FrameResDecoder, self).__init__()

        self.cropsize = cropsize
        self.dowsnsamples = downsamples
        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True) if upsample else nn.Identity()
        self.body = nn.ModuleList([ResBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel_size, padding=padding, cropsize=cropsize, downsamples=downsamples, n_fft=n_fft, attention=True, num_bands=num_bands) for i in range(num_res_blocks)])

    def __call__(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)

        for module in self.body:
            x = module(x)

        return x

class FrameResUNet(nn.Module):
    def __init__(self, nin, nout, cropsize=2048, n_fft=2048, num_res_blocks=2, num_bands=8):
        super(FrameResUNet, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.enc1 = FrameResEncoder(nin, out_channels=nout, kernel_size=3, stride=1, padding=1, downsamples=0, cropsize=cropsize, n_fft=n_fft, num_res_blocks=num_res_blocks)
        self.enc2 = FrameResEncoder(nout, out_channels=nout * 2, kernel_size=3, stride=2, padding=1, downsamples=0, cropsize=cropsize, n_fft=n_fft, num_res_blocks=num_res_blocks)
        self.enc3 = FrameResEncoder(nout * 2, out_channels=nout * 4, kernel_size=3, stride=2, padding=1, downsamples=1, cropsize=cropsize, n_fft=n_fft, num_res_blocks=num_res_blocks)
        self.enc4 = FrameResEncoder(nout * 4, out_channels=nout * 6, kernel_size=3, stride=2, padding=1, downsamples=2, cropsize=cropsize, n_fft=n_fft, num_res_blocks=num_res_blocks)
        self.enc5 = FrameResEncoder(nout * 6, out_channels=nout * 8, kernel_size=3, stride=2, padding=1, downsamples=3, cropsize=cropsize, n_fft=n_fft, num_res_blocks=num_res_blocks)
        self.dec4 = FrameResDecoder(nout * (6 + 8), nout * 6, kernel_size=3, padding=1, cropsize=cropsize, num_res_blocks=num_res_blocks, downsamples=3)
        self.dec3 = FrameResDecoder(nout * (4 + 6), nout * 4, kernel_size=3, padding=1, cropsize=cropsize, num_res_blocks=num_res_blocks, downsamples=2)
        self.dec2 = FrameResDecoder(nout * (2 + 4), nout * 2, kernel_size=3, padding=1, cropsize=cropsize, num_res_blocks=num_res_blocks, downsamples=1)
        self.dec1 = FrameResDecoder(nout * (1 + 2), nout, kernel_size=3, padding=1, cropsize=cropsize, num_res_blocks=num_res_blocks, downsamples=0)
        self.out = nn.Conv2d(nout, 2, kernel_size=1, padding=0, bias=False)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        h = self.dec4(e5, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)
        h = self.out(h)

        return h
