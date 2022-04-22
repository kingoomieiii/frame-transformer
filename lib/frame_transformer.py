import torch
from torch import nn
import torch.nn.functional as F
import math
from lib import spec_utils

class FrameTransformer(nn.Module):
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=4, num_encoders=1, num_decoders=1, cropsize=256, bias=False):
        super(FrameTransformer, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = Encoder(2, channels, n_fft=2048, downsamples=0, downsample=False, bias=bias)
        self.enc2 = Encoder(channels * 1, channels * 2, n_fft=2048, downsamples=0, bias=bias)
        self.enc3 = Encoder(channels * 2, channels * 4, n_fft=2048, downsamples=1, bias=bias)
        self.enc4 = Encoder(channels * 4, channels * 6, n_fft=2048, downsamples=2, bias=bias)
        self.enc5 = Encoder(channels * 6, channels * 8, n_fft=2048, downsamples=3, bias=bias)
        
        self.dec4_transformer = nn.ModuleList([FrameTransformerBlock(channels * 8 + i, channels * 8, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec4 = Decoder(channels * (6 + 8) + num_decoders, channels * 6, n_fft=2048, downsamples=4, bias=bias)

        self.dec3_transformer = nn.ModuleList([FrameTransformerBlock(channels * 6 + i, channels * 6, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec3 = Decoder(channels * (4 + 6) + num_decoders, channels * 4, n_fft=2048, downsamples=3, bias=bias)

        self.dec2_transformer = nn.ModuleList([FrameTransformerBlock(channels * 4 + i, channels * 4, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec2 = Decoder(channels * (2 + 4) + num_decoders, channels * 2, n_fft=2048, downsamples=2, bias=bias)

        self.dec1_transformer = nn.ModuleList([FrameTransformerBlock(channels * 2 + i, channels * 2, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec1 = Decoder(channels * (1 + 2) + num_decoders, channels * 1, n_fft=2048, downsamples=1, bias=bias)

        self.out_transformer = nn.ModuleList([FrameTransformerBlock(channels + i, channels, num_bands, cropsize, n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.out = Decoder(channels + num_decoders, 2, downsamples=0, upsample=False, activate_out=False)

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = e5
        for module in self.dec4_transformer:
            t = module(h, mem=e5)
            h = torch.cat((h, t), dim=1)
            
        h = self.dec4(h, e4)        
        for module in self.dec3_transformer:
            t = module(h, mem=e4)
            h = torch.cat((h, t), dim=1)

        h = self.dec3(h, e3)        
        for module in self.dec2_transformer:
            t = module(h, mem=e3)
            h = torch.cat((h, t), dim=1)

        h = self.dec2(h, e2)        
        for module in self.dec1_transformer:
            t = module(h, mem=e2)
            h = torch.cat((h, t), dim=1)

        h = self.dec1(h, e1)
        for module in self.out_transformer:
            t = module(h, mem=e1)
            h = torch.cat((h, t), dim=1)

        out = torch.sigmoid(self.out(h))

        return F.pad(
            input=out,
            pad=(0, 0, 0, self.output_bin - out.size()[2]),
            mode='replicate'
        )

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_fft=2048, downsamples=0, activ=nn.LeakyReLU, downsample=True, bias=True):
        super(Encoder, self).__init__()

        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.activate = activ(inplace=True)
        self.linear1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.linear2 = nn.Linear(bins, bins // 2 if downsample else bins, bias=bias)

    def __call__(self, x):
        h = self.activate(self.linear1(x.transpose(1,3)).transpose(1,3)).transpose(2,3)
        h = self.activate(self.linear2(h)).permute(0,1,3,2)

        return h

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_fft=2048, downsamples=0, activ=nn.LeakyReLU, bias=True, upsample=True, activate_out=True):
        super(Decoder, self).__init__()

        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.linear1 = nn.Linear(bins, bins * 2 if upsample else bins, bias=bias)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=bias)
        self.activate = activ(inplace=True)
        self.activate_out = activate_out

    def __call__(self, x, skip=None):
        h = self.activate(self.linear1(x.transpose(2,3)).transpose(2,3))

        if skip is not None:
            h = torch.cat((h, skip), dim=1)

        h = self.linear2(h.transpose(1,3)).transpose(1,3)

        if self.activate_out:
            h = self.activate(h)

        return h

class FrameTransformerBlock(nn.Module):
    def __init__(self, channels, mem_channels, num_bands=4, cropsize=256, n_fft=2048, feedforward_dim=2048, downsamples=0, bias=False, dropout=0.1):
        super(FrameTransformerBlock, self).__init__()

        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)
        
        self.bottleneck_linear = nn.Linear(channels, 1, bias=bias)
        self.bottleneck_norm = nn.LayerNorm(bins)
        self.mem_bottleneck_linear = nn.Linear(mem_channels, 1, bias=bias)
        self.mem_bottleneck_norm = nn.LayerNorm(bins)

        self.self_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.enc_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.norm1 = nn.LayerNorm(bins)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv1L = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=11, padding=5, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim * 2, kernel_size=1, padding=0, bias=bias))
        self.conv1R = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=7, padding=3, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 2, kernel_size=1, padding=0, bias=bias))
        self.norm2 = nn.LayerNorm(feedforward_dim * 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim * 2, feedforward_dim * 2, kernel_size=7, padding=3, groups=feedforward_dim*2, bias=bias),
            nn.Conv1d(feedforward_dim * 2, bins, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(bins)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.self_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.norm4 = nn.LayerNorm(bins)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.enc_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.norm5 = nn.LayerNorm(bins)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.swish = nn.SiLU(inplace=True)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.norm6 = nn.LayerNorm(bins)
        self.dropout5 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, mem):
        x = self.bottleneck_linear(x.transpose(1,3)).transpose(1,3)
        mem = self.mem_bottleneck_linear(mem.transpose(1,3)).transpose(1,3)

        b, _, h, w = x.shape
        x = x.transpose(2,3).reshape(b,w,h)
        mem = mem.transpose(2,3).reshape(b,w,h)

        x = self.relu(self.bottleneck_norm(x))
        mem = self.relu(self.mem_bottleneck_norm(mem))

        hs = self.self_attn1(x)
        hm = self.enc_attn1(x, mem=mem)
        x = self.norm1(x + self.dropout1(hs + hm))

        hL = self.relu(self.conv1L(x.transpose(1,2)).transpose(1,2))
        hR = self.conv1R(x.transpose(1,2)).transpose(1,2)
        h = self.norm2(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))

        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = self.norm3(x + h)

        h = self.dropout3(self.self_attn2(x))
        x = self.norm4(x + h)

        h = self.dropout4(self.enc_attn2(x, mem=mem))
        x = self.norm5(x + h)

        h = self.conv3(x)
        h = self.swish(h)
        h = self.dropout5(self.conv4(h))
        x = self.norm6(x + h)
                
        return x.transpose(1, 2).unsqueeze(1)

class MultibandFrameAttention(nn.Module):
    def __init__(self, num_bands, bins, cropsize):
        super().__init__()

        self.num_bands = num_bands
        self.q_proj = nn.Linear(bins, bins)
        self.k_proj = nn.Linear(bins, bins)
        self.v_proj = nn.Linear(bins, bins)
        self.o_proj = nn.Linear(bins, bins)
        self.er = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.er)

    def forward(self, x, mem=None):
        b,w,c = x.shape
        q = self.q_proj(x).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        k = self.k_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,3,1)
        v = self.v_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        p = F.pad(torch.matmul(q,self.er), (1,0)).transpose(2,3)[:,:,1:,:]
        qk = (torch.matmul(q,k)+p) / math.sqrt(c)
        a = F.softmax(qk, dim=-1)
        a = torch.matmul(a,v).transpose(1,2).reshape(b,w,-1)
        o = self.o_proj(a)
        return o