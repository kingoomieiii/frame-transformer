import torch
from torch import nn
import torch.nn.functional as F
import math
from lib import spec_utils

class FrameTransformer(nn.Module):
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=8, num_decoders=1, cropsize=256, bias=False):
        super(FrameTransformer, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameConv(2, channels, 3, 1, 1)
        self.enc2 = Encoder(channels * 1, channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc3 = Encoder(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1)
        self.enc4 = Encoder(channels * 4, channels * 6, kernel_size=3, stride=2, padding=1)
        self.enc5 = Encoder(channels * 6, channels * 8, kernel_size=3, stride=2, padding=1)
        
        self.dec4_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 8 + i, channels * 8, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec4 = Decoder(channels * (6 + 8) + num_decoders, channels * 6, kernel_size=3, padding=1)

        self.dec3_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 6 + i, channels * 6, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec3 = Decoder(channels * (4 + 6) + num_decoders, channels * 4, kernel_size=3, padding=1)

        self.dec2_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 4 + i, channels * 4, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec2 = Decoder(channels * (2 + 4) + num_decoders, channels * 2, kernel_size=3, padding=1)

        self.dec1_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 2 + i, channels * 2, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec1 = Decoder(channels * (1 + 2) + num_decoders, channels * 1, kernel_size=3, padding=1)

        self.out_transformer = nn.ModuleList([FrameTransformerDecoder(channels + i, channels, num_bands, cropsize, n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.out = nn.Linear(channels + num_decoders, 2, bias=bias)

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h, sa1, ea1, sa2, ea2 = e5, None, None, None, None
        for module in self.dec4_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e5, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)
        
        h, sa1, ea1, sa2, ea2 = self.dec4(h, e4), None, None, None, None
        for module in self.dec3_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e4, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)

        h, sa1, ea1, sa2, ea2 = self.dec3(h, e3), None, None, None, None
        for module in self.dec2_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e3, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)

        h, sa1, ea1, sa2, ea2 = self.dec2(h, e2), None, None, None, None
        for module in self.dec1_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e2, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)

        h, sa1, ea1, sa2, ea2 = self.dec1(h, e1), None, None, None, None
        for module in self.out_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e1, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)

        return F.pad(
            input=torch.sigmoid(self.out(h.transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )

class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, activate=nn.ReLU):
        super(FrameConv, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                dilation=(dilation, 1),
                groups=groups,
                bias=False),
            nn.BatchNorm2d(out_channels),
            activate(inplace=True))

    def __call__(self, x):
        h = self.body(x)

        return h
        
class Encoder(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = FrameConv(nin, nout, kernel_size, 1, padding, activate=activ)
        self.conv2 = FrameConv(nout, nout, kernel_size, stride, padding, activate=activ)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        return h

class Decoder(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, activ=nn.ReLU, dropout=False):
        super(Decoder, self).__init__()
        self.conv = FrameConv(nin, nout, kernel_size, 1, padding, activate=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, size=(skip.shape[2],skip.shape[3]), mode='bilinear', align_corners=True)

        if skip is not None:
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)

        h = self.conv(x)

        if self.dropout is not None:
            h = self.dropout(h)

        return h

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, num_bands=4, cropsize=256, n_fft=2048, feedforward_dim=2048, downsamples=0, bias=False, dropout=0.1):
        super(FrameTransformerEncoder, self).__init__()

        # these need to be updated; they make too many assumptions
        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)
        self.bottleneck_linear = nn.Linear(channels, 1, bias=bias)
        self.bottleneck_norm = nn.BatchNorm2d(1)
       
        # This isn't like how they have it in the evolved transformer paper; expanding the features by 2 allows for addition without padding, however they simply pad the output in the paper. Need to see if this performs better or worse.
        self.glu = nn.GLU()
        self.norm1 = nn.LayerNorm(bins)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv1L = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.conv1R = nn.Conv1d(bins, feedforward_dim // 2, kernel_size=3, padding=1, bias=bias)
        self.norm2 = nn.LayerNorm(feedforward_dim * 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim*2, feedforward_dim*2, kernel_size=9, padding=4, groups=feedforward_dim*2, bias=bias),
            nn.Conv1d(feedforward_dim*2, feedforward_dim//2, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(bins)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.attn = MultibandFrameAttention(num_bands, bins, cropsize)
        self.norm4 = nn.LayerNorm(bins)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.norm5 = nn.LayerNorm(bins)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x):
        x = self.relu(self.bottleneck_norm(self.bottleneck_linear(x.transpose(1,3)).transpose(1,3)))

        b, _, h, w = x.shape
        x = x.transpose(2,3).reshape(b,w,h)

        h = self.dropout1(self.glu(x))
        x = self.norm1(x + F.pad(input=h, pad=(0,x.shape[2]-h.shape[2])))

        hL = self.relu(self.conv1L(x))
        hR = self.relu(self.conv1R(x.transpose(1,2)).transpose(1,2))
        h = self.norm2(hL + F.pad(input=hR, pad=(0,hL.shape[2]-hR.shape[2])))
        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = self.norm3(x + F.pad(input=h, pad=(0,x.shape[2]-h.shape[2])))

        h = self.dropout3(self.attn(x))
        x = self.norm4(x + h)

        h = self.conv3(x)
        h = self.relu(h)
        h = self.dropout4(self.conv4(h))
        x = self.norm5(x + h)
                
        return x.transpose(1, 2).unsqueeze(1)

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, mem_channels, num_bands=4, cropsize=256, n_fft=2048, feedforward_dim=2048, downsamples=0, bias=False, dropout=0.1):
        super(FrameTransformerDecoder, self).__init__()

        bins = (n_fft // 2)
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)

        self.bottleneck_linear = nn.Linear(channels, 1, bias=bias)
        self.bottleneck_norm = nn.BatchNorm2d(1)
        self.mem_bottleneck_linear = nn.Linear(mem_channels, 1, bias=bias)
        self.mem_bottleneck_norm = nn.BatchNorm2d(1)

        self.self_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.enc_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.norm1 = nn.LayerNorm(bins)
        self.dropout1 = nn.Dropout(dropout)

        self.conv1L = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=11, padding=5, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim*2, kernel_size=1, padding=0, bias=bias))
        self.conv1R = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=7, padding=3, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 2, kernel_size=1, padding=0, bias=bias))
        self.norm2 = nn.LayerNorm(feedforward_dim*2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim*2, feedforward_dim*2, kernel_size=7, padding=3, groups=feedforward_dim, bias=bias),
            nn.Conv1d(feedforward_dim*2, bins, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(bins)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.norm4 = nn.LayerNorm(bins)
        self.dropout3 = nn.Dropout(dropout)

        self.enc_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.norm5 = nn.LayerNorm(bins)
        self.dropout4 = nn.Dropout(dropout)

        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.swish = nn.SiLU(inplace=True)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.norm6 = nn.LayerNorm(bins)
        self.dropout5 = nn.Dropout(dropout)

    def __call__(self, x, mem, sa1=None, ea1=None, sa2=None, ea2=None):
        x = self.relu(self.bottleneck_norm(self.bottleneck_linear(x.transpose(1,3)).transpose(1,3)))
        mem = self.relu(self.mem_bottleneck_norm(self.mem_bottleneck_linear(mem.transpose(1,3)).transpose(1,3)))
        b,_,h,w = x.shape

        x = x.transpose(2,3).reshape(b,w,h)
        mem = mem.transpose(2,3).reshape(b,w,h)

        hs1, sa1 = self.self_attn1(x, prev=sa1)
        hm1, ea1 = self.enc_attn1(x, mem=mem, prev=ea1)
        x = self.norm1(x + self.dropout1(hs1 + hm1))

        hL = self.relu(self.conv1L(x.transpose(1,2)).transpose(1,2))
        hR = self.conv1R(x.transpose(1,2)).transpose(1,2)
        h = self.norm2(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))

        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = self.norm3(x + h)

        hs2, sa2 = self.self_attn2(x, prev=sa2)
        hs2 = self.dropout3(hs2)
        x = self.norm4(x + hs2)

        hm2, ea2 = self.enc_attn2(x, mem=mem, prev=ea2)
        hm2 = self.dropout4(hm2)
        x = self.norm5(x + hm2)

        h = self.conv3(x)
        h = self.swish(h)
        h = self.dropout5(self.conv4(h))
        x = self.norm6(x + h)
                
        return x.transpose(1, 2).unsqueeze(1), sa1, ea1, sa2, ea2

class MultibandFrameAttention(nn.Module):
    def __init__(self, num_bands, bins, cropsize):
        super().__init__()

        self.num_bands = num_bands
        self.q_proj = nn.Linear(bins, bins)
        self.k_proj = nn.Linear(bins, bins)
        self.v_proj = nn.Linear(bins, bins)
        self.o_proj = nn.Linear(bins, bins)
        self.er = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        # self.distance_weight = nn.Parameter(torch.empty(num_bands).unsqueeze(1).expand((-1, cropsize)).unsqueeze(2).clone())
        # self.register_buffer('distances', torch.empty(num_bands, cropsize, cropsize))
        nn.init.kaiming_uniform_(self.er, a=math.sqrt(5))
        #nn.init.kaiming_uniform_(self.distance_weight, a=math.sqrt(5.0))

        # for i in range(cropsize):
        #     for j in range(cropsize):
        #         self.distances[:, i, j] = abs(i - j)

    def __call__(self, x, mem=None, prev=None):
        b,w,h = x.shape
        q = self.q_proj(x).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        k = self.k_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,3,1)
        v = self.v_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)

        p = F.pad(torch.matmul(q,self.er), (1,0)).transpose(2,3)[:,:,1:,:]
        a = (torch.matmul(q,k)+p) / math.sqrt(h)
        a = a + prev if prev is not None else a
        attn = F.softmax(a, dim=-1)

        v = torch.matmul(attn,v).transpose(1,2).reshape(b,w,-1)
        o = self.o_proj(v)

        return o, a