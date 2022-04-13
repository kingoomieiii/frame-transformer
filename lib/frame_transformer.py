import torch
from torch import nn
import torch.nn.functional as F
import math
from lib import spec_utils

class FrameTransformer(nn.Module):
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=8, num_encoders=1, num_decoders=1, cropsize=256, bias=False):
        super(FrameTransformer, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameConv(2, channels, 3, 1, 1)
        self.enc1_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 1 + i, num_bands, cropsize, n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc2 = Encoder(channels * 1 + num_encoders, channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 2 + i, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc3 = Encoder(channels * 2 + num_encoders, channels * 4, kernel_size=3, stride=2, padding=1)
        self.enc3_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 4 + i, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc4 = Encoder(channels * 4 + num_encoders, channels * 6, kernel_size=3, stride=2, padding=1)
        self.enc4_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 6 + i, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc5 = Encoder(channels * 6 + num_encoders, channels * 8, kernel_size=3, stride=2, padding=1)
        self.enc5_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 8 + i, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])
        
        self.dec4_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 8 + i, channels * 8 + num_encoders, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec4 = Decoder(channels * (6 + 8) + num_decoders + num_encoders, channels * 6, kernel_size=3, padding=1)

        self.dec3_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 6 + i, channels * 6 + num_encoders, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec3 = Decoder(channels * (4 + 6) + num_decoders + num_encoders, channels * 4, kernel_size=3, padding=1)

        self.dec2_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 4 + i, channels * 4 + num_encoders, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec2 = Decoder(channels * (2 + 4) + num_decoders + num_encoders, channels * 2, kernel_size=3, padding=1)

        self.dec1_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 2 + i, channels * 2 + num_encoders, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec1 = Decoder(channels * (1 + 2) + num_decoders + num_encoders, channels * 1, kernel_size=3, padding=1)

        self.out_transformer = nn.ModuleList([FrameTransformerDecoder(channels + i, channels + num_encoders, num_bands, cropsize, n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.out = nn.Linear(channels + num_decoders, 2, bias=bias)

    def __call__(self, x):
        x = x[:, :, :self.max_bin]
        sa, sa1, ea1, sa2, ea2 = None, None, None, None, None

        e1 = self.enc1(x)
        for module in self.enc1_transformer:
            t, sa = module(e1, sa=sa)
            e1 = torch.cat((e1, t), dim=1)

        e2 = self.enc2(e1)
        for module in self.enc2_transformer:
            t, sa = module(e2, sa=sa)
            e2 = torch.cat((e2, t), dim=1)

        e3 = self.enc3(e2)
        for module in self.enc3_transformer:
            t, sa = module(e3, sa=sa)
            e3 = torch.cat((e3, t), dim=1)

        e4 = self.enc4(e3)
        for module in self.enc4_transformer:
            t, sa = module(e4, sa=sa)
            e4 = torch.cat((e4, t), dim=1)

        h = self.enc5(e4)
        e5 = h
        for module in self.enc5_transformer:
            t, sa = module(e5, sa=sa)
            e5 = torch.cat((e5, t), dim=1)
        for module in self.dec4_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e5, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)
        
        h = self.dec4(h, e4)
        for module in self.dec3_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e4, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)

        h = self.dec3(h, e3)
        for module in self.dec2_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e3, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)

        h = self.dec2(h, e2)
        for module in self.dec1_transformer:
            t, sa1, ea1, sa2, ea2 = module(h, mem=e2, sa1=sa1, ea1=ea1, sa2=sa2, ea2=ea2)
            h = torch.cat((h, t), dim=1)

        h = self.dec1(h, e1)
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
            nn.BatchNorm2d(in_channels),
            activate(inplace=True),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                dilation=(dilation, 1),
                groups=groups,
                bias=False))

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
        self.bottleneck_norm = nn.BatchNorm2d(channels)
        self.bottleneck_linear = nn.Linear(channels, 1, bias=bias)

        self.norm1 = nn.LayerNorm(bins)
        self.glu_values = nn.Linear(bins, bins, bias=bias)
        self.glu_gates = nn.Linear(bins, bins, bias=bias)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.conv1R = nn.Conv1d(bins, feedforward_dim // 2, kernel_size=3, padding=1, bias=bias)
        self.norm3 = nn.LayerNorm(feedforward_dim * 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim*2, feedforward_dim*2, kernel_size=9, padding=4, groups=feedforward_dim*2, bias=bias),
            nn.Conv1d(feedforward_dim*2, feedforward_dim//2, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm4 = nn.LayerNorm(bins)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm5 = nn.LayerNorm(bins)
        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, sa=None):
        x = self.relu(self.bottleneck_linear(self.bottleneck_norm(x).transpose(1,3)).transpose(1,3))

        b, _, h, w = x.shape
        x = x.transpose(2,3).reshape(b,w,h)

        h = self.norm1(x)
        h = self.dropout1(self.glu_values(h) * torch.sigmoid(self.glu_gates(h)))
        x = x + F.pad(input=h, pad=(0,x.shape[2]-h.shape[2]))

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h))
        hR = self.relu(self.conv1R(h.transpose(1,2)).transpose(1,2))
        h = self.norm3(hL + F.pad(hR, (0,hL.shape[2]-hR.shape[2])))
        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = x + F.pad(h, (0,x.shape[2]-h.shape[2]))

        h = self.norm4(x)
        h, sa = self.attn(h, prev=sa)
        h = self.dropout3(h)
        x = x + h

        h = self.norm5(x)
        h = self.conv3(h)
        h = self.relu(h)
        h = self.dropout4(self.conv4(h))
        x = x + h
                
        return x.transpose(1, 2).unsqueeze(1), sa

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

        self.bottleneck_norm = nn.BatchNorm2d(channels)
        self.bottleneck_linear = nn.Linear(channels, 1, bias=bias)
        self.mem_bottleneck_norm = nn.BatchNorm2d(mem_channels)
        self.mem_bottleneck_linear = nn.Linear(mem_channels, 1, bias=bias)

        self.norm1 = nn.LayerNorm(bins)
        self.self_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.enc_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=11, padding=5, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim, kernel_size=1, padding=0, bias=bias))
        self.conv1R = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=7, padding=3, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 2, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(feedforward_dim)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim, feedforward_dim, kernel_size=7, padding=3, groups=feedforward_dim, bias=bias),
            nn.Conv1d(feedforward_dim, bins, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout)

        self.norm4 = nn.LayerNorm(bins)
        self.self_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout3 = nn.Dropout(dropout)

        self.norm5 = nn.LayerNorm(bins)
        self.enc_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout4 = nn.Dropout(dropout)

        self.norm6 = nn.LayerNorm(bins)
        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.swish = nn.SiLU(inplace=True)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.dropout5 = nn.Dropout(dropout)

    def __call__(self, x, mem, sa1=None, ea1=None, sa2=None, ea2=None):
        x = self.relu(self.bottleneck_linear(self.bottleneck_norm(x).transpose(1,3)).transpose(1,3))
        mem = self.relu(self.mem_bottleneck_linear(self.mem_bottleneck_norm(mem).transpose(1,3)).transpose(1,3))
        b,_,h,w = x.shape

        x = x.transpose(2,3).reshape(b,w,h)
        mem = mem.transpose(2,3).reshape(b,w,h)

        h = self.norm1(x)
        hs1, sa1 = self.self_attn1(h, prev=sa1)
        hm1, ea1 = self.enc_attn1(h, mem=mem, prev=ea1)
        x = x + self.dropout1(hs1 + hm1)

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h.transpose(1,2)).transpose(1,2))
        hR = self.conv1R(h.transpose(1,2)).transpose(1,2)
        h = self.norm3(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))
        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = x + h

        h = self.norm4(x)
        hs2, sa2 = self.self_attn2(h, prev=sa2)
        hs2 = self.dropout3(hs2)
        x = x + h

        h = self.norm5(x)
        hm2, ea2 = self.enc_attn2(h, mem=mem, prev=ea2)
        hm2 = self.dropout4(hm2)
        x = x + h

        h = self.norm6(x)
        h = self.conv3(h)
        h = self.swish(h)
        h = self.dropout5(self.conv4(h))
        x = x + h
                
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