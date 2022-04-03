import torch
from torch import nn
import torch.nn.functional as F
import math
from lib import spec_utils

class FrameTransformer(nn.Module):
    def __init__(self, n_fft, out_proj_width=8, num_encoders=4, num_decoders=4, num_bands=8, kernel_size=5, padding=2, bias=False, feedforward_dim=2048):
        super(FrameTransformer, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2

        self.transformer = FrameTransformerNet(2, out_proj_width, n_fft=n_fft, num_encoders=num_encoders, num_decoders=num_decoders, num_bands=num_bands, feedforward_dim=feedforward_dim, kernel_size=kernel_size, padding=padding, bias=bias)
        self.out = nn.Linear(in_features=out_proj_width, out_features=2, bias=bias)

    def forward(self, x):
        x = x[:, :, :self.max_bin]
        t = self.transformer(x)
        out = torch.sigmoid(self.out(t.transpose(1,3)).transpose(1,3))        
        out = F.pad(
            input=out,
            pad=(0, 0, 0, self.output_bin - out.size()[2]),
            mode='replicate'
        )

        return out

class FrameTransformerNet(nn.Module):
    def __init__(self, nin, nout, n_fft=2048, feedforward_dim=512, num_bands=4, num_encoders=1, num_decoders=1, cropsize=256, kernel_size=3, padding=1, bias=False):
        super(FrameTransformerNet, self).__init__()

        self.enc1 = FrameConv(nin, nout, kernel_size, 1, padding)
        
        self.enc2 = Encoder(nout * 1, nout * 2, kernel_size, stride=2, padding=padding)
        self.enc2_transformer = nn.ModuleList([FrameTransformerEncoder(nout * 2 + i, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc3 = Encoder(nout * 2 + num_encoders, nout * 4, kernel_size, stride=2, padding=padding)
        self.enc3_transformer = nn.ModuleList([FrameTransformerEncoder(nout * 4 + i, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc4 = Encoder(nout * 4 + num_encoders, nout * 6, kernel_size, stride=2, padding=padding)
        self.enc4_transformer = nn.ModuleList([FrameTransformerEncoder(nout * 6 + i, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])
        
        self.enc5 = Encoder(nout * 6 + num_encoders, nout * 8, kernel_size, stride=2, padding=padding)
        self.enc5_transformer = nn.ModuleList([FrameTransformerEncoder(nout * 8 + i, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])
        
        self.dec4_transformer = nn.ModuleList([FrameTransformerDecoder(nout * 8 + i + num_encoders, nout * 8 + num_encoders, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec4 = Decoder(nout * (6 + 8) + num_decoders + num_encoders * 2, nout * 6, kernel_size, padding=padding)

        self.dec3_transformer = nn.ModuleList([FrameTransformerDecoder(nout * 6 + i, nout * 6 + num_encoders, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec3 = Decoder(nout * (4 + 6) + num_decoders + num_encoders, nout * 4, kernel_size, padding=padding)

        self.dec2_transformer = nn.ModuleList([FrameTransformerDecoder(nout * 4 + i, nout * 4 + num_encoders, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec2 = Decoder(nout * (2 + 4) + num_decoders + num_encoders, nout * 2, kernel_size, padding=padding)

        self.dec1_transformer = nn.ModuleList([FrameTransformerDecoder(nout * 2 + i, nout * 2 + num_encoders, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec1 = Decoder(nout * (1 + 2) + num_decoders, nout * 1, kernel_size, padding=padding)

    def __call__(self, x):
        e1 = self.enc1(x)

        e2 = self.enc2(e1)
        for module in self.enc2_transformer:
            t = module(e2)
            e2 = torch.cat((e2, t), dim=1)

        e3 = self.enc3(e2)
        for module in self.enc3_transformer:
            t = module(e3)
            e3 = torch.cat((e3, t), dim=1) 

        e4 = self.enc4(e3)
        for module in self.enc4_transformer:
            t = module(e4)
            e4 = torch.cat((e4, t), dim=1)

        e5 = self.enc5(e4)
        for module in self.enc5_transformer:
            t = module(e5)
            e5 = torch.cat((e5, t), dim=1)

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
       
        self.glu = nn.Sequential(
            nn.Linear(bins, bins * 2, bias=bias),
            nn.GLU())
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

        self.attn = MultibandFrameAttention(num_bands, bins, cropsize, bias)
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

        self.self_attn1 = MultibandFrameAttention(num_bands, bins, cropsize, bias)
        self.enc_attn1 = MultibandFrameAttention(num_bands, bins, cropsize, bias)
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

        self.self_attn2 = MultibandFrameAttention(num_bands, bins, cropsize, bias)
        self.norm4 = nn.LayerNorm(bins)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.enc_attn2 = MultibandFrameAttention(num_bands, bins, cropsize, bias)
        self.norm5 = nn.LayerNorm(bins)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.swish = nn.SiLU(inplace=True)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.norm6 = nn.LayerNorm(bins)
        self.dropout5 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, mem):
        x = self.relu(self.bottleneck_norm(self.bottleneck_linear(x.transpose(1,3)).transpose(1,3)))
        mem = self.relu(self.mem_bottleneck_norm(self.mem_bottleneck_linear(mem.transpose(1,3)).transpose(1,3)))

        b, _, h, w = x.shape
        x = x.transpose(2,3).reshape(b,w,h)
        mem = mem.transpose(2,3).reshape(b,w,h)

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
    def __init__(self, num_bands, bins, cropsize, bias=False):
        super().__init__()

        self.num_bands = num_bands
        self.q_proj = nn.Linear(bins, bins, bias=bias)
        self.k_proj = nn.Linear(bins, bins, bias=bias)
        self.v_proj = nn.Linear(bins, bins, bias=bias)
        self.o_proj = nn.Linear(bins, bins, bias=bias)

        self.er = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.kaiming_uniform_(self.er, a=math.sqrt(5))

        self.register_buffer('distances', torch.empty(num_bands, cropsize, cropsize))
        self.distance_weight = nn.Parameter(torch.empty(num_bands).unsqueeze(1).expand((-1, cropsize)).unsqueeze(2).clone())
        nn.init.kaiming_uniform_(self.distance_weight, a=math.sqrt(5.0))

        for i in range(cropsize):
            for j in range(cropsize):
                self.distances[:, i, j] = abs(i - j) / 2

    def forward(self, x, mem=None):
        b,w,c = x.shape
        q = self.q_proj(x).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        k = self.k_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,3,1)
        v = self.v_proj(x if mem is None else mem).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        qk = torch.matmul(q,k) / math.sqrt(c)
        p = (self.distances * self.distance_weight) + F.pad(torch.matmul(q,self.er), (1,0)).transpose(2,3)[:,:,1:,:]
        a = F.softmax(qk+p, dim=-1)
        a = torch.matmul(a,v).transpose(1,2).reshape(b,w,-1)
        o = self.o_proj(a)
        return o

class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, activate=nn.ReLU, norm=True):
        super(FrameConv, self).__init__()

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