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

        self.register_buffer('tgt_mask', torch.triu(torch.ones(cropsize, cropsize) * float('-inf'), diagonal=1))

        self.src_enc1 = FrameConv(2, channels, kernel_size=3, padding=1, stride=1)
        self.enc1_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 1 + i, num_bands, cropsize, n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.src_enc2 = Encoder(channels * 1 + num_encoders, channels * 2, kernel_size=3, padding=1, stride=2)
        self.enc2_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 2 + i, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.src_enc3 = Encoder(channels * 2 + num_encoders, channels * 4, kernel_size=3, padding=1, stride=2)
        self.enc3_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 4 + i, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.src_enc4 = Encoder(channels * 4 + num_encoders, channels * 6, kernel_size=3, padding=1, stride=2)
        self.enc4_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 6 + i, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.src_enc5 = Encoder(channels * 6 + num_encoders, channels * 8, kernel_size=3, padding=1, stride=2)
        self.enc5_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 8 + i, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.tgt_enc1 = FrameConv(2, channels, kernel_size=3, padding=1, stride=1)
        self.tgt_enc2 = Encoder(channels * 1, channels * 2, kernel_size=3, padding=1, stride=2)
        self.tgt_enc3 = Encoder(channels * 2, channels * 4, kernel_size=3, padding=1, stride=2)
        self.tgt_enc4 = Encoder(channels * 4, channels * 6, kernel_size=3, padding=1, stride=2)
        self.tgt_enc5 = Encoder(channels * 6, channels * 8, kernel_size=3, padding=1, stride=2)
        
        self.dec4_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 8 + i, channels * 8 + num_encoders, num_bands, cropsize, n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec4 = Decoder(channels * (6 + 8) + num_decoders, channels * 6, kernel_size=3, padding=1)

        self.dec3_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 6 + i, channels * 6 + num_encoders, num_bands, cropsize, n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec3 = Decoder(channels * (4 + 6) + num_decoders, channels * 4, kernel_size=3, padding=1)

        self.dec2_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 4 + i, channels * 4 + num_encoders, num_bands, cropsize, n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec2 = Decoder(channels * (2 + 4) + num_decoders, channels * 2, kernel_size=3, padding=1)

        self.dec1_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 2 + i, channels * 2 + num_encoders, num_bands, cropsize, n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec1 = Decoder(channels * (1 + 2) + num_decoders, channels * 1, kernel_size=3, padding=1)

        self.out_transformer = nn.ModuleList([FrameTransformerDecoder(channels + i, channels + num_encoders, num_bands, cropsize, n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.out = nn.Linear(channels + num_decoders, 2, bias=bias)

    def __call__(self, src, tgt):
        src = src[:, :, :self.max_bin]
        tgt = tgt[:, :, :self.max_bin]

        se1 = self.src_enc1(src)
        for module in self.enc1_transformer:
            t = module(se1)
            se1 = torch.cat((se1, t), dim=1)

        se2 = self.src_enc2(se1)
        for module in self.enc2_transformer:
            t = module(se2)
            se2 = torch.cat((se2, t), dim=1)

        se3 = self.src_enc3(se2)
        for module in self.enc3_transformer:
            t = module(se3)
            se3 = torch.cat((se3, t), dim=1)

        se4 = self.src_enc4(se3)
        for module in self.enc4_transformer:
            t = module(se4)
            se4 = torch.cat((se4, t), dim=1)

        se5 = self.src_enc5(se4)
        for module in self.enc5_transformer:
            t = module(se5)
            se5 = torch.cat((se5, t), dim=1)

        te1 = self.tgt_enc1(tgt)
        te2 = self.tgt_enc2(te1)
        te3 = self.tgt_enc3(te2)
        te4 = self.tgt_enc4(te3)
        te5 = self.tgt_enc5(te4)

        h = te5
        for module in self.dec4_transformer:
            t = module(h, mem=se5, mask=self.tgt_mask)
            h = torch.cat((h, t), dim=1)
            
        h = self.dec4(h, te4)        
        for module in self.dec3_transformer:
            t = module(h, mem=se4, mask=self.tgt_mask)
            h = torch.cat((h, t), dim=1)

        h = self.dec3(h, te3)        
        for module in self.dec2_transformer:
            t = module(h, mem=se3, mask=self.tgt_mask)
            h = torch.cat((h, t), dim=1)

        h = self.dec2(h, te2)        
        for module in self.dec1_transformer:
            t = module(h, mem=se2, mask=self.tgt_mask)
            h = torch.cat((h, t), dim=1)

        h = self.dec1(h, te1)
        for module in self.out_transformer:
            t = module(h, mem=se1, mask=self.tgt_mask)
            h = torch.cat((h, t), dim=1)

        out = self.out(h.transpose(1,3)).transpose(1,3)

        return F.pad(
            input=torch.sigmoid(out),
            pad=(0, 0, 0, self.output_bin - out.size()[2]),
            mode='replicate'
        )
        
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
    def __init__(self, nin, nout, kernel_size=3, padding=1, activ=nn.LeakyReLU, dropout=False):
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

        self.norm1 = nn.LayerNorm(bins)
        self.glu = nn.Sequential(
            nn.Linear(bins, bins * 2, bias=bias),
            nn.GLU())
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.conv1R = nn.Conv1d(bins, feedforward_dim // 2, kernel_size=3, padding=1, bias=bias)
        self.norm3 = nn.LayerNorm(feedforward_dim * 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim*2, feedforward_dim*2, kernel_size=9, padding=4, groups=feedforward_dim*2, bias=bias),
            nn.Conv1d(feedforward_dim*2, bins, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm4 = nn.LayerNorm(bins)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm5 = nn.LayerNorm(bins)
        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, src):
        b, _, h, w = src.shape
        src = self.bottleneck_linear(src.transpose(1,3)).reshape(b,w,h)

        h = self.norm1(src)
        h = self.dropout1(self.glu(h))
        src = src + h

        h = self.norm2(src)
        hL = self.relu(self.conv1L(h))
        hR = self.relu(self.conv1R(h.transpose(1,2)).transpose(1,2))
        h = self.norm3(hL + F.pad(hR, (0,hL.shape[2]-hR.shape[2])))
        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2)) 
        src = src + h

        h = self.norm4(src)
        h = self.attn(h)
        h = self.dropout3(h)
        src = src + h

        h = self.norm5(src)
        h = self.conv3(h)
        h = self.relu(h)
        h = self.dropout4(self.conv4(h))
        src = src + h
                
        return src.transpose(1, 2).unsqueeze(1)

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

        self.relu = nn.LeakyReLU(inplace=True)
        
        self.bottleneck_linear = nn.Linear(channels, 1, bias=bias)
        self.mem_bottleneck_linear = nn.Linear(mem_channels, 1, bias=bias)

        self.norm1 = nn.LayerNorm(bins)
        self.self_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.enc_attn1 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Sequential(
            CausalConv1d(bins, bins, kernel_size=11, padding=5, groups=bins, bias=bias),
            CausalConv1d(bins, feedforward_dim * 2, kernel_size=1, padding=0, bias=bias))
        self.conv1R = nn.Sequential(
            CausalConv1d(bins, bins, kernel_size=7, padding=3, groups=bins, bias=bias),
            CausalConv1d(bins, feedforward_dim // 2, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(feedforward_dim * 2)
        self.conv2 = nn.Sequential(
            CausalConv1d(feedforward_dim * 2, feedforward_dim * 2, kernel_size=7, padding=3, groups=feedforward_dim*2, bias=bias),
            CausalConv1d(feedforward_dim * 2, bins, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm4 = nn.LayerNorm(bins)
        self.self_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm5 = nn.LayerNorm(bins)
        self.enc_attn2 = MultibandFrameAttention(num_bands, bins, cropsize)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm6 = nn.LayerNorm(bins)
        self.conv3 = nn.Linear(bins, feedforward_dim * 2, bias=bias)
        self.swish = nn.SiLU(inplace=True)
        self.conv4 = nn.Linear(feedforward_dim * 2, bins, bias=bias)
        self.dropout5 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, mem, mask=None):
        b, _, h, w = x.shape
        x = self.bottleneck_linear(x.transpose(1,3)).reshape(b,w,h)
        mem = self.mem_bottleneck_linear(mem.transpose(1,3)).reshape(b,w,h)

        h = self.norm1(x)
        hs = self.self_attn1(h, mask=mask)
        hm = self.enc_attn1(h, mem=mem)
        x = x + self.dropout1(hs + hm)

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h.transpose(1,2)).transpose(1,2))
        hR = self.conv1R(h.transpose(1,2)).transpose(1,2)
        h = self.norm3(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))

        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = x + h

        h = self.norm4(x)
        h = self.dropout3(self.self_attn2(h))
        x = x + h

        h = self.norm5(x)
        h = self.dropout4(self.enc_attn2(h, mem=mem))
        x = x + h

        h = self.norm6(x)
        h = self.conv3(h)
        h = self.swish(h)
        h = self.dropout5(self.conv4(h))
        x = x + h
                
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

class FrameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, activate=nn.LeakyReLU, norm=True):
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
        
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=1, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, ((kernel_size-1)//2)+1))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = F.pad(self.weight, (0, self.kernel_size - self.weight.shape[2]))
        return F.conv1d(x, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)