import math
import torch
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=2, num_layers=12):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False)
        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin)
        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2)
        self.enc4 = FrameEncoder(channels * 4, channels * 8, self.max_bin // 4)
        self.enc5 = FrameEncoder(channels * 8, channels * 16, self.max_bin // 8)        
        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels * 16, self.max_bin // 16, num_heads=num_heads, dropout=dropout, expansion=expansion) for _ in range(num_layers)])
        self.dec4 = FrameDecoder(channels * 16, channels * 8, self.max_bin // 8)
        self.dec3 = FrameDecoder(channels * 8, channels * 4, self.max_bin // 4)
        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2)
        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin)

        self.out = nn.Parameter(torch.empty(in_channels, channels))
        nn.init.uniform_(self.out, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        pqk = None
        for encoder in self.encoder:
            e5, pqk = encoder(e5, pqk)

        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)    

        return out

class SquaredReLU(nn.Module):
    def __call__(self, x):
        return torch.relu(x) ** 2

class FrameNorm(nn.Module):
    def __init__(self, channels, features, eps=0.00001):
        super(FrameNorm, self).__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(channels, 1, features))
        self.bias = nn.Parameter(torch.empty(channels, 1, features))
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def __call__(self, x):
        return torch.layer_norm(x, (self.weight.shape[-1],), eps=self.eps) * self.weight + self.bias

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, dropout=0.1):
        super(FrameEncoder, self).__init__()

        self.relu = SquaredReLU()
        self.norm = FrameNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=(2,1) if downsample else 1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(2,1) if downsample else 1, bias=False)

    def __call__(self, x):
        h = self.norm(x.transpose(2,3)).transpose(2,3)
        h = self.conv2(self.relu(self.conv1(h)))
        x = self.identity(x) + h

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, upsample=True, dropout=0.1, has_skip=True):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.activate = SquaredReLU()
        self.dropout = nn.Dropout2d(dropout)
        
        self.norm = FrameNorm(in_channels + out_channels, features)
        self.conv1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.identity = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def __call__(self, x, skip):
        x = torch.cat((self.upsample(x), self.dropout(skip)), dim=1)
        h = self.norm(x.transpose(2,3)).transpose(2,3)
        h = self.conv2(self.activate(self.conv1(h)))
        x = self.identity(x) + h

        return x

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, positionwise=True, depthwise=True):
        super(MultichannelLinear, self).__init__()

        self.weight_pw = None
        if in_features != out_features or positionwise:
            self.weight_pw = nn.Parameter(torch.empty(in_channels, out_features, in_features))
            nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features))

        self.weight_dw = None
        if in_channels != out_channels or depthwise:
            self.weight_dw = nn.Parameter(torch.empty(out_channels, in_channels))
            nn.init.uniform_(self.weight_dw, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        if self.weight_pw is not None:
            x = torch.matmul(x.transpose(2,3), self.weight_pw.transpose(1,2)).transpose(2,3)

        if self.weight_dw is not None:
            x = torch.matmul(x.transpose(1,3), self.weight_dw.t()).transpose(1,3) 
        
        return x

class FrameDrop(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout2d(dropout)

    def __call__(self, x):
        b,c,h,w = x.shape
        x = self.dropout(x.transpose(2,3).reshape(b,c*w,h,1))
        return x.reshape(b,c,w,h).transpose(2,3)

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, mixed_precision=False, dropout=0.1):
        super().__init__()

        self.mixed_precision = mixed_precision
        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        
        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            MultichannelLinear(channels, channels, features, features, depthwise=False))

        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            FrameDrop(dropout))
            
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            MultichannelLinear(channels, channels, features, features, depthwise=False))
            
        self.out_proj = MultichannelLinear(channels, channels, features, features, depthwise=True)

    def __call__(self, x, mem=None, pqk=None):
        b,c,h,w = x.shape

        q = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.rotary_embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)

            if pqk is not None:
                qk = qk + pqk

            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.out_proj(a)

        return x, qk

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, num_heads=4, dropout=0.1, expansion=4):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout2d(dropout)

        self.norm1 = FrameNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)

        self.norm2 = FrameNorm(channels, features)
        self.linear1 = MultichannelLinear(channels, channels, features, features * expansion, depthwise=True)
        self.linear2 = MultichannelLinear(channels, channels, features * expansion, features, depthwise=True)
        
    def __call__(self, x, pqk=None):
        z = self.norm1(x.transpose(2,3)).transpose(2,3)
        z, qk = self.attn(z, pqk=pqk)
        x = x + self.dropout(z)

        z = self.norm2(x.transpose(2,3)).transpose(2,3)
        z = self.linear2(self.activate(self.linear1(z)))
        x = x + self.dropout(z)

        return x, qk