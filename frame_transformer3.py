import math
import torch
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, out_channels=2, num_layers=12, expansion=4, num_heads=8, n_fft=2048, sinuisodal_embed=16, dropout=0.1):
        super().__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.encoder = FrameEncoder(in_channels, channels)
        self.transformer = nn.Sequential(*[TransformerEncoder(channels + i, n_fft // 4, expansion=expansion, num_heads=num_heads, dropout=dropout) for i in range(num_layers)])
        self.decoder = FrameDecoder(channels + num_layers, channels)
        self.out = nn.Conv2d(channels + in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def __call__(self, x):
        h = self.encoder(x)
        h = self.transformer(h)
        h = self.decoder(h)
        h = self.out(torch.cat((x, h), dim=1))

        return h

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, positionwise=True, depthwise=False, bias=False):
        super(MultichannelLinear, self).__init__()

        self.weight_pw = None
        self.bias_pw = None
        if in_features != out_features or positionwise:
            self.weight_pw = nn.Parameter(torch.empty(in_channels, out_features, in_features))
            nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features))

            if bias:
                self.bias_pw = nn.Parameter(torch.empty(in_channels, out_features, 1))
                bound = 1 / math.sqrt(in_features)
                nn.init.uniform_(self.bias_pw, -bound, bound)

        self.weight_dw = None
        self.bias_dw = None
        if in_channels != out_channels or depthwise:
            self.weight_dw = nn.Parameter(torch.empty(out_channels, in_channels))
            nn.init.uniform_(self.weight_dw, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

            if bias:
                self.bias_dw = nn.Parameter(torch.empty(out_channels, 1, 1))
                bound = 1 / math.sqrt(in_channels)
                nn.init.uniform_(self.bias_pw, -bound, bound)

    def __call__(self, x):
        if self.weight_pw is not None:
            x = torch.matmul(x.transpose(2,3), self.weight_pw.transpose(1,2)).transpose(2,3)

            if self.bias_pw is not None:
                x = x + self.bias_pw

        if self.weight_dw is not None:
            x = torch.matmul(x.transpose(1,3), self.weight_dw.t()).transpose(1,3)

            if self.bias_dw is not None:
                x = x + self.bias_dw
        
        return x
        
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
        return (torch.layer_norm(x.transpose(2,3), (self.weight.shape[-1],), eps=self.eps) * self.weight + self.bias).transpose(2,3)

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_fft=2048):
        super().__init__()

        self.norm1 = FrameNorm(in_channels, n_fft // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2, bias=False)
        self.activate = SquaredReLU()

    def __call__(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm1(x))))
        x = self.identity(x) + h

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_fft=2048):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.norm1 = FrameNorm(in_channels, n_fft // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.activate = SquaredReLU()

    def __call__(self, x):
        x = self.upsample(x)
        h = self.conv2(self.activate(self.conv1(self.norm1(x))))
        x = self.identity(x) + h

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, channels, features, expansion=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = FrameNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)

        self.norm2 = FrameNorm(channels, features)
        self.linear1 = MultichannelLinear(channels, channels + 1, features, features * expansion)
        self.linear2 = MultichannelLinear(channels + 1, channels + 1, features * expansion, features)
        self.identity = nn.Conv2d(channels, channels + 1, kernel_size=1, padding=0)

        self.activate = SquaredReLU()

    def __call__(self, x):
        h = self.attn(self.norm1(x))
        x = x + self.dropout(h)

        h = self.linear2(self.activate(self.linear1(self.norm2(x))))
        x = self.identity(x) + self.dropout(h)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        
        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))

        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))
            
        self.v_proj =  nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))
            
        self.out_proj = MultichannelLinear(channels, channels, features, features, depthwise=True)

    def __call__(self, x, mem=None):
        b,c,h,w = x.shape
        q = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.rotary_embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)
            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.out_proj(a)

        return x