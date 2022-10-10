import math
import numpy
import torch
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_layers=12, expansion=4, num_heads=8, n_fft=2048, sinuisodal_embed=16, dropout=0.1):
        super().__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.num_layers = num_layers
        self.encoder = TransformerEncoder(in_channels, n_fft // 2, expansion=expansion, num_heads=num_heads, dropout=dropout)
        self.out = nn.Conv2d(in_channels * 2, out_channels, 1)

    def __call__(self, x):
        idt, prev_qk = x, None

        num_layers = self.num_layers if not self.training else numpy.random.randint(6, self.num_layers)
        
        for _ in range(num_layers):
            x, prev_qk = self.encoder(x, prev_qk=prev_qk)

        x = self.out(torch.cat((x, idt), dim=1))

        return x

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

class TransformerEncoder(nn.Module):
    def __init__(self, channels, features, expansion=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout2d(dropout)

        self.norm1 = FrameNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)

        self.norm2 = FrameNorm(channels, features)
        self.linear1 = MultichannelLinear(channels, channels, features, features * expansion)
        self.linear2 = MultichannelLinear(channels, channels, features * expansion, features)

        self.activate = SquaredReLU()

    def __call__(self, x, prev_qk=None):
        h, prev_qk = self.attn(self.norm1(x), prev_qk=prev_qk)
        x = x + self.dropout(h)

        h = self.linear2(self.activate(self.linear1(self.norm2(x))))
        x = x + self.dropout(h)

        return x, prev_qk

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)

        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))

        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))
        
        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), bias=False))
        
        self.out_proj = MultichannelLinear(channels, channels, features, features, depthwise=True)

    def __call__(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.rotary_embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)

            if prev_qk is not None:
                qk = qk + prev_qk

            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.out_proj(a)

        return x, qk