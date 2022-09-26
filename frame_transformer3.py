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

        self.encoder = FrameEncoder(in_channels, channels, n_fft // 2, n_fft // 4)
        self.transformer = nn.ModuleList([TransformerEncoder(channels, n_fft // 4, expansion=expansion, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.decoder = FrameDecoder(channels, out_channels, n_fft // 4, n_fft // 2)

    def __call__(self, x):
        e = self.encoder(x)

        for encoder in self.transformer:
            e = encoder(e)

        d = self.decoder(e)

        return d

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, positionwise=True, depthwise=False):
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
    def __init__(self, in_channels, out_channels, in_features, out_features, n_fft=2048):
        super().__init__()

        self.features = n_fft // 2
        
        self.linear1 = MultichannelLinear(in_channels, out_channels, in_features, out_features, depthwise=True)
        self.linear2 = MultichannelLinear(out_channels, out_channels, out_features, out_features, depthwise=True)
        self.identity = MultichannelLinear(in_channels, out_channels, in_features, out_features, depthwise=True)
        self.activate = SquaredReLU()        

    def __call__(self, x):
        z = self.linear2(self.activate(self.linear1(x)))
        x = self.identity(x) + z

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features):
        super().__init__()

        self.out_channels = out_channels

        self.linear1 = MultichannelLinear(in_channels, out_channels, in_features, out_features, depthwise=True)
        self.linear2 = MultichannelLinear(out_channels, out_channels, out_features, out_features, depthwise=True)
        self.identity = MultichannelLinear(in_channels, out_channels, in_features, out_features, depthwise=True)
        self.activate = SquaredReLU()

    def __call__(self, x):
        z = self.linear2(self.activate(self.linear1(x)))
        x = self.identity(x) + z

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, channels, features, expansion=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = FrameNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)

        self.norm2 = FrameNorm(channels, features)
        self.linear1 = MultichannelLinear(channels, channels, features, features * expansion)
        self.linear2 = MultichannelLinear(channels, channels, features * expansion, features)

        self.activate = SquaredReLU()

    def __call__(self, x):
        h = self.norm1(x.transpose(2,3)).transpose(2,3)
        h = self.attn(h)
        x = x + self.dropout(h)

        h = self.norm2(x.transpose(2,3)).transpose(2,3)
        h = self.linear2(self.activate(self.linear1(h)))
        x = x + self.dropout(h)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, mixed_precision=False):
        super().__init__()

        self.mixed_precision = mixed_precision
        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        
        self.q_proj = MultichannelLinear(channels, channels, features, features, depthwise=True)
        self.k_proj = MultichannelLinear(channels, channels, features, features, depthwise=True)
        self.v_proj =  MultichannelLinear(channels, channels, features, features, depthwise=True)
        self.out_proj = MultichannelLinear(channels, channels, features, features)

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