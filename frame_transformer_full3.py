import math
import torch
from torch import nn
import torch.nn.functional as F
from frame_transformer4 import MultichannelLinear
from rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, out_channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=5):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.enc1 = FrameTransformerEncoder(in_channels, channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc2 = FrameTransformerEncoder(channels * 1, channels * 2, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc3 = FrameTransformerEncoder(channels * 2, channels * 4, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc4 = FrameTransformerEncoder(channels * 4, channels * 6, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc5 = FrameTransformerEncoder(channels * 6, channels * 8, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc6 = FrameTransformerEncoder(channels * 8, channels * 10, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec6 = FrameTransformerDecoder(channels * 10, channels * 8, channels * 8, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec5 = FrameTransformerDecoder(channels * 8, channels * 6, channels * 6, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec4 = FrameTransformerDecoder(channels * 6, channels * 4, channels * 4, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec3 = FrameTransformerDecoder(channels * 4, channels * 2, channels * 2, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec2 = FrameTransformerDecoder(channels * 2, channels * 1, channels * 1, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec1 = FrameTransformerDecoder(channels * 1, in_channels, out_channels, self.max_bin // 1, dropout=dropout, expansion=expansion, num_heads=num_heads)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        d6 = self.dec6(e6, e5)
        d5 = self.dec5(d6, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x)

        return d1

class FrameTransformerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = FrameNorm(in_channels, features)
        self.attn = MultichannelMultiheadAttention(in_channels, num_heads, features)

        self.norm2 = FrameNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1, bias=False, stride=(2,1))
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=(2,1))

    def __call__(self, x):
        h = self.attn(self.norm1(x))
        x = x + self.dropout(h)

        h = self.conv2(self.activate(self.conv1(self.norm2(x))))
        x = self.identity(x) + self.dropout(h)

        return x

class FrameTransformerDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerDecoder, self).__init__()
        
        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)

        self.norm1 = FrameNorm(in_channels, features)
        self.attn1 = MultichannelMultiheadAttention(in_channels, num_heads, features)

        self.norm2 = FrameNorm(in_channels, features)
        self.skip_bottleneck = nn.Conv2d(skip_channels, in_channels, 1) if skip_channels != in_channels else nn.Identity()
        self.attn2 = MultichannelMultiheadAttention(in_channels, num_heads, features)

        self.norm3 = FrameNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        
    def __call__(self, x, skip):
        x = self.upsample(x)

        h = self.attn1(self.norm1(x))
        x = x + self.dropout(h)

        h = self.attn2(self.norm2(x), mem=self.skip_bottleneck(skip))
        x = x + self.dropout(h)

        h = self.conv2(self.activate(self.conv1(self.norm3(x))))
        x = self.identity(x) + self.dropout(h)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        self.q_proj = MultichannelLinear(channels, channels, features, features)
        self.k_proj = MultichannelLinear(channels, channels, features, features)            
        self.v_proj = MultichannelLinear(channels, channels, features, features)
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