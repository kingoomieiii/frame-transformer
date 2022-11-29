import math
import torch
from torch import nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding
from multichannel_layernorm import MultichannelLayerNorm, FrameNorm
from multichannel_linear import MultichannelLinear

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc0_transformer = FrameTransformerEncoder(in_channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc1 = FrameEncoder(in_channels + 1, channels, self.max_bin, downsample=False)

        self.enc1_transformer = FrameTransformerEncoder(channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc2 = FrameEncoder(channels + 1, channels * 2, self.max_bin)

        self.enc2_transformer = FrameTransformerEncoder(channels * 2, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc3 = FrameEncoder(channels * 2 + 1, channels * 4, self.max_bin // 2)

        self.enc3_transformer = FrameTransformerEncoder(channels * 4, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc4 = FrameEncoder(channels * 4 + 1, channels * 6, self.max_bin // 4)

        self.enc4_transformer = FrameTransformerEncoder(channels * 6, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc5 = FrameEncoder(channels * 6 + 1, channels * 8, self.max_bin // 8)

        self.enc5_transformer = FrameTransformerEncoder(channels * 8, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc6 = FrameEncoder(channels * 8 + 1, channels * 10, self.max_bin // 16)

        self.enc6_transformer = FrameTransformerEncoder(channels * 10, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec5 = FrameDecoder(channels * 10 + 2, channels * 8, self.max_bin // 16)

        self.dec5_transformer = FrameTransformerDecoder(channels * 8, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec4 = FrameDecoder(channels * 8 + 2, channels * 6, self.max_bin // 8)

        self.dec4_transformer = FrameTransformerDecoder(channels * 6, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec3 = FrameDecoder(channels * 6 + 2, channels * 4, self.max_bin // 4)

        self.dec3_transformer = FrameTransformerDecoder(channels * 4, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec2 = FrameDecoder(channels * 4 + 2, channels * 2, self.max_bin // 2)

        self.dec2_transformer = FrameTransformerDecoder(channels * 2, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec1 = FrameDecoder(channels * 2 + 2, channels * 1, self.max_bin)

        self.dec1_transformer = FrameTransformerDecoder(channels * 1, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.out = nn.Parameter(torch.empty(out_channels, channels + 1))

        nn.init.uniform_(self.out, a=-1/math.sqrt(channels+1), b=1/math.sqrt(channels+1))

    def __call__(self, x):
        e0, _ = self.enc0_transformer(x)
        e1, a1 = self.enc1_transformer(self.enc1(e0))
        e2, a2 = self.enc2_transformer(self.enc2(e1))
        e3, a3 = self.enc3_transformer(self.enc3(e2))
        e4, a4 = self.enc4_transformer(self.enc4(e3))
        e5, a5 = self.enc5_transformer(self.enc5(e4))
        e6, _ = self.enc6_transformer(self.enc6(e5))
        d5 = self.dec5_transformer(self.dec5(e6, e5), a5)
        d4 = self.dec4_transformer(self.dec4(d5, e4), a4)
        d3 = self.dec3_transformer(self.dec3(d4, e3), a3)
        d2 = self.dec2_transformer(self.dec2(d3, e2), a2)
        d1 = self.dec1_transformer(self.dec1(d2, e1), a1)
        out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)

        return out

class SquaredReLU(nn.Module):
    def __call__(self, x):
        return torch.relu(x) ** 2

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=False):
        super(ResBlock, self).__init__()

        self.activate = SquaredReLU()
        self.norm = FrameNorm(features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=(2,1) if downsample else 1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(2,1) if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def __call__(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + h

        return x

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, num_blocks=1):
        super(FrameEncoder, self).__init__()

        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, features, downsample=True if i == num_blocks - 1 and downsample else False) for i in range(num_blocks)])

    def __call__(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, num_blocks=1):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = nn.Sequential(*[ResBlock(in_channels + out_channels if i == 0 else out_channels, out_channels, features) for i in range(num_blocks)])

    def __call__(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)        
        self.q_proj = MultichannelLinear(channels, channels, features, features)
        self.k_proj = MultichannelLinear(channels, channels, features, features)
        self.v_proj =  MultichannelLinear(channels, channels, features, features)
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
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, 1, 1)

        self.norm1 = FrameNorm(features)
        self.attn = MultichannelMultiheadAttention(1, num_heads, features)

        self.norm2 = FrameNorm(features)
        self.conv1 = MultichannelLinear(1, 1, features, features * expansion)
        self.conv2 = MultichannelLinear(1, 1, features * expansion, features)
        
    def __call__(self, x):      
        h = self.embed(x)

        z = self.attn(self.norm1(h))
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), h
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, 1, 1)

        self.norm1 = FrameNorm(features)
        self.attn1 = MultichannelMultiheadAttention(1, num_heads, features)

        self.norm2 = FrameNorm(features)
        self.attn2 = MultichannelMultiheadAttention(1, num_heads, features)

        self.norm3 = FrameNorm(features)
        self.conv1 = MultichannelLinear(1, 1, features, features * expansion)
        self.conv2 = MultichannelLinear(1, 1, features * expansion, features)
        
    def __call__(self, x, skip):      
        h = self.embed(x)

        z = self.attn1(self.norm1(h))
        h = h + self.dropout(z)

        z = self.attn2(self.norm1(h), mem=skip)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1)