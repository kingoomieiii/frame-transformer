import math
import torch
from torch import nn
import torch.nn.functional as F

from libft.multichannel_multihead_attention import MultichannelMultiheadAttention
from libft.multichannel_layernorm import MultichannelLayerNorm
from libft.multichannel_linear import MultichannelLinear
from libft.positional_embedding import PositionalEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, transformer_channels=[2,4,8,12,16,24,32]):
        super(FrameTransformer, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.positional_embedding = PositionalEmbedding(in_channels, self.max_bin)

        self.enc0_transformer = FrameTransformerEncoder(in_channels + 1, transformer_channels[0], self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc1 = FrameEncoder(in_channels + transformer_channels[0] + 1, channels, self.max_bin, downsample=False)

        self.enc1_transformer = FrameTransformerEncoder(channels, transformer_channels[1], self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc2 = FrameEncoder(channels + transformer_channels[1], channels * 2, self.max_bin)

        self.enc2_transformer = FrameTransformerEncoder(channels * 2, transformer_channels[2], self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc3 = FrameEncoder(channels * 2 + transformer_channels[2], channels * 4, self.max_bin // 2)

        self.enc3_transformer = FrameTransformerEncoder(channels * 4, transformer_channels[3], self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc4 = FrameEncoder(channels * 4 + transformer_channels[3], channels * 6, self.max_bin // 4)

        self.enc4_transformer = FrameTransformerEncoder(channels * 6, transformer_channels[4], self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc5 = FrameEncoder(channels * 6 + transformer_channels[4], channels * 8, self.max_bin // 8, linear=True)

        self.enc5_transformer = FrameTransformerEncoder(channels * 8, transformer_channels[5], self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc6 = FrameEncoder(channels * 8 + transformer_channels[5], channels * 10, self.max_bin // 16, linear=True)

        self.enc6_transformer = FrameTransformerEncoder(channels * 10, transformer_channels[6], self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec5 = FrameDecoder(channels * 10 + transformer_channels[6] + transformer_channels[5], channels * 8, self.max_bin // 16, linear=True)

        self.dec5_transformer = FrameTransformerDecoder(channels * 8, transformer_channels[5], self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec4 = FrameDecoder(channels * 8 + transformer_channels[5] + transformer_channels[4], channels * 6, self.max_bin // 8)

        self.dec4_transformer = FrameTransformerDecoder(channels * 6, transformer_channels[4], self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec3 = FrameDecoder(channels * 6 + transformer_channels[4] + transformer_channels[3], channels * 4, self.max_bin // 4)

        self.dec3_transformer = FrameTransformerDecoder(channels * 4, transformer_channels[3], self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec2 = FrameDecoder(channels * 4 + transformer_channels[3] + transformer_channels[2], channels * 2, self.max_bin // 2)

        self.dec2_transformer = FrameTransformerDecoder(channels * 2, transformer_channels[2], self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec1 = FrameDecoder(channels * 2 + transformer_channels[2] + transformer_channels[1], channels * 1, self.max_bin)

        self.dec1_transformer = FrameTransformerDecoder(channels * 1, transformer_channels[1], self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.out = nn.Parameter(torch.empty(out_channels, channels + transformer_channels[1]))

        nn.init.uniform_(self.out, a=-1/math.sqrt(channels+1), b=1/math.sqrt(channels+1))

    def __call__(self, x):
        x = torch.cat((x, self.positional_embedding(x)), dim=1)

        e0, _ = self.enc0_transformer(x)
        e1, a1 = self.enc1_transformer(self.enc1(e0))
        e2, a2 = self.enc2_transformer(self.enc2(e1))
        e3, a3 = self.enc3_transformer(self.enc3(e2))
        e4, a4 = self.enc4_transformer(self.enc4(e3))
        e5, a5 = self.enc5_transformer(self.enc5(e4))
        e6, _ = self.enc6_transformer(self.enc6(e5))
        d5 = self.dec5_transformer(self.dec5(e6, e5), skip=a5)
        d4 = self.dec4_transformer(self.dec4(d5, e4), skip=a4)
        d3 = self.dec3_transformer(self.dec3(d4, e3), skip=a3)
        d2 = self.dec2_transformer(self.dec2(d3, e2), skip=a2)
        d1 = self.dec1_transformer(self.dec1(d2, e1), a1)
        out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)

        return out
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1)

        self.norm1 = MultichannelLayerNorm(out_channels, features)
        self.attn = MultichannelMultiheadAttention(out_channels, num_heads, features)

        self.norm2 = MultichannelLayerNorm(out_channels, features)
        self.conv1 = MultichannelLinear(out_channels, out_channels, features, features * expansion)
        self.conv2 = MultichannelLinear(out_channels, out_channels, features * expansion, features)
        
    def __call__(self, x):
        h = self.embed(x)

        z, _ = self.attn(self.norm1(h))
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), h
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1)

        self.norm1 = MultichannelLayerNorm(out_channels, features)
        self.attn1 = MultichannelMultiheadAttention(out_channels, num_heads, features)

        self.norm2 = MultichannelLayerNorm(out_channels, features)
        self.attn2 = MultichannelMultiheadAttention(out_channels, num_heads, features)

        self.norm3 = MultichannelLayerNorm(out_channels, features)
        self.conv1 = MultichannelLinear(out_channels, out_channels, features, features * expansion)
        self.conv2 = MultichannelLinear(out_channels, out_channels, features * expansion, features)
        
    def __call__(self, x, skip):      
        h = self.embed(x)

        z, _ = self.attn1(self.norm1(h))
        h = h + self.dropout(z)

        z, _ = self.attn2(self.norm2(h), mem=skip)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm3(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1)

class SquaredReLU(nn.Module):
    def __call__(self, x):
        return torch.relu(x) ** 2

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=False, linear=False):
        super(ResBlock, self).__init__()

        self.activate = SquaredReLU()
        self.norm = MultichannelLayerNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False) if not linear else MultichannelLinear(in_channels, out_channels, features, features)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=(2,1) if downsample else 1, bias=False) if not linear else MultichannelLinear(out_channels, out_channels, features, features // 2 if downsample else features)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(2,1) if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def __call__(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + h

        return x

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, num_blocks=1, linear=False):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, linear=linear)

    def __call__(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, num_blocks=1, linear=False):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = ResBlock(in_channels + out_channels, out_channels, features, linear=linear)

    def __call__(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x