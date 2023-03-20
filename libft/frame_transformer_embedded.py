import math
import torch
from torch import nn
import torch.nn.functional as F

from libft.positional_embedding import PositionalEmbedding
from libft.rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, transformer_channels=[12,12,12,12,12,12,12]):
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

        nn.init.uniform_(self.out, a=-1/math.sqrt(channels + transformer_channels[1]), b=1/math.sqrt(channels + transformer_channels[1]))

    def __call__(self, x):
        x = torch.cat((x, self.positional_embedding(x)), dim=1)
        e0, _, s0 = self.enc0_transformer(x)
        e1, a1, s1 = self.enc1_transformer(self.enc1(e0), prev_qk=s0)
        e2, a2, s2 = self.enc2_transformer(self.enc2(e1), prev_qk=s1)
        e3, a3, s3 = self.enc3_transformer(self.enc3(e2), prev_qk=s2)
        e4, a4, s4 = self.enc4_transformer(self.enc4(e3), prev_qk=s3)
        e5, a5, s5 = self.enc5_transformer(self.enc5(e4), prev_qk=s4)
        e6, _, s6 = self.enc6_transformer(self.enc6(e5), prev_qk=s5)
        d5, prev_qk = self.dec5_transformer(self.dec5(e6, e5), skip=a5, prev_qk=s6, skip_qk=s5)
        d4, prev_qk = self.dec4_transformer(self.dec4(d5, e4), skip=a4, prev_qk=prev_qk, skip_qk=s4)
        d3, prev_qk = self.dec3_transformer(self.dec3(d4, e3), skip=a3, prev_qk=prev_qk, skip_qk=s3)
        d2, prev_qk = self.dec2_transformer(self.dec2(d3, e2), skip=a2, prev_qk=prev_qk, skip_qk=s2)
        d1, _ = self.dec1_transformer(self.dec1(d2, e1), skip=a1, prev_qk=prev_qk, skip_qk=s1)
        out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)

        return out
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1)

        self.norm1 = MultichannelLayerNorm(out_channels, features)
        self.attn = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=9, padding=4)

        self.norm2 = MultichannelLayerNorm(out_channels, features)
        self.conv1 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1)
        
    def __call__(self, x, prev_qk=None):
        h = self.embed(x)

        z, prev_qk = self.attn(self.norm1(h), prev_qk=prev_qk)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), h, prev_qk
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1)

        self.norm1 = MultichannelLayerNorm(out_channels, features)
        self.attn1 = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=9, padding=4)

        self.norm2 = MultichannelLayerNorm(out_channels, features)
        self.attn2 = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=9, padding=4)

        self.norm3 = MultichannelLayerNorm(out_channels, features)
        self.conv1 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1)
        
    def __call__(self, x, skip, prev_qk=None, skip_qk=None):
        h = self.embed(x)

        z, prev_qk = self.attn1(self.norm1(h), prev_qk=prev_qk)
        h = h + self.dropout(z)

        z, _ = self.attn2(self.norm2(h), mem=skip, prev_qk=skip_qk)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm3(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), prev_qk

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

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads)

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            MultichannelLinear(channels, channels, features, features))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            MultichannelLinear(channels, channels, features, features))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            MultichannelLinear(channels, channels, features, features))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features)

    def __call__(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=True):
            qk = torch.matmul(q,k) / math.sqrt(h)

            if prev_qk is not None:
                qk = qk + prev_qk

            a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.o_proj(a)

        return x, qk

class MultichannelLayerNorm(nn.Module):
    def __init__(self, channels, features, eps=1e-8, trainable=True):
        super(MultichannelLayerNorm, self).__init__()
        
        self.eps = eps

        if trainable:
            self.weight = nn.Parameter(torch.empty(channels, 1, features))
            self.bias = nn.Parameter(torch.empty(channels, 1, features))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.register_buffer('weight', torch.ones(channels, 1, features))
            self.register_buffer('bias', torch.zeros(channels, 1, features))

    def __call__(self, x):
        d = len(x.shape)

        if d == 2:
            x = x.unsqueeze(-1).unsqueeze(1)
        elif d == 3:
            x = x.unsqueeze(-1)

        x = (torch.layer_norm(x.transpose(2,3), (self.weight.shape[-1],), eps=self.eps) * self.weight + self.bias).transpose(2,3)

        if d == 2 or d == 3:
            x = x.squeeze(-1)

        return x   

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, positionwise=True, depthwise=False, bias=False, include_position=False):
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
                nn.init.uniform_(self.bias_dw, -bound, bound)

    def __call__(self, x):
        d = len(x.shape)

        if d == 2:
            x = x.unsqueeze(-1).unsqueeze(1)
        elif d == 3:
            x = x.unsqueeze(-1)

        if self.weight_pw is not None:
            x = torch.matmul(x.transpose(2,3), self.weight_pw.transpose(1,2)).transpose(2,3)

            if self.bias_pw is not None:
                x = x + self.bias_pw

        if self.weight_dw is not None:
            x = torch.matmul(x.transpose(1,3), self.weight_dw.t()).transpose(1,3)

            if self.bias_dw is not None:
                x = x + self.bias_dw

        if d == 2 or d == 3:
            x = x.squeeze(-1)
        
        return x