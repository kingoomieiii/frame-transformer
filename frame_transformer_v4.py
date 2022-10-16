import math
import torch
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=2, num_attention_maps=[4,6,8,12,14,16]):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps[0], self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc2 = FrameEncoder(channels * 1 + num_attention_maps[0], channels * 2, self.max_bin)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps[1], self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc3 = FrameEncoder(channels * 2 + num_attention_maps[1], channels * 4, self.max_bin // 2)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps[2], self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc4 = FrameEncoder(channels * 4 + num_attention_maps[2], channels * 6, self.max_bin // 4)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps[3], self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc5 = FrameEncoder(channels * 6 + num_attention_maps[3], channels * 8, self.max_bin // 8)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, num_attention_maps[4], self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc6 = FrameEncoder(channels * 8 + num_attention_maps[4], channels * 10, self.max_bin // 16)
        self.enc6_transformer = FrameTransformerEncoder(channels * 10, num_attention_maps[5], self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec5 = FrameDecoder(channels * (10 + 8) + num_attention_maps[5] + num_attention_maps[4], channels * 8, self.max_bin // 16)
        self.dec5_transformer = FrameTransformerDecoder(channels * 8, channels * 8 + num_attention_maps[4], num_attention_maps[4], self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec4 = FrameDecoder(channels * (8 + 6) + num_attention_maps[4] + num_attention_maps[3], channels * 6, self.max_bin // 8)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, channels * 6 + num_attention_maps[3], num_attention_maps[3], self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec3 = FrameDecoder(channels * (6 + 4) + num_attention_maps[3] + num_attention_maps[2], channels * 4, self.max_bin // 4)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, channels * 4 + num_attention_maps[2], num_attention_maps[2], self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec2 = FrameDecoder(channels * (4 + 2) + num_attention_maps[2] + num_attention_maps[1], channels * 2, self.max_bin // 2)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, channels * 2 + num_attention_maps[1], num_attention_maps[1], self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec1 = FrameDecoder(channels * (2 + 1) + num_attention_maps[1] + num_attention_maps[0], channels * 1, self.max_bin)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, channels * 1 + num_attention_maps[0], num_attention_maps[0], self.max_bin // 1, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.out = nn.Parameter(torch.empty(in_channels, channels + num_attention_maps[0]))
        nn.init.uniform_(self.out, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        e1 = self.enc1(x)
        e1 = torch.cat((e1, self.enc1_transformer(e1)), dim=1)

        e2 = self.enc2(e1)
        e2 = torch.cat((e2, self.enc2_transformer(e2)), dim=1)

        e3 = self.enc3(e2)
        e3 = torch.cat((e3, self.enc3_transformer(e3)), dim=1)

        e4 = self.enc4(e3)
        e4 = torch.cat((e4, self.enc4_transformer(e4)), dim=1)

        e5 = self.enc5(e4)
        e5 = torch.cat((e5, self.enc5_transformer(e5)), dim=1)

        e6 = self.enc6(e5)
        e6 = torch.cat((e6, self.enc6_transformer(e6)), dim=1)

        d5 = self.dec5(e6, e5)
        d5 = torch.cat((d5, self.dec5_transformer(d5, skip=e5)), dim=1)

        d4 = self.dec4(d5, e4)
        d4 = torch.cat((d4, self.dec4_transformer(d4, skip=e4)), dim=1)

        d3 = self.dec3(d4, e3)
        d3 = torch.cat((d3, self.dec3_transformer(d3, skip=e3)), dim=1)

        d2 = self.dec2(d3, e2)
        d2 = torch.cat((d2, self.dec2_transformer(d2, skip=e2)), dim=1)

        d1 = self.dec1(d2, e1)
        d1 = torch.cat((d1, self.dec1_transformer(d1, skip=e1)), dim=1)

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
        return (torch.layer_norm(x.transpose(2,3), (self.weight.shape[-1],), eps=self.eps) * self.weight + self.bias).transpose(2,3)

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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=False):
        super(ResBlock, self).__init__()

        self.activate = SquaredReLU()
        self.norm = FrameNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=(2,1) if downsample else 1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(2,1) if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def __call__(self, x):
        h = self.norm(x)
        h = self.conv2(self.activate(self.conv1(h)))
        x = self.identity(x) + h
        return x

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, dropout=0.1, num_blocks=1):
        super(FrameEncoder, self).__init__()

        self.body = nn.Sequential(*[ResBlock(in_channels if i == 0 else out_channels, out_channels, features, downsample=True if i == num_blocks - 1 and downsample else False) for i in range(num_blocks)])

    def __call__(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, upsample=True, dropout=0.1, has_skip=True, num_blocks=True):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.activate = SquaredReLU()
        self.dropout = nn.Dropout2d(dropout)

        self.body = nn.Sequential(*[ResBlock(in_channels, out_channels, features) for i in range(num_blocks)])

    def __call__(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x

class FrameTransformerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, expansion=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(in_channels, out_channels, 1)

        self.norm1 = FrameNorm(out_channels, features)
        self.attn = MultichannelMultiheadAttention(out_channels, num_heads, features)

        self.norm2 = FrameNorm(out_channels, features)
        self.linear1 = MultichannelLinear(out_channels, out_channels, features, features * expansion, depthwise=True)
        self.linear2 = MultichannelLinear(out_channels, out_channels, features * expansion, features)

        self.activate = SquaredReLU()

    def __call__(self, x):
        x = self.embed(x)

        h = self.attn(self.norm1(x))
        x = x + self.dropout(h)

        h = self.linear2(self.activate(self.linear1(self.norm2(x))))
        x = x + self.dropout(h)

        return x

class FrameTransformerDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, features, expansion=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.embed_self = nn.Conv2d(in_channels, out_channels, 1)
        self.embed_skip = nn.Conv2d(skip_channels, out_channels, 1)

        self.norm1 = FrameNorm(out_channels, features)
        self.self_attn = MultichannelMultiheadAttention(out_channels, num_heads, features)

        self.norm2 = FrameNorm(out_channels, features)
        self.skip_attn = MultichannelMultiheadAttention(out_channels, num_heads, features)

        self.norm3 = FrameNorm(out_channels, features)
        self.linear1 = MultichannelLinear(out_channels, out_channels, features, features * expansion, depthwise=True)
        self.linear2 = MultichannelLinear(out_channels, out_channels, features * expansion, features)

        self.activate = SquaredReLU()

    def __call__(self, x, skip):
        x = self.embed_self(x)
        skip = self.embed_skip(skip)

        h = self.self_attn(self.norm1(x))
        x = x + self.dropout(h)

        h = self.skip_attn(self.norm2(x), mem=skip)
        x = x + self.dropout(h)

        h = self.linear2(self.activate(self.linear1(self.norm3(x))))
        x = x + self.dropout(h)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1, separable=True):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        
        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), groups=channels if separable else 1))

        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), groups=channels if separable else 1))
            
        self.v_proj =  nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), groups=channels if separable else 1))

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