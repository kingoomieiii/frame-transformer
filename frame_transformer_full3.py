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

        self.enc1_transformer = FrameTransformerEncoder(in_channels, channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc2_transformer = FrameTransformerEncoder(channels * 1, channels * 2, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc3_transformer = FrameTransformerEncoder(channels * 2, channels * 4, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc4_transformer = FrameTransformerEncoder(channels * 4, channels * 6, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc5_transformer = FrameTransformerEncoder(channels * 6, channels * 8, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.enc6_transformer = FrameTransformerEncoder(channels * 8, channels * 10, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec5_transformer = FrameTransformerDecoder(channels * 10, channels * 8, channels * 8, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec4_transformer = FrameTransformerDecoder(channels * 8, channels * 6, channels * 6, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec3_transformer = FrameTransformerDecoder(channels * 6, channels * 4, channels * 4, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec2_transformer = FrameTransformerDecoder(channels * 4, channels * 2, channels * 2, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.dec1_transformer = FrameTransformerDecoder(channels * 2, channels * 1, channels * 1, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        self.out_transformer = FrameTransformerDecoder(channels * 1, in_channels, out_channels, self.max_bin // 1, dropout=dropout, expansion=expansion, num_heads=num_heads)

    def __call__(self, x):
        e1 = self.enc1_transformer(x)
        e2 = self.enc2_transformer(e1)
        e3 = self.enc3_transformer(e2)
        e4 = self.enc4_transformer(e3)
        e5 = self.enc5_transformer(e4)
        e6 = self.enc6_transformer(e5)
        d5 = self.dec5_transformer(e6, e5)
        d4 = self.dec4_transformer(d5, e4)
        d3 = self.dec3_transformer(d4, e3)
        d2 = self.dec2_transformer(d3, e2)
        d1 = self.dec1_transformer(d2, e1)
        out = self.out_transformer(d1, x)

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
    def __init__(self, in_channels, out_channels, features, upsample=True, dropout=0.1, has_skip=True, num_blocks=1):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = nn.Sequential(*[ResBlock(in_channels + out_channels if i == 0 else out_channels, out_channels, features) for i in range(num_blocks)])

    def __call__(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        
        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), groups=channels))

        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), groups=channels))
            
        self.v_proj =  nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), groups=channels))

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
    def __init__(self, in_channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, downsample=False):
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
        z = self.norm1(x)
        z = self.attn(z)
        x = x + self.dropout(z)

        z = self.norm2(x)
        z = self.conv2(self.activate(self.conv1(z)))
        x = self.identity(x) + self.dropout(z)

        return x

class FrameTransformerDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, upsample=False):
        super(FrameTransformerDecoder, self).__init__()
        
        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)

        self.norm1 = FrameNorm(in_channels, features)
        self.attn1 = MultichannelMultiheadAttention(in_channels, num_heads, features)

        self.norm2 = FrameNorm(in_channels, features)
        self.skip_bottleneck = nn.Conv2d(skip_channels, in_channels, 1) if skip_channels != in_channels else nn.Identity()
        self.attn2 = MultichannelMultiheadAttention(in_channels, num_heads, features)

        self.norm2 = FrameNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        
    def __call__(self, x, skip):
        x = self.upsample(x)

        z = self.norm1(x)
        z = self.attn1(z)
        x = x + self.dropout(z)

        z = self.norm2(x)
        z = self.attn2(z, mem=self.skip_bottleneck(skip))
        x = x + self.dropout(z)

        z = self.norm2(x)
        z = self.conv2(self.activate(self.conv1(z)))
        x = self.identity(x) + self.dropout(z)

        return x