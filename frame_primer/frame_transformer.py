import math
import torch
from torch import nn
import torch.nn.functional as F
from frame_primer.rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=[16, 16, 16, 16, 8, 4], expansion=16):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, n_fft=n_fft, downsamples=0, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(channels, num_heads=num_heads[0], n_fft=n_fft, downsamples=0, dropout=dropout, expansion=expansion)

        self.enc2 = FrameEncoder(channels, channels * 2, n_fft=n_fft, downsamples=0)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, num_heads=num_heads[1], n_fft=n_fft, downsamples=1, dropout=dropout, expansion=expansion)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, n_fft=n_fft, downsamples=1)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, num_heads=num_heads[2], n_fft=n_fft, downsamples=2, dropout=dropout, expansion=expansion)

        self.enc4 = FrameEncoder(channels * 4, channels * 8, n_fft=n_fft, downsamples=2)
        self.enc4_transformer = FrameTransformerEncoder(channels * 8, num_heads=num_heads[3], n_fft=n_fft, downsamples=3, dropout=dropout, expansion=expansion)

        self.enc5 = FrameEncoder(channels * 8, channels * 16, n_fft=n_fft, downsamples=3)
        self.enc5_transformer = FrameTransformerEncoder(channels * 16, num_heads=num_heads[4], n_fft=n_fft, downsamples=4, dropout=dropout, expansion=expansion)

        self.enc6 = FrameEncoder(channels * 16, channels * 32, n_fft=n_fft, downsamples=4)
        self.enc6_transformer = FrameTransformerEncoder(channels * 32, num_heads=num_heads[5], n_fft=n_fft, downsamples=5, dropout=dropout, expansion=expansion)

        self.dec5 = FrameDecoder(channels * 32, channels * 16, n_fft=n_fft, downsamples=4)
        self.dec5_transformer = FrameTransformerDecoder(channels * 16, num_heads=num_heads[4], n_fft=n_fft, downsamples=4, dropout=dropout, expansion=expansion)

        self.dec4 = FrameDecoder(channels * 16, channels * 8, n_fft=n_fft, downsamples=3)
        self.dec4_transformer = FrameTransformerDecoder(channels * 8, num_heads=num_heads[3], n_fft=n_fft, downsamples=3, dropout=dropout, expansion=expansion)

        self.dec3 = FrameDecoder(channels * 8, channels * 4, n_fft=n_fft, downsamples=2)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, num_heads=num_heads[2], n_fft=n_fft, downsamples=2, dropout=dropout, expansion=expansion)

        self.dec2 = FrameDecoder(channels * 4, channels * 2, n_fft=n_fft, downsamples=1)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, num_heads=num_heads[1], n_fft=n_fft, downsamples=1, dropout=dropout, expansion=expansion)

        self.dec1 = FrameDecoder(channels * 2, channels * 1, n_fft=n_fft, downsamples=0)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, num_heads=num_heads[0], n_fft=n_fft, downsamples=0, dropout=dropout, expansion=expansion)

        self.out = nn.Parameter(torch.empty(in_channels, channels))
        nn.init.uniform_(self.out, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        e1 = self.enc1_transformer(self.enc1(x))
        e2 = self.enc2_transformer(self.enc2(e1))
        e3 = self.enc3_transformer(self.enc3(e2))
        e4 = self.enc4_transformer(self.enc4(e3))
        e5 = self.enc5_transformer(self.enc5(e4))
        e6 = self.enc6_transformer(self.enc6(e5))
        d5 = self.dec5_transformer(self.dec5(e6, e5), e5)
        d4 = self.dec4_transformer(self.dec4(d5, e4), e4)
        d3 = self.dec3_transformer(self.dec3(d4, e3), e3)
        d2 = self.dec2_transformer(self.dec2(d3, e2), e2)
        d1 = self.dec1_transformer(self.dec1(d2, e1), e1)

        out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)
        
        return out

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, skip_redundant=False, depthwise=True):
        super(MultichannelLinear, self).__init__()

        self.weight_pw = None
        if in_features != out_features or not skip_redundant:
            self.weight_pw = nn.Parameter(torch.empty(out_channels, out_features, in_features))
            nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features))

        self.weight_dw = None
        if in_channels != out_channels or (depthwise and not skip_redundant):
            self.weight_dw = nn.Parameter(torch.empty(out_channels, in_channels))
            nn.init.uniform_(self.weight_dw, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        if self.weight_dw is not None:
            x = torch.matmul(x.transpose(1,3), self.weight_dw.t()).transpose(1,3)
        
        if self.weight_pw is not None:
            x = torch.matmul(x.transpose(2,3), self.weight_pw.transpose(1,2)).transpose(2,3)
        
        return x

class FrameNorm(nn.Module):
    def __init__(self, bins, channels=None):
        super(FrameNorm, self).__init__()

        self.norm = nn.LayerNorm((bins, channels)) if channels is not None else nn.LayerNorm(bins)

    def __call__(self, x):
        if len(self.norm.weight.shape) == 1:
            return self.norm(x.transpose(2,3)).transpose(2,3)
        else:
            return self.norm(x.transpose(1,3)).transpose(1,3)

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, downsamples=0, n_fft=2048, expansion=1):
        super(FrameEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.gelu = nn.GELU()
        self.norm = FrameNorm(bins)
        self.linear1 = MultichannelLinear(in_channels, out_channels, bins, bins * expansion)
        self.linear2 = MultichannelLinear(out_channels, out_channels, bins * expansion, bins // 2 if downsample else bins)
        self.identity = MultichannelLinear(in_channels, out_channels, bins, bins // 2 if downsample else bins, skip_redundant=True)

    def __call__(self, x):
        h = self.norm(x)
        h = self.linear2(self.gelu(self.linear1(h)))
        h = h + self.identity(x)
        
        return h

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, downsamples=0, n_fft=2048, upsample=True, expansion=2):
        super(FrameDecoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.upsample = MultichannelLinear(in_channels, in_channels, bins // 2, bins) if upsample else nn.Identity()

        self.gelu = nn.GELU()
        self.norm = FrameNorm(bins)
        self.linear1 = MultichannelLinear(in_channels + out_channels, out_channels, bins, bins * expansion)
        self.linear2 = MultichannelLinear(out_channels, out_channels, bins * expansion, bins)
        self.identity = MultichannelLinear(in_channels + out_channels, out_channels, bins, bins, skip_redundant=True)

    def __call__(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat((x, skip), dim=1)

        h = self.norm(x)
        h = self.linear2(self.gelu(self.linear1(h)))
        h = h + self.identity(x)
            
        return h

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, bins):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = bins // num_heads // 2, learned_freq=True, freqs_for='pixel')
        self.q_proj = MultichannelLinear(channels, channels, bins, bins, depthwise=False)
        self.k_proj = MultichannelLinear(channels, channels, bins, bins, depthwise=False)
        self.v_proj = MultichannelLinear(channels, channels, bins, bins, depthwise=False)
        self.out_proj = MultichannelLinear(channels, channels, bins, bins, depthwise=False)

    def forward(self, x, mem=None):
        b,c,h,w = x.shape

        q = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).contiguous()
        k = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4).contiguous()
        v = self.q_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4).contiguous()

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q,k) / math.sqrt(h)
            a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3).contiguous()
                
        x = self.out_proj(a)

        return x

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1, downsamples=0, n_fft=2048, expansion=4):
        super(FrameTransformerEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.gelu = nn.GELU()

        self.norm1 = FrameNorm(bins)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, bins)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(bins)
        self.linear1 = MultichannelLinear(channels, channels, bins, bins * expansion, skip_redundant=True)
        self.linear2 = MultichannelLinear(channels, channels, bins * expansion, bins, skip_redundant=True)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def __call__(self, x):
        z = self.norm1(x)
        z = self.attn(z)
        x = x + self.dropout1(z.transpose(2,3)).transpose(2,3)

        z = self.norm2(x)
        z = self.linear2(self.gelu(self.linear1(z)))
        x = x + self.dropout2(z.transpose(2,3)).transpose(2,3)

        return x

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.1, downsamples=0, n_fft=2048, expansion=4):
        super(FrameTransformerDecoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = bins // 2

        self.bins = bins
        self.gelu = nn.GELU()

        self.norm1 = FrameNorm(bins)
        self.attn1 = MultichannelMultiheadAttention(channels, num_heads, bins)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(bins)
        self.attn2 = MultichannelMultiheadAttention(channels, num_heads, bins)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm3 = FrameNorm(bins)
        self.linear1 = MultichannelLinear(channels, channels, bins, bins * expansion, skip_redundant=True)
        self.linear2 = MultichannelLinear(channels, channels, bins * expansion, bins, skip_redundant=True)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip):
        z = self.norm1(x)
        z = self.attn1(z)
        x = x + self.dropout1(z.transpose(2,3)).transpose(2,3)

        z = self.norm2(x)
        z = self.attn2(z, mem=skip)
        x = x + self.dropout2(z.transpose(2,3)).transpose(2,3)

        z = self.norm3(x)
        z = self.linear2(self.gelu(self.linear1(z)))
        x = x + self.dropout3(z.transpose(2,3)).transpose(2,3)

        return x