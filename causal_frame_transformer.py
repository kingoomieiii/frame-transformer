import math
import torch
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1):
        super(CausalConv2d, self).__init__()

        self.causal_padding = (kernel_size // 2) + padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(stride,1), padding=(padding, 0), bias=False, groups=groups)

    def __call__(self, x):
        return self.conv(F.pad(x, (self.causal_padding, 0)))

class CausalConv1xN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1):
        super(CausalConv1xN, self).__init__()

        self.causal_padding = (kernel_size // 2) + padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel_size), stride=(stride,1), bias=False, groups=groups)

    def __call__(self, x):
        return self.conv(F.pad(x, (self.causal_padding, 0)))

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=6):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False)
        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin)
        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2)
        self.enc4 = FrameEncoder(channels * 4, channels * 8, self.max_bin // 4)
        self.enc5 = FrameEncoder(channels * 8, channels * 16, self.max_bin // 8)
        self.enc6 = FrameEncoder(channels * 16, channels * 32, self.max_bin // 16)

        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels * 32, self.max_bin // 32, num_heads=num_heads, dropout=dropout, expansion=expansion) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([FrameTransformerDecoder(channels * 32, self.max_bin // 32, num_heads=num_heads, dropout=dropout, expansion=expansion) for _ in range(num_layers)])
        
        self.dec5 = FrameDecoder(channels * 32, channels * 16, self.max_bin // 16)
        self.dec4 = FrameDecoder(channels * 16, channels * 8, self.max_bin // 8)
        self.dec3 = FrameDecoder(channels * 8, channels * 4, self.max_bin // 4)
        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2)
        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin)

        self.mask = None

        self.out = nn.Parameter(torch.empty(in_channels, channels))
        nn.init.uniform_(self.out, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def _generate_mask(self, src):
        if self.mask is None or src.shape[3] != self.mask.shape[0]:
            self.mask = torch.triu(torch.ones(src.shape[3], src.shape[3]) * float('-inf'), diagonal=1).to(src.device)

        return self.mask

    def __call__(self, src, tgt):
        mask = self._generate_mask(src)

        se1 = self.enc1(src)
        se2 = self.enc2(se1)
        se3 = self.enc3(se2)
        se4 = self.enc4(se3)
        se5 = self.enc5(se4)
        se6 = self.enc6(se5)

        te1 = self.enc1(tgt)
        te2 = self.enc2(te1)
        te3 = self.enc3(te2)
        te4 = self.enc4(te3)
        te5 = self.enc5(te4)
        te6 = self.enc6(te5)

        for encoder in self.encoder:
            se6 = encoder(se6)
            
        for decoder in self.decoder:
            te5 = decoder(te6, se6, mask)

        d5 = self.dec5(te6, se5)
        d4 = self.dec4(d5, se4)
        d3 = self.dec3(d4, se3)
        d2 = self.dec2(d3, se2)
        d1 = self.dec1(d2, se1)

        out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)    

        return out

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, positionwise=True, depthwise=True):
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
    def __init__(self, in_channels, out_channels, features, downsample=True, dropout=0.1):
        super(FrameEncoder, self).__init__()

        self.relu = SquaredReLU()
        self.norm = FrameNorm(in_channels, features)
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2 if downsample else 1)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(2,1) if downsample else 1, bias=False)

    def __call__(self, x):
        h = self.norm(x.transpose(2,3)).transpose(2,3)
        h = self.conv2(self.relu(self.conv1(h)))
        x = self.identity(x) + h

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, upsample=True, dropout=0.1, has_skip=True):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)

        channels = in_channels + out_channels * 1
        self.channels = channels
        self.relu = SquaredReLU()
        self.norm = FrameNorm(channels, features)
        self.conv1 = CausalConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv2d(channels, out_channels, kernel_size=3, padding=1)
        self.identity = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0, bias=False)

    def __call__(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        h = self.norm(x.transpose(2,3)).transpose(2,3)
        h = self.conv2(self.relu(self.conv1(h)))
        x = self.identity(x) + h

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, mixed_precision=False):
        super().__init__()

        self.mixed_precision = mixed_precision
        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        
        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            CausalConv1xN(channels, channels, kernel_size=7, padding=3, groups=channels))

        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            CausalConv1xN(channels, channels, kernel_size=7, padding=3, groups=channels))
            
        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, depthwise=False),
            CausalConv1xN(channels, channels, kernel_size=7, padding=3, groups=channels))
            
        self.out_proj = MultichannelLinear(channels, channels, features, features, depthwise=False)

    def __call__(self, x, mem=None, mask=None):
        b,c,h,w = x.shape

        q = self.rotary_embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.rotary_embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(), k.float()) / math.sqrt(h)

            if mask is not None:
                qk = qk + mask

            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)

        x = self.out_proj(a)
        
        return x

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, num_heads=4, dropout=0.1, expansion=4):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()

        self.norm1 = FrameNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(channels, features)
        self.linear1 = MultichannelLinear(channels, channels, features, features * expansion, depthwise=False)
        self.linear2 = MultichannelLinear(channels, channels, features * expansion, features, depthwise=False)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def __call__(self, x):
        z = self.norm1(x.transpose(2,3)).transpose(2,3)
        z = self.attn(z)
        z = self.dropout1(z)
        x = x + z

        z = self.norm2(x.transpose(2,3)).transpose(2,3)
        z = self.linear2(self.activate(self.linear1(z)))
        z = self.dropout2(z)
        x = x + z

        return x

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, features, num_heads=4, dropout=0.1, expansion=4):
        super(FrameTransformerDecoder, self).__init__()
        
        self.activate = SquaredReLU()

        self.norm1 = FrameNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention(channels, num_heads, features)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = FrameNorm(channels, features)
        self.attn2 = MultichannelMultiheadAttention(channels, num_heads, features)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.norm3 = FrameNorm(channels, features)
        self.linear1 = MultichannelLinear(channels, channels, features, features * expansion, depthwise=False)
        self.linear2 = MultichannelLinear(channels, channels, features * expansion, features, depthwise=False)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, mem, mask=None):
        z = self.norm1(x.transpose(2,3)).transpose(2,3)
        z = self.attn1(z, mask=mask)
        z = self.dropout1(z)
        x = x + z

        z = self.norm2(x.transpose(2,3)).transpose(2,3)
        z = self.attn2(z, mem=mem)
        z = self.dropout2(z)
        x = x + z

        z = self.norm3(x.transpose(2,3)).transpose(2,3)
        z = self.linear2(self.activate(self.linear1(z)))
        z = self.dropout3(z)
        x = x + z

        return x