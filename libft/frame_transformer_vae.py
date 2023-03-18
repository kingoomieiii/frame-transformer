import math
import torch
from torch import nn
import torch.nn.functional as F

from libft.multichannel_layernorm import FrameNorm, MultichannelLayerNorm
from libft.multichannel_linear import MultichannelLinear
from libft.positional_embedding import PositionalEmbedding
from libft.rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_bridge_layers=4, num_attention_maps=[3,3,3,3,3]):
        super(FrameTransformer, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.positional_embedding = PositionalEmbedding(in_channels, self.max_bin)

        self.enc1 = FrameEncoder(in_channels + 1, channels, self.max_bin, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps[0], self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc2 = FrameEncoder(channels + num_attention_maps[0], channels * 2, self.max_bin)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps[1], self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc3 = FrameEncoder(channels * 2 + num_attention_maps[1], channels * 4, self.max_bin // 2)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps[2], self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc4 = FrameEncoder(channels * 4 + num_attention_maps[2], channels * 6, self.max_bin // 4)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps[3], self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc5 = FrameEncoder(channels * 6 + num_attention_maps[3], channels * 8, self.max_bin // 8)

        self.mu = nn.Sequential(
            *[FrameTransformerBridge(channels * 8 + num_attention_maps[4] * i, num_attention_maps[4], self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads) for i in range(num_bridge_layers)],
            ResBlock(channels * 8 + num_attention_maps[4] * num_bridge_layers, channels * 8, self.max_bin // 16))

        self.std = nn.Sequential(
            *[FrameTransformerBridge(channels * 8 + num_attention_maps[4] * i, num_attention_maps[4], self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads) for i in range(num_bridge_layers)],
            ResBlock(channels * 8 + num_attention_maps[4] * num_bridge_layers, channels * 8, self.max_bin // 16))

        self.dec4 = FrameDecoder(channels * 8 + num_attention_maps[3], channels * 6, self.max_bin // 8)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, num_attention_maps[3], self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec3 = FrameDecoder(channels * 6 + num_attention_maps[3] + num_attention_maps[2], channels * 4, self.max_bin // 4)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, num_attention_maps[2], self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec2 = FrameDecoder(channels * 4 + num_attention_maps[2] + num_attention_maps[1], channels * 2, self.max_bin // 2)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, num_attention_maps[1], self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec1 = FrameDecoder(channels * 2 + num_attention_maps[1] + num_attention_maps[0], channels * 1, self.max_bin // 1)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, num_attention_maps[0], self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.out = nn.Conv2d(channels + num_attention_maps[0], out_channels, 1)
        
    def forward(self, x):
        x = torch.cat((x, self.positional_embedding(x)), dim=1)

        e1 = self.enc1(x)
        e1, a1 = self.enc1_transformer(e1)

        e2 = self.enc2(e1)
        e2, a2 = self.enc2_transformer(e2)

        e3 = self.enc3(e2)
        e3, a3 = self.enc3_transformer(e3)

        e4 = self.enc4(e3)
        e4, a4 = self.enc4_transformer(e4)

        e5 = self.enc5(e4)
    
        mu = self.mu(e5)
        std = F.softplus(self.std(e5))
        logvar = torch.log(std) * 2
        eps = torch.randn_like(std)
        z = mu + eps * std

        h = self.dec4(z, e4)
        h = self.dec4_transformer(h, a4)
        
        h = self.dec3(h, e3)
        h = self.dec3_transformer(h, a3)

        h = self.dec2(h, e2)
        h = self.dec2_transformer(h, a2)

        h = self.dec1(h, e1)
        h = self.dec1_transformer(h, a1)

        out = self.out(h)

        return out, mu, logvar
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.SELU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

        self.norm1 = FrameNorm(features)
        self.attn = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=5, padding=2)

        self.norm2 = FrameNorm(features)
        self.conv1 = MultichannelLinear(out_channels, out_channels, features, features * expansion)
        self.conv2 = MultichannelLinear(out_channels, out_channels, features * expansion, features)
        
    def forward(self, x):
        h = self.embed(x)

        z, _ = self.attn(self.norm1(h))
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), h
        
class FrameTransformerBridge(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, concatenate=True):
        super(FrameTransformerBridge, self).__init__()

        self.concatenate = concatenate

        self.activate = nn.SELU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

        self.norm1 = FrameNorm(features)
        self.attn = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=5, padding=2)

        self.norm2 = FrameNorm(features)
        self.conv1 = MultichannelLinear(out_channels, out_channels, features, features * expansion)
        self.conv2 = MultichannelLinear(out_channels, out_channels, features * expansion, features)
        
    def forward(self, x):
        h = self.embed(x)

        z, _ = self.attn(self.norm1(h))
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1)
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = nn.SELU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

        self.norm1 = FrameNorm(features)
        self.attn1 = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=5, padding=2)

        self.norm2 = FrameNorm(features)
        self.attn2 = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=5, padding=2)

        self.norm3 = FrameNorm(features)
        self.conv1 = MultichannelLinear(out_channels, out_channels, features, features * expansion)
        self.conv2 = MultichannelLinear(out_channels, out_channels, features * expansion, features)
        
    def forward(self, x, skip):
        h = self.embed(x)

        z, _ = self.attn1(self.norm1(h))
        h = h + self.dropout(z)

        z, _ = self.attn2(self.norm2(h), mem=skip)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm3(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1)

class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 2

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=False, linear=False):
        super(ResBlock, self).__init__()

        self.activate = nn.SELU()
        self.norm = FrameNorm(features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=(2,1) if downsample else 1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=(2,1) if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + h

        return x

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, num_blocks=1, linear=False):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, linear=linear)

    def forward(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, num_blocks=1, linear=False):
        super(FrameDecoder, self).__init__()

        self.dropout = nn.Dropout2d(0.1)
        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = ResBlock(in_channels + out_channels, out_channels, features, linear=linear)

    def forward(self, x, skip):
        x = torch.cat((self.upsample(x), self.dropout(skip)), dim=1)
        x = self.body(x)

        return x

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads)

        self.norm = FrameNorm(features)

        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding)))
        
        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding)))
        
        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding)))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features)
        
    def __call__(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.norm(self.q_proj(x)).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.norm(self.k_proj(x if mem is None else mem)).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.norm(self.v_proj(x if mem is None else mem)).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)
        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)
        x = self.o_proj(a)

        return x, qk