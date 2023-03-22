import math
import torch
from torch import nn
import torch.nn.functional as F

from libft2.convolutional_multihead_attention import ConvolutionalMultiheadAttention
from libft2.multichannel_multihead_attention import MultichannelMultiheadAttention
from libft2.multichannel_layernorm import MultichannelLayerNorm
from libft2.multichannel_linear import MultichannelLinear
from libft2.frame_conv import FrameConv
from libft2.convolutional_embedding import ConvolutionalEmbedding
from libft2.res_block import ResBlock
from libft2.squared_relu import SquaredReLU

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_bridge_layers=4, num_attention_maps=4):
        super(FrameTransformer, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.positional_embedding = ConvolutionalEmbedding(in_channels, self.max_bin)

        self.enc1 = FrameEncoder(in_channels + 1, channels, self.max_bin, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc2 = FrameEncoder(channels + num_attention_maps, channels * 2, self.max_bin)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc3 = FrameEncoder(channels * 2 + num_attention_maps, channels * 4, self.max_bin // 2)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc4 = FrameEncoder(channels * 4 + num_attention_maps, channels * 6, self.max_bin // 4)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc5 = FrameEncoder(channels * 6 + num_attention_maps, channels * 8, self.max_bin // 8)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc6 = FrameEncoder(channels * 8 + num_attention_maps, channels * 10, self.max_bin // 16)
        self.enc6_transformer = FrameTransformerEncoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc7 = FrameEncoder(channels * 10 + num_attention_maps, channels * 12, self.max_bin // 32)
        self.enc7_transformer = FrameTransformerEncoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc8 = FrameEncoder(channels * 12 + num_attention_maps, channels * 14, self.max_bin // 64)
        self.enc8_transformer = FrameTransformerEncoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc9 = FrameEncoder(channels * 14 + num_attention_maps, channels * 16, self.max_bin // 128)
        self.enc9_transformer = nn.Sequential(*[ConvolutionalTransformerEncoder(channels * 16, dropout=dropout, expansion=4, num_heads=num_heads) for _ in range(num_bridge_layers)])

        self.dec8 = FrameDecoder(channels * 16 + num_attention_maps, channels * 14, self.max_bin // 128)
        self.dec8_transformer = FrameTransformerDecoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads, has_prev_skip=False)

        self.dec7 = FrameDecoder(channels * 14 + num_attention_maps + num_attention_maps, channels * 12, self.max_bin // 64)
        self.dec7_transformer = FrameTransformerDecoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec6 = FrameDecoder(channels * 12 + num_attention_maps + num_attention_maps, channels * 10, self.max_bin // 32)
        self.dec6_transformer = FrameTransformerDecoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec5 = FrameDecoder(channels * 10 + num_attention_maps + num_attention_maps, channels * 8, self.max_bin // 16)
        self.dec5_transformer = FrameTransformerDecoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec4 = FrameDecoder(channels * 8 + num_attention_maps + num_attention_maps, channels * 6, self.max_bin // 8)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec3 = FrameDecoder(channels * 6 + num_attention_maps + num_attention_maps, channels * 4, self.max_bin // 4)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec2 = FrameDecoder(channels * 4 + num_attention_maps + num_attention_maps, channels * 2, self.max_bin // 2)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec1 = FrameDecoder(channels * 2 + num_attention_maps + num_attention_maps, channels * 1, self.max_bin // 1)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.out = nn.Conv2d(channels + num_attention_maps, out_channels, 1)
        
    def forward(self, x):
        x = torch.cat((x, self.positional_embedding(x)), dim=1)

        e1 = self.enc1(x)
        e1, a1, qk1 = self.enc1_transformer(e1)

        e2 = self.enc2(e1)
        e2, a2, qk2 = self.enc2_transformer(e2, prev_qk=qk1)

        e3 = self.enc3(e2)
        e3, a3, qk3 = self.enc3_transformer(e3, prev_qk=qk2)

        e4 = self.enc4(e3)
        e4, a4, qk4 = self.enc4_transformer(e4, prev_qk=qk3)

        e5 = self.enc5(e4)
        e5, a5, qk5 = self.enc5_transformer(e5, prev_qk=qk4)

        e6 = self.enc6(e5)
        e6, a6, qk6 = self.enc6_transformer(e6, prev_qk=qk5)

        e7 = self.enc7(e6)
        e7, a7, qk7 = self.enc7_transformer(e7, prev_qk=qk6)

        e8 = self.enc8(e7)
        e8, a8, qk8 = self.enc8_transformer(e8, prev_qk=qk7)

        e9 = self.enc9_transformer(self.enc9(e8))

        h = self.dec8(e9, e8)
        h, pqk1, pqk2 = self.dec8_transformer(h, a8, prev_qk1=None, prev_qk2=None, skip_qk=qk8)

        h = self.dec7(h, e7)
        h, pqk1, pqk2 = self.dec7_transformer(h, a7, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=qk7)

        h = self.dec6(h, e6)
        h, pqk1, pqk2 = self.dec6_transformer(h, a6, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=qk6)

        h = self.dec5(h, e5)
        h, pqk1, pqk2 = self.dec5_transformer(h, a5, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=qk5)
    
        h = self.dec4(h, e4)
        h, pqk1, pqk2 = self.dec4_transformer(h, a4, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=qk4)
        
        h = self.dec3(h, e3)
        h, pqk1, pqk2 = self.dec3_transformer(h, a3, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=qk3)

        h = self.dec2(h, e2)
        h, pqk1, pqk2 = self.dec2_transformer(h, a2, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=qk2)

        h = self.dec1(h, e1)
        h, pqk1, pqk2 = self.dec1_transformer(h, a1, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=qk1)

        out = self.out(h)

        return out
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.embed = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

        self.norm1 = MultichannelLayerNorm(out_channels, features)
        self.glu = nn.Sequential(
            MultichannelLinear(out_channels, out_channels, features, features * 2),
            nn.GLU(dim=-2))
        
        self.norm2 = MultichannelLayerNorm(out_channels, features)
        self.conv1a = MultichannelLinear(out_channels, out_channels, features, features)
        self.conv1b = FrameConv(out_channels, out_channels, features, features, kernel_size=3, padding=1, groups=out_channels)

        self.norm3 = MultichannelLayerNorm(out_channels, features)
        self.conv2 = FrameConv(out_channels, out_channels, features, features, kernel_size=9, padding=4, groups=out_channels)

        self.norm4 = MultichannelLayerNorm(out_channels, features)
        self.attn = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm5 = MultichannelLayerNorm(out_channels, features)
        self.conv3 = MultichannelLinear(out_channels, out_channels, features, expansion)
        self.conv4 = MultichannelLinear(out_channels, out_channels, expansion, features)
        
    def forward(self, x, prev_qk=None):
        h = self.embed(x)

        z = self.glu(self.norm1(h))
        h = h + self.dropout(z)

        z = self.norm2(h)
        za = self.activate(self.conv1a(z))
        zb = self.activate(self.conv1b(z))
        h = h + self.dropout(za) + self.dropout(zb)

        z = self.conv2(self.norm3(h))
        h = h + self.dropout(z)

        z, prev_qk = self.attn(self.norm4(h), prev_qk=prev_qk)
        h = h + self.dropout(z)

        z = self.conv4(self.activate(self.conv3(self.norm5(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), h, prev_qk
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, has_prev_skip=True):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.gate1 = None if not has_prev_skip else nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            SquaredReLU(),
            nn.Conv3d(out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid())

        self.gate2 = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            SquaredReLU(),
            nn.Conv3d(out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid())

        self.embed = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

        self.norm1 = MultichannelLayerNorm(out_channels, features)
        self.attn1a = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)
        self.attn1b = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm2 = MultichannelLayerNorm(out_channels, features)
        self.conv1a = FrameConv(out_channels, out_channels, features, features, kernel_size=11, padding=5, groups=out_channels)
        self.conv1b = FrameConv(out_channels, out_channels, features, features, kernel_size=7, padding=3, groups=out_channels)
        self.norm3 = MultichannelLayerNorm(out_channels, features)
        self.conv2 = FrameConv(out_channels, out_channels, features, features, kernel_size=7, padding=3, groups=out_channels)

        self.norm4 = MultichannelLayerNorm(out_channels, features)
        self.attn2 = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm5 = MultichannelLayerNorm(out_channels, features)
        self.attn3 = MultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm6 = MultichannelLayerNorm(out_channels, features)
        self.conv3 = MultichannelLinear(out_channels, out_channels, features, expansion)
        self.conv4 = MultichannelLinear(out_channels, out_channels, expansion, features)
        
    def forward(self, x, skip, prev_qk1=None, prev_qk2=None, skip_qk=None):
        h = self.embed(x)

        z = self.norm1(h)
        za, prev_qk1 = self.attn1a(z, prev_qk=prev_qk1)

        if self.gate1 is not None:
            g = self.gate1(torch.cat((prev_qk2, skip_qk), dim=1))
            prev_qk2 = g * prev_qk2 + (1 - g) * skip_qk
        elif skip_qk is not None:
            prev_qk2 = skip_qk

        zb, prev_qk2 = self.attn1b(z, mem=skip, prev_qk=prev_qk2)
        h = h + self.dropout(za) + self.dropout(zb)

        z = self.norm2(h)
        za = self.activate(self.conv1a(z))
        zb = self.conv1b(z)
        z = self.conv2(self.norm3(za + zb))
        h = h + self.dropout(z)

        z, prev_qk1 = self.attn2(self.norm4(h), prev_qk=prev_qk1)
        h = h + self.dropout(z)

        g = self.gate2(torch.cat((prev_qk2, skip_qk), dim=1))
        prev_qk2 = g * prev_qk2 + (1 - g) * skip_qk
        z, prev_qk2 = self.attn3(self.norm5(h), mem=skip, prev_qk=prev_qk2)
        h = h + self.dropout(z)

        z = self.conv4(self.activate(self.conv3(self.norm6(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), prev_qk1, prev_qk2
        
class ConvolutionalTransformerEncoder(nn.Module):
    def __init__(self, channels, dropout=0.1, expansion=4, num_heads=8):
        super(ConvolutionalTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout2d(dropout)

        self.norm1 = nn.InstanceNorm2d(channels)
        self.glu = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, padding=0),
            nn.GLU(dim=1))
        
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv1a = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1b = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0))

        self.norm3 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=9, padding=4, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0))

        self.norm4 = nn.InstanceNorm2d(channels)
        self.attn = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=3, padding=1)

        self.norm5 = nn.InstanceNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        z = self.glu(self.norm1(x))
        h = x + self.dropout(z)

        z = self.norm2(h)
        za = self.activate(self.conv1a(z))
        zb = self.activate(self.conv1b(z))
        h = h + self.dropout(za) + self.dropout(zb)

        z = self.conv2(self.norm3(h))
        h = h + self.dropout(z)

        z = self.attn(self.norm4(h))
        h = x + self.dropout(z)

        z = self.conv4(self.activate(self.conv3(self.norm5(h))))
        h = h + self.dropout(z)

        return h

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample)

    def forward(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = ResBlock(in_channels + out_channels, out_channels, features)

    def forward(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())