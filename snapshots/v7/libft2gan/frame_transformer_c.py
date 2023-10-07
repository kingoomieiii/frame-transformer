import math
import torch
from torch import nn
import torch.nn.functional as F

from libft2gan.convolutional_multihead_attention import ComplexConvolutionalMultiheadAttention
from libft2gan.multichannel_multihead_attention import ComplexMultichannelMultiheadAttention
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.frame_conv import FrameConv
from libft2gan.convolutional_embedding import ConvolutionalEmbedding
from libft2gan.res_block import ResBlock
from libft2gan.squared_relu import SquaredReLU, Sigmoid, Upsample, Dropout, Dropout2d
from libft2gan.channel_norm import ChannelNorm

class FrameTransformerGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, latent_expansion=4, num_bridge_layers=0, num_attention_maps=1):
        super(FrameTransformerGenerator, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.positional_embedding = ConvolutionalEmbedding(in_channels, self.max_bin, dtype=torch.cfloat)

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
        self.enc9_transformer = nn.Sequential(*[ConvolutionalTransformerEncoder(channels * 16, dropout=dropout, expansion=latent_expansion, num_heads=num_heads, kernel_size=1, padding=0) for _ in range(num_bridge_layers)])

        self.dec8 = FrameDecoder(channels * 16 + num_attention_maps, channels * 14, self.max_bin // 128, dropout=0.5)
        self.dec8_transformer = FrameTransformerDecoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads, has_prev_skip=False)

        self.dec7 = FrameDecoder(channels * 14 + num_attention_maps + num_attention_maps, channels * 12, self.max_bin // 64, dropout=0.5)
        self.dec7_transformer = FrameTransformerDecoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec6 = FrameDecoder(channels * 12 + num_attention_maps + num_attention_maps, channels * 10, self.max_bin // 32, dropout=0.5)
        self.dec6_transformer = FrameTransformerDecoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec5 = FrameDecoder(channels * 10 + num_attention_maps + num_attention_maps, channels * 8, self.max_bin // 16, dropout=0.5)
        self.dec5_transformer = FrameTransformerDecoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec4 = FrameDecoder(channels * 8 + num_attention_maps + num_attention_maps, channels * 6, self.max_bin // 8)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec3 = FrameDecoder(channels * 6 + num_attention_maps + num_attention_maps, channels * 4, self.max_bin // 4)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec2 = FrameDecoder(channels * 4 + num_attention_maps + num_attention_maps, channels * 2, self.max_bin // 2)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec1 = FrameDecoder(channels * 2 + num_attention_maps + num_attention_maps, channels * 1, self.max_bin // 1)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.out = nn.Conv2d(channels + num_attention_maps, out_channels, 1, dtype=torch.cfloat)
        
    def forward(self, x):
        x = torch.cat((x, self.positional_embedding(x)), dim=1)

        e1 = self.enc1(x)
        e1, a1, qkm1, qkp1 = self.enc1_transformer(e1)

        e2 = self.enc2(e1)
        e2, a2, qkm2, qkp2 = self.enc2_transformer(e2, prev_qkm=qkm1, prev_qkp=qkp1)

        e3 = self.enc3(e2)
        e3, a3, qkm3, qkp3 = self.enc3_transformer(e3, prev_qkm=qkm2, prev_qkp=qkp2)

        e4 = self.enc4(e3)
        e4, a4, qkm4, qkp4 = self.enc4_transformer(e4, prev_qkm=qkm3, prev_qkp=qkp3)

        e5 = self.enc5(e4)
        e5, a5, qkm5, qkp5 = self.enc5_transformer(e5, prev_qkm=qkm4, prev_qkp=qkp4)

        e6 = self.enc6(e5)
        e6, a6, qkm6, qkp6 = self.enc6_transformer(e6, prev_qkm=qkm5, prev_qkp=qkp5)

        e7 = self.enc7(e6)
        e7, a7, qkm7, qkp7 = self.enc7_transformer(e7, prev_qkm=qkm6, prev_qkp=qkp6)

        e8 = self.enc8(e7)
        e8, a8, qkm8, qkp8 = self.enc8_transformer(e8, prev_qkm=qkm7, prev_qkp=qkp7)

        e9, pqkm, pqkp = self.enc9(e8), None, None
        for encoder in self.enc9_transformer:
            e9, pqkm, pqkp = encoder(e9, prev_qkm=pqkm, prev_qkp=pqkp)
            
        h = self.dec8(e9, e8)
        h, pqkm1, pqkp1 = self.dec8_transformer(h, a8, prev_qkm=None, prev_qkp=None, skip_qkm=qkm8, skip_qkp=qkp8)

        h = self.dec7(h, e7)
        h, pqkm2, pqkp2 = self.dec7_transformer(h, a7, prev_qkm=pqkm1, prev_qkp=pqkp1, skip_qkm=qkm7, skip_qkp=qkp7)

        h = self.dec6(h, e6)
        h, pqkm3, pqkp3 = self.dec6_transformer(h, a6, prev_qkm=pqkm2, prev_qkp=pqkp2, skip_qkm=qkm6, skip_qkp=qkp6)

        h = self.dec5(h, e5)
        h, pqkm4, pqkp4 = self.dec5_transformer(h, a5, prev_qkm=pqkm3, prev_qkp=pqkp3, skip_qkm=qkm5, skip_qkp=qkp5)
    
        h = self.dec4(h, e4)
        h, pqkm5, pqkp5 = self.dec4_transformer(h, a4, prev_qkm=pqkm4, prev_qkp=pqkp4, skip_qkm=qkm4, skip_qkp=qkp4)
        
        h = self.dec3(h, e3)
        h, pqkm6, pqkp6 = self.dec3_transformer(h, a3, prev_qkm=pqkm5, prev_qkp=pqkp5, skip_qkm=qkm3, skip_qkp=qkp3)

        h = self.dec2(h, e2)
        h, pqkm7, pqkp7 = self.dec2_transformer(h, a2, prev_qkm=pqkm6, prev_qkp=pqkp6, skip_qkm=qkm2, skip_qkp=qkp2)

        h = self.dec1(h, e1)
        h, pqkm8, pqkp8 = self.dec1_transformer(h, a1, prev_qkm=pqkm7, prev_qkp=pqkp7, skip_qkm=qkm1, skip_qkp=qkp1)

        out = self.out(h)

        return out#, vout
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = SquaredReLU(dtype=torch.cfloat)
        self.dropout = Dropout(dropout, dtype=torch.cfloat)

        self.embed = nn.Conv2d(channels, out_channels, 1, dtype=torch.cfloat) if channels != out_channels else nn.Identity()

        self.norm1 = MultichannelLayerNorm(out_channels, features, dtype=torch.cfloat)
        self.attn = ComplexMultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm2 = MultichannelLayerNorm(out_channels, features, dtype=torch.cfloat)
        self.conv1 = MultichannelLinear(out_channels, out_channels, features, expansion, depthwise=True, dtype=torch.cfloat)
        self.conv2 = MultichannelLinear(out_channels, out_channels, expansion, features, dtype=torch.cfloat)
        
    def forward(self, x, prev_qkm=None, prev_qkp=None):
        h = self.embed(x)

        z, prev_qkm, prev_qkp = self.attn(self.norm1(h), prev_qkm=prev_qkm, prev_qkp=prev_qkp)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), h, prev_qkm, prev_qkp
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, has_prev_skip=True):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = SquaredReLU(dtype=torch.cfloat)
        self.dropout = Dropout(dropout, dtype=torch.cfloat)

        self.gate1 = None if not has_prev_skip else nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            SquaredReLU(dtype=torch.cfloat),
            nn.Conv3d(out_channels, 1, kernel_size=3, padding=1, dtype=torch.cfloat),
            Sigmoid(dtype=torch.cfloat))

        self.embed = nn.Conv2d(channels, out_channels, 1, dtype=torch.cfloat) if channels != out_channels else nn.Identity()

        self.norm1 = MultichannelLayerNorm(out_channels, features, dtype=torch.cfloat)
        self.attn1 = ComplexMultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm2 = MultichannelLayerNorm(out_channels, features, dtype=torch.cfloat)
        self.attn2 = ComplexMultichannelMultiheadAttention(out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm3 = MultichannelLayerNorm(out_channels, features, dtype=torch.cfloat)
        self.conv1 = MultichannelLinear(out_channels, out_channels, features, expansion, depthwise=True, dtype=torch.cfloat)
        self.conv2 = MultichannelLinear(out_channels, out_channels, expansion, features, dtype=torch.cfloat)
        
    def forward(self, x, skip, prev_qkm=None, prev_qkp=None, skip_qkm=None, skip_qkp=None):
        h = self.embed(x)

        z = self.norm1(h)
        za, prev_qkm, prev_qkp = self.attn1(z, prev_qkm=prev_qkm, prev_qkp=prev_qkp)

        zb, _, _ = self.attn2(self.norm2(z), mem=skip, prev_qkm=skip_qkm, prev_qkp=skip_qkp)
        h = h + self.dropout(za) + self.dropout(zb)

        z = self.conv2(self.activate(self.conv1(self.norm3(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), prev_qkm, prev_qkp
        
class ConvolutionalTransformerEncoder(nn.Module):
    def __init__(self, channels, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1):
        super(ConvolutionalTransformerEncoder, self).__init__()

        self.activate = SquaredReLU(dtype=torch.cfloat)
        self.dropout = Dropout2d(dropout, dtype=torch.cfloat)

        self.norm1 = ChannelNorm(channels, dtype=torch.cfloat)
        self.attn = ComplexConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding)

        self.norm2 = ChannelNorm(channels, dtype=torch.cfloat)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1, dtype=torch.cfloat)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1, dtype=torch.cfloat)
        
    def forward(self, x, prev_qkm=None, prev_qkp=None):
        z, prev_qkm, prev_qkp = self.attn(self.norm1(x), prev_qkm=prev_qkm, prev_qkp=prev_qkp)
        h = x + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return h, prev_qkm, prev_qkp
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, stride=(2,1)):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, stride=stride, dtype=torch.cfloat)

    def forward(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, dropout=0):
        super(FrameDecoder, self).__init__()

        self.upsample = Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True, dtype=torch.cfloat)
        self.body = ResBlock(in_channels + out_channels, out_channels, features, dropout=dropout, dtype=torch.cfloat)

    def forward(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x