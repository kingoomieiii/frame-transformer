import math
import torch
from torch import nn
import torch.nn.functional as F

from libft2gan.convolutional_multihead_attention import ConvolutionalMultiheadAttention
from libft2gan.multichannel_multihead_attention import MultichannelMultiheadAttention2 as MultichannelMultiheadAttention
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.frame_conv import FrameConv
from libft2gan.convolutional_embedding import ConvolutionalEmbedding
from libft2gan.res_block import ResBlock, LinearResBlock
from libft2gan.squared_relu import SquaredReLU
from libft2gan.channel_norm import ChannelNorm

class FrameTransformerGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, latent_expansion=4, num_bridge_layers=0, num_attention_maps=1, n_mels=256):
        super(FrameTransformerGenerator, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.positional_embedding = ConvolutionalEmbedding(in_channels, self.max_bin)

        self.enc1 = FrameEncoder(in_channels + 1, channels, self.max_bin, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps, self.max_bin, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps, self.max_bin // 2, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps, self.max_bin // 4, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps, self.max_bin // 8, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, num_attention_maps, self.max_bin // 16, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc6 = FrameEncoder(channels * 8, channels * 10, self.max_bin // 16)
        self.enc6_transformer = FrameTransformerEncoder(channels * 10, num_attention_maps, self.max_bin // 32, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc7 = FrameEncoder(channels * 10, channels * 12, self.max_bin // 32)
        self.enc7_transformer = FrameTransformerEncoder(channels * 12, num_attention_maps, self.max_bin // 64, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc8 = FrameEncoder(channels * 12, channels * 14, self.max_bin // 64)
        self.enc8_transformer = FrameTransformerEncoder(channels * 14, num_attention_maps, self.max_bin // 128, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc9 = FrameEncoder(channels * 14, channels * 16, self.max_bin // 128)
        self.enc9_transformer = None #nn.Sequential(*[ConvolutionalTransformerEncoder(channels * 16 * 4, dropout=dropout, expansion=latent_expansion, num_heads=num_heads, kernel_size=1, padding=0) for _ in range(num_bridge_layers)])

        self.dec8 = FrameDecoder(channels * 16, channels * 14, self.max_bin // 128, dropout=0.5)
        self.dec8_transformer = FrameTransformerDecoder(channels * 14, num_attention_maps, self.max_bin // 128, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads, has_prev_skip=False)

        self.dec7 = FrameDecoder(channels * 14, channels * 12, self.max_bin // 64, dropout=0.5)
        self.dec7_transformer = FrameTransformerDecoder(channels * 12, num_attention_maps, self.max_bin // 64, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec6 = FrameDecoder(channels * 12, channels * 10, self.max_bin // 32, dropout=0.5)
        self.dec6_transformer = FrameTransformerDecoder(channels * 10, num_attention_maps, self.max_bin // 32, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec5 = FrameDecoder(channels * 10, channels * 8, self.max_bin // 16, dropout=0.5)
        self.dec5_transformer = FrameTransformerDecoder(channels * 8, num_attention_maps, self.max_bin // 16, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec4 = FrameDecoder(channels * 8, channels * 6, self.max_bin // 8)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, num_attention_maps, self.max_bin // 8, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec3 = FrameDecoder(channels * 6, channels * 4, self.max_bin // 4)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, num_attention_maps, self.max_bin // 4, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, num_attention_maps, self.max_bin // 2, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin // 1)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, num_attention_maps, self.max_bin, mem1_channels=in_channels, mem1_features=self.max_bin, mem2_channels=in_channels, mem2_features=n_mels, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.out = nn.Conv2d(channels, out_channels, 1)
        
    def forward(self, x, xp, xm):
        xh = torch.cat((x, self.positional_embedding(x)), dim=1)

        e1 = self.enc1(xh)
        e1, a1, qk1a, qk1b, qk1c, qk1d = self.enc1_transformer(e1, cross1=x, cross2=xp, cross3=xm)

        e2 = self.enc2(e1)
        e2, a2, qk2a, qk2b, qk2c, qk2d = self.enc2_transformer(e2, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk1a, prev_qk2=qk1b, prev_qk3=qk1c, prev_qk4=qk1d)

        e3 = self.enc3(e2)
        e3, a3, qk3a, qk3b, qk3c, qk3d = self.enc3_transformer(e3, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk2a, prev_qk2=qk2b, prev_qk3=qk2c, prev_qk4=qk2d)

        e4 = self.enc4(e3)
        e4, a4, qk4a, qk4b, qk4c, qk4d = self.enc4_transformer(e4, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk3a, prev_qk2=qk3b, prev_qk3=qk3c, prev_qk4=qk3d)

        e5 = self.enc5(e4)
        e5, a5, qk5a, qk5b, qk5c, qk5d = self.enc5_transformer(e5, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk4a, prev_qk2=qk4b, prev_qk3=qk4c, prev_qk4=qk4d)

        e6 = self.enc6(e5)
        e6, a6, qk6a, qk6b, qk6c, qk6d = self.enc6_transformer(e6, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk5a, prev_qk2=qk5b, prev_qk3=qk5c, prev_qk4=qk5d)

        e7 = self.enc7(e6)
        e7, a7, qk7a, qk7b, qk7c, qk7d = self.enc7_transformer(e7, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk6a, prev_qk2=qk6b, prev_qk3=qk6c, prev_qk4=qk6d)

        e8 = self.enc8(e7)
        e8, a8, qk8a, qk8b, qk8c, qk8d = self.enc8_transformer(e8, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk7a, prev_qk2=qk7b, prev_qk3=qk7c, prev_qk4=qk7d)

        e9, pqk = self.enc9(e8), None

        # _b,_c,_h,_w = e9.shape
        # e9 = e9.reshape(_b, _c * _h, 1, _w)

        # for encoder in self.enc9_transformer:
        #     e9, pqk = encoder(e9, prev_qk=pqk)

        # e9 = e9.reshape(_b, _c, _h, _w)
            
        h = self.dec8(e9, e8)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec8_transformer(h, skip=a8, cross1=x, cross2=xp, cross3=xm, prev_qk1=qk8a, prev_qk2=qk8b, prev_qk3=qk8c, prev_qk4=qk8d, skip_qk=qk8a)

        h = self.dec7(h, e7)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec7_transformer(h, skip=a7, cross1=x, cross2=xp, cross3=xm, prev_qk1=pqk1, prev_qk2=pqk2, prev_qk3=pqk3, prev_qk4=pqk4, skip_qk=qk7a)

        h = self.dec6(h, e6)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec6_transformer(h, skip=a6, cross1=x, cross2=xp, cross3=xm, prev_qk1=pqk1, prev_qk2=pqk2, prev_qk3=pqk3, prev_qk4=pqk4, skip_qk=qk6a)

        h = self.dec5(h, e5)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec5_transformer(h, skip=a5, cross1=x, cross2=xp, cross3=xm, prev_qk1=pqk1, prev_qk2=pqk2, prev_qk3=pqk3, prev_qk4=pqk4, skip_qk=qk5a)
    
        h = self.dec4(h, e4)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec4_transformer(h, skip=a4, cross1=x, cross2=xp, cross3=xm, prev_qk1=pqk1, prev_qk2=pqk2, prev_qk3=pqk3, prev_qk4=pqk4, skip_qk=qk4a)
        
        h = self.dec3(h, e3)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec3_transformer(h, skip=a3, cross1=x, cross2=xp, cross3=xm, prev_qk1=pqk1, prev_qk2=pqk2, prev_qk3=pqk3, prev_qk4=pqk4, skip_qk=qk3a)

        h = self.dec2(h, e2)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec2_transformer(h, skip=a2, cross1=x, cross2=xp, cross3=xm, prev_qk1=pqk1, prev_qk2=pqk2, prev_qk3=pqk3, prev_qk4=pqk4, skip_qk=qk2a)

        h = self.dec1(h, e1)
        h, pqk1, pqk2, pqk3, pqk4 = self.dec1_transformer(h, skip=a1, cross1=x, cross2=xp, cross3=xm, prev_qk1=pqk1, prev_qk2=pqk2, prev_qk3=pqk3, prev_qk4=pqk4, skip_qk=qk1a)

        out = self.out(h)

        return out#, vout

class FrameTransformerDiscriminator(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, latent_expansion=4, num_bridge_layers=0, num_attention_maps=1):
        super(FrameTransformerDiscriminator, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.positional_embedding = ConvolutionalEmbedding(in_channels, self.max_bin)

        self.enc1 = FrameEncoder(in_channels + 1, channels, self.max_bin, downsample=False)
        self.enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc6 = FrameEncoder(channels * 8, channels * 10, self.max_bin // 16)
        self.enc6_transformer = FrameTransformerEncoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc7 = FrameEncoder(channels * 10, channels * 12, self.max_bin // 32)
        self.enc7_transformer = FrameTransformerEncoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc8 = FrameEncoder(channels * 12, channels * 14, self.max_bin // 64)
        self.enc8_transformer = FrameTransformerEncoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.enc9 = FrameEncoder(channels * 14, channels * 16, self.max_bin // 128)
        self.enc9_transformer = nn.Sequential(*[ConvolutionalTransformerEncoder(channels * 16 * 4, dropout=dropout, expansion=latent_expansion, num_heads=num_heads) for _ in range(num_bridge_layers)])

        self.out = nn.Sequential(
            nn.Conv2d(channels * 16, channels * 32, 1),
            nn.GELU(),
            nn.Conv2d(channels * 32, in_channels, 1))
        
    def from_generator(self, gen: FrameTransformerGenerator):
        self.positional_embedding = gen.positional_embedding

        self.enc1 = gen.enc1
        self.enc1_transformer = gen.enc1_transformer

        self.enc2 = gen.enc2
        self.enc2_transformer = gen.enc2_transformer

        self.enc3 = gen.enc3
        self.enc3_transformer = gen.enc3_transformer
        
        self.enc4 = gen.enc4
        self.enc4_transformer = gen.enc4_transformer
        
        self.enc5 = gen.enc5
        self.enc5_transformer = gen.enc5_transformer
        
        self.enc6 = gen.enc6
        self.enc6_transformer = gen.enc6_transformer
        
        self.enc7 = gen.enc7
        self.enc7_transformer = gen.enc7_transformer
        
        self.enc8 = gen.enc8
        self.enc8_transformer = gen.enc8_transformer
        
        self.enc9 = gen.enc9
        self.enc9_transformer = gen.enc9_transformer
        
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

        e9, pqk = self.enc9(e8), None

        _b,_c,_h,_w = e9.shape
        e9 = e9.reshape(_b, _c * _h, 1, _w)

        for encoder in self.enc9_transformer:
            e9, pqk = encoder(e9, prev_qk=pqk)
            
        out = self.out(e9.reshape(_b, _c, _h, _w))

        return out
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, mem1_channels, mem1_features, mem2_channels, mem2_features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.attn2 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=mem1_channels, mem_features=mem1_features)

        self.norm3 = MultichannelLayerNorm(channels, features)
        self.attn3 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=mem1_channels, mem_features=mem1_features)

        self.norm4 = MultichannelLayerNorm(channels, features)
        self.attn4 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=mem2_channels, mem_features=mem2_features)

        self.norm4 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, cross1, cross2, cross3, prev_qk1=None, prev_qk2=None, prev_qk3=None, prev_qk4=None):
        z, prev_qk1 = self.attn(self.norm1(x), prev_qk=prev_qk1)
        h = x + self.dropout(z)

        z, prev_qk2 = self.attn2(self.norm2(h), prev_qk=prev_qk2, mem=cross1)
        h = h + self.dropout(z)

        z, prev_qk3 = self.attn3(self.norm3(h), prev_qk=prev_qk3, mem=cross2)
        h = h + self.dropout(z)

        z, prev_qk4 = self.attn4(self.norm4(h), prev_qk=prev_qk4, mem=cross3)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm4(h))))
        h = h + self.dropout(z)

        return h, h, prev_qk1, prev_qk2, prev_qk3, prev_qk4
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, mem1_channels, mem1_features, mem2_channels, mem2_features, dropout=0.1, expansion=4, num_heads=8, has_prev_skip=True):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.attn2 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm3 = MultichannelLayerNorm(channels, features)
        self.attn3 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=mem1_channels, mem_features=mem1_features)

        self.norm4 = MultichannelLayerNorm(channels, features)
        self.attn4 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=mem1_channels, mem_features=mem1_features)

        self.norm5 = MultichannelLayerNorm(channels, features)
        self.attn5 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=3, padding=1, mem_channels=mem2_channels, mem_features=mem2_features)

        self.norm6 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, skip, cross1, cross2, cross3, prev_qk1=None, prev_qk2=None, prev_qk3=None, prev_qk4=None, skip_qk=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1)
        h = x + self.dropout(z)

        z, _ = self.attn2(self.norm2(h), mem=skip, prev_qk=skip_qk)
        h = h + self.dropout(z)

        z, prev_qk2 = self.attn3(self.norm3(h), mem=cross1, prev_qk=prev_qk2)
        h = h + self.dropout(z)

        z, prev_qk3 = self.attn4(self.norm4(h), mem=cross2, prev_qk=prev_qk3)
        h = h + self.dropout(z)

        z, prev_qk4 = self.attn5(self.norm5(h), mem=cross3, prev_qk=prev_qk4)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm5(h))))
        h = h + self.dropout(z)

        return h, prev_qk1, prev_qk2, prev_qk3, prev_qk4
        
class ConvolutionalTransformerEncoder(nn.Module):
    def __init__(self, channels, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1):
        super(ConvolutionalTransformerEncoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

        self.norm1 = ChannelNorm(channels)
        self.attn = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding)

        self.norm2 = ChannelNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, prev_qk=None):
        z, prev_qk = self.attn(self.norm1(x), prev_qk=prev_qk)
        h = x + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return h, prev_qk

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, stride=(2,1)):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, stride=stride)

    def forward(self, x):
        x = self.body(x)

        return x

class LinearFrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features):
        super(LinearFrameEncoder, self).__init__()

        self.body = LinearResBlock(in_channels, out_channels, in_features, out_features, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, dropout=0):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = ResBlock(in_channels + out_channels, out_channels, features, dropout=dropout)

    def forward(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x