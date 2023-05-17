import math
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as A

from libft2gan.rotary_embedding_torch import RotaryEmbedding
from libft2gan.convolutional_multihead_attention import ConvolutionalMultiheadAttention
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.convolutional_embedding import ConvolutionalEmbedding
from libft2gan.res_block import ResBlock2 as ResBlock, ResBlock1d, SquaredReLU
from libft2gan.channel_norm import ChannelNorm
from libft2gan.tempogram import Tempogram
from libft2gan.baseline_phase_difference import BasebandPhaseDifference, InverseBasebandPhaseDifference, phase_derivative, magnitude_derivative

class FrameTransformerGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, hop_length=1024, sr=44100, num_heads=4, expansion=4, latent_expansion=4, num_bridge_layers=9, num_attention_maps=1, quantizer_levels=128):
        super(FrameTransformerGenerator, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.positional_embedding = ConvolutionalEmbedding(in_channels * 1, self.max_bin)

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
        self.enc9_transformer = FrameTransformerEncoder(channels * 16, num_attention_maps, self.max_bin // 256, dropout=dropout, expansion=expansion, num_heads=num_heads // 2)

        self.dec8 = FrameDecoder(channels * 16, channels * 14, self.max_bin // 128)
        self.dec8_transformer = FrameTransformerDecoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads, has_prev_skip=False)

        self.dec7 = FrameDecoder(channels * 14, channels * 12, self.max_bin // 64)
        self.dec7_transformer = FrameTransformerDecoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec6 = FrameDecoder(channels * 12, channels * 10, self.max_bin // 32)
        self.dec6_transformer = FrameTransformerDecoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec5 = FrameDecoder(channels * 10, channels * 8, self.max_bin // 16)
        self.dec5_transformer = FrameTransformerDecoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads)

        self.dec4 = FrameDecoder(channels * 8, channels * 6, self.max_bin // 8)
        self.dec4_transformer = FrameTransformerDecoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec3 = FrameDecoder(channels * 6, channels * 4, self.max_bin // 4)
        self.dec3_transformer = FrameTransformerDecoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2)
        self.dec2_transformer = FrameTransformerDecoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin // 1)
        self.dec1_transformer = FrameTransformerDecoder(channels * 1, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads)
        
        self.out = nn.Sequential(
            MultichannelLinear(channels, channels * 2, self.max_bin, self.max_bin * 2),
            nn.GELU(),
            MultichannelLinear(channels * 2, quantizer_levels * 2, self.max_bin * 2, self.max_bin))
        
        self.out_norm = ChannelNorm(quantizer_levels)
        
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

        e9 = self.enc9(e8)
        e9, a9, qk9 = self.enc9_transformer(e9)
            
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

        L = F.softmax(self.out_norm(out[:, :(out.shape[1] // 2), :, :]), dim=1).unsqueeze(-1)
        R = F.softmax(self.out_norm(out[:, (out.shape[1] // 2):, :, :]), dim=1).unsqueeze(-1)

        LMI = torch.min(L)
        LMA = torch.max(L)
        RMI = torch.min(R)
        RMA = torch.max(R)
        LSU = torch.mean(torch.sum(L, dim=1))
        RSU = torch.mean(torch.sum(R, dim=1))

        return torch.cat((
            L,
            R
        ), dim=-1).permute(0,1,4,2,3), LMI, LMA, RMI, RMA, LSU, RSU
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, prev_qk=None):
        z, prev_qk = self.attn(x, prev_qk=prev_qk)
        h = self.norm1(x + self.dropout(z))

        z = self.conv2(self.activate(self.conv1(h)))
        h = self.norm2(h + self.dropout(z))

        return h, h, prev_qk
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, has_prev_skip=True):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.attn2 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1)

        self.norm3 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, skip, prev_qk1=None, prev_qk2=None, skip_qk=None):
        z, prev_qk1 = self.attn1(x, prev_qk=prev_qk1)
        h = self.norm1(x + self.dropout(z))

        z, prev_qk2 = self.attn2(z, mem=skip, prev_qk=skip_qk)
        h = self.norm2(h + self.dropout(z))

        z = self.conv2(self.activate(self.conv1(h)))
        h = self.norm3(h + self.dropout(z))

        return h, prev_qk1, prev_qk2

class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, stride=(2,1)):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, stride=stride)

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

class MultichannelMultiheadAttention2(nn.Module):
    def __init__(self, channels, attention_maps, num_heads, features, kernel_size=3, padding=1, dtype=torch.float):
        super().__init__()

        self.attention_maps = attention_maps
        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads, dtype=dtype)

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features, features, dtype=dtype))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features, features, dtype=dtype))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features, features, dtype=dtype))
        
        self.o_linear = MultichannelLinear(attention_maps, attention_maps, features, features, depthwise=True)
        self.o_proj = nn.Conv2d(channels + attention_maps, channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,self.attention_maps,w,-1).transpose(2,3)
        x = self.o_proj(torch.cat((x, self.o_linear(a)), dim=1))

        return x, qk
