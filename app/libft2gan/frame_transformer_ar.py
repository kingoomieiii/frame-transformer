import math
import torch
from torch import nn
import torch.nn.functional as F

from libft2gan.rotary_embedding_torch import RotaryEmbedding
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.convolutional_embedding import ConvolutionalEmbedding
from libft2gan.squared_relu import SquaredReLU

def generate_square_subsequent_mask(sz: int):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class FrameTransformerGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_attention_maps=1, quantizer_dims=256):
        super(FrameTransformerGenerator, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.mask = None

        self.positional_embedding = ConvolutionalEmbedding(in_channels, self.max_bin)

        self.src_enc1 = FrameEncoder(in_channels + 1, channels, self.max_bin, downsample=False)
        self.tgt_enc1 = FrameEncoder(in_channels + 1, channels, self.max_bin, downsample=False, causal=True)
        self.src_enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc1_transformer = FrameTransformerEncoder(channels, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc2 = FrameEncoder(channels, channels * 2, self.max_bin)
        self.tgt_enc2 = FrameEncoder(channels, channels * 2, self.max_bin, causal=True)
        self.src_enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc2_transformer = FrameTransformerEncoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2)
        self.tgt_enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2, causal=True)
        self.src_enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc3_transformer = FrameTransformerEncoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4)
        self.tgt_enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4, causal=True)
        self.src_enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc4_transformer = FrameTransformerEncoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8)
        self.tgt_enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8, causal=True)
        self.src_enc5_transformer = FrameTransformerEncoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc5_transformer = FrameTransformerEncoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc6 = FrameEncoder(channels * 8, channels * 10, self.max_bin // 16)
        self.tgt_enc6 = FrameEncoder(channels * 8, channels * 10, self.max_bin // 16, causal=True)
        self.src_enc6_transformer = FrameTransformerEncoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc6_transformer = FrameTransformerEncoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc7 = FrameEncoder(channels * 10, channels * 12, self.max_bin // 32)
        self.tgt_enc7 = FrameEncoder(channels * 10, channels * 12, self.max_bin // 32, causal=True)
        self.src_enc7_transformer = FrameTransformerEncoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc7_transformer = FrameTransformerEncoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc8 = FrameEncoder(channels * 12, channels * 14, self.max_bin // 64)
        self.tgt_enc8 = FrameEncoder(channels * 12, channels * 14, self.max_bin // 64, causal=True)
        self.src_enc8_transformer = FrameTransformerEncoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads, mem_causal=True)
        self.tgt_enc8_transformer = FrameTransformerEncoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem_causal=False)

        self.src_enc9 = FrameEncoder(channels * 14, channels * 16, self.max_bin // 128)
        self.tgt_enc9 = FrameEncoder(channels * 14, channels * 16, self.max_bin // 128, causal=True)
        self.src_enc9_transformer = FrameTransformerEncoder(channels * 16, num_attention_maps, self.max_bin // 256, dropout=dropout, expansion=expansion, num_heads=num_heads // 2, mem_causal=True)
        self.tgt_enc9_transformer = FrameTransformerEncoder(channels * 16, num_attention_maps, self.max_bin // 256, dropout=dropout, expansion=expansion, num_heads=num_heads // 2, causal=True, mem_causal=False)

        self.tgt_dec8 = FrameDecoder(channels * 32, channels * 14, self.max_bin // 128, causal=True)
        self.tgt_dec8_transformer = FrameTransformerDecoder(channels * 14, num_attention_maps, self.max_bin // 128, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)

        self.tgt_dec7 = FrameDecoder(channels * 14, channels * 12, self.max_bin // 64, causal=True)
        self.tgt_dec7_transformer = FrameTransformerDecoder(channels * 12, num_attention_maps, self.max_bin // 64, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)

        self.tgt_dec6 = FrameDecoder(channels * 12, channels * 10, self.max_bin // 32, causal=True)
        self.tgt_dec6_transformer = FrameTransformerDecoder(channels * 10, num_attention_maps, self.max_bin // 32, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)

        self.tgt_dec5 = FrameDecoder(channels * 10, channels * 8, self.max_bin // 16, causal=True)
        self.tgt_dec5_transformer = FrameTransformerDecoder(channels * 8, num_attention_maps, self.max_bin // 16, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)

        self.tgt_dec4 = FrameDecoder(channels * 8, channels * 6, self.max_bin // 8, causal=True)
        self.tgt_dec4_transformer = FrameTransformerDecoder(channels * 6, num_attention_maps, self.max_bin // 8, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)
        
        self.tgt_dec3 = FrameDecoder(channels * 6, channels * 4, self.max_bin // 4, causal=True)
        self.tgt_dec3_transformer = FrameTransformerDecoder(channels * 4, num_attention_maps, self.max_bin // 4, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)
        
        self.tgt_dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2, causal=True)
        self.tgt_dec2_transformer = FrameTransformerDecoder(channels * 2, num_attention_maps, self.max_bin // 2, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)
        
        self.tgt_dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin // 1, causal=True)
        self.tgt_dec1_transformer = FrameTransformerDecoder(channels * 1, num_attention_maps, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads, causal=True, mem1_causal=False, mem2_causal=True)
        
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            SquaredReLU(),
            nn.Conv2d(channels * 2, quantizer_dims * 2, 1))
        
    def forward(self, src, tgt):
        if self.mask is None or self.mask.shape[1] != src.shape[3]:
            self.mask = generate_square_subsequent_mask(src.shape[3]).to(src.device)

        pos = self.positional_embedding(src)

        src_h = torch.cat((src, pos), dim=1)
        tgt_h = torch.cat((tgt, pos), dim=1)

        se1 = self.src_enc1(src_h)
        te1 = self.tgt_enc1(tgt_h)
        se1, sqk1a, sqk1b = self.src_enc1_transformer(se1, te1, mask2=self.mask)
        te1, tqk1a, tqk1b = self.tgt_enc1_transformer(te1, se1, mask1=self.mask)
        
        se2 = self.src_enc2(se1)
        te2 = self.tgt_enc2(te1)
        se2, sqk2a, sqk2b = self.src_enc2_transformer(se2, te2, prev_qk1=sqk1a, prev_qk2=sqk1b, mask2=self.mask)
        te2, tqk2a, tqk2b = self.tgt_enc2_transformer(te2, se2, prev_qk1=tqk1a, prev_qk2=tqk1b, mask1=self.mask)
        
        se3 = self.src_enc3(se2)
        te3 = self.tgt_enc3(te2)
        se3, sqk3a, sqk3b = self.src_enc3_transformer(se3, te3, prev_qk1=sqk2a, prev_qk2=sqk2b, mask2=self.mask)
        te3, tqk3a, tqk3b = self.tgt_enc3_transformer(te3, se3, prev_qk1=tqk2a, prev_qk2=tqk2b, mask1=self.mask)

        se4 = self.src_enc4(se3)
        te4 = self.tgt_enc4(te3)
        se4, sqk4a, sqk4b = self.src_enc4_transformer(se4, te4, prev_qk1=sqk3a, prev_qk2=sqk3b, mask2=self.mask)
        te4, tqk4a, tqk4b = self.tgt_enc4_transformer(te4, se4, prev_qk1=tqk3a, prev_qk2=tqk3b, mask1=self.mask)

        se5 = self.src_enc5(se4)
        te5 = self.tgt_enc5(te4)
        se5, sqk5a, sqk5b = self.src_enc5_transformer(se5, te5, prev_qk1=sqk4a, prev_qk2=sqk4b, mask2=self.mask)
        te5, tqk5a, tqk5b = self.tgt_enc5_transformer(te5, se5, prev_qk1=tqk4a, prev_qk2=tqk4b, mask1=self.mask)

        se6 = self.src_enc6(se5)
        te6 = self.tgt_enc6(te5)
        se6, sqk6a, sqk6b = self.src_enc6_transformer(se6, te6, prev_qk1=sqk5a, prev_qk2=sqk5b, mask2=self.mask)
        te6, tqk6a, tqk6b = self.tgt_enc6_transformer(te6, se6, prev_qk1=tqk5a, prev_qk2=tqk5b, mask1=self.mask)

        se7 = self.src_enc7(se6)
        te7 = self.tgt_enc7(te6)
        se7, sqk7a, sqk7b = self.src_enc7_transformer(se7, te7, prev_qk1=sqk6a, prev_qk2=sqk6b, mask2=self.mask)
        te7, tqk7a, tqk7b = self.tgt_enc7_transformer(te7, se7, prev_qk1=tqk6a, prev_qk2=tqk6b, mask1=self.mask)

        se8 = self.src_enc8(se7)
        te8 = self.tgt_enc8(te7)
        se8, _, _ = self.src_enc8_transformer(se8, te8, prev_qk1=sqk7a, prev_qk2=sqk7b, mask2=self.mask)
        te8, tqk8a, _ = self.tgt_enc8_transformer(te8, se8, prev_qk1=tqk7a, prev_qk2=tqk7b, mask1=self.mask)

        se9 = self.src_enc9(se8)
        te9 = self.tgt_enc9(te8)
        se9, _, _ = self.src_enc9_transformer(se9, te9, mask2=self.mask)
        te9, _, _ = self.tgt_enc9_transformer(te9, se9, mask1=self.mask)

        h = self.tgt_dec8(torch.cat((se9, te9), dim=1), torch.cat((se8, te8), dim=1))
        h, pqk1, pqk2 = self.tgt_dec8_transformer(h, skip1=se8, skip2=te8, prev_qk1=None, prev_qk2=None, skip_qk=tqk8a, mask1=self.mask)

        h = self.tgt_dec7(h, torch.cat((se7, te7), dim=1))
        h, pqk1, pqk2 = self.tgt_dec7_transformer(h, skip1=se7, skip2=te7, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=tqk7a, mask1=self.mask)

        h = self.tgt_dec6(h, torch.cat((se6, te6), dim=1))
        h, pqk1, pqk2 = self.tgt_dec6_transformer(h, skip1=se6, skip2=te6, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=tqk6a, mask1=self.mask)

        h = self.tgt_dec5(h, torch.cat((se5, te5), dim=1))
        h, pqk1, pqk2 = self.tgt_dec5_transformer(h, skip1=se5, skip2=te5, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=tqk5a, mask1=self.mask)

        h = self.tgt_dec4(h, torch.cat((se4, te4), dim=1))
        h, pqk1, pqk2 = self.tgt_dec4_transformer(h, skip1=se4, skip2=te4, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=tqk4a, mask1=self.mask)

        h = self.tgt_dec3(h, torch.cat((se3, te3), dim=1))
        h, pqk1, pqk2 = self.tgt_dec3_transformer(h, skip1=se3, skip2=te3, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=tqk3a, mask1=self.mask)

        h = self.tgt_dec2(h, torch.cat((se2, te2), dim=1))
        h, pqk1, pqk2 = self.tgt_dec2_transformer(h, skip1=se2, skip2=te2, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=tqk2a, mask1=self.mask)

        h = self.tgt_dec1(h, torch.cat((se1, te1), dim=1))
        h, pqk1, pqk2 = self.tgt_dec1_transformer(h, skip1=se1, skip2=te1, prev_qk1=pqk1, prev_qk2=pqk2, skip_qk=tqk1a, mask1=self.mask)
             
        out = self.out(h)

        return torch.cat((
            F.softmax(out[:, :(out.shape[1] // 2), :, :], dim=1).unsqueeze(-1),
            F.softmax(out[:, (out.shape[1] // 2):, :, :], dim=1).unsqueeze(-1)
        ), dim=-1).permute(0,1,4,2,3)  # B,C,H,W,num_levels

class MultichannelMultiheadAttention2(nn.Module): 
    def __init__(self, channels, attention_maps, num_heads, features, kernel_size=3, padding=1, causal=False, mem_channels=None, mem_features=None, mem_causal=None):
        super().__init__()

        self.attention_maps = attention_maps
        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads)

        self.q_proj = nn.Sequential(
            (nn.Conv2d if not causal else CausalConv2d)(channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features, features))
        
        self.k_proj = nn.Sequential(
            (nn.Conv2d if not (causal if mem_causal is None else mem_causal) else CausalConv2d)(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features))
        
        self.v_proj = nn.Sequential(
            (nn.Conv2d if not (causal if mem_causal is None else mem_causal) else CausalConv2d)(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features))
        
        self.o_linear = MultichannelLinear(attention_maps, attention_maps, features, features, depthwise=True)
        self.o_proj = (nn.Conv2d if not causal else CausalConv2d)(channels + attention_maps, channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x, mem=None, prev_qk=None, mask=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        if mask is not None:
            qk = qk + mask

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,self.attention_maps,w,-1).transpose(2,3)
        x = self.o_proj(torch.cat((x, self.o_linear(a)), dim=1))

        return x, qk
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, causal=False, mem_causal=False, mem_channels=None, mem_features=None):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1, causal=causal)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.attn2 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1, causal=causal, mem_causal=mem_causal, mem_channels=mem_channels, mem_features=mem_features)

        self.norm3 = MultichannelLayerNorm(channels, features)
        self.conv1 = (nn.Conv2d if not causal else CausalConv2d)(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = (nn.Conv2d if not causal else CausalConv2d)(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, skip, prev_qk1=None, prev_qk2=None, mask1=None, mask2=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1, mask=mask1)
        h = x + self.dropout(z)

        z, prev_qk2 = self.attn2(self.norm2(z), mem=skip, prev_qk=prev_qk2, mask=mask2)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm3(h))))
        h = h + self.dropout(z)

        return h, prev_qk1, prev_qk2
        
class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, causal=False, mem1_causal=False, mem2_causal=False):
        super(FrameTransformerDecoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1, causal=causal)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.attn2 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1, causal=causal, mem_causal=mem1_causal)

        self.norm3 = MultichannelLayerNorm(channels, features)
        self.attn3 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=3, padding=1, causal=causal, mem_causal=mem2_causal)

        self.norm4 = MultichannelLayerNorm(channels, features)
        self.conv1 = (nn.Conv2d if not causal else CausalConv2d)(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = (nn.Conv2d if not causal else CausalConv2d)(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, skip1, skip2, prev_qk1=None, prev_qk2=None, skip_qk=None, mask1=None, mask2=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1, mask=mask1)
        h = x + self.dropout(z)

        z, _ = self.attn2(self.norm2(z), mem=skip1, prev_qk=skip_qk, mask=mask2)
        h = h + self.dropout(z)

        z, prev_qk2 = self.attn2(self.norm3(z), mem=skip2, prev_qk=prev_qk2, mask=mask1)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm4(h))))
        h = h + self.dropout(z)

        return h, prev_qk1, prev_qk2

class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, groups=1, dilation=1, bias=True):
        super(CausalConv2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, (padding, kernel_size - 1), dilation, groups, bias)

    def forward(self, x):
        return self.conv(x)[:, :, :, :-(self.kernel_size - 1)]

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, kernel_size=3, padding=1, downsample=False, stride=(2,1), dropout=0, causal=False):
        super(ResBlock, self).__init__()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activate = nn.GELU()
        self.norm = MultichannelLayerNorm(in_channels, features)
        self.conv1 = (nn.Conv2d if not causal else CausalConv2d)(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = (nn.Conv2d if not causal else CausalConv2d)(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride if downsample else 1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + self.dropout(h)

        return x
    
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, stride=(2,1), causal=False):
        super(FrameEncoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, stride=stride, causal=causal)

    def forward(self, x):
        x = self.body(x)

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, dropout=0, causal=False):
        super(FrameDecoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.body = ResBlock(in_channels + (out_channels * 2), out_channels, features, dropout=dropout, causal=causal)

    def forward(self, x, skip):
        x = torch.cat((self.upsample(x), skip), dim=1)
        x = self.body(x)

        return x