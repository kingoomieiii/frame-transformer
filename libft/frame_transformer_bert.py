import math
import torch
from torch import nn
import torch.nn.functional as F

from libft.multichannel_linear import MultichannelLinear
from libft.multichannel_layernorm import MultichannelLayerNorm
from libft.positional_embedding import PositionalEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=15, repeats=1, hidden_size=1024):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.features = hidden_size
        self.channels = channels
        self.out_channels = out_channels
        self.repeats = repeats
        
        self.positional_embedding = PositionalEmbedding(in_channels, self.max_bin)
        self.embed = MultichannelLinear(in_channels + 1, channels, self.max_bin, self.features, depthwise=True)
        self.transformer = nn.ModuleList([FrameTransformerEncoder(channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads) for _ in range(num_layers)]).requires_grad_(False)
        self.out = MultichannelLinear(channels, out_channels, self.features, self.max_bin, depthwise=True)

    def __call__(self, x):
        p = self.positional_embedding(x)
        h = self.embed(torch.cat((x, p), dim=1))

        prev_qk = None
        for encoder in self.transformer:
            h, prev_qk = encoder(h, prev_qk=prev_qk)

        out = self.out(h)

        return out

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features):
        super().__init__()

        self.num_heads = num_heads
        self.q_proj = MultichannelLinear(channels, channels, features, features, bias=True)
        self.k_proj = MultichannelLinear(channels, channels, features, features, bias=True)
        self.v_proj = MultichannelLinear(channels, channels, features, features, bias=True)
        self.o_proj = MultichannelLinear(channels, channels, features, features, bias=True)

    def __call__(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        k = self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,4,2)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q, k) / math.sqrt(h)

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)
        x = self.o_proj(a)

        return x, qk
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.attn = MultichannelMultiheadAttention(channels, num_heads, features)
        self.norm1 = MultichannelLayerNorm(channels, features)

        self.conv1 = MultichannelLinear(channels, channels, features, features * expansion, bias=True)
        self.conv2 = MultichannelLinear(channels, channels, features * expansion, features, bias=True)
        self.norm2 = MultichannelLayerNorm(channels, features)
                
    def __call__(self, x, prev_qk=None):       
        z, prev_qk = self.attn(x, prev_qk=prev_qk)
        h = self.norm1(x + self.dropout(z))

        z = self.conv2(self.gelu(self.conv1(h)))
        h = self.norm2(h + self.dropout(z))

        return h, prev_qk