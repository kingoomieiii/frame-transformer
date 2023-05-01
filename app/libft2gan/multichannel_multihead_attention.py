import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from libft2gan.rotary_embedding_torch import RotaryEmbedding
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.multichannel_layernorm import MultichannelLayerNorm

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1, dtype=torch.float):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads, dtype=dtype)

        self.q_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, dtype=dtype),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), dtype=dtype))
        
        self.k_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, dtype=dtype),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), dtype=dtype))
        
        self.v_proj = nn.Sequential(
            MultichannelLinear(channels, channels, features, features, dtype=dtype),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), padding=(0,padding), dtype=dtype))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features, dtype=dtype)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)
        x = self.o_proj(a)

        return x, qk

class MultichannelMultiheadAttention2(nn.Module):
    def __init__(self, channels, attention_maps, num_heads, features, kernel_size=3, padding=1, mem_channels=None, mem_features=None, mem_kernel_size=None, mem_padding=None):
        super().__init__()

        self.attention_maps = attention_maps
        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads)
        self.features = features

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features, features))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size if mem_kernel_size is None else mem_kernel_size, padding=padding if mem_padding is None else mem_padding),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size if mem_kernel_size is None else mem_kernel_size, padding=padding if mem_padding is None else mem_padding),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features))
        
        self.o_proj = MultichannelLinear(channels + attention_maps, channels, features, features)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        w2 = w if mem is None else mem.shape[3]
        
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w2,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w2,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,self.attention_maps,w,-1).transpose(2,3)
        x = self.o_proj(torch.cat((x, a), dim=1))

        return x, qk

class ComplexMultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, num_heads, features, kernel_size=3, padding=1):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads, dtype=torch.cfloat)

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=torch.cfloat),
            MultichannelLinear(channels, channels, features, features, dtype=torch.cfloat))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=torch.cfloat),
            MultichannelLinear(channels, channels, features, features, dtype=torch.cfloat))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dtype=torch.cfloat),
            MultichannelLinear(channels, channels, features, features, dtype=torch.cfloat))
        
        self.o_proj = MultichannelLinear(channels, channels, features, features, dtype=torch.cfloat)
        
    def forward(self, x, mem=None, prev_qkm=None, prev_qkp=None):
        b,c,h,w = x.shape
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,c,w,self.num_heads,-1).permute(0,1,3,2,4)

        qm, qp = torch.abs(q), torch.angle(q)
        km, kp = torch.abs(k), torch.angle(k)
        vm, vp = torch.abs(v), torch.angle(v)

        qkm = torch.matmul(qm,km) / math.sqrt(h)
        qkp = torch.matmul(qp,kp) / math.sqrt(h)

        if prev_qkm is not None:
            qkm = qkm + prev_qkm
            qkp = qkp + prev_qkp

        qkt = qkm * (torch.cos(qkp) + 1.j * torch.sin(qkp))
        scores = F.softmax(qkt.abs(), dim=-1)
        out_mag = torch.matmul(scores, vm)
        out_phase = torch.matmul(scores, vp)

        a = out_mag * (torch.cos(out_phase) + 1.j * torch.sin(out_phase))
        a = a.type_as(x).transpose(2,3).reshape(b,c,w,-1).transpose(2,3)
        x = self.o_proj(a)

        return x, qkm, qkp