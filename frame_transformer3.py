import math
import torch
from torch import nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, num_layers=12, expansion=4, num_heads=8, n_fft=2048):
        super().__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.encoder = FrameEncoder(n_fft)
        self.transformer = nn.Sequential(*[TransformerEncoder(n_fft // 2, expansion=expansion, num_heads=num_heads) for _ in range(num_layers)])
        self.decoder = FrameDecoder(n_fft)

    def __call__(self, x):
        b,c,h,w = x.shape
        x = x.reshape(b,c*h,w)

        e = self.encoder(x)
        t = self.transformer(e)
        d = self.decoder(t)
        return d

class SquaredReLU(nn.Module):
    def __call__(self, x):
        return torch.relu(x) ** 2

class FrameEncoder(nn.Module):
    def __init__(self, n_fft=2048):
        super().__init__()

        self.bins = n_fft // 2
        self.conv1 = nn.Conv1d(self.bins * 2, self.bins * 2, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(self.bins * 2, self.bins, kernel_size=3, padding=1, bias=False)
        self.activate = SquaredReLU()

    def __call__(self, x):
        x = self.conv2(self.activate(self.conv1(x)))

        return x

class FrameDecoder(nn.Module):
    def __init__(self, n_fft=2048):
        super().__init__()

        self.bins = n_fft // 2
        self.conv1 = nn.Conv1d(self.bins, self.bins * 2, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(self.bins * 2, self.bins * 2, kernel_size=3, padding=1, bias=False)
        self.activate = SquaredReLU()

    def __call__(self, x):
        b,h,w = x.shape
        x = self.conv2(self.activate(self.conv1(x)))
        x = x.reshape(b,2,h,w)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, features, expansion=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(features)
        self.attn = MultiheadAttention(features, num_heads)

        self.norm2 = nn.LayerNorm(features)
        self.linear1 = nn.Linear(features, features * expansion, bias=False)
        self.linear2 = nn.Linear(features * expansion, features, bias=False)

        self.activate = SquaredReLU()

    def __call__(self, x):
        h = self.norm1(x.transpose(1,2)).transpose(1,2)
        h = self.attn(h)
        x = x + self.dropout(h)

        h = self.norm2(x.transpose(1,2)).transpose(1,2)
        h = self.linear2(self.activate(self.linear1(h.transpose(1,2)))).transpose(1,2)
        x = x + self.dropout(h)

        return x

class MultiheadAttention(nn.Module):
    def __init__(self, features, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim = features // num_heads)
        
        self.q_proj = nn.Linear(features, features, bias=False)
        self.q_conv = nn.Conv1d(features, features, kernel_size=7, padding=3, bias=False, groups=features)

        self.k_proj = nn.Linear(features, features, bias=False)
        self.k_conv = nn.Conv1d(features, features, kernel_size=7, padding=3, bias=False, groups=features)

        self.v_proj = nn.Linear(features, features, bias=False)
        self.v_conv = nn.Conv1d(features, features, kernel_size=7, padding=3, bias=False, groups=features)

        self.out_proj = nn.Linear(features, features, bias=False)

    def __call__(self, x, mem=None):
        b,h,w = x.shape

        q = self.q_proj(x.transpose(1,2)).transpose(1,2)
        q = self.q_conv(x).transpose(1,2)

        k = self.k_proj(x.transpose(1,2)).transpose(1,2)
        k = self.k_conv(x).transpose(1,2)

        v = self.v_proj(x.transpose(1,2)).transpose(1,2)
        v = self.v_conv(v).transpose(1,2)

        q = self.rotary_embedding.rotate_queries_or_keys(q.reshape(b,w,self.num_heads,-1).permute(0,2,1,3))
        k = self.rotary_embedding.rotate_queries_or_keys(k.reshape(b,w,self.num_heads,-1).permute(0,2,1,3)).transpose(2,3)
        v = v.reshape(b,w,self.num_heads,-1).permute(0,2,1,3)

        with torch.cuda.amp.autocast_mode.autocast(enabled=False):
            qk = torch.matmul(q.float(),k.float()) / math.sqrt(h)
            a = torch.matmul(F.softmax(qk, dim=-1),v.float()).transpose(1,2).reshape(b,w,-1)

        x = self.out_proj(a).transpose(1,2)

        return x