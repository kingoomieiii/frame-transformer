import math
import torch
from torch import nn
import torch.nn.functional as F

from libft.multichannel_multihead_attention import MultichannelMultiheadAttention
from libft.multichannel_layernorm import MultichannelLayerNorm

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=15, repeats=1, num_embeddings=1024, use_local_embedding=True):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.channels = channels
        self.out_channels = out_channels
        self.repeats = repeats
        
        self.positional_embedding = PositionalEmbedding(in_channels, self.max_bin) if use_local_embedding else None
        self.embed = Conv2d(in_channels, channels, kernel_size=7, padding=3, bias=False)
        self.transformer = nn.ModuleList([FrameTransformerEncoder(channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads) for _ in range(num_layers)])
        self.out = Conv2d(channels, out_channels, kernel_size=1, padding=0, bias=False)

    def __call__(self, x):
        x = x + self.positional_embedding(x)

        h = self.embed(x)

        prev_qk = None
        for encoder in self.transformer:
            h, prev_qk = encoder(h, prev_qk=prev_qk)

        out = self.out(h)

        return out
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features, depthwise=True)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.conv1 = Conv2d(channels, channels * expansion, kernel_size=3, padding=1, bias=False)
        self.conv2 = Conv2d(channels * expansion, channels, kernel_size=3, padding=1, bias=False)
                
    def __call__(self, x, prev_qk=None):       
        z, prev_qk = self.attn(self.norm1(x), prev_qk=prev_qk)
        h = x + self.dropout(z)

        z = self.conv2(torch.relu(self.conv1(self.norm2(h))) ** 2)
        h = h + self.dropout(z)

        return h, prev_qk
       
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups=1, stride=1, bias=True, transpose=False):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, stride=stride, bias=bias) if not transpose else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, stride=stride, bias=bias)
        self.register_buffer('idx_dw', torch.arange(in_channels))
        self.embedding_dw = nn.Embedding(in_channels, in_channels)
        self.conv_dw = nn.Conv1d(in_channels, 1, kernel_size=9, padding=4)

    def forward(self, x):
        if self.embedding_dw is not None:
            x = x + self.conv_dw(self.embedding_dw(self.idx_dw).unsqueeze(0)).transpose(1,2).unsqueeze(-1)

        return self.conv(x)
    
class ResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, features, kernel_size=3, padding=1, downsample=False):
        super(ResBlock2, self).__init__()
        self.norm = MultichannelLayerNorm(in_channels, features)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=2 if downsample else 1, bias=False)
        self.identity = Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2 if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def __call__(self, x):
        h = self.conv2(torch.relu(self.conv1(self.norm(x))) ** 2)
        x = self.identity(x) + h

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, channels, features, max_seq_length=4096):
        super(PositionalEmbedding, self).__init__()

        self.extract1 = ResBlock2(channels, 1, features, kernel_size=11, padding=5)
        self.extract2 = ResBlock2(channels * 2, 1, features // 2, kernel_size=11, padding=5)
        self.extract3 = ResBlock2(channels * 4, 1, features // 4, kernel_size=11, padding=5)
        self.extract4 = ResBlock2(channels * 6, 1, features // 8, kernel_size=11, padding=5)
        self.extract5 = ResBlock2(channels * 8, 1, features // 16, kernel_size=11, padding=5)
        self.extract6 = ResBlock2(channels * 10, 1, features // 32, kernel_size=11, padding=5)
        self.extract7 = ResBlock2(channels * 12, 1, features // 64, kernel_size=11, padding=5)
        self.extract8 = ResBlock2(channels * 14, 1, features // 128, kernel_size=11, padding=5)
        self.extract9 = ResBlock2(channels * 16, 1, features // 256, kernel_size=11, padding=5)

        self.encoder1 = ResBlock2(channels, channels * 2, features, kernel_size=3, padding=1, downsample=True)
        self.encoder2 = ResBlock2(channels * 2, channels * 4, features // 2, kernel_size=3, padding=1, downsample=True)
        self.encoder3 = ResBlock2(channels * 4, channels * 6, features // 4, kernel_size=3, padding=1, downsample=True)
        self.encoder4 = ResBlock2(channels * 6, channels * 8, features // 8, kernel_size=3, padding=1, downsample=True)
        self.encoder5 = ResBlock2(channels * 8, channels * 10, features // 16, kernel_size=3, padding=1, downsample=True)
        self.encoder6 = ResBlock2(channels * 10, channels * 12, features // 32, kernel_size=3, padding=1, downsample=True)
        self.encoder7 = ResBlock2(channels * 12, channels * 14, features // 64, kernel_size=3, padding=1, downsample=True)
        self.encoder8 = ResBlock2(channels * 14, channels * 16, features // 128, kernel_size=3, padding=1, downsample=True)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, features, 2) * (-math.log(10000.0) / features))
        pe = torch.zeros(max_seq_length, 1, features)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.embedding = nn.Embedding(max_seq_length, features)
        self.register_buffer('indices', position.squeeze(-1))

        self.out = ResBlock2(11, 1, features, kernel_size=3, padding=1)

    def __call__(self, x):
        e1 = self.extract1(x)
        h = self.encoder1(x)

        e2 = F.interpolate(self.extract2(h), size=x.shape[2:], mode='bilinear', align_corners=True)
        h = self.encoder2(h)

        e3 = F.interpolate(self.extract3(h), size=x.shape[2:], mode='bilinear', align_corners=True)
        h = self.encoder3(h)

        e4 = F.interpolate(self.extract4(h), size=x.shape[2:], mode='bilinear', align_corners=True)
        h = self.encoder4(h)
        
        e5 = F.interpolate(self.extract5(h), size=x.shape[2:], mode='bilinear', align_corners=True)
        h = self.encoder5(h)
        
        e6 = F.interpolate(self.extract6(h), size=x.shape[2:], mode='bilinear', align_corners=True)
        h = self.encoder6(h)

        e7 = F.interpolate(self.extract7(h), size=x.shape[2:], mode='bilinear', align_corners=True)
        h = self.encoder7(h)

        e8 = F.interpolate(self.extract8(h), size=x.shape[2:], mode='bilinear', align_corners=True)
        h = self.encoder8(h)

        e9 = F.interpolate(self.extract9(h), size=x.shape[2:], mode='bilinear', align_corners=True)

        sinusoidal = self.pe[:x.shape[3]].transpose(0,1).unsqueeze(0).transpose(2,3).expand((x.shape[0], -1, -1, -1))
        embedding = self.embedding(self.indices[:x.shape[3]]).transpose(0,1).unsqueeze(0).unsqueeze(0).expand((x.shape[0], -1, -1, -1))

        return self.out(torch.cat((e1, e2, e3, e4, e5, e6, e7, e8, e9, sinusoidal, embedding), dim=1))