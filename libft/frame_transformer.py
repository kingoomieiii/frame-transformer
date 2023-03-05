import math
import torch
from torch import nn
import torch.nn.functional as F

from libft.multichannel_multihead_attention import MultichannelMultiheadAttention
from libft.multichannel_layernorm import MultichannelLayerNorm
from libft.multihead_channel_attention import MultiheadChannelAttention
from libft.positional_embedding import PositionalEmbedding

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=12):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.channels = channels
        self.out_channels = out_channels

        self.positional_embedding = PositionalEmbedding(in_channels, self.max_bin)

        self.norm = MultichannelLayerNorm(in_channels + 1, self.max_bin)
        self.conv1 = nn.Conv2d(in_channels + 1, channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.identity = nn.Conv2d(in_channels+1, channels, kernel_size=1, padding=0)
        
        self.transformer = nn.ModuleList([FrameTransformerEncoder(channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads) for _ in range(num_layers)])
        self.out = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0)

    def __call__(self, x):
        x = torch.cat((x, self.positional_embedding(x)), dim=1)
        x = self.identity(x) + self.conv2(torch.relu(self.conv1(self.norm(x))) ** 2)        

        prev_qk = None
        for encoder in self.transformer:
            x, prev_qk = encoder(x, prev_qk=prev_qk)

        x = self.out(x)

        return x
            
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, features, dropout=0.1, expansion=4, num_heads=8):
        super(FrameTransformerEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels, num_heads, features, kernel_size=9, padding=4)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
                
    def __call__(self, x, prev_qk=None):
        h, prev_qk = self.attn(self.norm1(x), prev_qk=prev_qk)
        x = x + self.dropout(h)

        h = self.conv2(torch.relu(self.conv1(self.norm2(x))) ** 2)
        x = x + self.dropout(h)

        return x, prev_qk