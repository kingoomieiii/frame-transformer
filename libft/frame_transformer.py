import torch
from torch import nn

from libft.multichannel_multihead_attention import MultichannelMultiheadAttention
from libft.multichannel_layernorm import MultichannelLayerNorm

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=4, num_layers=15, repeats=1, num_embeddings=1024):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.channels = channels
        self.out_channels = out_channels
        self.repeats = repeats
        
        self.embed = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.transformer = nn.ModuleList([FrameTransformerEncoder(channels, self.max_bin, dropout=dropout, expansion=expansion, num_heads=num_heads) for _ in range(num_layers)])
        self.out = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0, bias=False)

    def __call__(self, x):
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
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1, bias=False)
                
    def __call__(self, x, prev_qk=None):       
        z, prev_qk = self.attn(self.norm1(x), prev_qk=prev_qk)
        h = x + self.dropout(z)

        z = self.conv2(torch.relu(self.conv1(self.norm2(h))) ** 2)
        h = h + self.dropout(z)

        return h, prev_qk