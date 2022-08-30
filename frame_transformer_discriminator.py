import torch
import torch.nn as nn
from frame_transformer import FrameEncoder, FrameTransformerEncoder

class FrameTransformerDiscriminator(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=2):
        super(FrameTransformerDiscriminator, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        # output to shape [B,1,1,W]
        self.encoder = nn.Sequential(
            FrameEncoder(in_channels, channels, self.max_bin, downsample=False, expansion=expansion),
            FrameTransformerEncoder(channels, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion),
            FrameEncoder(channels, channels * 2, self.max_bin, expansion=expansion),
            FrameTransformerEncoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion),
            FrameEncoder(channels * 2, channels * 4, self.max_bin // 2, expansion=expansion),
            FrameTransformerEncoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion),
            FrameEncoder(channels * 4, channels * 6, self.max_bin // 4, expansion=expansion),
            FrameTransformerEncoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion),
            FrameEncoder(channels * 6, channels * 8, self.max_bin // 8, expansion=expansion),
            FrameTransformerEncoder(channels * 8, self.max_bin // 16, num_heads=num_heads, dropout=dropout, expansion=expansion),
            FrameEncoder(channels * 8, channels * 10, self.max_bin // 16, expansion=expansion),
            FrameTransformerEncoder(channels * 10, self.max_bin // 32, num_heads=num_heads, dropout=dropout, expansion=expansion),
            FrameEncoder(channels * 10, channels * 12, self.max_bin // 32, expansion=expansion),
            FrameEncoder(channels * 10, channels * 12, self.max_bin // 32, expansion=expansion),
            FrameEncoder(channels * 12, channels * 14, self.max_bin // 64, expansion=expansion),
            FrameEncoder(channels * 14, channels * 16, self.max_bin // 128, expansion=expansion), 
            FrameEncoder(channels * 16, channels * 18, self.max_bin // 256, expansion=expansion),
            FrameEncoder(channels * 18, 1, self.max_bin // 512, expansion=expansion))

    def __call__(self, x):
        return self.encoder(x)