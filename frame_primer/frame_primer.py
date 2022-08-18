import torch
from torch import nn
import torch.nn.functional as F
import math
from frame_primer.common import FrameDecoder, FrameEncoder, FramePrimerEncoder, FramePrimerDecoder

class FramePrimer2(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, feedforward_dim=2048, num_res_blocks=1, num_heads=[8, 8, 8, 8, 8, 4], expansion=16):
        super(FramePrimer2, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, n_fft=n_fft, downsamples=0, stride=1, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.enc1_primer = FramePrimerEncoder(num_heads=num_heads[0], n_fft=n_fft, downsamples=0, dropout=dropout, expansion=expansion)

        self.enc2 = FrameEncoder(channels, channels * 2, n_fft=n_fft, downsamples=0, stride=2, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.enc2_primer = FramePrimerEncoder(num_heads=num_heads[1], n_fft=n_fft, downsamples=1, dropout=dropout, expansion=expansion)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, n_fft=n_fft, downsamples=1, stride=2, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.enc3_primer = FramePrimerEncoder(num_heads=num_heads[2], n_fft=n_fft, downsamples=2, dropout=dropout, expansion=expansion)

        self.enc4 = FrameEncoder(channels * 4, channels * 8, n_fft=n_fft, downsamples=2, stride=2, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.enc4_primer = FramePrimerEncoder(num_heads=num_heads[3], n_fft=n_fft, downsamples=3, dropout=dropout, expansion=expansion)

        self.enc5 = FrameEncoder(channels * 8, channels * 16, n_fft=n_fft, downsamples=3, stride=2, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.enc5_primer = FramePrimerEncoder(num_heads=num_heads[4], n_fft=n_fft, downsamples=4, dropout=dropout, expansion=expansion)

        self.enc6 = FrameEncoder(channels * 16, channels * 32, n_fft=n_fft, downsamples=4, stride=2, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.enc6_primer = FramePrimerEncoder(num_heads=num_heads[5], n_fft=n_fft, downsamples=5, dropout=dropout, expansion=expansion)

        self.dec5 = FrameDecoder(channels * (32 + 16), channels * 16, n_fft=n_fft, downsamples=4, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.dec5_primer = FramePrimerDecoder(num_heads=num_heads[4], n_fft=n_fft, downsamples=4, dropout=dropout, expansion=expansion)

        self.dec4 = FrameDecoder(channels * (16 + 8), channels * 8, n_fft=n_fft, downsamples=3, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.dec4_primer = FramePrimerDecoder(num_heads=num_heads[3], n_fft=n_fft, downsamples=3, dropout=dropout, expansion=expansion)

        self.dec3 = FrameDecoder(channels * (8 + 4), channels * 4, n_fft=n_fft, downsamples=2, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.dec3_primer = FramePrimerDecoder(num_heads=num_heads[2], n_fft=n_fft, downsamples=2, dropout=dropout, expansion=expansion)

        self.dec2 = FrameDecoder(channels * (4 + 2), channels * 2, n_fft=n_fft, downsamples=1, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.dec2_primer = FramePrimerDecoder(num_heads=num_heads[1], n_fft=n_fft, downsamples=1, dropout=dropout, expansion=expansion)

        self.dec1 = FrameDecoder(channels * (2 + 1), channels * 1, n_fft=n_fft, downsamples=0, kernel_size=9, padding=4, num_res_blocks=num_res_blocks)
        self.dec1_primer = FramePrimerDecoder(num_heads=num_heads[0], n_fft=n_fft, downsamples=0, dropout=dropout, expansion=expansion)

        self.out = nn.Conv2d(channels, in_channels, kernel_size=1, padding=0)

    def __call__(self, x):
        e1 = self.enc1_primer(self.enc1(x))
        e2 = self.enc2_primer(self.enc2(e1))
        e3 = self.enc3_primer(self.enc3(e2))
        e4 = self.enc4_primer(self.enc4(e3))
        e5 = self.enc5_primer(self.enc5(e4))
        e6 = self.enc6_primer(self.enc6(e5))

        d5 = self.dec5_primer(self.dec5(e6, e5), e5)
        d4 = self.dec4_primer(self.dec4(d5, e4), e4)
        d3 = self.dec3_primer(self.dec3(d4, e3), e3)
        d2 = self.dec2_primer(self.dec2(d3, e2), e2)
        d1 = self.dec1_primer(self.dec1(d2, e1), e1)

        return self.out(d1)