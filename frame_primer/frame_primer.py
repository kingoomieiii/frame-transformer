import torch
from torch import nn
import torch.nn.functional as F
import math
from frame_primer.common import FrameDecoder, FrameEncoder, FramePrimerEncoder, FramePrimerDecoder

class FramePrimer2(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, feedforward_dim=2048, num_res_blocks=1, num_bands=[16, 16, 16, 8, 4, 2], bottlenecks=[1, 2, 4, 8, 12, 14]):
        super(FramePrimer2, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.channels = channels

        self.enc1 = FrameEncoder(in_channels, channels, n_fft=n_fft, downsamples=0, stride=1, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.enc1_primer = FramePrimerEncoder(channels, num_bands=num_bands[0], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=0, dropout=dropout, bottleneck=bottlenecks[0])

        self.enc2 = FrameEncoder(channels + bottlenecks[0], channels * 2, n_fft=n_fft, downsamples=0, stride=2, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.enc2_primer = FramePrimerEncoder(channels * 2, num_bands=num_bands[1], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=1, dropout=dropout, bottleneck=bottlenecks[1])

        self.enc3 = FrameEncoder(channels * 2 + bottlenecks[1], channels * 4, n_fft=n_fft, downsamples=1, stride=2, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.enc3_primer = FramePrimerEncoder(channels * 4, num_bands=num_bands[2], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=2, dropout=dropout, bottleneck=bottlenecks[2])

        self.enc4 = FrameEncoder(channels * 4 + bottlenecks[2], channels * 6, n_fft=n_fft, downsamples=2, stride=2, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.enc4_primer = FramePrimerEncoder(channels * 6, num_bands=num_bands[3], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=3, dropout=dropout, bottleneck=bottlenecks[3])

        self.enc5 = FrameEncoder(channels * 6 + bottlenecks[3], channels * 8, n_fft=n_fft, downsamples=3, stride=2, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.enc5_primer = FramePrimerEncoder(channels * 8, num_bands=num_bands[4], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=4, dropout=dropout, bottleneck=bottlenecks[4])

        self.enc6 = FrameEncoder(channels * 8 + bottlenecks[4], channels * 10, n_fft=n_fft, downsamples=4, stride=2, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.enc6_primer = FramePrimerEncoder(channels * 10, num_bands=num_bands[5], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=5, dropout=dropout, bottleneck=bottlenecks[5])

        self.dec5 = FrameDecoder(channels * (10 + 8) + bottlenecks[5] + bottlenecks[4], channels * 8, n_fft=n_fft, downsamples=4, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.dec5_primer = FramePrimerDecoder(channels * 8, channels * 8 + bottlenecks[4], num_bands=num_bands[4], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=4, dropout=dropout, bottleneck=bottlenecks[4])

        self.dec4 = FrameDecoder(channels * (8 + 6) + bottlenecks[4] + bottlenecks[3], channels * 6, n_fft=n_fft, downsamples=3, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.dec4_primer = FramePrimerDecoder(channels * 6, channels * 6 + bottlenecks[3], num_bands=num_bands[3], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=3, dropout=dropout, bottleneck=bottlenecks[3])

        self.dec3 = FrameDecoder(channels * (6 + 4) + bottlenecks[3] + bottlenecks[2], channels * 4, n_fft=n_fft, downsamples=2, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.dec3_primer = FramePrimerDecoder(channels * 4, channels * 4 + bottlenecks[2], num_bands=num_bands[2], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=2, dropout=dropout, bottleneck=bottlenecks[2])

        self.dec2 = FrameDecoder(channels * (4 + 2) + bottlenecks[2] + bottlenecks[1], channels * 2, n_fft=n_fft, downsamples=1, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.dec2_primer = FramePrimerDecoder(channels * 2, channels * 2 + bottlenecks[1], num_bands=num_bands[1], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=1, dropout=dropout, bottleneck=bottlenecks[1])

        self.dec1 = FrameDecoder(channels * (2 + 1) + bottlenecks[1] + bottlenecks[0], channels * 1, n_fft=n_fft, downsamples=0, column_kernel=False, kernel_size=3, padding=1, num_res_blocks=num_res_blocks)
        self.dec1_primer = FramePrimerDecoder(channels * 1, channels * 1 + bottlenecks[0], num_bands=num_bands[0], n_fft=n_fft, feedforward_dim=feedforward_dim, downsamples=0, dropout=dropout, bottleneck=bottlenecks[0])

        self.out = nn.Conv2d(channels + 1, in_channels, kernel_size=1, padding=0)

    def __call__(self, x):
        e1 = self.enc1(x)
        e1 = torch.cat((e1, self.enc1_primer(e1)), dim=1)

        e2 = self.enc2(e1)
        e2 = torch.cat((e2, self.enc2_primer(e2)), dim=1)

        e3 = self.enc3(e2)
        e3 = torch.cat((e3, self.enc3_primer(e3)), dim=1)

        e4 = self.enc4(e3)
        e4 = torch.cat((e4, self.enc4_primer(e4)), dim=1)

        e5 = self.enc5(e4)
        e5 = torch.cat((e5, self.enc5_primer(e5)), dim=1)

        e6 = self.enc6(e5)
        e6 = torch.cat((e6, self.enc6_primer(e6)), dim=1)

        h = self.dec5(e6, e5)
        h = torch.cat((h, self.dec5_primer(h, skip=e5)), dim=1)

        h = self.dec4(h, e4)
        h = torch.cat((h, self.dec4_primer(h, skip=e4)), dim=1)

        h = self.dec3(h, e3)
        h = torch.cat((h, self.dec3_primer(h, skip=e3)), dim=1)

        h = self.dec2(h, e2)
        h = torch.cat((h, self.dec2_primer(h, skip=e2)), dim=1)

        h = self.dec1(h, e1)
        h = torch.cat((h, self.dec1_primer(h, skip=e1)), dim=1)

        return self.out(h)