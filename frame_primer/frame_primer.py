import torch
from torch import nn
import torch.nn.functional as F
import math
from frame_primer.common import FrameDecoder, FrameEncoder, FramePrimerEncoder, FramePrimerDecoder, FrameTransformerDecoder, FrameTransformerEncoder

class FramePrimer(nn.Module):
    def __init__(self, in_channels=2, channels=2, depth=4, num_transformer_encoders=0, num_transformer_decoders=0, dropout=0.1, n_fft=2048, cropsize=2048, num_bands=8, feedforward_dim=2048, bias=True, scale_factor=1, num_res_blocks=1, column_kernel=True):
        super(FramePrimer, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.encoder = [FrameEncoder(in_channels, channels * scale_factor, kernel_size=3, padding=1, stride=1, downsamples=0, num_res_blocks=num_res_blocks, column_kernel=column_kernel)]
        self.decoder = [FrameDecoder(channels * scale_factor + num_transformer_decoders + in_channels, channels, kernel_size=1, padding=0, downsamples=0, num_res_blocks=num_res_blocks, upsample=False, column_kernel=column_kernel)]
        self.transformer_encoder = [nn.ModuleList([FramePrimerEncoder(channels * scale_factor + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_encoders)])]
        self.transformer_decoder = [nn.ModuleList([FramePrimerDecoder(channels * scale_factor + i, channels * scale_factor + num_transformer_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_decoders)])]

        for i in range(depth - 1):
            self.encoder.append(FrameEncoder(channels * scale_factor * (i + 1) + num_transformer_encoders, channels * scale_factor * (i + 2), stride=2, downsamples=i, num_res_blocks=num_res_blocks, column_kernel=column_kernel))
            self.decoder.append(FrameDecoder(channels * scale_factor * (i + 2) + num_transformer_decoders + (num_transformer_encoders if i == depth - 2 else 0) + channels * scale_factor * (i + 1) + num_transformer_encoders, channels * scale_factor * (i + 1), downsamples=i, num_res_blocks=num_res_blocks, column_kernel=column_kernel))
            self.transformer_encoder.append(nn.ModuleList([FramePrimerEncoder(channels * scale_factor * (i + 2) + j, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for j in range(num_transformer_encoders)]))
            self.transformer_decoder.append(nn.ModuleList([FramePrimerDecoder(channels * scale_factor * (i + 2) + j + (num_transformer_encoders if i == depth - 2 else 0), scale_factor * channels * (i + 2) + num_transformer_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for j in range(num_transformer_decoders)]))

        self.encoder = nn.ModuleList(self.encoder)
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        self.decoder = nn.ModuleList(self.decoder[::-1])
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder[::-1])

        self.out = nn.Conv2d(channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x

        skips = []
        prev_qk = None
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            for transformer_encoder in self.transformer_encoder[i]:
                h, prev_qk = transformer_encoder(x, prev_qk=prev_qk)
                x = torch.cat((x, h), dim=1)

            skips.append(x)

        skips = skips[::-1]
        prev_qk1, prev_qk2 = None, None
        for i, decoder in enumerate(self.decoder):
            for transformer_decoder in self.transformer_decoder[i]:
                h, prev_qk1, prev_qk2 = transformer_decoder(x, skip=skips[i], prev_qk1=prev_qk1, prev_qk2=prev_qk2)
                x = torch.cat((x, h), dim=1)

            x = decoder(x, skips[i+1]) if i < len(self.decoder) - 1 else decoder(x, identity)

        x = self.out(x)

        return x

class FramePrimer2(nn.Module):
    def __init__(self, in_channels=2, channels=2, depth=4, num_transformer_encoders=0, num_transformer_decoders=0, dropout=0.1, n_fft=2048, cropsize=2048, num_bands=8, feedforward_dim=2048, bias=True, scale_factor=1, num_res_blocks=1, column_kernel=True):
        super(FramePrimer2, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.channels = channels

        self.enc1 = FrameEncoder(in_channels, channels, n_fft=n_fft, downsamples=0, stride=1, column_kernel=False)
        self.enc1_primer = FramePrimerEncoder(channels, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=0, dropout=dropout, bottleneck=1, residual_attention=False)

        self.enc2 = FrameEncoder(channels + 1, channels * 2, n_fft=n_fft, downsamples=0, stride=2, column_kernel=False)
        self.enc2_primer = FramePrimerEncoder(channels * 2, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=1, dropout=dropout, bottleneck=3, residual_attention=False)

        self.enc3 = FrameEncoder(channels * 2 + 3, channels * 4, n_fft=n_fft, downsamples=1, stride=2, column_kernel=False)
        self.enc3_primer = FramePrimerEncoder(channels * 4, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=2, dropout=dropout, bottleneck=6, residual_attention=False)

        self.enc4 = FrameEncoder(channels * 4 + 6, channels * 6, n_fft=n_fft, downsamples=2, stride=2, column_kernel=False)
        self.enc4_primer = FramePrimerEncoder(channels * 6, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=3, dropout=dropout, bottleneck=9, residual_attention=False)

        self.enc5 = FrameEncoder(channels * 6 + 9, channels * 8, n_fft=n_fft, downsamples=3, stride=2, column_kernel=False)
        self.enc5_primer = FramePrimerEncoder(channels * 8, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=4, dropout=dropout, bottleneck=12, residual_attention=False)

        self.enc6 = FrameEncoder(channels * 8 + 12, channels * 10, n_fft=n_fft, downsamples=4, stride=2, column_kernel=False)
        self.enc6_primer = FramePrimerEncoder(channels * 10, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=5, dropout=dropout, bottleneck=15, residual_attention=False)

        self.dec5 = FrameDecoder(channels * (10 + 8) + 15 + 12, channels * 8, n_fft=n_fft, downsamples=4, column_kernel=False)
        self.dec5_primer = FramePrimerDecoder(channels * 8, channels * 8 + 12, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=4, dropout=dropout, bottleneck=12, residual_attention=False)

        self.dec4 = FrameDecoder(channels * (8 + 6) + 12 + 9, channels * 6, n_fft=n_fft, downsamples=3, column_kernel=False)
        self.dec4_primer = FramePrimerDecoder(channels * 6, channels * 6 + 9, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=3, dropout=dropout, bottleneck=9, residual_attention=False)

        self.dec3 = FrameDecoder(channels * (6 + 4) + 9 + 6, channels * 4, n_fft=n_fft, downsamples=2, column_kernel=False)
        self.dec3_primer = FramePrimerDecoder(channels * 4, channels * 4 + 6, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=2, dropout=dropout, bottleneck=6, residual_attention=False)

        self.dec2 = FrameDecoder(channels * (4 + 2) + 6 + 3, channels * 2, n_fft=n_fft, downsamples=1, column_kernel=False)
        self.dec2_primer = FramePrimerDecoder(channels * 2, channels * 2 + 3, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=1, dropout=dropout, bottleneck=3, residual_attention=False)

        self.dec1 = FrameDecoder(channels * (2 + 1) + 3 + 1, channels * 1, n_fft=n_fft, downsamples=0, column_kernel=False)
        self.dec1_primer = FramePrimerDecoder(channels * 1, channels * 1 + 1, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=0, dropout=dropout, bottleneck=1, residual_attention=False)

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