import torch
from torch import nn
import torch.nn.functional as F
import math
from frame_primer.common import FrameDecoder, FrameEncoder, FramePrimerEncoder, FramePrimerDecoder, FramePrimerEncoder2

class FramePrimer(nn.Module):
    def __init__(self, in_channels=2, channels=2, depth=4, num_transformer_encoders=0, num_transformer_decoders=0, dropout=0.1, n_fft=2048, cropsize=2048, num_bands=8, feedforward_dim=2048, bias=True, scale_factor=1):
        super(FramePrimer, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.encoder = [FrameEncoder(in_channels, channels, kernel_size=3, padding=1, stride=1, cropsize=cropsize, downsamples=0)]
        self.decoder = [nn.Conv2d(channels + num_transformer_decoders, in_channels, kernel_size=1, padding=0)]
        self.transformer_encoder = [nn.ModuleList([FramePrimerEncoder(channels + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_encoders)])]
        self.transformer_decoder = [nn.ModuleList([FramePrimerDecoder(channels + i, channels + num_transformer_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_decoders)])]

        for i in range(depth - 1):
            self.encoder.append(FrameEncoder(channels * (scale_factor * i + 1) + num_transformer_encoders, channels * (scale_factor * i + 2), stride=2, cropsize=cropsize, downsamples=i))
            self.decoder.append(FrameDecoder(channels * (scale_factor * i + 2) + num_transformer_decoders + (num_transformer_encoders if i == depth - 2 else 0) + channels * (scale_factor * i + 1) + num_transformer_encoders, channels * (scale_factor * i + 1), cropsize=cropsize, downsamples=i))
            self.transformer_encoder.append(nn.ModuleList([FramePrimerEncoder(channels * (i + 2) + j, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for j in range(num_transformer_encoders)]))
            self.transformer_decoder.append(nn.ModuleList([FramePrimerDecoder(channels * (i + 2) + j + (num_transformer_encoders if i == depth - 2 else 0), channels * (i + 2) + num_transformer_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for j in range(num_transformer_decoders)]))

        self.encoder = nn.ModuleList(self.encoder)
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        self.decoder = nn.ModuleList(self.decoder[::-1])
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder[::-1])

    def forward(self, x):
        mem = []
        prev_qk = None
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            for transformer_encoder in self.transformer_encoder[i]:
                h, prev_qk = transformer_encoder(x, prev_qk=prev_qk)
                x = torch.cat((x, h), dim=1)

            mem.append(x)

        mem = mem[::-1]
        prev_qk1, prev_qk2 = None, None
        for i, decoder in enumerate(self.decoder):
            for transformer_decoder in self.transformer_decoder[i]:
                h, prev_qk1, prev_qk2 = transformer_decoder(x, mem=mem[i], prev_qk1=prev_qk1, prev_qk2=prev_qk2)
                x = torch.cat((x, h), dim=1)

            x = decoder(x, mem[i+1]) if i < len(self.decoder) - 1 else decoder(x)

        return torch.sigmoid(x)

class FramePrimer2(nn.Module):
    def __init__(self, in_channels=2, channels=2, depth=4, scale_factor=1, num_transformer_blocks=0, feedforward_expansion=4, dropout=0.1, n_fft=2048, cropsize=2048, num_bands=16, bias=True, num_res_blocks=1):
        super(FramePrimer2, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.encoder = [FrameEncoder(in_channels, channels * scale_factor, kernel_size=3, padding=1, stride=1, cropsize=cropsize, downsamples=0, num_res_blocks=num_res_blocks)]
        self.decoder = [FrameDecoder(channels * scale_factor, in_channels, kernel_size=3, padding=1, upsample=False)]

        for i in range(depth - 1):
            self.encoder.append(FrameEncoder(channels * scale_factor * (i + 1), channels * scale_factor * (i + 2), stride=2, cropsize=cropsize, downsamples=i, num_res_blocks=num_res_blocks))
            self.decoder.append(FrameDecoder(channels * scale_factor * (i + 2) + channels * scale_factor * (i + 1), channels * scale_factor * (i + 1), cropsize=cropsize, downsamples=i, num_res_blocks=num_res_blocks))

        self.transformer = nn.ModuleList([FramePrimerEncoder2(channels * scale_factor * ((depth - 2) + 2), num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=depth-1, bias=bias, dropout=dropout, expansion=feedforward_expansion) for i in range(num_transformer_blocks)])

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder[::-1])

    def forward(self, x):
        mem = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            mem.append(x)

        prev_qk = None
        for module in self.transformer:
            x, prev_qk = module(x, prev_qk=prev_qk)

        mem = mem[::-1]
        for i, decoder in enumerate(self.decoder):
            x = decoder(x, mem[i+1]) if i < len(self.decoder) - 1 else decoder(x)

        return x
