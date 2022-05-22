import torch
from torch import nn
import torch.nn.functional as F
from lib import spec_utils
from lib.frame_transformer_common import FrameConv, FrameEncoder, FrameTransformerDecoder, FrameTransformerEncoder

class FrameTransformerDiscriminator(nn.Module):
    def __init__(self, in_channels=4, channels=4, depth=6, num_transformer_blocks=2, dropout=0.1, n_fft=2048, cropsize=1024, num_bands=8, feedforward_dim=2048, bias=True, pretraining=True):
        super(FrameTransformerDiscriminator, self).__init__()

        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.encoder = [FrameEncoder(in_channels, channels, kernel_size=3, padding=1, stride=1, cropsize=cropsize)]
        self.transformer_encoder = [nn.ModuleList([FrameTransformerEncoder(channels + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])]

        for i in range(depth - 1):
            self.encoder.append(FrameEncoder(channels * (i + 1) + num_transformer_blocks, channels * (i + 2), stride=2, cropsize=cropsize))
            self.transformer_encoder.append(nn.ModuleList([FrameTransformerEncoder(channels * (i + 2) + j, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for j in range(num_transformer_blocks)]))

        self.encoder = nn.ModuleList(self.encoder)
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        self.out = FrameConv(channels * depth + num_transformer_blocks, 1, kernel_size=3, padding=1, norm=False, activate=None)

    def forward(self, masked, unmasked):
        x = torch.cat((masked, unmasked), dim=1)

        x = x[:, :, :self.max_bin]

        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            for transformer_encoder in self.transformer_encoder[i]:
                x = torch.cat((x, transformer_encoder(x)), dim=1)

        return self.out(x)