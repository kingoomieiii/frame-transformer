import torch
from torch import nn
import torch.nn.functional as F
from lib import spec_utils
from lib.frame_transformer_common import FrameConv, FrameTransformerDecoder, FrameTransformerEncoder

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

        self.out = FrameConv(channels * (depth) + num_transformer_blocks, 1, kernel_size=3, padding=1, norm=False, activate=None)

    def forward(self, masked, unmasked):
        x = torch.cat((masked, unmasked), dim=1)

        x = x[:, :, :self.max_bin]

        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            for transformer_encoder in self.transformer_encoder[i]:
                x = torch.cat((x, transformer_encoder(x)), dim=1)

        return self.out(x)
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU, cropsize=1024):
        super(FrameEncoder, self).__init__()

        self.conv1 = FrameConv(in_channels, out_channels, kernel_size, 1, padding, activate=activ, cropsize=cropsize)
        self.conv2 = FrameConv(out_channels, out_channels, kernel_size, stride, padding, activate=activ, cropsize=cropsize)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        return h

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activ=nn.LeakyReLU, norm=True, dropout=False, cropsize=1024):
        super(FrameDecoder, self).__init__()

        self.conv1 = FrameConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, activate=activ, norm=norm, cropsize=cropsize)
        self.conv2 = FrameConv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, activate=activ, norm=norm, cropsize=cropsize)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=(skip.shape[2],skip.shape[3]), mode='bilinear', align_corners=True)
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)

        h = self.conv1(x)
        h = self.conv2(h)

        if self.dropout is not None:
            h = self.dropout(h)

        return h