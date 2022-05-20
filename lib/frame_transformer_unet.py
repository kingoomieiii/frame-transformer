import torch
from torch import nn
import torch.nn.functional as F
from lib import spec_utils
from lib.frame_transformer_common import FrameConv, FrameTransformerDecoder, FrameTransformerEncoder

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, num_stages=6, num_transformer_blocks=2, num_bridge_encoders=8, channel_heads=1, feature_heads=1, dropout=0.3, out=False, activate=nn.PReLU, n_fft=2048, cropsize=1024, num_bands=8, feedforward_dim=2048, bias=True):
        super(FrameTransformer, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.num_stages = num_stages
        self.encoder = [FrameEncoder(in_channels, channels, kernel_size=3, padding=1, stride=1)]
        self.decoder = [FrameDecoder(channels + num_transformer_blocks, channels, kernel_size=1, padding=0)]
        self.transformer_encoder = [nn.ModuleList([FrameTransformerEncoder(channels + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_transformer_blocks)])]
        self.transformer_decoder = [nn.ModuleList([FrameTransformerDecoder(channels + i, channels + num_transformer_blocks, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_transformer_blocks)])]

        for i in range(num_stages - 1):
            self.encoder.append(FrameEncoder(channels * (i + 1) + num_transformer_blocks, channels * (i + 2), stride=2))
            self.decoder.append(FrameDecoder(channels * (2 * i + 3) + 2 * num_transformer_blocks + (num_transformer_blocks if i == num_stages - 2 else 0), channels * (i + 1)))
            self.transformer_encoder.append(nn.ModuleList([FrameTransformerEncoder(channels * (i + 2) + j, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias) for j in range(num_transformer_blocks)]))
            self.transformer_decoder.append(nn.ModuleList([FrameTransformerDecoder(channels * (i + 2) + j + (num_transformer_blocks if i == num_stages - 2 else 0), channels * (i + 2) + num_transformer_blocks, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias) for j in range(num_transformer_blocks)]))

        self.encoder = nn.ModuleList(self.encoder)
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        
        self.decoder = nn.ModuleList(self.decoder[::-1])
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder[::-1])

        self.out = nn.Linear(channels, in_channels, bias=False)        
        self.is_next = nn.Sequential(
            nn.Linear(channels * (i + 2) + num_transformer_blocks, 2, bias=bias),
            nn.LogSoftmax(dim=-1))

    def forward(self, x):
        x = x[:, :, :self.max_bin]

        skips = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            for transformer_encoder in self.transformer_encoder[i]:
                x = torch.cat((x, transformer_encoder(x)), dim=1)

            skips.append(x)

        is_next = F.adaptive_avg_pool2d(self.is_next(x.transpose(1,3)).transpose(1,3), (1,1)).squeeze(-1).squeeze(-1)

        skips = skips[::-1]
        for i, decoder in enumerate(self.decoder):
            for transformer_decoder in self.transformer_decoder[i]:
                x = torch.cat((x, transformer_decoder(x, skip=skips[i])), dim=1)

            x = decoder(x, skips[i+1] if i < len(self.decoder) - 1 else None)

        return F.pad(
            input=torch.sigmoid(self.out(x.transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        ), is_next
        
class FrameEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU):
        super(FrameEncoder, self).__init__()

        self.idnt = FrameConv(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, activate=None, norm=False)
        self.conv1 = FrameConv(in_channels, out_channels, kernel_size, 1, padding, activate=activ)
        self.conv2 = FrameConv(out_channels, out_channels, kernel_size, stride, padding, activate=activ)

    def __call__(self, x):
        idnt = self.idnt(x)
        h = self.conv1(x)
        h = self.conv2(h)
        x = idnt + h

        return x

class FrameDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activ=nn.LeakyReLU, norm=True, dropout=False):
        super(FrameDecoder, self).__init__()

        self.idnt = FrameConv(in_channels, out_channels, kernel_size=1, padding=0, activate=None, norm=False)
        self.conv1 = FrameConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, activate=activ, norm=norm)
        self.conv2 = FrameConv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, activate=activ, norm=norm)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=(skip.shape[2],skip.shape[3]), mode='bilinear', align_corners=True)
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)

        idnt = self.idnt(x)
        h = self.conv1(x)
        h = self.conv2(h)

        if self.dropout is not None:
            h = self.dropout(h)

        x = idnt + h

        return x