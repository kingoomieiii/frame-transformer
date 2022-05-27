import torch
from torch import nn
import torch.nn.functional as F
from lib.frame_transformer_common import FrameConv, FrameDecoder, FrameEncoder, FrameTransformerDecoder, FrameTransformerEncoder

class FrameTransformerUnet(nn.Module):
    def __init__(self, in_channels=2, channels=2, depth=6, num_transformer_blocks=2, dropout_enc=0.1, dropout_dec=0.5, n_fft=2048, cropsize=1024, num_bands=8, feedforward_dim=2048, bias=True, pretraining=True):
        super(FrameTransformerUnet, self).__init__()

        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.encoder = [FrameEncoder(in_channels, channels, kernel_size=3, padding=1, stride=1, cropsize=cropsize)]
        self.decoder = [FrameConv(channels + num_transformer_blocks, in_channels, kernel_size=1, padding=0, norm=False, activate=None)]
        self.transformer_encoder = [nn.ModuleList([FrameTransformerEncoder(channels + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout_enc) for i in range(num_transformer_blocks)])]
        self.transformer_decoder = [nn.ModuleList([FrameTransformerDecoder(channels + i, channels + num_transformer_blocks, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias, dropout=0) for i in range(num_transformer_blocks)])]

        for i in range(depth - 1):
            self.encoder.append(FrameEncoder(channels * (i + 1) + num_transformer_blocks, channels * (i + 2), stride=2, cropsize=cropsize))
            self.decoder.append(FrameDecoder(channels * (2 * i + 3) + 2 * num_transformer_blocks + (num_transformer_blocks if i == depth - 2 else 0), channels * (i + 1), cropsize=cropsize, dropout=dropout_dec if i < 3 else 0))
            self.transformer_encoder.append(nn.ModuleList([FrameTransformerEncoder(channels * (i + 2) + j, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout_enc) for j in range(num_transformer_blocks)]))
            self.transformer_decoder.append(nn.ModuleList([FrameTransformerDecoder(channels * (i + 2) + j + (num_transformer_blocks if i == depth - 2 else 0), channels * (i + 2) + num_transformer_blocks, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=i+1, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout_dec if i < 3 else 0) for j in range(num_transformer_blocks)]))

        self.encoder = nn.ModuleList(self.encoder)
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        
        self.decoder = nn.ModuleList(self.decoder[::-1])
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder[::-1])

        self.is_next = nn.Linear(channels * (i + 2) + num_transformer_blocks, 2, bias=bias)

    def forward(self, x):
        x = x[:, :, :self.max_bin]

        skips = []
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            for transformer_encoder in self.transformer_encoder[i]:
                x = torch.cat((x, transformer_encoder(x)), dim=1)

            skips.append(x)

        #is_next = F.adaptive_avg_pool2d(self.is_next(x.transpose(1,3)).transpose(1,3), (1,1)).squeeze(-1).squeeze(-1) if self.pretraining else None

        skips = skips[::-1]
        for i, decoder in enumerate(self.decoder):
            for transformer_decoder in self.transformer_decoder[i]:
                x = torch.cat((x, transformer_decoder(x, skip=skips[i])), dim=1)

            x = decoder(x, skips[i+1]) if i < len(self.decoder) - 1 else decoder(x)

        return F.pad(
            input=torch.sigmoid(x),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )