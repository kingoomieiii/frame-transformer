import torch
from torch import log_softmax, nn
import torch.nn.functional as F
import math

from lib.frame_transformer_common import MultibandFrameAttention

class FramePrimer(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True):
        super(FramePrimer, self).__init__()
        
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        self.encoder = nn.ModuleList([FramePrimerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])

        self.out_norm = nn.BatchNorm2d(channels + num_transformer_blocks)
        
        self.out = nn.Linear(channels + num_transformer_blocks, 2)
                
    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        for module in self.encoder:
            h = module(x)
            x = torch.cat((x, h), dim=1)

        x = self.out_norm(x)

        return F.pad(
            input=torch.sigmoid(self.out(x.transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )

class FramePrimerDiscriminator(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True):
        super(FramePrimerDiscriminator, self).__init__()
        
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        self.encoder = nn.ModuleList([FramePrimerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])

        self.out_norm = nn.BatchNorm2d(channels + num_transformer_blocks)
        self.out_channels = nn.Linear(channels + num_transformer_blocks, 1)

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        for module in self.encoder:
            h = module(x)
            x = torch.cat((x, h), dim=1)

        x = self.out_norm(x)

        return torch.mean(self.out_channels(x.transpose(1,3)).transpose(1,3), dim=2, keepdim=True)

class FramePrimerEncoder(nn.Module):
    def __init__(self, channels, bins=0, num_bands=4, cropsize=1024, feedforward_dim=2048, bias=False, dropout=0.1, downsamples=0, n_fft=2048):
        super(FramePrimerEncoder, self).__init__()

        bins = n_fft // 2
        if downsamples > 0:
            for _ in range(downsamples):
                bins = ((bins - 1) // 2) + 1

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.relu = nn.ReLU(inplace=True)

        self.in_norm = nn.BatchNorm2d(channels)
        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.norm1 = nn.BatchNorm2d(1)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize, bias=bias)

        self.norm2 = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.linear2 = nn.Linear(feedforward_dim, bins, bias=bias)

    def __call__(self, x):
        x = self.in_norm(x)
        x = self.in_project(x.transpose(1,3)).squeeze(-1)

        h = self.norm1(x.unsqueeze(-1).transpose(1,3)).transpose(1,3).squeeze(-1)
        h = self.attn(h)
        x = x + h
        
        h = self.norm2(x.unsqueeze(-1).transpose(1,3)).transpose(1,3).squeeze(-1)
        h = self.linear2(torch.square(self.relu(self.linear1(h))))
        x = x + h

        return x.transpose(1,2).unsqueeze(1)