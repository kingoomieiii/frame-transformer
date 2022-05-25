import torch
from torch import log_softmax, nn
import torch.nn.functional as F
import math

from lib.frame_transformer_common import FrameTransformerEncoder

class FrameTransformer(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True):
        super(FrameTransformer, self).__init__()
        
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])

        self.out = nn.Linear(channels + num_transformer_blocks, 2, bias=bias)
        
        self.is_next_channels = nn.Linear(channels + num_transformer_blocks, 1)
        self.is_next_bins = nn.Linear(self.max_bin, 1)
        self.is_next_frames = nn.Linear(self.cropsize, 1)
        
        self.activate = out_activate if out_activate is not None else nn.Identity()

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        for module in self.encoder:
            h = module(x)
            x = torch.cat((x, h), dim=1)

        c = self.is_next_channels(x.transpose(1,3))
        b = self.is_next_bins(c.transpose(2,3)).squeeze(-1).squeeze(-1)
        is_next = self.is_next_frames(b)

        return F.pad(
            input=self.activate(self.out(x.transpose(1,3)).transpose(1,3)),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        ), is_next

class FrameTransformerCritic(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True):
        super(FrameTransformerCritic, self).__init__()
        
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        self.encoder = nn.ModuleList([FrameTransformerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])

        self.out_channels = nn.Linear(channels + num_transformer_blocks, 1)
        self.out = nn.Linear(self.max_bin, 1, bias=bias)

        self.activate = out_activate if out_activate is not None else nn.Identity()

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        for module in self.encoder:
            h = module(x)
            x = torch.cat((x, h), dim=1)

        return self.out(self.out_channels(x.transpose(1,3)).transpose(2,3)).squeeze(-1).squeeze(-1)