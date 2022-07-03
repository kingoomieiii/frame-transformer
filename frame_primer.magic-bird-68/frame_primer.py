import torch
from torch import nn
import torch.nn.functional as F
import math

from frame_primer.common import MultibandFrameAttention
from frame_primer.primer import Primer
from frame_primer.quantizer import FrameQuantizer

from frame_primer.common import FramePrimerEncoder as FramePrimerEncoder2

class FramePrimer(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True):
        super(FramePrimer, self).__init__()
        
        self.num_classes = 128
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        self.encoder = nn.ModuleList([FramePrimerEncoder2(channels + i * 3, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])
        self.out_norm = nn.BatchNorm2d(channels + num_transformer_blocks * 3)
        self.out = nn.Linear(channels + num_transformer_blocks * 3, 2)

    def __call__(self, x):

        prev_qk = None
        for module in self.encoder:
            h, prev_qk = module(x, prev_qk=prev_qk)
            x = torch.cat((x, h), dim=1)

        return torch.sigmoid(self.out(torch.relu(self.out_norm(x)).transpose(1,3)).transpose(1,3))

class VQFramePrimer(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True, num_embeddings=16384):
        super(VQFramePrimer, self).__init__()
        
        self.num_classes = 128
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize
        
        self.encoder = nn.ModuleList([FramePrimerEncoder(channels + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])
        self.bottleneck_norm = nn.BatchNorm2d(channels + num_transformer_blocks)
        self.bottleneck = nn.Linear(channels + num_transformer_blocks, 1)

        self.quantizer = FrameQuantizer(1, num_embeddings)
        
        self.decoder = nn.ModuleList([FramePrimerEncoder(1 + i, bins=self.max_bin, num_bands=num_bands, cropsize=cropsize, feedforward_dim=feedforward_dim, bias=bias, dropout=dropout) for i in range(num_transformer_blocks)])
        self.out_norm = nn.BatchNorm2d(1 + num_transformer_blocks)
        self.out = nn.Linear(1 + num_transformer_blocks, 2)

    def __call__(self, x):

        prev_qk = None
        for module in self.encoder:
            h, prev_qk = module(x, prev_qk=prev_qk)
            x = torch.cat((x, h), dim=1)

        x = self.bottleneck(self.bottleneck_norm(x).transpose(1,3)).transpose(1,3)

        x_q, loss, _ = self.quantizer(x)

        prev_qk = None
        for module in self.decoder:
            h, prev_qk = module(x_q, prev_qk=prev_qk)
            x_q = torch.cat((x_q, h), dim=1)

        return torch.relu(self.out(self.out_norm(x_q).transpose(1,3)).transpose(1,3)), loss
        
class VQFramePrimer2(nn.Module):
    def __init__(self, channels=2, n_fft=2048, feedforward_dim=512, num_bands=4, num_transformer_blocks=1, cropsize=1024, bias=False, out_activate=nn.Sigmoid(), dropout=0.1, pretraining=True, num_embeddings=16384):
        super(VQFramePrimer2, self).__init__()
        
        self.num_classes = 128
        self.pretraining = pretraining
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.cropsize = cropsize

        self.encoder = Primer(channels * self.max_bin, feedforward_dim=feedforward_dim, num_heads=num_bands, num_transformer_blocks=num_transformer_blocks, cropsize=cropsize, bias=bias, dropout=dropout)
        self.quantizer = FrameQuantizer(channels, num_embeddings)
        self.decoder = Primer(channels * self.max_bin, feedforward_dim=feedforward_dim, num_heads=num_bands, num_transformer_blocks=num_transformer_blocks, cropsize=cropsize, bias=bias, dropout=dropout)

    def __call__(self, x):
        b,c,h,w = x.shape
        x = self.encoder(x.reshape(b,c*h,w).transpose(1,2))
        x, q_loss, _ = self.quantizer(x)
        x = self.decoder(x).transpose(1,2).reshape(b,c,h,w)

        return torch.relu(x), q_loss

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

        self.in_norm = nn.InstanceNorm2d(cropsize, affine=True)
        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.norm1 = nn.InstanceNorm2d(cropsize, affine=True)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize, bias=bias)

        self.norm2 = nn.InstanceNorm2d(cropsize, affine=True)
        self.linear1 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.linear2 = nn.Linear(feedforward_dim, bins, bias=bias)

    def __call__(self, x, prev_qk=None):
        x = self.in_project(self.relu(self.in_norm(x.transpose(1,3)))).squeeze(-1)

        h = self.norm1(x.unsqueeze(-1)).squeeze(-1)
        h, prev_qk = self.attn(h, prev_qk=prev_qk)
        x = x + h
        
        h = self.norm2(x.unsqueeze(-1)).squeeze(-1)
        h = self.linear2(torch.square(self.relu(self.linear1(h))))
        x = x + h

        return x.transpose(1,2).unsqueeze(1), prev_qk