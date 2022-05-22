import torch
from torch import nn
import torch.nn.functional as F
from lib import spec_utils
from lib.frame_transformer_common import FrameConv, FrameTransformerDecoder, FrameTransformerEncoder

class ConvDiscriminator(nn.Module):
    def __init__(self, in_channels=4, channels=4, depth=6, n_fft=2048):
        super(ConvDiscriminator, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.encoder = [Encoder(in_channels, channels, kernel_size=3, padding=1, stride=1)]

        m = 1
        for i in range(depth - 1):
            self.encoder.append(Encoder(channels * m, channels * 2 * m, stride=2))
            m = 2 * m

        self.encoder = nn.ModuleList(self.encoder)
        self.out = FrameConv(channels * m, 1, kernel_size=3, padding=1, norm=False, activate=None)

    def forward(self, masked, unmasked):
        x = torch.cat((masked, unmasked), dim=1)

        x = x[:, :, :self.max_bin]

        for encoder in self.encoder:
            x = encoder(x)

        return self.out(x)
        
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), 
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True))

    def __call__(self, x):
        return self.body(x)