import torch
import torch.nn as nn

from libft2gancurr.multichannel_layernorm import MultichannelLayerNorm
from libft2gancurr.squared_relu import SquaredReLU

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, kernel_size=3, padding=1, downsample=False, stride=(2,1), dropout=0):
        super(ResBlock, self).__init__()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activate = SquaredReLU()
        self.norm = MultichannelLayerNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride if downsample else 1, bias=False)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride if downsample else 1, bias=False) if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + self.dropout(h)

        return x