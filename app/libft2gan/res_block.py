import torch
import torch.nn as nn

from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.channel_norm import ChannelNorm
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.squared_relu import SquaredReLU, Dropout2d, Cardioid

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, kernel_size=3, padding=1, downsample=False, stride=(2,1), dropout=0, channel_norm=False, dtype=torch.float):
        super(ResBlock, self).__init__()

        self.dropout = Dropout2d(dropout, dtype=dtype) if dropout > 0 else nn.Identity()
        self.activate = SquaredReLU(dtype=dtype)
        self.norm = MultichannelLayerNorm(in_channels, features, dtype=dtype) if not channel_norm else ChannelNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, dtype=dtype)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride if downsample else 1, bias=False, dtype=dtype)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride if downsample else 1, bias=False, dtype=dtype) if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + self.dropout(h)

        return x

class LinearResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, bias=False, dropout=0, depthwise=True, positionwise=True):
        super(LinearResBlock, self).__init__()

        self.dropout = Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activate = SquaredReLU()
        self.norm = MultichannelLayerNorm(in_channels, in_features)
        self.conv1 = MultichannelLinear(in_channels, out_channels, in_features, out_features, positionwise=positionwise, depthwise=depthwise, bias=bias)
        self.conv2 = MultichannelLinear(out_channels, out_channels, out_features, out_features, positionwise=positionwise, depthwise=depthwise, bias=bias)
        self.identity = MultichannelLinear(in_channels, out_channels, in_features, out_features) if in_channels != out_channels or in_features != out_features else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x))))
        x = self.identity(x) + self.dropout(h)

        return x

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dropout=0):
        super(ResBlock1d, self).__init__()

        self.activate = SquaredReLU()
        self.norm = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.identity = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False) if in_channels != out_channels or stride > 1 else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.activate(self.conv1(self.norm(x.transpose(1,2)).transpose(1,2))))
        x = self.identity(x) + h

        return x