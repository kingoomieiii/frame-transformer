import torch
import torch.nn as nn

from libft.multichannel_layernorm import MultichannelLayerNorm

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=False, expansion=1):
        super(ResBlock, self).__init__()

        self.norm = MultichannelLayerNorm(in_channels, features)
        self.conv1 = nn.Conv2d(in_channels, out_channels * expansion, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels * expansion, out_channels, kernel_size=3, padding=1, bias=False, stride=2 if downsample else 1)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=2 if downsample else 1) if in_channels != out_channels or downsample else nn.Identity()

    def __call__(self, x):
        return self.identity(x) + self.conv2(torch.relu(self.conv1(self.norm(x))) ** 2)