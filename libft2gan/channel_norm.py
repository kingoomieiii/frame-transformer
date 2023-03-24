import torch
import torch.nn as nn

class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super(ChannelNorm, self).__init__()

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1,3)).transpose(1,3)