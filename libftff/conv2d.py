import torch
import torch.nn as nn

class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False, lr=1e-4):
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.optim = torch.optim.Adam(self.conv.parameters(), lr=lr)

    def __call__(self, x):
        pass