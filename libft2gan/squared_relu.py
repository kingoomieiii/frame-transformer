import torch
import torch.nn as nn

class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 2