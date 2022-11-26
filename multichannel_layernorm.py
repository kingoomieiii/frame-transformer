import torch
import torch.nn as nn

class FrameNorm(nn.Module):
    def __init__(self, features):
        super(FrameNorm, self).__init__()
        
        self.norm = nn.LayerNorm(features)

    def __call__(self, x):
        return self.norm(x.transpose(2,3)).transpose(2,3)

class MultichannelLayerNorm(nn.Module):
    def __init__(self, channels, features, eps=0.00001):
        super(MultichannelLayerNorm, self).__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(channels, 1, features))
        self.bias = nn.Parameter(torch.empty(channels, 1, features))
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def __call__(self, x):
        return (torch.layer_norm(x.transpose(2,3), (self.weight.shape[-1],), eps=self.eps) * self.weight + self.bias).transpose(2,3)