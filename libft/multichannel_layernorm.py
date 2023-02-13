import torch
import torch.nn as nn

class MultichannelLayerNorm(nn.Module):
    def __init__(self, channels, features, eps=0.00001, trainable=True):
        super(MultichannelLayerNorm, self).__init__()
        
        self.eps = eps

        if trainable:
            self.weight = nn.Parameter(torch.empty(channels, 1, features))
            self.bias = nn.Parameter(torch.empty(channels, 1, features))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        else:
            self.register_buffer('weight', torch.ones(channels, 1, features))
            self.register_buffer('bias', torch.zeros(channels, 1, features))

    def __call__(self, x):
        d = len(x.shape)

        if d == 2:
            x = x.unsqueeze(-1).unsqueeze(1)
        elif d == 3:
            x = x.unsqueeze(-1)

        x = (torch.layer_norm(x.transpose(2,3), (self.weight.shape[-1],), eps=self.eps) * self.weight + self.bias).transpose(2,3)

        if d == 2 or d == 3:
            x = x.squeeze(-1)

        return x   

class FrameNorm(nn.Module):
    def __init__(self, features):
        super(FrameNorm, self).__init__()
        
        self.norm = nn.LayerNorm(features)

    def __call__(self, x):
        d = len(x.shape)
        h = x

        if d == 2:
            h = h.unsqueeze(-1).unsqueeze(1)
        elif d == 3:
            h = h.unsqueeze(-1)

        h = self.norm(h.transpose(2,3)).transpose(2,3)

        if d == 2 or d == 3:
            h = h.squeeze(-1)

        return h