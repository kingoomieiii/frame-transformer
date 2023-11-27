import torch
import torch.nn as nn

class MultichannelLayerNorm(nn.Module):
    def __init__(self, channels, features, eps=1e-8, trainable=True):
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
        x = (torch.layer_norm(x.transpose(2,3), (self.weight.shape[-1],), eps=self.eps) * self.weight + self.bias).transpose(2,3)

        return x