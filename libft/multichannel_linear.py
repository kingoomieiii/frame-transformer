import torch
import torch.nn as nn
import math

class MultichannelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, positionwise=True, depthwise=False, bias=False):
        super(MultichannelLinear, self).__init__()

        self.weight_pw = None
        self.bias_pw = None
        if in_features != out_features or positionwise:
            self.weight_pw = nn.Parameter(torch.empty(in_channels, out_features, in_features))
            nn.init.uniform_(self.weight_pw, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features))

            if bias:
                self.bias_pw = nn.Parameter(torch.empty(in_channels, out_features, 1))
                bound = 1 / math.sqrt(in_features)
                nn.init.uniform_(self.bias_pw, -bound, bound)

        self.weight_dw = None
        self.bias_dw = None
        if in_channels != out_channels or depthwise:
            self.weight_dw = nn.Parameter(torch.empty(out_channels, in_channels))
            nn.init.uniform_(self.weight_dw, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

            if bias:
                self.bias_dw = nn.Parameter(torch.empty(out_channels, 1, 1))
                bound = 1 / math.sqrt(in_channels)
                nn.init.uniform_(self.bias_pw, -bound, bound)

    def __call__(self, x):
        d = len(x.shape)

        if d == 2:
            x = x.unsqueeze(-1).unsqueeze(1)
        elif d == 3:
            x = x.unsqueeze(-1)

        if self.weight_pw is not None:
            x = torch.matmul(x.transpose(2,3), self.weight_pw.transpose(1,2)).transpose(2,3)

            if self.bias_pw is not None:
                x = x + self.bias_pw

        if self.weight_dw is not None:
            x = torch.matmul(x.transpose(1,3), self.weight_dw.t()).transpose(1,3)

            if self.bias_dw is not None:
                x = x + self.bias_dw

        if d == 2 or d == 3:
            x = x.squeeze(-1)
        
        return x