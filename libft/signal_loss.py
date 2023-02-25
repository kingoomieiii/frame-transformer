import math
import torch

def sdr_loss(X, Y, eps=1e-10):
    signal = torch.sum(Y ** 2)
    residual = torch.sum((X - Y) ** 2)
    sdr = 10 * torch.log10((signal + eps) / (residual + eps))
    return -sdr