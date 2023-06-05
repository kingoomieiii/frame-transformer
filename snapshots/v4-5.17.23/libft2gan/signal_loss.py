import torch

def snr_loss(X, Y, eps=1e-10):
    signal = torch.sum(Y ** 2)
    residual = torch.sum((X - Y) ** 2)
    sdr = 10 * torch.log10((signal + eps) / (residual + eps))
    return -sdr

def lsd_loss(X, Y, eps=1e-10):
    X = 10 * torch.log10(torch.abs(X) ** 2 + eps)
    Y = 10 * torch.log10(torch.abs(Y) ** 2 + eps)
    return torch.mean((X - Y) ** 2)

def sdr_loss(X, Y, eps=1e-10):
    alpha = torch.sum(X * Y) / (torch.sum(Y ** 2) + eps)
    signal = alpha * Y
    distortion = X - signal
    return -(10 * torch.log10(((torch.sum(signal ** 2) + eps) / torch.sum(distortion ** 2) + eps)))

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())