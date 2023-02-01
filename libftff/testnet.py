import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.lr_scheduler_linear_warmup import LinearWarmupScheduler
from lib.lr_scheduler_polynomial_decay import PolynomialDecayScheduler

from libft.frame_transformer2 import FrameTransformerEncoder
from libft.multichannel_layernorm import MultichannelLayerNorm
from libft.multichannel_linear import MultichannelLinear

class SquaredReLU(nn.Module):
    def __call__(self, x):
        return torch.relu(x) ** 2

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, out_features, num_heads=16, lr=1e-4, decay_lr=1e-12, warmup=8000, decay=100000, curr_step=0):
        super(Layer, self).__init__()

        self.encoder = FrameTransformerEncoder(in_channels, in_features, expansion=4)
        self.norm = MultichannelLayerNorm(out_channels, out_features, trainable=False)
        self.optim = torch.optim.Adam(self.encoder.parameters(), lr=lr)    
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        model_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)

        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            LinearWarmupScheduler(self.optim, target_lr=lr, num_steps=warmup, current_step=curr_step, verbose_skip_steps=10000),
            PolynomialDecayScheduler(self.optim, target=decay_lr, power=1, num_decay_steps=decay, start_step=warmup, current_step=curr_step, verbose_skip_steps=10000)
        ])

    def __call__(self, x, activity=None):
        if self.training:
            self.zero_grad()
            h = self.encoder(x)
            a = h.reshape(h.shape[0], -1)
            a = torch.cat([a.min(dim=1, keepdim=True).values, a.max(dim=1, keepdim=True).values, a.mean(dim=1, keepdim=True), a.median(dim=1, keepdim=True).values, a.var(dim=1, keepdim=True)], dim=1)
            l = F.mse_loss(a, activity)
            self.scaler.scale(l).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()
            self.zero_grad()
            return self.norm(h.detach())
        else:
            return self.norm(self.encoder(x))

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, in_features, kernel_size=3, padding=1, lr=1e-4, decay_lr=1e-12, warmup=8000, decay=100000, curr_step=0):
        super(Conv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU())

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)    
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)

        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            LinearWarmupScheduler(self.optim, target_lr=lr, num_steps=warmup, current_step=curr_step, verbose_skip_steps=10000),
            PolynomialDecayScheduler(self.optim, target=decay_lr, power=1, num_decay_steps=decay, start_step=warmup, current_step=curr_step, verbose_skip_steps=10000)
        ])

    def __call__(self, x, activity=None):
        if self.training:
            self.zero_grad()
            h = self.conv(x)
            a = h.reshape(h.shape[0], -1)
            a = torch.cat([a.min(dim=1, keepdim=True).values, a.max(dim=1, keepdim=True).values, a.mean(dim=1, keepdim=True), a.median(dim=1, keepdim=True).values, a.var(dim=1, keepdim=True)], dim=1)
            l = F.mse_loss(a, activity)
            self.scaler.scale(l).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()
            self.zero_grad()
            return h.detach()
        else:
            return self.conv(x)
            
class TestNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels=2,  num_heads=8, n_fft=2048, num_layers=8, lr=1e-4, decay_lr=1e-12, warmup=0, decay=100000, curr_step=0):
        super(TestNet, self).__init__()

        self.features = n_fft // 2
        self.in_layer = Conv2d(in_channels, channels, self.features, lr=lr, decay_lr=decay_lr, warmup=warmup, decay=decay, curr_step=curr_step)
        self.norm = MultichannelLayerNorm(channels, self.features, trainable=False)
        self.positive_layers = nn.ModuleList([Layer(channels, channels, self.features, self.features, num_heads, lr=lr) for _ in range(num_layers)])
        self.negative_layers = nn.ModuleList([Layer(channels, channels, self.features, self.features, num_heads, lr=lr) for _ in range(num_layers)])
        self.out = MultichannelLinear(channels * num_layers * 2, out_channels, self.features, self.features) # nn.Conv2d(channels * num_layers * 2, out_channels, kernel_size=1, padding=0, bias=False)
        self.optim = torch.optim.Adam(self.out.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()        

        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            LinearWarmupScheduler(self.optim, target_lr=lr, num_steps=warmup, current_step=curr_step, verbose_skip_steps=10000),
            PolynomialDecayScheduler(self.optim, target=decay_lr, power=1, num_decay_steps=decay, start_step=warmup, current_step=curr_step, verbose_skip_steps=10000)
        ])

    def __call__(self, x, y=None, activity=None):
        if self.training:
            h = self.norm(self.in_layer(x, activity))

            out = None
            for layer in self.positive_layers:
                h = layer(h, activity)
                out = h if out is None else torch.cat((out, h), dim=1)

            for layer in self.negative_layers:
                h = layer(h, 1 - activity)
                out = h if out is None else torch.cat((out, h), dim=1)

            self.optim.zero_grad()
            out = torch.sigmoid(self.out(out))
            l = F.l1_loss(x * out, y)
            self.scaler.scale(l).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.scheduler.step()
            self.optim.zero_grad()

            return x * out
        else:
            h = self.in_layer(x)

            out = None
            for layer in self.positive_layers:
                h = layer(h)
                out = h if out is None else torch.cat((out, h), dim=1)

            out = torch.sigmoid(self.out(out))

            return x * out