import torch
from torch import nn
import torch.nn.functional as F
import math
from lib import spec_utils
from lib.frame_transformer_common import FrameConv, FrameTransformerDecoder, FrameTransformerEncoder

class FrameTransformerUNet(nn.Module):
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=4, num_encoders=1, num_decoders=1, cropsize=1024, bias=False, autoregressive=False, out_activate=nn.Sigmoid()):
        super(FrameTransformerUNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        
        self.register_buffer('mask', torch.triu(torch.ones(cropsize, cropsize) * float('-inf'), diagonal=1))
        self.autoregressive = autoregressive
        self.out_activate = out_activate

        self.enc1 = FrameConv(2, channels, kernel_size=3, padding=1, stride=1)
        self.enc1_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 1 + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc2 = FrameEncoder(channels * 1 + num_encoders, channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 2 + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc3 = FrameEncoder(channels * 2 + num_encoders, channels * 4, kernel_size=3, stride=2, padding=1)
        self.enc3_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 4 + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc4 = FrameEncoder(channels * 4 + num_encoders, channels * 6, kernel_size=3, stride=2, padding=1)
        self.enc4_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 6 + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])

        self.enc5 = FrameEncoder(channels * 6 + num_encoders, channels * 8, kernel_size=3, stride=2, padding=1)
        self.enc5_transformer = nn.ModuleList([FrameTransformerEncoder(channels * 8 + i, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_encoders)])
        
        self.dec4_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 8 + i, channels * 8 + num_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=4, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec4 = FrameDecoder(channels * (6 + 8) + num_decoders + num_encoders, channels * 6, kernel_size=3, padding=1)

        self.dec3_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 6 + i, channels * 6 + num_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=3, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec3 = FrameDecoder(channels * (4 + 6) + num_decoders + num_encoders, channels * 4, kernel_size=3, padding=1)

        self.dec2_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 4 + i, channels * 4 + num_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=2, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec2 = FrameDecoder(channels * (2 + 4) + num_decoders + num_encoders, channels * 2, kernel_size=3, padding=1)

        self.dec1_transformer = nn.ModuleList([FrameTransformerDecoder(channels * 2 + i, channels * 2 + num_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=1, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.dec1 = FrameDecoder(channels * (1 + 2) + num_decoders + num_encoders, channels * 1, kernel_size=3, padding=1)

        self.out_transformer = nn.ModuleList([FrameTransformerDecoder(channels + i, channels + num_encoders, num_bands=num_bands, cropsize=cropsize, n_fft=n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])
        self.out = nn.Linear(channels + num_decoders, 2)
        
        self.is_next = nn.Sequential(
            nn.Linear(channels + num_encoders, 2, bias=bias),
            nn.LogSoftmax(dim=-1))

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        e1 = self.enc1(x)
        for module in self.enc1_transformer:
            t = module(e1)
            e1 = torch.cat((e1, t), dim=1)

        e2 = self.enc2(e1)
        for module in self.enc2_transformer:
            t = module(e2)
            e2 = torch.cat((e2, t), dim=1)

        e3 = self.enc3(e2)
        for module in self.enc3_transformer:
            t = module(e3)
            e3 = torch.cat((e3, t), dim=1)

        e4 = self.enc4(e3)
        for module in self.enc4_transformer:
            t = module(e4)
            e4 = torch.cat((e4, t), dim=1)

        e5 = self.enc5(e4)
        h = e5
        for module in self.enc5_transformer:
            t = module(e5)
            e5 = torch.cat((e5, t), dim=1)
        for module in self.dec4_transformer:
            t = module(h, skip=e5)
            h = torch.cat((h, t), dim=1)
            
        h = self.dec4(h, e4)
        for module in self.dec3_transformer:
            t = module(h, skip=e4)
            h = torch.cat((h, t), dim=1)

        h = self.dec3(h, e3)        
        for module in self.dec2_transformer:
            t = module(h, skip=e3)
            h = torch.cat((h, t), dim=1)

        h = self.dec2(h, e2)        
        for module in self.dec1_transformer:
            t = module(h, skip=e2)
            h = torch.cat((h, t), dim=1)

        h = self.dec1(h, e1)
        for module in self.out_transformer:
            t = module(h, skip=e1)
            h = torch.cat((h, t), dim=1)

        out = self.out(h.transpose(1,3)).transpose(1,3)

        return F.pad(
            input=self.out_activate(out),
            pad=(0, 0, 0, self.output_bin - out.size()[2]),
            mode='replicate'
        ), F.adaptive_avg_pool2d(self.is_next(h.transpose(1,3)).transpose(1,3), (1,1)).squeeze(-1).squeeze(-1)
        
class FrameEncoder(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, activ=nn.LeakyReLU):
        super(FrameEncoder, self).__init__()
        self.conv1 = FrameConv(nin, nout, kernel_size, 1, padding, activate=activ)
        self.conv2 = FrameConv(nout, nout, kernel_size, stride, padding, activate=activ)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        return h

class FrameDecoder(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, activ=nn.LeakyReLU, norm=True, dropout=False):
        super(FrameDecoder, self).__init__()
        self.conv = FrameConv(nin, nout, kernel_size, 1, padding, activate=activ, norm=norm)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=(skip.shape[2],skip.shape[3]), mode='bilinear', align_corners=True)
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)

        h = self.conv(x)

        if self.dropout is not None:
            h = self.dropout(h)

        return h