import torch
from torch import nn
import torch.nn.functional as F

from frame_primer.common import FrameEncoder, FrameDecoder, FramePrimerEncoder, FramePrimerDecoder

from lib import layers

class PrimerNet(nn.Module):
    def __init__(self, nin, nout, n_fft, cropsize, num_bands=8, feedforward_dim=2048, dropout=0.1):
        super(PrimerNet, self).__init__()

        self.nout = nout

        self.enc1 = FrameEncoder(nin, nout, n_fft=n_fft, downsamples=0, stride=1, column_kernel=False, column_stride=False)
        self.enc1_primer = FramePrimerEncoder(nout, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=0, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.enc2 = FrameEncoder(nout + 1, nout * 2, n_fft=n_fft, downsamples=0, stride=2, column_kernel=False, column_stride=False)
        self.enc2_primer = FramePrimerEncoder(nout * 2, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=1, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.enc3 = FrameEncoder(nout * 2 + 1, nout * 4, n_fft=n_fft, downsamples=1, stride=2, column_kernel=False, column_stride=False)
        self.enc3_primer = FramePrimerEncoder(nout * 4, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=2, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.enc4 = FrameEncoder(nout * 4 + 1, nout * 6, n_fft=n_fft, downsamples=2, stride=2, column_kernel=False, column_stride=False)
        self.enc4_primer = FramePrimerEncoder(nout * 6, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=3, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.enc5 = FrameEncoder(nout * 6 + 1, nout * 8, n_fft=n_fft, downsamples=3, stride=2, column_kernel=False, column_stride=False)
        self.enc5_primer = FramePrimerEncoder(nout * 8, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=4, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.enc6 = FrameEncoder(nout * 8 + 1, nout * 10, n_fft=n_fft, downsamples=4, stride=2, column_kernel=False, column_stride=False)
        self.enc6_primer = FramePrimerEncoder(nout * 10, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=5, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.dec5 = FrameDecoder(nout * (10 + 8) + 2, nout * 8, n_fft=n_fft, downsamples=4, column_kernel=False, column_stride=False)
        self.dec5_primer = FramePrimerDecoder(nout * 8, nout * 8 + 1, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=4, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.dec4 = FrameDecoder(nout * (8 + 6) + 2, nout * 6, n_fft=n_fft, downsamples=3, column_kernel=False, column_stride=False)
        self.dec4_primer = FramePrimerDecoder(nout * 6, nout * 6 + 1, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=3, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.dec3 = FrameDecoder(nout * (6 + 4) + 2, nout * 4, n_fft=n_fft, downsamples=2, column_kernel=False, column_stride=False)
        self.dec3_primer = FramePrimerDecoder(nout * 4, nout * 4 + 1, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=2, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.dec2 = FrameDecoder(nout * (4 + 2) + 2, nout * 2, n_fft=n_fft, downsamples=1, column_kernel=False, column_stride=False)
        self.dec2_primer = FramePrimerDecoder(nout * 2, nout * 2 + 1, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=1, dropout=dropout, downsample_cropsize=True, residual_attention=False)

        self.dec1 = FrameDecoder(nout * (2 + 1) + 2, nout * 1, n_fft=n_fft, downsamples=0, column_kernel=False, column_stride=False)
        self.dec1_primer = FramePrimerDecoder(nout * 1, nout * 1 + 1, num_bands=num_bands, n_fft=n_fft, cropsize=cropsize, feedforward_dim=feedforward_dim, downsamples=0, dropout=dropout, downsample_cropsize=True, residual_attention=False)

    def __call__(self, x):
        e1 = self.enc1(x)
        e1 = torch.cat((e1, self.enc1_primer(e1)), dim=1)

        e2 = self.enc2(e1)
        e2 = torch.cat((e2, self.enc2_primer(e2)), dim=1)

        e3 = self.enc3(e2)
        e3 = torch.cat((e3, self.enc3_primer(e3)), dim=1)

        e4 = self.enc4(e3)
        e4 = torch.cat((e4, self.enc4_primer(e4)), dim=1)

        e5 = self.enc5(e4)
        e5 = torch.cat((e5, self.enc5_primer(e5)), dim=1)

        e6 = self.enc6(e5)
        e6 = torch.cat((e6, self.enc6_primer(e6)), dim=1)

        h = self.dec5(e6, e5)
        h = torch.cat((h, self.dec5_primer(h, skip=e5)), dim=1)

        h = self.dec4(h, e4)
        h = torch.cat((h, self.dec4_primer(h, skip=e4)), dim=1)

        h = self.dec3(h, e3)
        h = torch.cat((h, self.dec3_primer(h, skip=e3)), dim=1)

        h = self.dec2(h, e2)
        h = torch.cat((h, self.dec2_primer(h, skip=e2)), dim=1)

        h = self.dec1(h, e1)
        h = torch.cat((h, self.dec1_primer(h, skip=e1)), dim=1)

        return h

class BaseNet(nn.Module):

    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = layers.Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(nout * 6, nout * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):

    def __init__(self, n_fft):
        super(CascadedNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, 16, self.nin_lstm // 2, 128),
            layers.Conv2DBNActiv(16, 8, 1, 1, 0)
        )
        self.stg1_high_band_net = BaseNet(2, 8, self.nin_lstm // 2, 64)

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(10, 32, self.nin_lstm // 2, 128),
            layers.Conv2DBNActiv(32, 16, 1, 1, 0)
        )
        self.stg2_high_band_net = BaseNet(10, 16, self.nin_lstm // 2, 64)

        self.stg3_full_band_net = BaseNet(26, 32, self.nin_lstm, 128)

        self.out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(24, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, :self.max_bin]

        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        mask = torch.sigmoid(self.out(f3))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate'
        )

        if self.training:
            aux = torch.cat([aux1, aux2], dim=1)
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode='replicate'
            )
            return mask, aux
        else:
            return mask

    def predict_mask(self, x):
        mask = self.forward(x)

        if self.offset > 0:
            mask = mask[:, :, :, self.offset:-self.offset]
            assert mask.size()[3] > 0

        return mask

    def predict(self, x):
        mask = self.forward(x)
        pred_mag = x * mask

        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset:-self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag

class CascadedPrimerNet(nn.Module):
    def __init__(self, n_fft, cropsize):
        super(CascadedPrimerNet, self).__init__()

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        self.stg1_full = PrimerNet(nin=2, nout=8, n_fft=n_fft, cropsize=cropsize, num_bands=16, feedforward_dim=8192, dropout=0.1)
        self.stg2_full = PrimerNet(nin=11, nout=12, n_fft=n_fft, cropsize=cropsize, num_bands=16, feedforward_dim=8192, dropout=0.1)
        self.stg3_full = PrimerNet(nin=24, nout=16, n_fft=n_fft, cropsize=cropsize, num_bands=16, feedforward_dim=8192, dropout=0.1)
        self.out = nn.Conv2d(17, 2, 1, bias=False)

    def forward(self, x):

        l1 = self.stg1_full(x)
        l2 = self.stg2_full(torch.cat((x, l1), dim=1))
        l3 = self.stg3_full(torch.cat((x, l1, l2), dim=1))

        mask = torch.sigmoid(self.out(l3))

        return mask