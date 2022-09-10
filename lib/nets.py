import torch
from torch import nn
import torch.nn.functional as F

from lib import layers


class BaseNet(nn.Module):

    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6)), perceptual_loss=False, skip_decoder=False):
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

        self.skip_decoder = skip_decoder
        self.perceptual_loss = perceptual_loss

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        l = None
        if self.perceptual_loss:
            l = h

        if self.skip_decoder:
            return l

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        if l is not None:
            return h, l
        else:
            return h

class CascadedNet(nn.Module):

    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 0

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0)
        )
        self.stg1_high_band_net = BaseNet(
            2, nout // 4, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0)
        )
        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm
        )

        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, :self.max_bin]

        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        l1 = self.stg1_low_band_bn(l1)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        l2 = self.stg2_low_band_bn(l2)
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


class CascadedNetPerceptualLoss(nn.Module):
    def __init__(self, cascade_net: CascadedNet):
        super(CascadedNetPerceptualLoss, self).__init__()

        self.stg1_low_band_net = cascade_net.stg1_low_band_net[0].requires_grad_(False)
        self.stg1_low_band_bn = cascade_net.stg1_low_band_net[1].requires_grad_(False)
        self.stg1_high_band_net = cascade_net.stg1_high_band_net.requires_grad_(False)
        self.stg1_low_band_net.perceptual_loss = True
        self.stg1_high_band_net.perceptual_loss = True

        self.stg2_low_band_net = cascade_net.stg2_low_band_net[0].requires_grad_(False)
        self.stg2_low_band_bn = cascade_net.stg2_low_band_net[1].requires_grad_(False)
        self.stg2_high_band_net = cascade_net.stg2_high_band_net.requires_grad_(False)
        self.stg2_low_band_net.perceptual_loss = True
        self.stg2_high_band_net.perceptual_loss = True

        self.stg3_full_band_net = cascade_net.stg3_full_band_net.requires_grad_(False)
        self.stg3_full_band_net.perceptual_loss = True
        self.stg3_full_band_net.skip_decoder = True

    def forward(self, x, y):
        x1a, x1b, x2a, x2b, x3 = self.calculate_encoded(x)
        y1a, y1b, y2a, y2b, y3 = self.calculate_encoded(y)
        return (F.l1_loss(x1a, y1a) + F.l1_loss(x1b, y1b) + F.l1_loss(x2a, y2a) + F.l1_loss(x2b, y2b) + F.l1_loss(x3, y3)) * 0.1

    def calculate_encoded(self, x):
        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1, p1l = self.stg1_low_band_net(l1_in)
        l1 = self.stg1_low_band_bn(l1)
        h1, p1h = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2, p2l = self.stg2_low_band_net(l2_in)
        l2 = self.stg2_low_band_bn(l2)
        h2, p2h = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        p3 = self.stg3_full_band_net(f3_in)

        return p1l, p1h, p2l, p2h, p3