# inspired from https://arxiv.org/abs/2001.08248

import torch
import torch.nn as nn
import torch.nn.functional as F

from libft.res_block import ResBlock

class PositionalEmbedding(nn.Module):
    def __init__(self, channels, features):
        super(PositionalEmbedding, self).__init__()

        self.extract1 = ResBlock(channels, 1, features, kernel_size=9, padding=4)
        self.extract2 = ResBlock(channels * 2, 1, features // 2, kernel_size=9, padding=4)
        self.extract3 = ResBlock(channels * 4, 1, features // 4, kernel_size=9, padding=4)
        self.extract4 = ResBlock(channels * 6, 1, features // 8, kernel_size=9, padding=4)
        self.extract5 = ResBlock(channels * 8, 1, features // 16, kernel_size=9, padding=4)
        self.extract6 = ResBlock(channels * 10, 1, features // 32, kernel_size=9, padding=4)
        self.extract7 = ResBlock(channels * 12, 1, features // 64, kernel_size=9, padding=4)
        self.extract8 = ResBlock(channels * 14, 1, features // 128, kernel_size=9, padding=4)
        self.extract9 = ResBlock(channels * 16, 1, features // 256, kernel_size=9, padding=4)

        self.encoder1 = ResBlock(channels, channels * 2, features, True)
        self.encoder2 = ResBlock(channels * 2, channels * 4, features // 2, True)
        self.encoder3 = ResBlock(channels * 4, channels * 6, features // 4, True)
        self.encoder4 = ResBlock(channels * 6, channels * 8, features // 8, True)
        self.encoder5 = ResBlock(channels * 8, channels * 10, features // 16, True)
        self.encoder6 = ResBlock(channels * 10, channels * 12, features // 32, True)
        self.encoder7 = ResBlock(channels * 12, channels * 14, features // 64, True)
        self.encoder8 = ResBlock(channels * 14, channels * 16, features // 128, True)

        self.out = nn.Conv2d(9, 1, kernel_size=1, padding=0)

    def __call__(self, x):
        e1 = self.extract1(x)
        h = self.encoder1(x)

        e2 = F.interpolate(self.extract2(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        h = self.encoder2(h)

        e3 = F.interpolate(self.extract3(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        h = self.encoder3(h)

        e4 = F.interpolate(self.extract4(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        h = self.encoder4(h)
        
        e5 = F.interpolate(self.extract5(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        h = self.encoder5(h)
        
        e6 = F.interpolate(self.extract6(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        h = self.encoder6(h)

        e7 = F.interpolate(self.extract7(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        h = self.encoder7(h)

        e8 = F.interpolate(self.extract8(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        h = self.encoder8(h)

        e9 = F.interpolate(self.extract9(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        return self.out(torch.cat((e1, e2, e3, e4, e5, e6, e7, e8, e9), dim=1))