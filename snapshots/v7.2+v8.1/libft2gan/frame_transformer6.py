import torch
from torch import nn

from libft2gan.multichannel_multihead_attention import MultichannelMultiheadAttention
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.convolutional_embedding import ConvolutionalEmbedding
from libft2gan.res_block import ResBlock
from libft2ganv9.linear2d import Linear2d

class FrameTransformer(nn.Module):
    def __init__(self, in_channels=2, stacks=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, feature_expansion=2, channel_expansion=2, num_attention_maps=1, expansions=[4,4,3,3,4,4,5], num_layers=9, freeze_layers=2):
        super(FrameTransformer, self).__init__(),
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.stacks = stacks

        self.positional_embedding = ConvolutionalEmbedding(in_channels, self.max_bin)
        self.transformer = nn.Sequential(*[FrameTransformerEncoder(in_channels * stacks + 1, num_attention_maps, self.max_bin, dropout=dropout, feature_expansion=feature_expansion, channel_expansion=channel_expansion, num_heads=num_heads, prev_attn=i*num_attention_maps, freeze_layers=freeze_layers) for i in range(num_layers)])
        
    def forward(self, x):
        p = self.positional_embedding(x)
        h = torch.cat([*[x for _ in range(self.stacks)], p], dim=1)

        pattn, pqk = None, None
        for encoder in self.transformer:
            h, pattn, pqk = encoder(h, prev_attn=pattn, prev_qk=pqk)
 
        return h[:, :2, :, :]
        
class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, feature_expansion=4, channel_expansion=4, num_heads=8, kernel_size=3, padding=1, prev_attn=0, freeze_layers=2):
        super(FrameTransformerEncoder, self).__init__()

        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn = MultichannelMultiheadAttention(channels + prev_attn, out_channels, num_heads, features, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(channels + out_channels + prev_attn, int((channels + out_channels) * channel_expansion), kernel_size=kernel_size, padding=padding, bias=False)
        self.linear1 = Linear2d(int((channels + out_channels) * channel_expansion), features, int(features * feature_expansion), bias=False)
        self.linear2 = Linear2d(int((channels + out_channels) * channel_expansion), int(features * feature_expansion), features, bias=False)
        self.conv2 = nn.Conv2d(int((channels + out_channels) * channel_expansion), channels, kernel_size=kernel_size, padding=padding, bias=False)

        self.residual_weight = nn.Parameter(torch.ones(channels, features, 1))
        self.residual_weight.data[:freeze_layers] = 0
        
    def forward(self, x, prev_attn=None, prev_qk=None):
        h = self.norm1(x)
        a, prev_qk = self.attn(torch.cat((h, prev_attn), dim=1) if prev_attn is not None else h, prev_qk=prev_qk)
        z = self.conv2(self.linear2(self.activate(self.linear1(self.conv1(torch.cat((h, a, prev_attn), dim=1) if prev_attn is not None else torch.cat((h, a), dim=1))))))
        h = x + self.residual_weight * self.dropout(z)

        return h, torch.cat((prev_attn, a), dim=1) if prev_attn is not None else a, prev_qk