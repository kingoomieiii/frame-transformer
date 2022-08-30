import math
import torch
import torch.nn as nn
import numpy as np

from frame_transformer import FrameDecoder, FrameEncoder, FrameTransformerEncoder

# adapted from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

class FrameQuantizer(nn.Module):
    def __init__(self, channels, bins, num_embeddings, beta=0.25):
        super().__init__()

        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, bins * channels)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        b,c,h,w = z.shape

        z = z.permute(0,3,1,2).reshape(b,w,c*h).contiguous()

        z_flattened = z.view(-1, c*h)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened,  self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.reshape(b,w,c,h).permute(0,2,3,1).contiguous()

        min_encoding_indices = min_encoding_indices.reshape(b, w)

        return z_q, loss, min_encoding_indices

    def encode(self, x):       
        b,c,h,w = z.shape
        z = z.permute(0,3,1,2).reshape(b,w,c*h).contiguous()
        z_flattened = z.view(-1, c*h)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened,  self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).reshape(b,w)

        return min_encoding_indices

    def decode(self, indices, channels, features):
        b,w = indices.shape[0], indices.shape[1]
        z_q = self.embedding(indices).reshape(b,w,channels,features).permute(0,2,3,1).contiguous()

        return z_q

class VQFrameTransformer(nn.Module):
    def __init__(self, in_channels=2, channels=2, dropout=0.1, n_fft=2048, num_heads=4, expansion=2, num_embeddings=2048):
        super(VQFrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.enc1 = FrameEncoder(in_channels, channels, self.max_bin, downsample=False, expansion=expansion)
        self.enc1_transformer = FrameTransformerEncoder(channels, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc2 = FrameEncoder(channels, channels * 2, self.max_bin, expansion=expansion)
        self.enc2_transformer = FrameTransformerEncoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc3 = FrameEncoder(channels * 2, channels * 4, self.max_bin // 2, expansion=expansion)
        self.enc3_transformer = FrameTransformerEncoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc4 = FrameEncoder(channels * 4, channels * 6, self.max_bin // 4, expansion=expansion)
        self.enc4_transformer = FrameTransformerEncoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.enc5 = FrameEncoder(channels * 6, channels * 8, self.max_bin // 8, expansion=expansion)
        self.enc5_transformer = FrameTransformerEncoder(channels * 8, self.max_bin // 16, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.fq = FrameQuantizer(channels * 8, self.max_bin // 16, num_embeddings)

        self.dec4 = FrameDecoder(channels * 8, channels * 6, self.max_bin // 8, expansion=expansion, has_skip=False)
        self.dec4_transformer = FrameTransformerEncoder(channels * 6, self.max_bin // 8, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec3 = FrameDecoder(channels * 6, channels * 4, self.max_bin // 4, expansion=expansion, has_skip=False)
        self.dec3_transformer = FrameTransformerEncoder(channels * 4, self.max_bin // 4, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec2 = FrameDecoder(channels * 4, channels * 2, self.max_bin // 2, expansion=expansion, has_skip=False)
        self.dec2_transformer = FrameTransformerEncoder(channels * 2, self.max_bin // 2, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.dec1 = FrameDecoder(channels * 2, channels * 1, self.max_bin, expansion=expansion, has_skip=False)
        self.dec1_transformer = FrameTransformerEncoder(channels * 1, self.max_bin, num_heads=num_heads, dropout=dropout, expansion=expansion)

        self.out = nn.Parameter(torch.empty(in_channels, channels)) if in_channels != channels else None
        nn.init.uniform_(self.out, a=-1/math.sqrt(in_channels), b=1/math.sqrt(in_channels))

    def __call__(self, x):
        e1 = self.enc1_transformer(self.enc1(x))
        e2 = self.enc2_transformer(self.enc2(e1))
        e3 = self.enc3_transformer(self.enc3(e2))
        e4 = self.enc4_transformer(self.enc4(e3))
        e5 = self.enc5_transformer(self.enc5(e4))
        q, ql, _ = self.fq(e5)
        d4 = self.dec4_transformer(self.dec4(q))
        d3 = self.dec3_transformer(self.dec3(d4))
        d2 = self.dec2_transformer(self.dec2(d3))
        d1 = self.dec1_transformer(self.dec1(d2))
 
        if self.out is not None:
            out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)

        return torch.sigmoid(out), ql

    def encode(self, x):
        e1 = self.enc1_transformer(self.enc1(x))
        e2 = self.enc2_transformer(self.enc2(e1))
        e3 = self.enc3_transformer(self.enc3(e2))
        e4 = self.enc4_transformer(self.enc4(e3))
        e5 = self.enc5_transformer(self.enc5(e4))
        _, _, indices = self.fq(e5)

        return indices

    def decode(self, x):
        d4 = self.dec4_transformer(self.dec4(self.fq.decode(x)))
        d3 = self.dec3_transformer(self.dec3(d4))
        d2 = self.dec2_transformer(self.dec2(d3))
        d1 = self.dec1_transformer(self.dec1(d2))
 
        if self.out is not None:
            out = torch.matmul(d1.transpose(1,3), self.out.t()).transpose(1,3)

        return torch.sigmoid(out)