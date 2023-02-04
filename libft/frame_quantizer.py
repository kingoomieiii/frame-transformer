import math
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

# adapted from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py

class FrameQuantizer(nn.Module):
    def __init__(self, bins, num_embeddings, beta=0.25):
        super().__init__()

        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, bins)
        self.embedding.weight.data.uniform_()

    def forward(self, z):
        b,c,h,w = z.shape
        
        z = z.permute(0,3,1,2).reshape(b,w,c,h).contiguous()

        z_flattened = z.view(-1, h)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened,  self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q - z.detach()) ** 2) + 0.25 * torch.mean((z_q.detach() - z) ** 2)
        z_q = z_q.reshape(b,w,c,h).permute(0,2,3,1).contiguous()

        min_encoding_indices = min_encoding_indices.reshape(b,w,c).transpose(1,2)

        return z_q, loss, min_encoding_indices

    def encode(self, x):       
        b,c,h,w = z.shape
        z = z.permute(0,3,1,2).reshape(b,w,c,h).contiguous()
        z_flattened = z.view(-1, h)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened,  self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).reshape(b,w,c)

        return min_encoding_indices

    def decode(self, indices, channels, features):
        b,w,c = indices.shape[0], indices.shape[1]
        z_q = self.embedding(indices).reshape(b,w,c,features).permute(0,2,3,1).contiguous()

        return z_q