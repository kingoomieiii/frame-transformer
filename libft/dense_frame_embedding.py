import torch
import torch.nn as nn

from libft.frame_quantizer import FrameQuantizer
from libft.res_block import ResBlock, LinearResBlock

class DenseFrameEmbedding(nn.Module):
    def __init__(self, channels, num_quantizers, bins, num_embeddings):
        super().__init__()
        
        self.num_quantizers = num_quantizers
        self.project = ResBlock(2, 2 * num_quantizers, bins, kernel_size=(9,1), padding=(4,0), expansion=1)
        self.quantizers = nn.Sequential(*[FrameQuantizer(bins, num_embeddings) for _ in range(2 * num_quantizers)])

    def forward(self, x):
        h = self.project(x)
        h = h.reshape((h.shape[0], 1, h.shape[1], h.shape[2], h.shape[3]))

        embedded, loss = None, None
        for i in range(len(self.quantizers)):
            q, ql, _ = self.quantizers[i](h[:, :, i])
            embedded = q if embedded is None else torch.cat((q, embedded), dim=1)
            loss = ql if loss is None else loss + ql
            

        return embedded, loss