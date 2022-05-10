import torch
from torch import nn
import torch.nn.functional as F
import math

class FrameTransformer(nn.Module):
    def __init__(self, channels, n_fft=2048, feedforward_dim=512, num_bands=4, num_encoders=1, num_decoders=1, cropsize=1024, bias=False, autoregressive=True, out_activate=nn.Sigmoid()):
        super(FrameTransformer, self).__init__()
        
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        
        self.register_buffer('mask', torch.triu(torch.ones(cropsize, cropsize) * float('-inf'), diagonal=1))
        self.autoregressive = autoregressive
        self.out_activate = out_activate
        self.out_transformer = nn.ModuleList([FrameTransformerEncoder(channels + i, self.max_bin, num_bands, cropsize, n_fft, downsamples=0, feedforward_dim=feedforward_dim, bias=bias) for i in range(num_decoders)])

    def __call__(self, x):
        x = x[:, :, :self.max_bin]

        for module in self.out_transformer:
            t = module(x, mask=self.mask if self.autoregressive else None)
            x = torch.cat((x, t), dim=1)

        return F.pad(
            input=self.out_activate(x[:, -2:]),
            pad=(0, 0, 0, self.output_bin - self.max_bin),
            mode='replicate'
        )
        
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.conv1d(F.pad(x, (self.kernel_size - 1, 0)), weight=self.weight, bias=self.bias, stride=self.stride, groups=self.groups)

class FrameTransformerEncoder(nn.Module):
    def __init__(self, channels, bins, num_bands=4, cropsize=1024, n_fft=2048, feedforward_dim=2048, downsamples=0, bias=False, dropout=0.1):
        super(FrameTransformerEncoder, self).__init__()

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.in_norm = nn.LayerNorm((bins, channels))
        self.in_project = nn.Linear(channels, 1, bias=bias)

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(bins)
        self.glu = nn.Sequential(
            nn.Linear(bins, bins * 2, bias=bias),
            nn.GLU())

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Linear(bins, feedforward_dim, bias=bias)
        self.conv1R = nn.Conv1d(bins, feedforward_dim//4, kernel_size=3, padding=1, bias=bias)
        self.norm3 = nn.LayerNorm(feedforward_dim)
        self.conv1M = nn.Sequential(
            nn.Conv1d(feedforward_dim, feedforward_dim, kernel_size=9, padding=4, groups=feedforward_dim, bias=bias),
            nn.Conv1d(feedforward_dim, bins, kernel_size=1, padding=0, bias=bias))

        self.norm4 = nn.LayerNorm(bins)
        self.attn = MultibandFrameAttention(num_bands, bins, cropsize, kernel_size=7, padding=3)

        self.norm5 = nn.LayerNorm(bins)
        self.conv2 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.conv3 = nn.Linear(feedforward_dim, bins, bias=bias)

    def __call__(self, x, mask=None):
        x = self.in_project(self.relu(self.in_norm(x.transpose(1,3)))).squeeze(3)

        h = self.norm1(x)
        h = self.glu(h)
        x = x + h

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h))
        hR = self.relu(self.conv1R(h.transpose(1,2))).transpose(1,2)

        h = self.norm3(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))
        h = self.conv1M(h.transpose(1,2)).transpose(1,2)
        x = x + h

        h = self.norm4(x)
        h = self.attn(h, mask=mask)
        x = x + h

        h = self.norm5(x)
        h = self.conv3(torch.square(self.relu(self.conv2(h))))
        x = x + h

        return x.transpose(1,2).unsqueeze(1)

class FrameTransformerDecoder(nn.Module):
    def __init__(self, channels, skip_channels, num_bands=4, cropsize=1024, n_fft=2048, feedforward_dim=2048, downsamples=0, bias=False, dropout=0.1):
        super(FrameTransformerDecoder, self).__init__()

        bins = get_bins(n_fft=n_fft, downsamples=downsamples)

        self.bins = bins
        self.cropsize = cropsize
        self.num_bands = num_bands

        self.in_norm = nn.LayerNorm((bins, channels))
        self.skip_norm = nn.LayerNorm((bins, channels))

        self.in_project = nn.Linear(channels, 1, bias=bias)
        self.skip_project = nn.Linear(skip_channels, 1, bias=bias)

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(bins)
        self.self_attn1 = MultibandFrameAttention(num_bands, bins, cropsize, kernel_size=7, padding=3)
        self.skip_attn1 = MultibandFrameAttention(num_bands, bins, cropsize, kernel_size=7, padding=3)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(bins)
        self.conv1L = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=11, padding=5, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 2, kernel_size=1, padding=0, bias=bias))
        self.conv1R = nn.Sequential(
            nn.Conv1d(bins, bins, kernel_size=7, padding=3, groups=bins, bias=bias),
            nn.Conv1d(bins, feedforward_dim // 4, kernel_size=1, padding=0, bias=bias))
        self.norm3 = nn.LayerNorm(feedforward_dim // 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(feedforward_dim // 2, feedforward_dim // 2, kernel_size=7, padding=3, groups=feedforward_dim // 2, bias=bias),
            nn.Conv1d(feedforward_dim // 2, bins, kernel_size=1, padding=0, bias=bias))
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm4 = nn.LayerNorm(bins)
        self.self_attn2 = MultibandFrameAttention(num_bands, bins, cropsize, kernel_size=7, padding=3)
        self.norm4 = nn.LayerNorm(bins)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm5 = nn.LayerNorm(bins)
        self.skip_attn2 = MultibandFrameAttention(num_bands, bins, cropsize, kernel_size=7, padding=3)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm6 = nn.LayerNorm(bins)
        self.conv3 = nn.Linear(bins, feedforward_dim, bias=bias)
        self.silu = nn.SiLU(inplace=True)
        self.conv4 = nn.Linear(feedforward_dim, bins, bias=bias)
        self.dropout5 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __call__(self, x, skip, mask=None):
        x = self.in_project(self.relu(self.in_norm(x.transpose(1,3)))).squeeze(3)
        skip = self.skip_project(self.relu(self.skip_norm(skip.transpose(1,3)))).squeeze(3)

        h = self.norm1(x)
        hs = self.self_attn1(h, mask=mask)
        hm = self.skip_attn1(h, mem=skip, mask=mask)
        x = x + self.dropout1(hs + hm)

        h = self.norm2(x)
        hL = self.relu(self.conv1L(h.transpose(1,2)).transpose(1,2))
        hR = self.conv1R(h.transpose(1,2)).transpose(1,2)

        h = self.norm3(hL + F.pad(hR, (0, hL.shape[2]-hR.shape[2])))
        h = self.dropout2(self.conv2(h.transpose(1,2)).transpose(1,2))
        x = x + h

        h = self.norm4(x)
        h = self.dropout3(self.self_attn2(h, mask=mask))
        x = x + h

        h = self.norm5(x)
        h = self.dropout4(self.skip_attn2(h, mem=skip, mask=mask))
        x = x + h

        h = self.norm6(x)
        h = torch.square(torch.relu(self.conv3(h)))
        h = self.dropout5(self.conv4(h))
        x = x + h

        return x.transpose(1,2).unsqueeze(1)

class MultibandFrameAttention(nn.Module):
    def __init__(self, num_bands, bins, cropsize, kernel_size=3, padding=1):
        super().__init__()

        self.num_bands = num_bands
        self.q_proj = nn.Linear(bins, bins)
        self.q_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins)

        self.k_proj = nn.Linear(bins, bins)
        self.k_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins)

        self.v_proj = nn.Linear(bins, bins)
        self.v_conv = nn.Conv1d(bins, bins, kernel_size=kernel_size, padding=padding, groups=bins)

        self.o_proj = nn.Linear(bins, bins)

        self.er = nn.Parameter(torch.empty(bins // num_bands, cropsize))
        nn.init.normal_(self.er)

    def forward(self, x, mem=None, mask=None):
        b,w,c = x.shape
        q = self.q_conv(self.q_proj(x).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        k = self.k_conv(self.k_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,3,1)
        v = self.v_conv(self.v_proj(x if mem is None else mem).transpose(1,2)).transpose(1,2).reshape(b, w, self.num_bands, -1).permute(0,2,1,3)
        p = F.pad(torch.matmul(q,self.er), (1,0)).transpose(2,3)[:,:,1:,:]
        qk = (torch.matmul(q,k)+p) / math.sqrt(c)

        if mask is not None:
            qk = qk + mask

        a = F.softmax(qk, dim=-1)
        a = torch.matmul(a,v).transpose(1,2).reshape(b,w,-1)
        o = self.o_proj(a)
        return o

class FrameNorm(nn.Module):
    def __init__(self, channels, cropsize=1024, n_fft=2048, downsamples=0):
        super(FrameNorm, self).__init__()

        bins = get_bins(n_fft, downsamples)
        self.norm = nn.LayerNorm((bins, channels))

    def __call__(self, x):
        return self.norm(x.transpose(1,3)).transpose(1,3)

def get_bins(n_fft, downsamples):    
    bins = (n_fft // 2)
    if downsamples > 0:
        for _ in range(downsamples):
            bins = ((bins - 1) // 2) + 1

    return bins