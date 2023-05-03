import math
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as A

from libft2gan.rotary_embedding_torch import RotaryEmbedding
from libft2gan.convolutional_multihead_attention import ConvolutionalMultiheadAttention
from libft2gan.multichannel_multihead_attention import MultichannelMultiheadAttention2
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.frame_conv import FrameConv
from libft2gan.convolutional_embedding import ConvolutionalEmbedding
from libft2gan.res_block import ResBlock, ResBlock1d
from libft2gan.squared_relu import SquaredReLU
from libft2gan.channel_norm import ChannelNorm

from libft2gan.audio_scales import melscale_fbanks, octavescale_fbanks, linear_fbanks

class MelScale(nn.Module):
    def __init__(self, n_filters=128, sample_rate=44100, n_stft=1025, min_freq=0, max_freq=None, learned_filters=True):
        super().__init__()
        
        if learned_filters:
            self.fb = nn.Parameter(melscale_fbanks(n_stft, min_freq, max_freq if max_freq is not None else float(sample_rate // 2), n_mels=n_filters, sample_rate=sample_rate))
        else:
            self.register_buffer('fb', melscale_fbanks(n_stft, min_freq, max_freq if max_freq is not None else float(sample_rate // 2), n_mels=n_filters, sample_rate=sample_rate))

    def forward(self, x):
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

class OctaveScale(nn.Module):
    def __init__(self, n_filters=128, sample_rate=44100, n_stft=1025, learned_filters=True, limit_to_freqs=False, min_freq=0, max_freq=None):
        super().__init__()
        
        if learned_filters:
            self.fb = nn.Parameter(octavescale_fbanks(n_stft, n_filters=n_filters, sample_rate=sample_rate, limit_to_freqs=limit_to_freqs, f_min=min_freq, f_max=max_freq))
        else:
            self.register_buffer('fb', octavescale_fbanks(n_stft, n_filters=n_filters, sample_rate=sample_rate, limit_to_freqs=limit_to_freqs, f_min=min_freq, f_max=max_freq))

    def forward(self, x, reshape_as=None):
        x = torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

        if reshape_as is not None:
            x = x.reshape_as(reshape_as)

        return x

class BandScale(nn.Module):
    def __init__(self, n_filters, min_freq, max_freq, sample_rate=44100, n_stft=1025, learned_filters=True):
        super().__init__()
        
        if learned_filters:
            self.fb = nn.Parameter(linear_fbanks(n_stft, f_min=min_freq, f_max=max_freq, n_filters=n_filters, sample_rate=sample_rate))
        else:
            self.register_buffer('fb', linear_fbanks(n_stft, f_min=min_freq, f_max=max_freq, n_filters=n_filters, sample_rate=sample_rate))

    def forward(self, x):
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2)

class SpecWaveTransformer(nn.Module):
    def __init__(self, wave_in_channels=2, dropout=0.1, encoder_layers=8, encoder_heads=8, encoder_expansion=4, decoder_layers=8, decoder_heads=8, decoder_expansion=4, encoder_attention_maps=1, decoder_attention_maps=1, num_octave_maps=3, num_mel_maps=3, num_band_maps=3, n_mels=128, n_fft=2048, hop_length=1024, sr=44100, inferencing=False):
        super(SpecWaveTransformer, self).__init__(),
        
        self.inferencing = inferencing
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.to_spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.to_wave = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        self.num_octave_maps = num_octave_maps
        self.num_mel_maps = num_mel_maps
        self.num_band_maps = num_band_maps
        self.num_scale_channels = wave_in_channels * ((12 * self.num_band_maps) + self.num_mel_maps + self.num_octave_maps + 2)
        self.to_lbass = nn.Sequential(*[BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=16.0, max_freq=60.0) for _ in range(self.num_band_maps)])
        self.to_ubass = nn.Sequential(*[BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=60.0, max_freq=250.0) for _ in range(self.num_band_maps)])
        self.to_lmids = nn.Sequential(*[BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=250.0, max_freq=2000.0) for _ in range(self.num_band_maps)])
        self.to_umids = nn.Sequential(*[BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=2000.0, max_freq=4000.0) for _ in range(self.num_band_maps)])
        self.to_presence = nn.Sequential(*[BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=4000.0, max_freq=6000.0) for _ in range(self.num_band_maps)])
        self.to_brilliance = nn.Sequential(*[BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=6000.0, max_freq=16000.0) for _ in range(self.num_band_maps)])
        self.to_lbasso = nn.Sequential(*[OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=16.0, max_freq=60.0) for _ in range(self.num_band_maps)])
        self.to_ubasso = nn.Sequential(*[OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=60.0, max_freq=250.0) for _ in range(self.num_band_maps)])
        self.to_lmidso = nn.Sequential(*[OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=250.0, max_freq=2000.0) for _ in range(self.num_band_maps)])
        self.to_umidso = nn.Sequential(*[OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=2000.0, max_freq=4000.0) for _ in range(self.num_band_maps)])
        self.to_presenceo = nn.Sequential(*[OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=4000.0, max_freq=6000.0) for _ in range(self.num_band_maps)])
        self.to_brillianceo = nn.Sequential(*[OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=6000.0, max_freq=16000.0) for _ in range(self.num_band_maps)])
        # self.to_lbass2 = BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=16.0, max_freq=60.0, learned_filters=False)
        # self.to_ubass2 = BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=60.0, max_freq=250.0, learned_filters=False)
        # self.to_lmids2 = BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=250.0, max_freq=2000.0, learned_filters=False)
        # self.to_umids2 = BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=2000.0, max_freq=4000.0, learned_filters=False)
        # self.to_presence2 = BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=4000.0, max_freq=6000.0, learned_filters=False)
        # self.to_brilliance2 = BandScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, min_freq=6000.0, max_freq=16000.0, learned_filters=False)
        self.to_mel_fixed = MelScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, learned_filters=False)
        self.to_mel = nn.Sequential(*[MelScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin) for _ in range(self.num_mel_maps)])
        self.to_octave_fixed = OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, learned_filters=False, limit_to_freqs=True)
        self.to_octave = nn.Sequential(*[OctaveScale(n_filters=n_mels, sample_rate=sr, n_stft=self.output_bin, limit_to_freqs=True) for _ in range(self.num_octave_maps)])

        self.encode_scales = nn.Sequential(*[MultichannelTransformerEncoder(self.num_scale_channels, encoder_attention_maps, n_mels, dropout=dropout, expansion=encoder_expansion, num_heads=encoder_heads, kernel_size=3, padding=1) for _ in range(encoder_layers)])
        self.positional_embedding = ConvolutionalEmbedding(wave_in_channels * 2, self.max_bin)
        self.decode_spectrogram = nn.Sequential(*[MultichannelTransformerDecoder(wave_in_channels * 2 + 1, decoder_attention_maps, self.max_bin, mem_channels=self.num_scale_channels, mem_features=n_mels, dropout=dropout, expansion=decoder_expansion, num_heads=decoder_heads, kernel_size=3, padding=1 ) for _ in range(decoder_layers)])
        # self.mask_out = nn.Conv2d(wave_in_channels * 2, wave_in_channels, 1)

    def forward(self, w, c):
        s = self.to_spec(w)
        p = (torch.angle(s) + torch.pi) / (2 * torch.pi)
        s = torch.abs(s)

        m = torch.cat((
            *[self.to_lbass[i](s) for i in range(self.num_band_maps)],
            *[self.to_ubass[i](s) for i in range(self.num_band_maps)],
            *[self.to_lmids[i](s) for i in range(self.num_band_maps)],
            *[self.to_umids[i](s) for i in range(self.num_band_maps)],
            *[self.to_presence[i](s) for i in range(self.num_band_maps)],
            *[self.to_brilliance[i](s) for i in range(self.num_band_maps)],
            *[self.to_lbasso[i](s) for i in range(self.num_band_maps)],
            *[self.to_ubasso[i](s) for i in range(self.num_band_maps)],
            *[self.to_lmidso[i](s) for i in range(self.num_band_maps)],
            *[self.to_umidso[i](s) for i in range(self.num_band_maps)],
            *[self.to_presenceo[i](s) for i in range(self.num_band_maps)],
            *[self.to_brillianceo[i](s) for i in range(self.num_band_maps)],
            # self.to_lbass2(s),
            # self.to_ubass2(s),
            # self.to_lmids2(s),
            # self.to_umids2(s),
            # self.to_presence2(s),
            # self.to_brilliance2(s),
            self.to_mel_fixed(s),
            *[self.to_mel[i](s) for i in range(self.num_mel_maps)],
            self.to_octave_fixed(s),
            *[self.to_octave[i](s) for i in range(self.num_octave_maps)]
        ), dim=1) / c

        s = s[:, :, :-1] / c
        p = p[:, :, :-1]
        x = torch.cat((s, p), dim=1)
        x = torch.cat((x, self.positional_embedding(x)), dim=1)

        prev_qk1 = None
        for i in range(len(self.encode_scales)):
            m, prev_qk1 = self.encode_scales[i](m, prev_qk1=prev_qk1)
        
        prev_qk1, prev_qk2 = None, None
        for i in range(len(self.decode_spectrogram)):
            x, prev_qk1, prev_qk2 = self.decode_spectrogram[i](x, cross=m, prev_qk1=prev_qk1, prev_qk2=prev_qk2)

        mask = torch.sigmoid(x[:, :2])
        phase_mask = torch.sigmoid(x[:, 2:4]) * 2 - 1

        mag = s * mask
        phase = p + phase_mask

        mag_p = F.pad(input=mag * c, pad=(0, 0, 0, 1), mode='replicate')
        phase_p = (phase * 2 - 1) * torch.pi
        phase_p = F.pad(input=phase, pad=(0, 0, 0, 1), mode='replicate')
        spec = mag_p * torch.exp(1.j * phase_p)
        wave = self.to_wave(spec)

        mmin = torch.min(mag)
        mmax = torch.max(mag)
        mel = self.to_mel_fixed(F.pad(input=mag * c, pad=(0, 0, 0, 1), mode='replicate')) / c

        if self.inferencing:
            return wave
        else:
            return mag_p, phase, wave, mmin, mmax
    
class WaveEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(WaveEncoder, self).__init__()

        self.body = ResBlock1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.body(x)

        return x

class WaveDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(WaveDecoder, self).__init__()

        self.body = ResBlock1d(in_channels + out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2], mode='linear', align_corners=True)
        x = torch.cat((x, skip), dim=1)
        x = self.body(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, downsample=True, stride=(2,1), channel_norm=False, kernel_size=3, padding=1):
        super(Encoder, self).__init__()

        self.body = ResBlock(in_channels, out_channels, features, downsample=downsample, stride=stride, channel_norm=channel_norm, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.body(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, features, dropout=0, channel_norm=False, kernel_size=3, padding=1):
        super(Decoder, self).__init__()

        self.body = ResBlock(in_channels + out_channels, out_channels, features, dropout=dropout, channel_norm=channel_norm, kernel_size=kernel_size, padding=padding)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, skip), dim=1)
        x = self.body(x)

        return x

class MultichannelMultiheadAttention(nn.Module):
    def __init__(self, channels, attention_maps, num_heads, features, kernel_size=3, padding=1, mem_channels=None, mem_features=None, mem_kernel_size=None, mem_padding=None):
        super().__init__()

        self.attention_maps = attention_maps
        self.num_heads = num_heads
        self.embedding = RotaryEmbedding(features // num_heads)
        self.features = features

        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, attention_maps, kernel_size=kernel_size, padding=padding),
            MultichannelLinear(attention_maps, attention_maps, features, features))
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size if mem_kernel_size is None else mem_kernel_size, padding=padding if mem_padding is None else mem_padding),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features))
        
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels if mem_channels is None else mem_channels, attention_maps, kernel_size=kernel_size if mem_kernel_size is None else mem_kernel_size, padding=padding if mem_padding is None else mem_padding),
            MultichannelLinear(attention_maps, attention_maps, features if mem_features is None else mem_features, features))
        
        self.o_proj = MultichannelLinear(channels + attention_maps, channels, features, features)
        
    def forward(self, x, mem=None, prev_qk=None):
        b,c,h,w = x.shape
        w2 = w if mem is None else mem.shape[3]
        
        q = self.embedding.rotate_queries_or_keys(self.q_proj(x).transpose(2,3).reshape(b,self.attention_maps,w,self.num_heads,-1).permute(0,1,3,2,4))
        k = self.embedding.rotate_queries_or_keys(self.k_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w2,self.num_heads,-1).permute(0,1,3,2,4)).transpose(3,4)
        v = self.v_proj(x if mem is None else mem).transpose(2,3).reshape(b,self.attention_maps,w2,self.num_heads,-1).permute(0,1,3,2,4)
        qk = torch.matmul(q,k) / math.sqrt(h)

        if prev_qk is not None:
            qk = qk + prev_qk

        a = torch.matmul(F.softmax(qk, dim=-1),v).transpose(2,3).reshape(b,self.attention_maps,w,-1).transpose(2,3)
        x = self.o_proj(torch.cat((x, a), dim=1))

        return x, qk
    
class MultichannelTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1):
        super(MultichannelTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding)

        self.norm5 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, prev_qk1=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1)
        h = x + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm5(h))))
        h = h + self.dropout(z)

        return h, prev_qk1
        
class MultichannelTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, mem_channels, mem_features, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1):
        super(MultichannelTransformerDecoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding)

        self.norm3 = MultichannelLayerNorm(channels, features)
        self.attn3 = MultichannelMultiheadAttention(channels, out_channels, num_heads, features, mem_channels=mem_channels, mem_features=mem_features, kernel_size=kernel_size, padding=padding)

        self.norm6 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, cross, prev_qk1=None, prev_qk2=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1)
        h = x + self.dropout(z)

        z, prev_qk2 = self.attn3(self.norm3(h), mem=cross, prev_qk=prev_qk2)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm6(h))))
        h = h + self.dropout(z)

        return h, prev_qk1, prev_qk2