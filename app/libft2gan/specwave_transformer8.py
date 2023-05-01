import math
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T

from libft2gan.convolutional_multihead_attention import ConvolutionalMultiheadAttention
from libft2gan.multichannel_multihead_attention import MultichannelMultiheadAttention2
from libft2gan.multichannel_layernorm import MultichannelLayerNorm
from libft2gan.multichannel_linear import MultichannelLinear
from libft2gan.frame_conv import FrameConv
from libft2gan.convolutional_embedding import ConvolutionalEmbedding
from libft2gan.res_block import ResBlock, ResBlock1d
from libft2gan.squared_relu import SquaredReLU
from libft2gan.channel_norm import ChannelNorm

class SpecWaveTransformer(nn.Module):
    def __init__(self, wave_in_channels=2, frame_in_channels=4, frame_out_channels=2, wave_out_channels=2, wave_channels=8, frame_channels=8, dropout=0.1, n_fft=2048, hop_length=1024, wave_embedding=256, wave_heads=8, wave_expansion=4, frame_heads=8, frame_expansion=4, num_attention_maps=1, n_mels=128):
        super(SpecWaveTransformer, self).__init__(),
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.wave_embedding = wave_embedding
        encoder_layers = 4

        self.to_wave = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        self.to_spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None, return_complex=True)
        self.to_mel = T.MelScale(n_mels=n_mels, sample_rate=44100, n_stft=self.output_bin)

        self.encode_wave1 = Encoder(wave_in_channels, wave_channels, 1, downsample=True, stride=(1,2), kernel_size=(1,3), padding=(0,1), channel_norm=True)
        self.encode_wave2 = Encoder(wave_channels, wave_channels * 2, 1, downsample=True, stride=(2,1), kernel_size=(1,3), padding=(0,1), channel_norm=True)
        self.encode_wave = nn.Sequential(*[MultichannelTransformerEncoder(wave_channels * 2, num_attention_maps, wave_embedding, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads) for _ in range(encoder_layers)])  
        self.encode_mel = nn.Sequential(*[MultichannelTransformerEncoder(frame_in_channels, num_attention_maps, n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads, kernel_size=3, padding=1) for _ in range(encoder_layers)])
        
        self.frame_enc1 = Encoder(wave_in_channels * 2, frame_channels * 1, self.max_bin, downsample=False)
        self.enc1_wave_transformer = MultichannelTransformerEncoder(frame_channels * 1, num_attention_maps, self.max_bin, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)

        self.frame_enc2 = Encoder(frame_channels * 1, frame_channels * 2, self.max_bin)
        self.enc2_wave_transformer = MultichannelTransformerEncoder(frame_channels * 2, num_attention_maps, self.max_bin // 2, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)

        self.frame_enc3 = Encoder(frame_channels * 2, frame_channels * 4, self.max_bin // 2)
        self.enc3_wave_transformer = MultichannelTransformerEncoder(frame_channels * 4, num_attention_maps, self.max_bin // 4, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)

        self.frame_enc4 = Encoder(frame_channels * 4, frame_channels * 6, self.max_bin // 4)
        self.enc4_wave_transformer = MultichannelTransformerEncoder(frame_channels * 6, num_attention_maps, self.max_bin // 8, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)

        self.frame_enc5 = Encoder(frame_channels * 6, frame_channels * 8, self.max_bin // 8)
        self.enc5_wave_transformer = MultichannelTransformerEncoder(frame_channels * 8, num_attention_maps, self.max_bin // 16, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)

        self.frame_enc6 = Encoder(frame_channels * 8, frame_channels * 10, self.max_bin // 16)
        self.enc6_wave_transformer = MultichannelTransformerEncoder(frame_channels * 10, num_attention_maps, self.max_bin // 32, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)

        self.frame_enc7 = Encoder(frame_channels * 10, frame_channels * 12, self.max_bin // 32)
        self.enc7_wave_transformer = MultichannelTransformerEncoder(frame_channels * 12, num_attention_maps, self.max_bin // 64, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)

        self.frame_enc8 = Encoder(frame_channels * 12, frame_channels * 14, self.max_bin // 64)
        self.enc8_wave_transformer = MultichannelTransformerEncoder(frame_channels * 14, num_attention_maps, self.max_bin // 128, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=wave_expansion, num_heads=wave_heads)
        
        self.frame_dec7 = Decoder(frame_channels * 14, frame_channels * 12, self.max_bin // 64, dropout=0.5)
        self.dec7_wave_transformer = MultichannelTransformerDecoder(frame_channels * 12, num_attention_maps, self.max_bin // 64, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=frame_expansion, num_heads=frame_heads)

        self.frame_dec6 = Decoder(frame_channels * 12, frame_channels * 10, self.max_bin // 32, dropout=0.5)
        self.dec6_wave_transformer = MultichannelTransformerDecoder(frame_channels * 10, num_attention_maps, self.max_bin // 32, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=frame_expansion, num_heads=frame_heads)
        
        self.frame_dec5 = Decoder(frame_channels * 10, frame_channels * 8, self.max_bin // 16, dropout=0.5)
        self.dec5_wave_transformer = MultichannelTransformerDecoder(frame_channels * 8, num_attention_maps, self.max_bin // 16, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=frame_expansion, num_heads=frame_heads)
        
        self.frame_dec4 = Decoder(frame_channels * 8, frame_channels * 6, self.max_bin // 8)
        self.dec4_wave_transformer = MultichannelTransformerDecoder(frame_channels * 6, num_attention_maps, self.max_bin // 8, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=frame_expansion, num_heads=frame_heads)
        
        self.frame_dec3 = Decoder(frame_channels * 6, frame_channels * 4, self.max_bin // 4)
        self.dec3_wave_transformer = MultichannelTransformerDecoder(frame_channels * 4, num_attention_maps, self.max_bin // 4, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=frame_expansion, num_heads=frame_heads)

        self.frame_dec2 = Decoder(frame_channels * 4, frame_channels * 2, self.max_bin // 2)
        self.dec2_wave_transformer = MultichannelTransformerDecoder(frame_channels * 2, num_attention_maps, self.max_bin // 2, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=frame_expansion, num_heads=frame_heads)
        
        self.frame_dec1 = Decoder(frame_channels * 2, frame_channels * 1, self.max_bin // 1)
        self.dec1_wave_transformer = MultichannelTransformerDecoder(frame_channels * 1, num_attention_maps, self.max_bin, mem_channels=wave_channels * 2, mem_features=self.wave_embedding, mem2_channels=wave_in_channels, mem2_features=n_mels, dropout=dropout, expansion=frame_expansion, num_heads=frame_heads)
        
        self.spec_out = nn.Conv2d(frame_channels, wave_in_channels * 2, kernel_size=1)
        self.wave_out = nn.Conv2d(frame_channels, wave_out_channels, kernel_size=1)
        self.frame_out = nn.Conv2d(frame_channels, frame_out_channels, kernel_size=1)
        
    def forward(self, w, c):
        s = self.to_spec(w)
        p = (torch.angle(s) + torch.pi) / (2 * torch.pi)
        s = torch.abs(s)
        m = self.to_mel(s) / c
        s = s / c
        s = s[:, :, :-1]
        p = p[:, :, :-1]

        hw = self.encode_wave1(w.unsqueeze(2))
        hw = self.encode_wave2(hw)
        hw = hw.reshape(hw.shape[0], hw.shape[1], self.wave_embedding, hw.shape[3] // self.wave_embedding)

        hwqk1, mqk1 = None, None
        for i in range(len(self.encode_wave)):
            hw, hwqk1, _, _, _ = self.encode_wave[i](hw, prev_qk1=hwqk1)
            m, mqk1, _, _, _ = self.encode_mel[i](m, prev_qk1=mqk1)

        e1w = self.frame_enc1(torch.cat((s, p), dim=1))
        e1w, qk1w1, qk1w2, qk1w3, qk1w4 = self.enc1_wave_transformer(e1w, cross=hw, cross2=m)

        e2w = self.frame_enc2(e1w)
        e2w, qk2w1, qk2w2, qk2w3, qk2w4 = self.enc2_wave_transformer(e2w, cross=hw, cross2=m, prev_qk1=qk1w1, prev_qk2=qk1w2, prev_qk3=qk1w3, prev_qk4=qk1w4)

        e3w = self.frame_enc3(e2w)
        e3w, qk3w1, qk3w2, qk3w3, qk3w4 = self.enc3_wave_transformer(e3w, cross=hw, cross2=m, prev_qk1=qk2w1, prev_qk2=qk2w2, prev_qk3=qk2w3, prev_qk4=qk2w4)

        e4w = self.frame_enc4(e3w)
        e4w, qk4w1, qk4w2, qk4w3, qk4w4 = self.enc4_wave_transformer(e4w, cross=hw, cross2=m, prev_qk1=qk3w1, prev_qk2=qk3w2, prev_qk3=qk3w3, prev_qk4=qk3w4)

        e5w = self.frame_enc5(e4w)
        e5w, qk5w1, qk5w2, qk5w3, qk5w4 = self.enc5_wave_transformer(e5w, cross=hw, cross2=m, prev_qk1=qk4w1, prev_qk2=qk4w2, prev_qk3=qk4w3, prev_qk4=qk4w4)

        e6w = self.frame_enc6(e5w)
        e6w, qk6w1, qk6w2, qk6w3, qk6w4 = self.enc6_wave_transformer(e6w, cross=hw, cross2=m, prev_qk1=qk5w1, prev_qk2=qk5w2, prev_qk3=qk5w3, prev_qk4=qk5w4)

        e7w = self.frame_enc7(e6w)
        e7w, qk7w1, qk7w2, qk7w3, qk7w4 = self.enc7_wave_transformer(e7w, cross=hw, cross2=m, prev_qk1=qk6w1, prev_qk2=qk6w2, prev_qk3=qk6w3, prev_qk4=qk6w4)

        e8w = self.frame_enc8(e7w)
        e8w, qk8w1, qk8w2, qk8w3, qk8w4 = self.enc8_wave_transformer(e8w, cross=hw, cross2=m, prev_qk1=qk7w1, prev_qk2=qk7w2, prev_qk3=qk7w3, prev_qk4=qk7w4)

        h = self.frame_dec7(e8w, e7w)
        h, pqk1w, pqk2w, pqk3w, pqk4w = self.dec7_wave_transformer(h, skip=e7w, cross=hw, cross2=m, prev_qk1=qk8w1, prev_qk2=qk8w2, prev_qk3=qk8w3, prev_qk4=qk8w4, skip_qk=qk7w1)

        h = self.frame_dec6(h, e6w)
        h, pqk1w, pqk2w, pqk3w, pqk4w = self.dec6_wave_transformer(h, skip=e6w, cross=hw, cross2=m, prev_qk1=pqk1w, prev_qk2=pqk2w, prev_qk3=pqk3w, prev_qk4=pqk4w, skip_qk=qk6w1)

        h = self.frame_dec5(h, e5w)
        h, pqk1w, pqk2w, pqk3w, pqk4w = self.dec5_wave_transformer(h, skip=e5w, cross=hw, cross2=m, prev_qk1=pqk1w, prev_qk2=pqk2w, prev_qk3=pqk3w, prev_qk4=pqk4w, skip_qk=qk5w1)

        h = self.frame_dec4(h, e4w)
        h, pqk1w, pqk2w, pqk3w, pqk4w = self.dec4_wave_transformer(h, skip=e4w, cross=hw, cross2=m, prev_qk1=pqk1w, prev_qk2=pqk2w, prev_qk3=pqk3w, prev_qk4=pqk4w, skip_qk=qk4w1)

        h = self.frame_dec3(h, e3w)
        h, pqk1w, pqk2w, pqk3w, pqk4w = self.dec3_wave_transformer(h, skip=e3w, cross=hw, cross2=m, prev_qk1=pqk1w, prev_qk2=pqk2w, prev_qk3=pqk3w, prev_qk4=pqk4w, skip_qk=qk3w1)

        h = self.frame_dec2(h, e2w)
        h, pqk1w, pqk2w, pqk3w, pqk4w = self.dec2_wave_transformer(h, skip=e2w, cross=hw, cross2=m, prev_qk1=pqk1w, prev_qk2=pqk2w, prev_qk3=pqk3w, prev_qk4=pqk4w, skip_qk=qk2w1)

        h = self.frame_dec1(h, e1w)
        h, pqk1w, pqk2w, pqk3w, pqk4w = self.dec1_wave_transformer(h, skip=e1w, cross=hw, cross2=m, prev_qk1=pqk1w, prev_qk2=pqk2w, prev_qk3=pqk3w, prev_qk4=pqk4w, skip_qk=qk1w1)

        h = self.spec_out(h)

        mag = s * torch.sigmoid(h[:, :2])
        phase = p + (torch.sigmoid(h[:, 2:]) * 2 - 1)
        phase = (phase * 2 - 1) * torch.pi
        spec = mag * c * torch.exp(1.j * phase)
        spec = F.pad(input=spec, pad=(0, 0, 0, 1), mode='replicate')
        nw = self.to_wave(spec)

        return nw, s, p
    
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
    
class MultichannelTransformerEncoder(nn.Module):
    def __init__(self, channels, out_channels, features, mem_channels=None, mem_features=None, mem2_channels=None, mem2_features=None, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1):
        super(MultichannelTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding)

        if mem_channels is not None:
            self.norm2 = MultichannelLayerNorm(channels, features)
            self.attn2 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding, mem_channels=mem_channels, mem_features=mem_features, mem_kernel_size=(5,1), mem_padding=(2,0))

            self.norm3 = MultichannelLayerNorm(channels, features)
            self.attn3 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, kernel_size=kernel_size, padding=padding, mem_channels=mem2_channels, mem_features=mem2_features, mem_kernel_size=5, mem_padding=2)

        self.norm5 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, cross=None, cross2=None, prev_qk1=None, prev_qk2=None, prev_qk3=None, prev_qk4=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1)
        h = x + self.dropout(z)

        if cross is not None:
            z, prev_qk2 = self.attn2(self.norm2(h), mem=cross, prev_qk=prev_qk2)
            h = h + self.dropout(z)

            z, prev_qk3 = self.attn3(self.norm3(h), mem=cross2, prev_qk=prev_qk3)
            h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm5(h))))
        h = h + self.dropout(z)

        return h, prev_qk1, prev_qk2, prev_qk3, prev_qk4
        
class MultichannelTransformerDecoder(nn.Module):
    def __init__(self, channels, out_channels, features, mem_channels, mem_features, mem2_channels, mem2_features, dropout=0.1, expansion=4, num_heads=8, has_prev_skip=True):
        super(MultichannelTransformerDecoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = MultichannelLayerNorm(channels, features)
        self.attn1 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features)

        self.norm2 = MultichannelLayerNorm(channels, features)
        self.attn2 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features)

        self.norm3 = MultichannelLayerNorm(channels, features)
        self.attn3 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, mem_channels=mem_channels, mem_features=mem_features, mem_kernel_size=5, mem_padding=2)

        self.norm4 = MultichannelLayerNorm(channels, features)
        self.attn4 = MultichannelMultiheadAttention2(channels, out_channels, num_heads, features, mem_channels=mem2_channels, mem_features=mem2_features, mem_kernel_size=5, mem_padding=2)

        self.norm6 = MultichannelLayerNorm(channels, features)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, skip, cross, cross2, prev_qk1=None, prev_qk2=None, prev_qk3=None, prev_qk4=None, skip_qk=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1)
        h = x + self.dropout(z)

        z, _ = self.attn2(self.norm2(h), mem=skip, prev_qk=skip_qk)
        h = h + self.dropout(z)

        z, prev_qk2 = self.attn3(self.norm3(h), mem=cross, prev_qk=prev_qk2)
        h = h + self.dropout(z)

        z, prev_qk3 = self.attn4(self.norm4(h), mem=cross2, prev_qk=prev_qk3)
        h = h + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm6(h))))
        h = h + self.dropout(z)

        return h, prev_qk1, prev_qk2, prev_qk3, prev_qk4
        
class ConvolutionalTransformerEncoder(nn.Module):
    def __init__(self, channels, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1):
        super(ConvolutionalTransformerEncoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout2d(dropout)

        self.norm1 = ChannelNorm(channels)
        self.attn = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding)

        self.norm2 = ChannelNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, prev_qk=None):
        z, prev_qk = self.attn(self.norm1(x), prev_qk=prev_qk)
        h = x + self.dropout(z)

        z = self.conv2(self.activate(self.conv1(self.norm2(h))))
        h = h + self.dropout(z)

        return h, prev_qk
        
class ConvolutionalTransformerDecoder(nn.Module):
    def __init__(self, channels, dropout=0.1, expansion=4, num_heads=8, kernel_size=3, padding=1, mem_channels=None):
        super(ConvolutionalTransformerDecoder, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout2d(dropout)

        self.norm1 = ChannelNorm(channels)
        self.attn1 = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding)

        self.norm2 = ChannelNorm(channels)
        self.attn2 = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding, mem_channels=mem_channels)

        self.norm3 = ChannelNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1)
        
    def forward(self, x, cross, prev_qk1=None, prev_qk2=None):
        z, prev_qk1 = self.attn1(self.norm1(x), prev_qk=prev_qk1)
        h = x + self.dropout(z)

        z, prev_qk2 = self.attn2(self.norm2(h), mem=cross, prev_qk=prev_qk2)
        h = h + self.dropout(z) 

        z = self.conv2(self.activate(self.conv1(self.norm3(h))))
        h = h + self.dropout(z)

        return h, prev_qk1, prev_qk2
        
class ConvolutionalTransformerDecoder2(nn.Module):
    def __init__(self, channels, features, mem_channels, mem_features, dropout=0.1, kernel_size=3, padding=1, expansion=4, num_heads=8):
        super(ConvolutionalTransformerDecoder2, self).__init__()

        self.activate = SquaredReLU()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = ChannelNorm(channels, features)
        self.attn1a = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding)
        self.attn1b = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding)

        self.norm2 = ChannelNorm(channels)
        self.conv1a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=11, padding=5, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0))
        
        self.conv1b = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0))
        
        self.norm3 = ChannelNorm(channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0))

        self.norm4 = ChannelNorm(channels)
        self.attn2 = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding)

        self.norm5 = ChannelNorm(channels)
        self.attn3 = ConvolutionalMultiheadAttention(channels, num_heads, kernel_size=kernel_size, padding=padding, mem_channels=mem_channels)

        self.norm6 = ChannelNorm(channels)
        self.conv3 = nn.Conv2d(channels, channels * expansion, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(channels * expansion, channels, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x, skip, cross, prev_qk1=None, prev_qk2=None, skip_qk=None):
        h = self.embed(x)

        z = self.norm1(h)
        za, prev_qk1 = self.attn1a(z, prev_qk=prev_qk1)
        zb, _ = self.attn1b(z, mem=skip, prev_qk=skip_qk)
        h = h + self.dropout(za) + self.dropout(zb)

        z = self.norm2(h)
        za = self.activate(self.conv1a(z))
        zb = self.conv1b(z)
        z = self.conv2(self.norm3(za + zb))
        h = h + self.dropout(z)

        z, prev_qk1 = self.attn2(self.norm4(h), prev_qk=prev_qk1)
        h = h + self.dropout(z)

        z, _ = self.attn3(self.norm5(h), mem=skip, prev_qk=skip_qk)
        h = h + self.dropout(z)

        z = self.conv4(self.activate(self.conv3(self.norm6(h))))
        h = h + self.dropout(z)

        return torch.cat((x, h), dim=1), prev_qk1, prev_qk2