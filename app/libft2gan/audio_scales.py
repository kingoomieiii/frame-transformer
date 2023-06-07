
import math
import torch
import torch.nn as nn

def _hz_to_mel(freq: float):
    return 2595.0 * math.log10(1.0 + (freq / 700.0))

def _mel_to_hz(mels: torch.Tensor):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

def note_to_hertz(note, ref_note=69, ref_freq=440.0):
    n = note - ref_note
    return ref_freq * 2 ** (n / 12)

def hertz_to_note(hz, ref_note=69, ref_freq=440.0):
    note = ref_note + 12 * math.log2(hz / ref_freq)
    return round(note)

# condensed from torchaudio
def _create_triangular_filterbank(
    all_freqs,
    f_pts):
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb

# condensed from torchaudio
def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int
):
    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min)
    m_max = _hz_to_mel(f_max)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        print(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb

def octavescale_fbanks(
    n_freqs: int,
    n_filters: int,
    sample_rate: int,
    f_min = 0,
    f_max = None,
    limit_to_freqs = False
):
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    max_note = hertz_to_note(f_max if f_max is not None else sample_rate // 2)
    min_note = hertz_to_note(f_min) if f_min is not 0 else f_min
    octave_pts = torch.linspace(min_note, n_filters + 2 if (max_note - min_note) > n_filters and limit_to_freqs else max_note, n_filters + 2)
    f_pts = note_to_hertz(octave_pts)

    fb = _create_triangular_filterbank(all_freqs, f_pts)

    return fb

def octavescale_fbanks2(
    n_freqs: int,
    n_filters: int,
    sample_rate: int,
    f_min = 0,
    f_max = None,
    limit_to_freqs = False
):
    max_note = hertz_to_note(f_max if f_max is not None else sample_rate // 2)
    min_note = hertz_to_note(f_min)
    all_notes = torch.linspace(min_note, max_note if max_note < n_freqs or not limit_to_freqs else n_freqs, n_freqs)
    all_freqs = note_to_hertz(all_notes)

    octave_pts = torch.linspace(min_note, max_note if max_note < n_freqs or not limit_to_freqs else n_freqs, n_filters + 2)
    f_pts = note_to_hertz(octave_pts)

    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        print(
            "At least one octave filterbank has all zero values. "
            f"The value for `n_octaves` ({n_filters}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb

def linear_fbanks(n_freqs: int, f_min: float, f_max: float, n_filters: int, sample_rate: int):
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    f_pts = torch.linspace(f_min, f_max, n_filters + 2)
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    return fb
    
class Tempogram(nn.Module):
    def __init__(self, n_fft, win_length=None, learnable=False):
        super(Tempogram, self).__init__()

        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.tempo_bins = torch.arange(n_fft // 2 + 1, dtype=torch.float).unsqueeze(0).unsqueeze(1)
        self.learnable = learnable

        if learnable:
            self.tempo_kernel = nn.Parameter(torch.abs(torch.exp(-2j * torch.pi * self.tempo_bins * torch.arange(n_fft // 2 + 1, dtype=torch.float).unsqueeze(0).unsqueeze(2) / (n_fft // 2 + 1))))
        else:
            self.register_buffer('tempo_kernel', torch.exp(-2j * torch.pi * self.tempo_bins * torch.arange(n_fft // 2 + 1, dtype=torch.float).unsqueeze(0).unsqueeze(2) / (n_fft // 2 + 1)))

    def forward(self, magnitude, phase):
        phase_diff = torch.zeros_like(phase)
        phase_diff[:, :, :, 1:] = torch.diff(phase, dim=3)
        onset_strength = torch.sum(magnitude * torch.maximum(torch.zeros_like(phase_diff), -phase_diff), dim=1)
        tempogram = torch.abs(torch.matmul(onset_strength.to(dtype=torch.complex64 if not self.learnable else None).transpose(1,2).unsqueeze(1), self.tempo_kernel).transpose(2,3))
        return tempogram

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