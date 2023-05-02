
import math
import torch

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
    sample_rate: int
):
    max_note = hertz_to_note(sample_rate // 2)
    all_notes = torch.linspace(0, max_note, n_freqs)
    all_freqs = note_to_hertz(all_notes)

    octave_pts = torch.linspace(0, max_note, n_filters + 2)
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