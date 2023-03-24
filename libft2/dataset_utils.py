import torch
import torch.nn.functional as F
import numpy as np
import librosa

def apply_random_eq(X, P, min=0, max=2):
    arr1 = F.interpolate(torch.rand((1, 1, 512,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr2 = F.interpolate(torch.rand((1, 1, 256,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr3 = F.interpolate(torch.rand((1, 1, 128,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr4 = F.interpolate(torch.rand((1, 1, 64,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr5 = F.interpolate(torch.rand((1, 1, 32,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr6 = F.interpolate(torch.rand((1, 1, 16,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr7 = F.interpolate(torch.rand((1, 1, 8,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr8 = F.interpolate(torch.rand((1, 1, 4,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr9 = F.interpolate(torch.rand((1, 1, 2,)) * (max - min) + min, size=(X.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    eq = (arr1 + arr2 + arr3 + arr4 + arr5 + arr6 + arr7 + arr8 + arr9) / 9.0
    eq = np.clip(eq, min, max)
    eq = np.expand_dims(eq, (0, 2))
    return X * eq, P

def apply_harmonic_distortion(M, P, c, num_harmonics=2, gain=0.1, n_fft=2048, hop_length=1024):
    left_M = M[0] / c
    right_M = M[1] / c
    left_P = P[0]
    right_P = P[1]
    
    left_X = left_M * np.exp(1.j * left_P)
    right_X = right_M * np.exp(1.j * right_P)

    left_s = librosa.istft(left_X, hop_length=hop_length)
    right_s = librosa.istft(right_X, hop_length=hop_length)

    left_ds = np.copy(left_s)
    right_ds = np.copy(right_s)
    for h in range(2, num_harmonics + 1):
        left_hs = np.roll(left_s, h, axis=-1)
        right_hs = np.roll(right_s, h, axis=-1)
        left_ds += (gain ** (h - 1)) * left_hs
        right_ds += (gain ** (h - 1)) * right_hs

    left_X = librosa.stft(left_ds, n_fft=n_fft, hop_length=hop_length)
    right_X = librosa.stft(right_ds, n_fft=n_fft, hop_length=hop_length)

    return np.array([np.abs(left_X), np.abs(right_X)]) * c, np.array([np.angle(left_X), np.angle(right_X)])

def apply_pitch_shift(M, P, c, n_fft=2048, hop_length=1024, sr=44100, n_steps=1):
    left_M = M[0] / c
    right_M = M[1] / c
    left_P = P[0]
    right_P = P[1]
    
    left_X = left_M * np.exp(1.j * left_P)
    right_X = right_M * np.exp(1.j * right_P)

    left_s = librosa.istft(left_X, hop_length=hop_length)
    right_s = librosa.istft(right_X, hop_length=hop_length)

    left_s = librosa.effects.pitch_shift(left_s, sr=sr, n_steps=n_steps)
    right_s = librosa.effects.pitch_shift(right_s, sr=sr, n_steps=n_steps)

    left_X = librosa.stft(left_s, n_fft=n_fft, hop_length=hop_length)
    right_X = librosa.stft(right_s, n_fft=n_fft, hop_length=hop_length)

    return np.array([np.abs(left_X), np.abs(right_X)]) * c, np.array([np.angle(left_X), np.angle(right_X)])

def apply_emphasis(M, P, c, emphasis_coef, n_fft=2048, hop_length=1024):
    left_M = M[0] / c
    right_M = M[1] / c
    left_P = P[0]
    right_P = P[1]
    
    left_X = left_M * np.exp(1.j * left_P)
    right_X = right_M * np.exp(1.j * right_P)

    left_s = librosa.istft(left_X, hop_length=hop_length)
    right_s = librosa.istft(right_X, hop_length=hop_length)
    
    if np.random.uniform() < 0.5:
        left_s = librosa.effects.preemphasis(left_s, coef=emphasis_coef)
        right_s = librosa.effects.preemphasis(right_s, coef=emphasis_coef)
    else:
        left_s = librosa.effects.deemphasis(left_s, coef=emphasis_coef)
        right_s = librosa.effects.deemphasis(right_s, coef=emphasis_coef)

    left_X = librosa.stft(left_s, n_fft=n_fft, hop_length=hop_length)
    right_X = librosa.stft(right_s, n_fft=n_fft, hop_length=hop_length)

    return np.array([np.abs(left_X), np.abs(right_X)]) * c, np.array([np.angle(left_X), np.angle(right_X)])

def apply_random_phase_noise(X, P, strength=0.1):
    random_phase = np.random.uniform(-np.pi, np.pi, size=P.shape)
    return X, P + strength * random_phase

def apply_stereo_spatialization(X, P, alpha=1):
    left, right = X[0], X[1]
    avg = (left + right) * 0.5
    left = alpha * left + (1 - alpha) * avg 
    right = alpha * right + (1 - alpha) * avg
    return np.clip(np.stack([left, right], axis=0), 0, 1), P

def apply_multiplicative_noise(X, P, loc=1, scale=0.1):
    eps = np.random.normal(loc, scale, size=X.shape)
    return X * eps, P

def apply_dynamic_range_mod(X, P, c, threshold=0.5, ratio=4):
    X = X / c

    if np.random.uniform() < 0.5:
        clipped = np.clip(X - threshold, 0, None)
        compressed = clipped / ratio
        return np.where(X > threshold, compressed + threshold, X) * c, P
    else:
        clipped = np.clip(X - threshold, None, 0)
        expanded = clipped * ratio
        return np.where(X < threshold, expanded + threshold, X) * c, P
    
def apply_channel_drop(X, P, channel):
    X[channel] = 0
    return X, P

def apply_time_stretch(X, target_size):
    if X.shape[2] > target_size:
        size = np.random.randint(target_size // 16, X.shape[2])
        start = np.random.randint(0, X.shape[2] - size)
        cropped = X[:, :, start:start+size]
        H = X[:, :, :target_size]
        H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(X.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
        H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(X.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
    else:
        if np.random.uniform() < 0.5:
            padded = np.pad(X, ((0, 0), (0, 0), (np.random.randint(0, target_size // 4), (target_size - X.shape[2]) + np.random.randint(0, target_size // 4))))
            size = np.random.randint(target_size, padded.shape[2])
            start = np.random.randint(0, padded.shape[2] - size)
            cropped = padded[:, :, start:start+size]
            H = padded[:, :, :target_size]
            H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(padded.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
            H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(padded.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
        else: 
            size = np.random.randint(target_size // 8, target_size)
            start = np.random.randint(0, X.shape[2] - size)
            cropped = X[:, :, start:start+size]
            H = X[:, :, :target_size]
            H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(X.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
            H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(X.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()

    return H
        
def apply_time_masking(X, P, max_mask_percentage=0.2):
    mask_percentage = np.random.uniform(0, max_mask_percentage)
    mask_width = int(X.shape[2] * mask_percentage)
    mask_start = np.random.randint(0, X.shape[2] - mask_width)
    X[:, :, mask_start:mask_start+mask_width] = 0
    return X, P

def apply_frequency_masking(X, P, max_mask_percentage=0.2):
    mask_percentage = np.random.uniform(0, max_mask_percentage)
    mask_height = int(X.shape[1] * mask_percentage)
    mask_start = np.random.randint(0, X.shape[1] - mask_height)
    X[:, mask_start:mask_start+mask_height, :] = 0
    return X, P