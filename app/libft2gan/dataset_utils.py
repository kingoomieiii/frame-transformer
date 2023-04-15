import torch
import torch.nn.functional as F
import numpy as np
import librosa

def apply_random_eq(M, P, random, min=0, max=2):
    arr1 = F.interpolate(torch.rand((1, 1, 512,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr2 = F.interpolate(torch.rand((1, 1, 256,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr3 = F.interpolate(torch.rand((1, 1, 128,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr4 = F.interpolate(torch.rand((1, 1, 64,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr5 = F.interpolate(torch.rand((1, 1, 32,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr6 = F.interpolate(torch.rand((1, 1, 16,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr7 = F.interpolate(torch.rand((1, 1, 8,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr8 = F.interpolate(torch.rand((1, 1, 4,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    arr9 = F.interpolate(torch.rand((1, 1, 2,)) * (max - min) + min, size=(M.shape[1]), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy()
    eq = (arr1 + arr2 + arr3 + arr4 + arr5 + arr6 + arr7 + arr8 + arr9) / 9.0
    eq = np.clip(eq, min, max)
    eq = np.expand_dims(eq, (0, 2))

    return M * eq, P

def apply_random_phase_noise(M, P, random, strength=0.1):
    random_phase = np.random.uniform(-np.pi, np.pi, size=P.shape)
    return M, P + strength * random_phase

def apply_random_volume(M, P, random, gain=0.1):
    a = random.uniform(-gain, gain)
    return M + (M * a), P

def apply_stereo_spatialization(M, P, random, c, alpha=1):
    left, right = M[0], M[1]
    mid = (left + right) * 0.5
    left = alpha * left + (1 - alpha) * mid 
    right = alpha * right + (1 - alpha) * mid

    return np.stack([left, right], axis=0), P

def apply_multiplicative_noise(M, P, random, loc=1, scale=0.1):
    eps = np.random.normal(loc, scale, size=M.shape)

    return M * eps, P

def apply_additive_noise(M, P, random, c, loc=0, scale=0.1):
    X = M / c
    eps = np.random.normal(loc, scale, size=M.shape)

    return X + eps * c, P

def apply_dynamic_range_mod(M, P, random, c, threshold=0.5, gain=0.1):
    M = M / c
    
    if random.uniform(0,1) < 0.5:
        if random.uniform(0,1) < 0.5:
            return np.where(M > threshold, M - (M * gain), M) * c, P
        else:
            return np.where(M < threshold, M + (M * gain), M) * c, P
    else:
        if random.uniform(0,1) < 0.5:
            return np.where(M > threshold, M + (M * gain), M) * c, P
        else:
            return np.where(M < threshold, M - (M * gain), M) * c, P
    
def apply_channel_drop(M, P, random, channel, alpha=1):
    H = np.copy(M)

    if channel == 2:
        H[:, :, :] = 0
    else:
        H[channel, :, :] = 0
        
    return alpha * H + (1 - alpha) * M, P

def apply_time_stretch(M, random, target_size):
    if M.shape[2] > target_size:
        size = random.randint(target_size // 16, M.shape[2])
        start = random.randint(0, M.shape[2] - size)
        cropped = M[:, :, start:start+size]
        H = M[:, :, :target_size]
        H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
        H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
    else:
        if random.uniform(0,1) < 0.5:
            padded = np.pad(M, ((0, 0), (0, 0), (random.randint(0, target_size // 4), (target_size - M.shape[2]) + random.randint(0, target_size // 4))))
            size = random.randint(target_size, padded.shape[2])
            start = random.randint(0, padded.shape[2] - size)
            cropped = padded[:, :, start:start+size]
            H = padded[:, :, :target_size]
            H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(padded.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
            H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(padded.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
        else: 
            size = random.randint(target_size // 8, target_size)
            start = random.randint(0, M.shape[2] - size)
            cropped = M[:, :, start:start+size]
            H = M[:, :, :target_size]
            H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
            H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()

    return H

def apply_harmonic_distortion(M, P, random, c, num_harmonics=2, gain=0.1, n_fft=2048, hop_length=1024):
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

    left_s = np.nan_to_num(left_s, nan=0, neginf=-1, posinf=1)
    right_s = np.nan_to_num(right_s, nan=0, neginf=-1, posinf=1)

    left_X = librosa.stft(left_ds, n_fft=n_fft, hop_length=hop_length)
    right_X = librosa.stft(right_ds, n_fft=n_fft, hop_length=hop_length)
    
    left_M = np.abs(left_X) * c
    right_M = np.abs(right_X) * c

    return np.array([left_M, right_M]), np.array([np.angle(left_X), np.angle(right_X)])

def apply_pitch_shift(M, P, random, pitch_shift):
    _, num_bins, num_frames = M.shape
    scaling_factor = 2 ** (pitch_shift / 12)

    H_L = np.zeros_like(M[0])
    H_R = np.zeros_like(M[1])

    for i in range(num_frames):
        H_L[:, i] = np.interp(np.arange(num_bins) * scaling_factor, np.arange(num_bins), M[0, :, i])
        H_R[:, i] = np.interp(np.arange(num_bins) * scaling_factor, np.arange(num_bins), M[1, :, i])

    G_L, G_L_accum = np.zeros_like(P[0]), np.zeros(num_bins)
    G_R, G_R_accum = np.zeros_like(P[1]), np.zeros(num_bins)

    for i in range(1, num_frames):
        dphase = P[0, :, i] - P[0, :, i - 1]
        dphase = dphase - 2 * np.pi * np.floor((dphase + np.pi) / (2 * np.pi))
        dphase = dphase / scaling_factor
        G_L_accum += dphase
        G_L[:, i] = P[0, :, i - 1] + G_L_accum

        dphase = P[1, :, i] - P[1, :, i - 1]
        dphase = dphase - 2 * np.pi * np.floor((dphase + np.pi) / (2 * np.pi))
        dphase = dphase / scaling_factor
        G_R_accum += dphase
        G_R[:, i] = P[1, :, i - 1] + G_R_accum

    return np.array([H_L, H_R]), np.array([G_L, G_R])

def apply_emphasis(M, P, random, coef):
    left_M = M[0]
    right_M = M[1]
    
    _, num_bins, _ = M.shape
    filter = 1 - coef * np.linspace(0, 1, num_bins)
    left_M = left_M * filter[:, None]
    right_M = right_M * filter[:, None]

    return np.array([left_M, right_M]), P

def apply_deemphasis(M, P, random, coef):
    left_M = M[0]
    right_M = M[1]
    
    _, num_bins, _ = M.shape
    filter = 1 / (1 - coef * np.linspace(0, 1, num_bins))
    left_M = left_M * filter[:, None]
    right_M = right_M * filter[:, None]

    return np.array([left_M, right_M]), P

def apply_masking(M, P, random, c, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M) / c
    N = np.random.uniform(size=H.shape)

    for _ in range(num_masks):
        mask_percentage = random.uniform(0, max_mask_percentage)
        mask_height = int(M.shape[1] * mask_percentage)
        mask_start_h = random.randint(0, M.shape[1] - mask_height)
        mask_width = int(M.shape[2] * mask_percentage)
        mask_start_w = random.randint(0, M.shape[2] - mask_width)
        H[:, mask_start_h:mask_start_h+mask_height, mask_start_w:mask_start_w+mask_width] = (alpha * N[:, mask_start_h:mask_start_h+mask_height, mask_start_w:mask_start_w+mask_width]) + (1 - alpha) * H[:, mask_start_h:mask_start_h+mask_height, mask_start_w:mask_start_w+mask_width]

    return H * c, P

def apply_time_masking(M, P, random, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = random.uniform(0, max_mask_percentage)
        mask_width = int(M.shape[2] * mask_percentage)
        mask_start = random.randint(0, M.shape[2] - mask_width)
        H[:, :, mask_start:mask_start+mask_width] = 0

    return alpha * H + (1 - alpha) * M, P
        
def apply_time_masking2(M, P, random, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = random.uniform(0, max_mask_percentage)
        mask_width = int(M.shape[2] * mask_percentage)
        mask_start = random.randint(0, M.shape[2] - mask_width)
        H[:, :, mask_start:mask_start+mask_width] = H[:, :, mask_start:mask_start+mask_width] * (1 - alpha)

    return H, P

def apply_frequency_masking(M, P, random, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = random.uniform(0, max_mask_percentage)
        mask_height = int(M.shape[1] * mask_percentage)
        mask_start = random.randint(0, M.shape[1] - mask_height)
        H[:, mask_start:mask_start+mask_height, :] = 0

    return alpha * H + (1 - alpha) * M, P

def apply_frequency_masking2(M, P, random, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = random.uniform(0, max_mask_percentage)
        mask_height = int(M.shape[1] * mask_percentage)
        mask_start = random.randint(0, M.shape[1] - mask_height)
        H[:, mask_start:mask_start+mask_height, :] = H[:, mask_start:mask_start+mask_height, :] * (1 - alpha)

    return H, P

def apply_frame_masking(M, P, random, c, num_masks=1, max_mask_percentage=0.2, alpha=1, type=0):
    M = M / c
    H = np.copy(M)
    N = np.random.uniform(size=H.shape)

    for _ in range(num_masks):
        mask_percentage = random.uniform(0.01, max_mask_percentage)
        w = int(M.shape[2] * mask_percentage)
        s = random.randint(0, H.shape[2] - w)

        if type == 0:
            m = N[:, :, s:s+w]
        elif type == 1:
            m = np.zeros_like(N[:, :, s:s+w])
        elif type == 2:
            m = np.ones_like(N[:, :, s:s+w])
        elif type == 3:
            m = np.full_like(N[:, :, s:s+w], fill_value=H[:, :, s:s+w].mean())
        elif type == 4:
            m = np.full_like(N[:, :, s:s+w], fill_value=H[:, :, s:s+w].min())
        elif type == 5:
            m = np.full_like(N[:, :, s:s+w], fill_value=H[:, :, s:s+w].max())
        elif type == 6:
            m = np.full_like(N[:, :, s:s+w], fill_value=H[:, :, s:s+w].var())

        H[:, :, s:s+w] = alpha * m + (1 - alpha) * H[:, :, s:s+w]

    return np.clip(H, 0, 1) * c, P