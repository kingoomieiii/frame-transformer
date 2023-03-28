import torch
import torch.nn.functional as F
import numpy as np
import librosa

def apply_random_eq(M, P, min=0, max=2):
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

def apply_random_phase_noise(M, P, strength=0.1):
    random_phase = np.random.uniform(-np.pi, np.pi, size=P.shape)

    return M, P + strength * random_phase

def apply_stereo_spatialization(M, P, c, alpha=1):
    left, right = M[0] / c, M[1] / c
    mid = (left + right) * 0.5
    left = alpha * left + (1 - alpha) * mid 
    right = alpha * right + (1 - alpha) * mid

    return np.clip(np.stack([left, right], axis=0), 0, 1) * c, P

def apply_multiplicative_noise(M, P, loc=1, scale=0.1):
    eps = np.random.normal(loc, scale, size=M.shape)

    return M * eps, P

def apply_additive_noise(M, P, c, loc=0, scale=0.1):
    X = M / c
    eps = np.random.normal(loc, scale, size=M.shape)

    return np.clip(X + eps, 0, 1) * c, P

def apply_dynamic_range_mod(M, P, c, threshold=0.5, gain=0.1):
    M = M / c
    
    if np.random.uniform() < 0.5:
        if np.random.uniform() < 0.5:
            return np.clip(np.where(M > threshold, M - (M * gain), M), 0, 1) * c, P
        else:
            return np.clip(np.where(M < threshold, M + (M * gain), M), 0, 1) * c, P
    else:
        if np.random.uniform() < 0.5:
            return np.clip(np.where(M > threshold, M + (M * gain), M), 0, 1) * c, P
        else:
            return np.clip(np.where(M < threshold, M - (M * gain), M), 0, 1) * c, P
    
def apply_channel_drop(M, P, channel, alpha=1):
    H = np.copy(M)

    if channel == 2:
        H[:, :, :] = 0
    else:
        H[channel, :, :] = 0
        
    return alpha * H + (1 - alpha) * M, P

def apply_time_stretch(M, target_size):
    if M.shape[2] > target_size:
        size = np.random.randint(target_size // 16, M.shape[2])
        start = np.random.randint(0, M.shape[2] - size)
        cropped = M[:, :, start:start+size]
        H = M[:, :, :target_size]
        H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
        H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
    else:
        if np.random.uniform() < 0.5:
            padded = np.pad(M, ((0, 0), (0, 0), (np.random.randint(0, target_size // 4), (target_size - M.shape[2]) + np.random.randint(0, target_size // 4))))
            size = np.random.randint(target_size, padded.shape[2])
            start = np.random.randint(0, padded.shape[2] - size)
            cropped = padded[:, :, start:start+size]
            H = padded[:, :, :target_size]
            H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(padded.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
            H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(padded.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
        else: 
            size = np.random.randint(target_size // 8, target_size)
            start = np.random.randint(0, M.shape[2] - size)
            cropped = M[:, :, start:start+size]
            H = M[:, :, :target_size]
            H.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()
            H.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(M.shape[1], target_size), mode='bilinear', align_corners=True).squeeze(0).numpy()

    return H
        
def apply_time_masking(M, P, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = np.random.uniform(0, max_mask_percentage)
        mask_width = int(M.shape[2] * mask_percentage)
        mask_start = np.random.randint(0, M.shape[2] - mask_width)
        H[:, :, mask_start:mask_start+mask_width] = 0

    return alpha * H + (1 - alpha) * M, P
        
def apply_time_masking2(M, P, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = np.random.uniform(0, max_mask_percentage)
        mask_width = int(M.shape[2] * mask_percentage)
        mask_start = np.random.randint(0, M.shape[2] - mask_width)
        H[:, :, mask_start:mask_start+mask_width] = H[:, :, mask_start:mask_start+mask_width] * (1 - alpha)

    return H, P

def apply_frequency_masking(M, P, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = np.random.uniform(0, max_mask_percentage)
        mask_height = int(M.shape[1] * mask_percentage)
        mask_start = np.random.randint(0, M.shape[1] - mask_height)
        H[:, mask_start:mask_start+mask_height, :] = 0

    return alpha * H + (1 - alpha) * M, P

def apply_frequency_masking2(M, P, num_masks=1, max_mask_percentage=0.2, alpha=1):
    H = np.copy(M)

    for _ in range(num_masks):
        mask_percentage = np.random.uniform(0, max_mask_percentage)
        mask_height = int(M.shape[1] * mask_percentage)
        mask_start = np.random.randint(0, M.shape[1] - mask_height)
        H[:, mask_start:mask_start+mask_height, :] = H[:, mask_start:mask_start+mask_height, :] * (1 - alpha)

    return H, P