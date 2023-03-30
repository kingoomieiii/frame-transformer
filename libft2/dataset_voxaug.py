import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from libft2.dataset_utils import apply_channel_drop, apply_dynamic_range_mod, apply_multiplicative_noise, apply_random_eq, apply_stereo_spatialization, apply_time_stretch, apply_random_phase_noise, apply_time_masking, apply_frequency_masking, apply_time_masking2, apply_frequency_masking2, apply_emphasis, apply_deemphasis, apply_pitch_shift, apply_harmonic_distortion
import librosa

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path=[], is_validation=False, n_fft=2048, hop_length=1024, cropsize=256, sr=44100, seed=0, inst_rate=0.01, data_limit=None, predict_vocals=False, time_scaling=True):
        self.is_validation = is_validation
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.inst_rate = inst_rate
        self.predict_vocals = predict_vocals
        self.time_scaling = time_scaling

        self.max_bin = n_fft // 2
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cropsize = cropsize

        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                self.curr_list.append(m)
            
        if not is_validation and len(vocal_path) != 0:
            for vp in vocal_path:
                vox = [os.path.join(vp, f) for f in os.listdir(vp) if os.path.isfile(os.path.join(vp, f))]

                for v in vox:
                    self.vocal_list.append(v)

            random.Random(seed).shuffle(self.vocal_list)

        random.Random(seed+1).shuffle(self.curr_list)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.curr_list)

    def _get_vocals(self, idx):
        path = str(self.vocal_list[(self.epoch + idx) % len(self.vocal_list)])
        vdata = np.load(path)
        V, Vc = vdata['X'], vdata['c']

        if np.random.uniform() < 0.5:
            V = apply_time_stretch(V, self.cropsize)
        elif V.shape[2] > self.cropsize:
            start = np.random.randint(0, V.shape[2] - self.cropsize)
            V = V[:, :, start:start+self.cropsize]

        P = np.angle(V)
        M = np.abs(V)

        augmentations = [
            (0.04, apply_channel_drop, { "channel": np.random.randint(1,3), "alpha": np.random.uniform() }),
            (0.2, apply_dynamic_range_mod, { "c": Vc, "threshold": np.random.uniform(), "gain": np.random.uniform(), }),
            (0.2, apply_multiplicative_noise, { "loc": 1, "scale": np.random.uniform(0, 0.5), }),
            (0.2, apply_random_eq, { "min": np.random.uniform(0, 1), "max": np.random.uniform(1, 2), }),
            (0.2, apply_stereo_spatialization, { "c": Vc, "alpha": np.random.uniform(0, 2) }),
            (0.1, apply_time_masking, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.1, apply_frequency_masking, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.1, apply_time_masking2, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.1, apply_frequency_masking2, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.2, apply_random_phase_noise, { "strength": np.random.uniform(0, 0.5)}),
            (0.2, apply_harmonic_distortion, { "c": Vc, "num_harmonics": np.random.randint(1,4), "gain": np.random.uniform(0, 0.1), "hop_length": self.hop_length, "n_fft": self.n_fft }),
            (0.2, apply_pitch_shift, { "pitch_shift": np.random.uniform(-12, 12) }),
            (0.2, apply_emphasis, { "coef": np.random.uniform(0.9, 1) }),
            (0.2, apply_deemphasis, { "coef": np.random.uniform(0.9, 1) }),
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if np.random.uniform() < p:
                M, P = aug(M, P, **args)

        V = M * np.exp(1.j * P)

        if np.random.uniform() < 0.5:
            V = V[::-1]

        return V

    def _augment_instruments(self, X, c):
        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize)
            X = X[:, :, start:start+self.cropsize]

        P = np.angle(X)
        M = np.abs(X)

        augmentations = [
            (0.04, apply_channel_drop, { "channel": np.random.randint(1,3), "alpha": np.random.uniform() }),
            (0.2, apply_dynamic_range_mod, { "c": c, "threshold": np.random.uniform(), "gain": np.random.uniform(), }),
            (0.2, apply_multiplicative_noise, { "loc": 1, "scale": np.random.uniform(0, 0.5), }),
            (0.2, apply_random_eq, { "min": np.random.uniform(0, 1), "max": np.random.uniform(1, 2), }),
            (0.2, apply_stereo_spatialization, { "c": c, "alpha": np.random.uniform(0, 2) }),
            (0.1, apply_time_masking, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.1, apply_frequency_masking, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.1, apply_time_masking2, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.1, apply_frequency_masking2, { "num_masks": np.random.randint(0, 5), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
            (0.2, apply_random_phase_noise, { "strength": np.random.uniform(0, 0.5)}),
            (0.2, apply_harmonic_distortion, { "c": c, "num_harmonics": np.random.randint(1,4), "gain": np.random.uniform(0, 0.1), "hop_length": self.hop_length, "n_fft": self.n_fft }),
            (0.2, apply_pitch_shift, { "pitch_shift": np.random.uniform(-12, 12) }),
            (0.2, apply_emphasis, { "coef": np.random.uniform(0.9, 1) }),
            (0.2, apply_deemphasis, { "coef": np.random.uniform(0.9, 1) }),
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if np.random.uniform() < p:
                M, P = aug(M, P, **args)

        X = M * np.exp(1.j * P)

        if np.random.uniform() < 0.5:
            X = X[::-1]

        return X
    
    def _get_wave(self, X, c):
        left_s = np.pad(librosa.istft((np.abs(X[0]) / c) + np.exp(1.j * np.angle(X[0])), hop_length=self.hop_length), ((0, self.hop_length)))
        right_s = np.pad(librosa.istft((np.abs(X[1]) / c) + np.exp(1.j * np.angle(X[1])), hop_length=self.hop_length), ((0, self.hop_length)))
        S = np.expand_dims(np.stack([left_s, right_s], axis=0), axis=2).reshape((2, left_s.shape[0] // X.shape[2], -1))
        return S
    
    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])
        data = np.load(path)
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']
        V = None
        
        if not self.is_validation:
            Y = self._augment_instruments(Y, c)
            V = self._get_vocals(idx)
            X = Y + V
            c = np.max([c, np.abs(X).max()])

        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)

        return X.astype(np.float32), Y.astype(np.float32)