import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from libft2gan.dataset_utils import apply_channel_drop, apply_dynamic_range_mod, apply_masking, apply_multiplicative_noise, apply_random_eq, apply_stereo_spatialization, apply_time_stretch, apply_random_phase_noise, apply_time_masking, apply_emphasis, apply_deemphasis, apply_pitch_shift, apply_harmonic_distortion, apply_random_volume
import librosa

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, instrumental_lib=[], vocal_lib=[], is_validation=False, n_fft=2048, hop_length=1024, cropsize=256, sr=44100, seed=0, inst_rate=0.01, data_limit=None, predict_vocals=False, time_scaling=True, vocal_threshold=0.00125, vout_bands=4):
        self.is_validation = is_validation
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.inst_rate = inst_rate
        self.predict_vocals = predict_vocals
        self.time_scaling = time_scaling
        self.vocal_threshold = vocal_threshold
        self.vout_bands = vout_bands

        self.max_bin = n_fft // 2
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cropsize = cropsize

        self.random = random.Random(seed)

        for mp in instrumental_lib:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                self.curr_list.append(m)
            
        if not is_validation and len(vocal_lib) != 0:
            for vp in vocal_lib:
                vox = [os.path.join(vp, f) for f in os.listdir(vp) if os.path.isfile(os.path.join(vp, f))]

                for v in vox:
                    self.vocal_list.append(v)

        def key(p):
            return os.path.basename(p)
        
        self.vocal_list.sort(key=key)
        self.curr_list.sort(key=key)
        self.random.shuffle(self.vocal_list)
        self.random.shuffle(self.curr_list)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.curr_list)

    def _get_vocals(self, idx):
        path = str(self.vocal_list[(self.epoch + idx) % len(self.vocal_list)])
        vdata = np.load(path)
        V, Vc = vdata['X'], vdata['c']

        if self.random.uniform(0,1) < 0.5:
            V = apply_time_stretch(V, self.random, self.cropsize)
        elif V.shape[2] > self.cropsize:
            start = self.random.randint(0, V.shape[2] - self.cropsize - 1)
            V = V[:, :, start:start+self.cropsize]

        if np.random.uniform() < 0.04:
            if np.random.uniform() < 0.5:
                V[0] = 0
            else:
                V[1] = 0

        VP = V[:, :-1, :]
        VP = (np.abs(VP) / Vc).reshape((VP.shape[0], self.vout_bands, VP.shape[1] // self.vout_bands, VP.shape[2]))
        VP = VP.mean(axis=2)
        VP = np.where(VP > self.vocal_threshold, 1, (self.vocal_threshold - VP) ** 2)

        P = np.angle(V)
        M = np.abs(V)

        augmentations = [
            (0.2, apply_dynamic_range_mod, { "c": Vc, "threshold": self.random.uniform(0,1), "gain": self.random.uniform(0,0.25), }),
            (0.2, apply_multiplicative_noise, { "loc": 1, "scale": self.random.uniform(0,0.25), }),
            (0.2, apply_random_eq, { "min": self.random.uniform(0, 1), "max": self.random.uniform(1, 2), }),
            (0.2, apply_stereo_spatialization, { "c": Vc, "alpha": self.random.uniform(0, 2) }),
            (0.2, apply_random_phase_noise, { "strength": self.random.uniform(0, 0.5)}),
            (0.2, apply_harmonic_distortion, { "c": Vc, "num_harmonics": self.random.randint(1,4), "gain": self.random.uniform(0, 0.2), "hop_length": self.hop_length, "n_fft": self.n_fft }),
            (0.2, apply_pitch_shift, { "pitch_shift": self.random.uniform(-12, 12) }),
            (0.2, apply_emphasis, { "coef": self.random.uniform(0.8, 1) }),
            (0.2, apply_deemphasis, { "coef": self.random.uniform(0.8, 1) }),
            (0.2, apply_random_volume, { "gain": self.random.uniform(0.5, 1.5) }),
            (0.2, apply_masking, { "c": Vc, "num_masks": np.random.randint(0, 6), "max_mask_percentage": np.random.uniform(0, 0.2), "alpha": np.random.uniform() }),
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if self.random.uniform(0,1) < p:
                M, P = aug(M, P, self.random, **args)
                M = np.clip(M / Vc, 0, 1) * Vc

        V = M * np.exp(1.j * P)

        if self.random.uniform(0,1) < 0.5:
            V = V[::-1]

        return V, Vc, VP

    def _augment_instruments(self, X, c):
        if X.shape[2] > self.cropsize:
            start = self.random.randint(0, X.shape[2] - self.cropsize - 1)
            X = X[:, :, start:start+self.cropsize]

        P = np.angle(X)
        M = np.abs(X)

        augmentations = [
            (0.02, apply_channel_drop, { "channel": self.random.randint(0,2), "alpha": self.random.uniform(0,1) }),
            (0.2, apply_dynamic_range_mod, { "c": c, "threshold": self.random.uniform(0,1), "gain": self.random.uniform(0,0.125), }),
            (0.2, apply_multiplicative_noise, { "loc": 1, "scale": self.random.uniform(0, 0.125), }),
            (0.2, apply_random_eq, { "min": self.random.uniform(0.5, 1), "max": self.random.uniform(1, 1.5), }),
            (0.2, apply_stereo_spatialization, { "c": c, "alpha": self.random.uniform(0.5, 1.5) }),
            (0.2, apply_random_phase_noise, { "strength": self.random.uniform(0, 0.25)}),
            (0.2, apply_pitch_shift, { "pitch_shift": self.random.uniform(-6, 6) }),
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if self.random.uniform(0,1) < p:
                M, P = aug(M, P, self.random, **args)
                M = np.clip(M / c, 0, 1) * c

        X = M * np.exp(1.j * P)

        if self.random.uniform(0,1) < 0.5:
            X = X[::-1]

        return X
    
    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])
        data = np.load(path)
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']
        V, VP = None, np.zeros((X.shape[0], 4, X.shape[2]))

        if not self.is_validation:
            Y = self._augment_instruments(Y, c)
            V, Vc, VP = self._get_vocals(idx)
            X = Y + V
            c = np.max([np.abs(X).max(), c, Vc])

        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)

        return X.astype(np.float32), Y.astype(np.float32), VP.astype(np.float32)