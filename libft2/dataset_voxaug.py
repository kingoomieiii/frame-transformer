import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from libft2.dataset_utils import apply_channel_drop, apply_dynamic_range_mod, apply_harmonic_distortion, apply_multiplicative_noise, apply_random_eq, apply_stereo_spatialization, apply_time_stretch, apply_pitch_shift, apply_random_phase_noise, apply_time_masking, apply_frequency_masking, apply_emphasis

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path=[], is_validation=False, n_fft=2048, hop_length=1024, cropsize=256, sr=44100, seed=0, inst_rate=0.01, data_limit=None, predict_vocals=False, time_scaling=True):
        self.is_validation = is_validation
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.inst_rate = inst_rate
        self.predict_vocals = predict_vocals
        self.time_scaling = time_scaling

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
            (0.025, apply_channel_drop, {}),
            (0.5, apply_dynamic_range_mod, { "threshold": np.random.uniform(0, 0.5), "ratio": np.random.randint(2,8) }),
            (0.2, apply_harmonic_distortion, { "P": P, "num_harmonics": np.random.randint(1, 5), "gain": np.random.uniform(0, 0.5), "n_fft": self.n_fft, "hop_length": self.hop_length }),
            (0.5, apply_multiplicative_noise, { "loc": 1, "scale": np.random.uniform(0, 0.25) }),
            (0.5, apply_random_eq, { "min": np.random.uniform(0, 1), "max": np.random.uniform(1, 2) }),
            (0.5, apply_stereo_spatialization, { "c": Vc, "alpha": np.random.uniform(0, 2) }),
            (0.2, apply_pitch_shift, { "P": P, "n_fft": self.n_fft, "hop_length": self.hop_length, "sr": self.sr, "n_steps": np.random.uniform(-5, 5) }),
            (0.2, apply_time_masking, { "max_mask_percentage": np.random.uniform(0, 0.3) }),
            (0.2, apply_frequency_masking, { "max_mask_percentage": np.random.uniform(0, 0.3) }),
            (0.2, apply_emphasis, { "P": P, "coef": np.random.uniform(0, 0.3), "n_fft": self.n_fft, "hop_length": self.hop_length })
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if np.random.uniform() < p:
                M = aug(M, **args)

        if np.random.uniform() < 0.5:
            P = apply_random_phase_noise(P, np.random.uniform(0, 0.3))

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
            (0.01, apply_channel_drop, {}),
            (0.5, apply_random_eq, { "min": np.random.uniform(0.8,1), "max": np.random.uniform(1, 1.2) }),
            (0.5, apply_stereo_spatialization, { "c": c, "alpha": np.random.uniform(0.8, 1.2) })
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if np.random.uniform() < p:
                M = aug(M, **args)

        X = M * np.exp(1.j * P)

        if np.random.uniform() < 0.5:
            X = X[::-1]

        return X

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

            if np.random.uniform() < self.inst_rate:
                X = Y

        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)
        
        return X.astype(np.float32), Y.astype(np.float32)