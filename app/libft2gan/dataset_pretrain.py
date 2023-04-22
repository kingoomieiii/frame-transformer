import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from libft2gan.dataset_utils import apply_channel_drop, apply_dynamic_range_mod, apply_multiplicative_noise, apply_random_eq, apply_stereo_spatialization, apply_time_stretch, apply_random_phase_noise, apply_time_masking, apply_frequency_masking, apply_emphasis, apply_deemphasis, apply_pitch_shift, apply_masking, apply_harmonic_distortion, apply_random_volume, apply_frame_mag_masking, apply_frame_phase_masking
import librosa

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, instrumental_lib=[], pretraining_lib=[], vocal_lib=[], is_validation=False, n_fft=2048, hop_length=1024, cropsize=256, sr=44100, seed=0, data_limit=None, max_frames_per_mask=16, max_masks_per_item=16, max_mask_percentage=0.2, predict_phase=False):
        self.is_validation = is_validation
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.predict_phase = predict_phase

        self.max_mask_percentage = max_mask_percentage
        self.max_frames_per_mask = max_frames_per_mask
        self.max_masks_per_item = max_masks_per_item

        self.max_bin = n_fft // 2
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cropsize = cropsize

        for mp in instrumental_lib:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                if m.endswith('.npz'):
                    self.curr_list.append(m)

        for mp in pretraining_lib:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                if m.endswith('.npz'):
                    self.curr_list.append(m)
            
        if not is_validation and len(vocal_lib) != 0:
            for vp in vocal_lib:
                vox = [os.path.join(vp, f) for f in os.listdir(vp) if os.path.isfile(os.path.join(vp, f))]

                for v in vox:
                    if v.endswith('.npz'):
                        self.vocal_list.append(v)

        self.random = random.Random(seed)

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
        vdata = np.load(path, allow_pickle=True)
        V, Vc = vdata['X'], vdata['c']
        VCr, VCi = vdata['cr'], vdata['ci']

        if self.random.uniform(0,1) < 0.5:
            V = apply_time_stretch(V, self.random, self.cropsize)
        elif V.shape[2] > self.cropsize:
            start = self.random.randint(0, V.shape[2] - self.cropsize - 1)
            V = V[:, :, start:start+self.cropsize]

        P = np.angle(V)
        M = np.abs(V)

        augmentations = [
            (0.01, apply_channel_drop, { "channel": self.random.randint(0,2), "alpha": self.random.uniform(0,1) }),
            (0.2, apply_dynamic_range_mod, { "c": Vc, "threshold": self.random.uniform(0,1), "gain": self.random.uniform(0,0.25), }),
            (0.2, apply_multiplicative_noise, { "loc": 1, "scale": self.random.uniform(0,0.25), }),
            (0.2, apply_random_eq, { "min": self.random.uniform(0, 1), "max": self.random.uniform(1, 2), }),
            (0.2, apply_stereo_spatialization, { "c": Vc, "alpha": self.random.uniform(0, 2) }),
            (0.2, apply_random_phase_noise, { "strength": self.random.uniform(0, 0.2)}),
            (0.2, apply_pitch_shift, { "pitch_shift": self.random.uniform(-12, 12) }),
            (0.2, apply_emphasis, { "coef": self.random.uniform(0.8, 1) }),
            (0.2, apply_deemphasis, { "coef": self.random.uniform(0.8, 1) })
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if self.random.uniform(0,1) < p:
                M, P = aug(M, P, self.random, **args)

        V = M * np.exp(1.j * P)

        if self.random.uniform(0,1) < 0.5:
            V = V[::-1]

        VCr = np.max([VCr, np.abs(V.real).max()])
        VCi = np.max([VCi, np.abs(V.imag).max()])
        V.imag = V.imag / VCi
        V.real = V.real / VCr

        return V

    def _augment_mix(self, X, c):
        if X.shape[2] > self.cropsize:
            start = self.random.randint(0, X.shape[2] - self.cropsize - 1)
            X = X[:, :, start:start+self.cropsize]

        P = np.angle(X)
        M = np.abs(X)

        if self.predict_phase:
            M, P = apply_frame_phase_masking(M, P, self.random, c, num_masks=np.random.randint(0, self.max_masks_per_item), max_mask_percentage=self.max_mask_percentage, type=np.random.randint(0,8), alpha=self.random.uniform(0.5,1))
        else:
            M, P = apply_frame_mag_masking(M, P, self.random, c, num_masks=np.random.randint(0, self.max_masks_per_item), max_mask_percentage=self.max_mask_percentage, type=np.random.randint(0,8), alpha=self.random.uniform(0.5,1))
        
        X = M * np.exp(1.j * P)

        if self.random.uniform(0,1) < 0.5:
            X = X[::-1]

        return X

    def _get_instruments(self, X, c):
        if X.shape[2] > self.cropsize:
            start = self.random.randint(0, X.shape[2] - self.cropsize - 1)
            X = X[:, :, start:start+self.cropsize]

        P = np.angle(X)
        M = np.abs(X)

        augmentations = [
            (0.01, apply_channel_drop, { "channel": self.random.randint(0,2), "alpha": self.random.uniform(0,1) })
        ]

        random.shuffle(augmentations)

        for p, aug, args in augmentations:
            if self.random.uniform(0,1) < p:
                M, P = aug(M, P, self.random, **args)

        X = M * np.exp(1.j * P)

        if self.random.uniform(0,1) < 0.5:
            X = X[::-1]

        return X
    
    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])
        data = np.load(path, allow_pickle=True)
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']
        V = None
        
        Y = self._get_instruments(Y, c)
        X = Y

        if not self.is_validation and str.lower(path).find('pretraining') == -1:
            if np.random.uniform(0,1) < 0.2:
                V = self._get_vocals(idx)
                Y = Y + V
                
            c = np.max([c, np.abs(Y).max()])

        X = Y if (self.random.uniform(0,1) < 0.075 and not self.is_validation) else self._augment_mix(Y, c)

        XP = (np.angle(X) + np.pi) / (2 * np.pi)
        YP = (np.angle(Y) + np.pi) / (2 * np.pi)
        X = np.abs(X) / c
        Y = np.abs(Y) / c

        if self.predict_phase:
            Y = YP

        X = np.concatenate((X, XP), axis=0)

        return X.astype(np.float32), Y.astype(np.float32)