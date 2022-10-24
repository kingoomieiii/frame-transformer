from math import nan
import math
import os
import random
import numpy as np
import torch
import torch.utils.data
from scipy.ndimage.filters import gaussian_filter, uniform_filter

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path=[], pair_path=[], is_validation=False, mul=1, pair_mul=1, downsamples=0, epoch_size=None, cropsize=256, vocal_mix_rate=0, mixup_rate=0, mixup_alpha=1, include_phase=False, force_voxaug=False, use_noise=True, noise_rate=1.0, gamma=0.5, sigma=0.4, alpha=-0.8, clip_validation=False):
        self.epoch_size = epoch_size
        self.mul = mul
        self.cropsize = cropsize
        self.vocal_mix_rate = vocal_mix_rate
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.include_phase = include_phase
        self.force_voxaug = force_voxaug
        self.use_noise = use_noise
        self.noise_rate = noise_rate
        self.vidx = 0
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.clip_validation = clip_validation

        if self.force_voxaug:
            print('# Forcing vocal augmentations')
        
        patch_list = []
        self.vocal_list = []

        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                patch_list.append(m)

        for mp in pair_path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                for _ in range(pair_mul):
                    patch_list.append(m)
        
        if not is_validation and len(vocal_path) != 0:
            for vp in vocal_path:
                vox = [os.path.join(vp, f) for f in os.listdir(vp) if os.path.isfile(os.path.join(vp, f))]

                for v in vox:
                    self.vocal_list.append(v)

        self.is_validation = is_validation

        self.curr_list = []
        for p in patch_list:
            self.curr_list.append(p)

        self.downsamples = downsamples
        self.full_list = self.curr_list

        if not is_validation and self.epoch_size is not None:
            self.curr_list = self.full_list[:self.epoch_size]            
            self.rebuild()

    def rebuild(self):
        self.vidx = 0
        random.shuffle(self.vocal_list)

        if self.epoch_size is not None:
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:self.epoch_size]

    def __len__(self):
        return len(self.curr_list) * self.mul

    def _get_vocals(self):          
        vpath = self.vocal_list[np.random.randint(len(self.vocal_list))]
        vdata = np.load(str(vpath))
        V, c = vdata['X'], vdata['c']

        if V.shape[2] > self.cropsize:
            start = np.random.randint(0, V.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            V = V[:,:,start:stop]
        
        if np.random.uniform() < 0.5:
            V = V[::-1]

        if np.random.uniform() < 0.025:
            if np.random.uniform() < 0.5:
                V[0] = 0
            else:
                V[1] = 0

        return V

    def __getitem__(self, idx):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]
            Y = Y[:,:,start:stop]

        Y = np.abs(Y) / c

        if np.random.uniform() < 0.5:
            Y = Y[::-1]

        if np.random.uniform() > 0.025:
            V = self._get_vocals()

            gamma1 = np.random.uniform(0.01, 0.2)
            gamma2 = np.random.uniform(0.01, 0.2)
            gamma3 = np.random.uniform(0.01, 0.2)
            
            v0 = np.abs(V) / c
            vm = v0 * 2 - 1
            xm = Y * 2 - 1
            v1 = uniform_filter(np.where(vm > self.alpha, v0 * np.sqrt(gamma1) * np.random.normal(size=vm.shape), 0), size=3)
            v2 = uniform_filter(np.where(vm > self.alpha, v0 * np.sqrt(gamma2) * np.random.normal(size=vm.shape), 0), size=7)
            v3 = np.where(vm > self.alpha, v0 * np.sqrt(gamma3) * np.random.normal(scale=1, size=vm.shape), 0)
            xm = xm + v1 + v2 + v3
            xm = np.clip((xm + 1) * 0.5, 0, 1)

            X = xm
        else:
            X = Y
            
        return X.astype(np.float32), Y.astype(np.float32)