from math import nan
import math
import os
import random
import numpy as np
import torch
import torch.utils.data

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path=[], is_validation=False, mul=1, downsamples=0, epoch_size=None, cropsize=256, vocal_mixup_rate=0, mixup_rate=0, mixup_alpha=1, include_phase=False, force_voxaug=False, use_noise=True, gamma=0.95, sigma=0.25):
        self.epoch_size = epoch_size
        self.mul = mul
        self.cropsize = cropsize
        self.vocal_mixup_rate = vocal_mixup_rate
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.include_phase = include_phase
        self.force_voxaug = force_voxaug
        self.use_noise = use_noise
        self.gamma = gamma
        self.sigma = sigma
        self.vidx = 0

        if self.force_voxaug:
            print('# Forcing vocal augmentations')
        
        patch_list = []
        self.vocal_list = []

        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
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
        print('shuffle')
        random.shuffle(self.vocal_list)

        if self.epoch_size is not None:
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:self.epoch_size]

    def __len__(self):
        return len(self.curr_list) * self.mul

    def _get_vocals(self, root=True):
        vidx = np.random.randint(len(self.vocal_list))                
        vpath = self.vocal_list[vidx]
        vdata = np.load(str(vpath))
        V = vdata['X']

        if V.shape[2] > self.cropsize:
            start = np.random.randint(0, V.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            V = V[:,:,start:stop]

        if np.random.uniform() < 0.5:
            V = V[::-1]

        if np.random.uniform() < 0.025:
            if np.random.uniform() < 0.5:
                V[0] = V[0] * 0
            else:
                V[1] = V[1] * 0

        if np.random.uniform() < self.vocal_mixup_rate and root:
            V2 = self._get_vocals(np.random.randint(len(self.vocal_list)), root=False)
            V = V + V2

        return V

    def __getitem__(self, idx, root=True):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        aug = 'Y' not in data.files
        X, Xc = data['X'], data['c']
        Y = X if aug else data['Y']

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]
            Y = Y[:,:,start:stop]

        if not self.is_validation:
            if self.force_voxaug:
                X, aug = Y, True

            if aug and np.random.uniform() > 0.02:
                V = self._get_vocals()
                X = Y + V
                c = np.max([Xc, np.abs(X).max()])

                if np.random.uniform() < 0.025:
                    X = Y
                    c = Xc
            else:
                c = Xc

            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]
        else:
            if len(self.vocal_list) > 0:                              
                vpath = self.vocal_list[idx % len(self.vocal_list)]
                vdata = np.load(str(vpath))
                V = vdata['X']
                X = Y + V
                c = np.max([Xc, np.abs(X).max()])
            else:
                c = Xc

        if np.random.uniform() < self.mixup_rate and root and not self.is_validation:
            MX, MY = self.__getitem__(np.random.randint(len(self)), root=False)
            a = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            X = X * a + (1 - a) * MX
            Y = Y * a + (1 - a) * MY
            
        if root:
            X = (np.clip(np.abs(X) / c, 0, 1) * 2) - 1
            Y = np.clip(np.abs(Y) / c, 0, 1)

        return X, Y