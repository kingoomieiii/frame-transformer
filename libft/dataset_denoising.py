from math import nan
import math
import os
import random
import numpy as np
import torch
import torch.utils.data

class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, path, gamma=0.95, sigma=0.4, epoch_size=None, cropsize=256, is_validation=False, half_range=True):
        self.epoch_size = epoch_size
        self.cropsize = cropsize
        self.gamma = gamma
        self.sigma = sigma
        self.is_validation = is_validation
        self.half_range = half_range

        self.curr_list = []
        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                self.curr_list.append(m)
        
        self.full_list = self.curr_list

        self.rebuild()

    def rebuild(self):
        if self.epoch_size is not None:
            end = math.ceil(len(self.full_list) * self.epoch_size)
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:end]

    def __len__(self):
        return len(self.curr_list)

    def __getitem__(self, idx, root=True):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        X, Xc = data['X'], data['c']
        
        if not self.is_validation:
            c = Xc

            if np.random.uniform() < 0.5:
                X = X[::-1]
        else:
            c = Xc

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]

        X = (np.abs(X) / c) * 2 - 1.0
        Y = X.copy()
        eps = np.random.normal(scale=self.sigma, size=X.shape)
        X = np.sqrt(self.gamma) * X + np.sqrt(1 - self.gamma) * eps

        X = np.clip((X + 1) * 0.5, 0, 1)
        Y = np.clip((Y + 1) * 0.5, 0, 1)
        
        return X.astype(np.float32), Y.astype(np.float32)