from math import nan
import math
import os
import random
import numpy as np
import torch
import torch.utils.data

class InstMixDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], cropsize=256, num_quantizers=1, num_embeddings=1024, seed=0, batch_size=1, limit=True, iterations=1):
        self.patch_list = []
        self.cropsize = cropsize

        for mp in path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                self.patch_list.append(m)
        
        random.Random(seed+1).shuffle(self.patch_list)

        self.iterations = iterations

        if limit:
            self.patch_list = self.patch_list[:(batch_size * num_quantizers * num_embeddings)]

    def __len__(self):
        return len(self.patch_list) * self.iterations

    def __getitem__(self, idx):
        data = np.load(str(self.patch_list[idx % len(self.patch_list)]))

        X, Y, Xc, Yc = data['X'], data['Y'], data['Xc'], data['Yc']

        c = np.max([Xc, Yc])

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]
            Y = Y[:,:,start:stop]

        if np.random.uniform() < 0.5:
            X = X[::-1]
            Y = Y[::-1]
        
        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)
        
        return X, Y