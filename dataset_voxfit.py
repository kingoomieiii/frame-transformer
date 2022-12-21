import numpy as np
import torch
import torch.utils.data

from lib.dataset import make_overfit_dataset

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, song, dir, mul=1, cropsize=256, sr=44100, hop_length=1024, n_fft=2048):
        self.X_set, self.Y_set, self.X_path, self.Y_path, self.c = make_overfit_dataset(song, dir, cropsize, sr, hop_length, n_fft)
        self.mul = mul
        self.cropsize = cropsize
        
    def __len__(self):
        return len(self.X_set) * self.mul

    def __getitem__(self, idx):
        X = self.X_set[idx % len(self.X_set)]
        Y = self.Y_set[idx % len(self.X_set)]

        if X.shape[2] > self.cropsize:
            start = np.random.randint(0, X.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            X = X[:,:,start:stop]
            Y = Y[:,:,start:stop]
            
        X = np.abs(X)
        Y = np.abs(Y)
        X = np.where(Y > X, Y, X)

        X = np.clip(X / self.c, 0, 1)
        Y = np.clip(Y / self.c, 0, 1)

        return X, Y