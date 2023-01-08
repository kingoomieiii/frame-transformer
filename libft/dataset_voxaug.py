import os
import random
import numpy as np
import torch
import torch.utils.data

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path=[], is_validation=False, cropsize=256, seed=0, inst_rate=0.025, data_limit=None):
        self.is_validation = is_validation
        self.cropsize = cropsize
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.inst_rate = inst_rate

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

        if data_limit is not None:
            self.curr_list = self.curr_list[:data_limit]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.curr_list)

    def _get_vocals(self, idx, root=True):
        path = str(self.vocal_list[(self.epoch + idx) % len(self.vocal_list)])
        vdata = np.load(path)
        V = vdata['X']

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

        if np.random.uniform() < 0.075:
            V2 = self._get_vocals(np.random.randint(len(self.vocal_list)))
            V = V + V2

        return V

    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])
        data = np.load(path)
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']
        
        if not self.is_validation:
            if Y.shape[2] > self.cropsize:
                start = np.random.randint(0, Y.shape[2] - self.cropsize + 1)
                stop = start + self.cropsize
                Y = Y[:,:,start:stop]

            V = self._get_vocals(idx)
            X = Y + V
            c = np.max([c, np.abs(X).max()])

            if np.random.uniform() < self.inst_rate:
                X = Y

            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]
        else:
            if len(self.vocal_list) > 0:
                vpath = self.vocal_list[idx % len(self.vocal_list)]
                vdata = np.load(str(vpath))
                V = vdata['X']
                X = Y + V

        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)
        
        return X, Y