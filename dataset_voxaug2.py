from math import nan
import math
import os
import random
import numpy as np
import torch
import torch.utils.data

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path=[], is_validation=False, cropsize=256, seed=0, chunks=2):
        self.is_validation = is_validation
        self.cropsize = cropsize
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.chunks = chunks

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

    def _get_vocals(self, idx, chunks):
        path = str(self.vocal_list[(self.epoch + idx) % len(self.vocal_list)])
        vdata = np.load(path)
        V = vdata['X']

        p2 = path[::-1]
        idx = p2.index('p_')
        chunk = int(path[-idx:-4])
        root = path[:-(idx+1)-1]

        i, j = 1, 1
        curr_chunks = 1
        while curr_chunks < chunks:
            ci, cj = chunk + i, chunk - j
            ir, jr = f'{root}_p{ci}.npz', f'{root}_p{cj}.npz'
            ie, je = os.path.exists(ir), os.path.exists(jr)

            if not ie and not je:
                Vj = np.zeros_like(V)
                V = np.concatenate((V, Vj), 2)
                curr_chunks += 1
                print('zero')

            if ie and np.random.uniform() < 0.5:
                i += 1
                curr_chunks += 1
                di = np.load(str(ir))
                Vi = di['Y'] if 'Y' in di.files else di['X']
                V = np.concatenate((V, Vi), 2)

            if je and curr_chunks < self.chunks:
                j += 1
                curr_chunks += 1
                dj = np.load(str(jr))
                Vj = dj['Y'] if 'Y' in dj.files else dj['X']
                V = np.concatenate((Vj, V), 2)

        if V.shape[2] > self.cropsize:
            start = np.random.randint(0, V.shape[2] - self.cropsize + 1)
            stop = start + self.cropsize
            V = V[:,:,start:stop]

        if np.random.uniform() < 0.5:
            V = V * (np.random.uniform(0.5, 1.5))

        if np.random.uniform() < 0.5:
            V = V[::-1]

        if np.random.uniform() < 0.025:
            if np.random.uniform() < 0.5:
                V[0] = 0
            else:
                V[1] = 0

        return V

    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])
        data = np.load(path)
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']
        
        if not self.is_validation:
            p2 = path[::-1]
            idx = p2.index('p_')
            chunk = int(path[-idx:-4])
            root = path[:-(idx+1)-1]

            i, j = 1, 1
            curr_chunks = 1
            while curr_chunks < self.chunks:
                ci, cj = chunk + i, chunk - j
                ir, jr = f'{root}_p{ci}.npz', f'{root}_p{cj}.npz'
                ie, je = os.path.exists(ir), os.path.exists(jr)

                if not ie and not je:
                    Yj = np.zeros_like(Y)
                    Y = np.concatenate((Y, Yj), 2)
                    curr_chunks += 1
                    print('zero')

                if ie and np.random.uniform() < 0.5:
                    i += 1
                    curr_chunks += 1
                    di = np.load(str(ir))
                    Yi = di['Y'] if 'Y' in di.files else di['X']
                    Y = np.concatenate((Y, Yi), 2)

                if je and curr_chunks < self.chunks:
                    j += 1
                    curr_chunks += 1
                    dj = np.load(str(jr))
                    Yj = dj['Y'] if 'Y' in dj.files else dj['X']
                    Y = np.concatenate((Yj, Y), 2)

            if Y.shape[2] > self.cropsize:
                start = np.random.randint(0, Y.shape[2] - self.cropsize + 1)
                stop = start + self.cropsize
                Y = Y[:,:,start:stop]

            V = self._get_vocals(idx, curr_chunks)
            X = Y + V

            if np.random.uniform() < 0.025:
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