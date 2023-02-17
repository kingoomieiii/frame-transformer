import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import scipy.ndimage

import opensimplex

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, path=[], vocal_path=[], is_validation=False, cropsize=256, seed=0, inst_rate=0.025, data_limit=None, predict_vocals=False, time_scaling=True):
        self.is_validation = is_validation
        self.cropsize = cropsize
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.inst_rate = inst_rate
        self.predict_vocals = predict_vocals
        self.time_scaling = time_scaling

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

        # # if data_limit is not None:
        # self.curr_list = self.curr_list[:512]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.curr_list)

    def _get_vocals(self, idx):
        path = str(self.vocal_list[(self.epoch + idx) % len(self.vocal_list)])
        vdata = np.load(path)
        V, Vc = vdata['X'], vdata['c']

        if V.shape[2] > self.cropsize:
            if np.random.uniform() < 0.3:
                curr_factor = self.cropsize / V.shape[2]
                scale_factor = np.random.uniform(curr_factor, 1)
                size = int(V.shape[2] * scale_factor)
                start = np.random.randint(0, V.shape[2] - size)
                cropped = V[:, :, start:start+size]
                V = V[:,:,:self.cropsize]
                V.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(V.shape[1], self.cropsize), mode='bilinear', align_corners=True).squeeze(0).numpy()
                V.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(V.shape[1], self.cropsize), mode='bilinear', align_corners=True).squeeze(0).numpy()
            else:
                if np.random.uniform() < 0.2:
                    scale_factor = np.random.uniform(0.33, 1)
                    size = int(V.shape[2] * scale_factor)
                    start = np.random.randint(0, V.shape[2] - size)
                    cropped = V[:, :, start:start+size]
                    V.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(V.shape[1], V.shape[2]), mode='bilinear', align_corners=True).squeeze(0).numpy()
                    V.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(V.shape[1], V.shape[2]), mode='bilinear', align_corners=True).squeeze(0).numpy()

                start = np.random.randint(0, V.shape[2] - self.cropsize + 1)
                stop = start + self.cropsize
                V = V[:,:,start:stop]
        else:
            if np.random.uniform() < 0.2:
                scale_factor = np.random.uniform(0.33, 1)
                size = int(V.shape[2] * scale_factor)
                start = np.random.randint(0, V.shape[2] - size)
                cropped = V[:, :, start:start+size]
                V.real = F.interpolate(torch.from_numpy(cropped.real).unsqueeze(0), size=(V.shape[1], V.shape[2]), mode='bilinear', align_corners=True).squeeze(0).numpy()
                V.imag = F.interpolate(torch.from_numpy(cropped.imag).unsqueeze(0), size=(V.shape[1], V.shape[2]), mode='bilinear', align_corners=True).squeeze(0).numpy()

        if np.random.uniform() < 0.2:
            arr1 = F.interpolate(torch.rand((1, 1, 512,)) + 0.5, size=(1025), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy() * np.random.uniform(0.75, 1.5)
            arr2 = F.interpolate(torch.rand((1, 1, 256,)) + 0.5, size=(1025), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy() * np.random.uniform(0.75, 1.5)
            arr3 = F.interpolate(torch.rand((1, 1, 128,)) + 0.5, size=(1025), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy() * np.random.uniform(0.75, 1.5)
            arr4 = F.interpolate(torch.rand((1, 1, 64,)) + 0.5, size=(1025), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy() * np.random.uniform(0.75, 1.5)
            arr5 = F.interpolate(torch.rand((1, 1, 32,)) + 0.5, size=(1025), mode='linear', align_corners=True).squeeze(0).squeeze(0).numpy() * np.random.uniform(0.75, 1.5)
            eq = (arr1 + arr2 + arr3 + arr4 + arr5) * (1.0 / 5.0)
            eq = scipy.ndimage.gaussian_filter1d(eq, sigma=1)
            eq = np.expand_dims(eq, (0, 2))
            Vs = np.abs(V) / Vc
            V = (Vs * eq) * Vc * np.exp(1.j * np.angle(V))

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
        V = None
        
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
        
        PX = (np.angle(X) * np.pi) / (2 * np.pi)
        PY = (np.angle(Y) * np.pi) / (2 * np.pi)
        X = np.clip(np.abs(X) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)

        if V is None:
            V = np.zeros_like(X)

            if self.predict_vocals:
                Y = V
        else:
            V = np.clip(np.abs(V) / c, 0, 1)

            if self.predict_vocals:
                Y = V
        
        return X, PX, Y, PY