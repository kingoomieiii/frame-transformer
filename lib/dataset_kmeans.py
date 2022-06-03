import os
import random
import numpy as np
import torch
import torch.utils.data

class KMeansPreprocessingDataset(torch.utils.data.Dataset):
    def __init__(self, path, extra_path=None, pair_path=None, mix_path=[], vocal_path=None, is_validation=False, mul=1, downsamples=0, epoch_size=None, pair_mul=1, slide=True, cropsize=256, mixup_rate=0, mixup_alpha=1, mask_rate=0.15, next_frame_chunk_size=0, token_size=16, current_step=0, num_steps=16000):
        self.epoch_size = epoch_size
        self.slide = slide
        patch_list = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        self.cropsize = cropsize
        pair_list = []

        self.warmup_steps = num_steps
        self.token_size = token_size
        self.separator_size = token_size // 4
        self.current_step = current_step

        if pair_path is not None and pair_mul > 0:
            pairs = [os.path.join(pair_path, f) for f in os.listdir(pair_path) if os.path.isfile(os.path.join(pair_path, f))]

            for p in pairs:
                if pair_mul > 1:
                    for _ in range(pair_mul):
                        pair_list.append(p)
                else:
                    if np.random.uniform() < pair_mul:
                        pair_list.append(p)

        for mp in mix_path:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                if pair_mul > 1:
                    for _ in range(pair_mul):
                        pair_list.append(m)
                else:
                    if np.random.uniform() < pair_mul:
                        pair_list.append(m)

        if extra_path is not None:
            extra_list = [os.path.join(extra_path, f) for f in os.listdir(extra_path) if os.path.isfile(os.path.join(extra_path, f))]

            for f in extra_list:
                patch_list.append(f) 

        self.curr_list = []
        for p in patch_list:
            self.curr_list.append(p)

        for p in pair_list:
            self.curr_list.append(p)

        self.full_list = self.curr_list

        if not is_validation and self.epoch_size is not None:
            self.curr_list = self.full_list[:self.epoch_size]            
            self.rebuild()

    def rebuild(self):
        if self.epoch_size is not None:
            random.shuffle(self.full_list)
            self.curr_list = self.full_list[:self.epoch_size]

    def __len__(self):
        return len(self.curr_list)

    def __getitem__(self, idx, root=True):
        path = self.curr_list[idx % len(self.curr_list)]
        data = np.load(str(path))
        X, c = data['X'], data['c']

        if np.random.uniform() < 0.5:
            X = X[::-1]
            
        start = np.random.randint(0, X.shape[2] - self.cropsize - 1)
        stop = start + self.cropsize
        X = X[:,:,start:stop]            
        X = np.clip(np.abs(X) / c, 0, 1)
        X = np.transpose(X, (2, 0, 1))
        X = np.reshape(X, (X.shape[0] // self.token_size, -1, X.shape[1], X.shape[2])) # (Token,Frame,Channel,Bin)

        return X