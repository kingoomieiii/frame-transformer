import os
import random
import numpy as np
import torch
import re
import torch.utils.data
import torch.nn.functional as F
from libft2gan.dataset_utils import apply_channel_drop, apply_dynamic_range_mod, apply_random_phase_noise, apply_random_phase_noise_pair, apply_multiplicative_noise, apply_multiplicative_noise_pair, apply_random_eq, apply_stereo_spatialization, apply_time_stretch, apply_pitch_shift, apply_dynamic_range_mod_pair, apply_random_eq_pair, apply_channel_drop_pair, apply_stereo_spatialization_pair
import librosa

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, instrumental_lib=[], vocal_lib=[], is_validation=False, n_fft=2048, hop_length=1024, cropsize=256, sr=44100, seed=0, inst_rate=0.01, data_limit=None, predict_vocals=False, time_scaling=True, vocal_threshold=0.001, vout_bands=4, predict_phase=False):
        self.is_validation = is_validation
        self.vocal_list = []
        self.curr_list = []
        self.epoch = 0
        self.inst_rate = inst_rate
        self.predict_vocals = predict_vocals
        self.time_scaling = time_scaling
        self.vocal_threshold = vocal_threshold
        self.vout_bands = vout_bands
        self.predict_phase = predict_phase

        self.max_bin = n_fft // 2
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cropsize = cropsize

        self.random = random.Random(seed)

        for mp in instrumental_lib:
            mixes = [os.path.join(mp, f) for f in os.listdir(mp) if os.path.isfile(os.path.join(mp, f))]

            for m in mixes:
                if m.endswith('.npz'):
                    self.curr_list.append(m)
            
        if not is_validation and (vocal_lib != None and len(vocal_lib) != 0):
            for vp in vocal_lib:
                vox = [os.path.join(vp, f) for f in os.listdir(vp) if os.path.isfile(os.path.join(vp, f))]

                for v in vox:
                    if v.endswith('.npz'):
                        self.vocal_list.append(v)

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

        if V.shape[2] > self.cropsize:
            start = self.random.randint(0, V.shape[2] - self.cropsize - 1)
            V = V[:, :, start:start+self.cropsize]

        P = np.angle(V)
        M = np.abs(V)

        if np.random.uniform() < 0.02:
            if np.random.uniform() < 0.5:
                M[0, :, :] = 0
            else:
                M[1, :, :] = 0

        V = M * np.exp(1.j * P)

        if self.random.uniform(0,1) < 0.5:
            V = V[::-1]

        return V

    def _augment_instruments(self, X, path, is_validation=False):
        m = re.search(r"_p(\d+)\.npz$", path)
        m2 = re.search(r"(.*_p)\d+\.npz$", path)
        curr_idx = int(m.group(1))
        base_file = m2.group(1)
        prev_idx = curr_idx - 1

        P = np.angle(X)
        M = np.abs(X)

        if prev_idx >= 0:
            prev_file = f"{base_file}{prev_idx}.npz"

            if os.path.exists(prev_file):
                PX = np.load(prev_file, allow_pickle=True)['X' if not is_validation else 'Y']
                PP = np.angle(PX)
                PM = np.abs(PX)
            else:
                print(f'{prev_file} does not exist; treating as beginning')
                PX = np.zeros_like(M) * np.exp(1.j * P)
                PP = np.angle(PX)
                PM = np.abs(PX)
        else:
            PX = np.zeros_like(M) * np.exp(1.j * P)
            PP = np.angle(PX)
            PM = np.abs(PX)

        if not is_validation and np.random.uniform() < 0.02:
            if np.random.uniform() < 0.5:
                M[0, :, :] = 0
                PM[0, :, :] = 0
            else:
                M[1, :, :] = 0
                PM[1, :, :] = 0

        X = M * np.exp(1.j * P)
        PX = PM * np.exp(1.j * PP)

        if not is_validation and self.random.uniform(0,1) < 0.5:
            X = X[::-1]
            PX = PX[::-1]

        return X, PX

    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])     

        data = np.load(path, allow_pickle=True)
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']
        V, VP = None, np.zeros((X.shape[0], self.vout_bands, X.shape[2]))

        Y, PY = self._augment_instruments(Y, path, self.is_validation)

        if not self.is_validation:
            V = self._get_vocals(idx)
            X = Y + V

        c = np.max([c, np.abs(X).max()])

        X = np.clip(np.abs(X) / c, 0, 1)
        PY = np.clip(np.abs(PY) / c, 0, 1)
        Y = np.clip(np.abs(Y) / c, 0, 1)

        X = np.concatenate((X, PY), axis=0)

        return X.astype(np.float32), Y.astype(np.float32)