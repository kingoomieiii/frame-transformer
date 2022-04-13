from math import nan
import math
import os
import random
import time

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

try:
    from lib import spec_utils
except ModuleNotFoundError:
    import spec_utils

class VocalAugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, inst_a_path, inst_b_path=None, pair_path=None, vocal_path=None, is_validation=False, epoch_size=None, fake_data_prob=math.nan, vocal_recurse_prob=0.25, vocal_recurse_prob_decay=0.5, vocal_noise_prob=0.5, vocal_noise_magnitude=0.5, vocal_pan_prob=0.5, instrumental_mixup_rate=0.25):
        self.epoch_size = epoch_size
        self.is_validation = is_validation

        self.fake_data_prob = fake_data_prob
        self.vocal_recurse_prob = vocal_recurse_prob
        self.vocal_recurse_prob_decay = vocal_recurse_prob_decay
        self.vocal_noise_prob = vocal_noise_prob
        self.vocal_noise_magnitude = vocal_noise_magnitude
        self.vocal_pan_prob = vocal_pan_prob
        self.instrumental_mixup_rate = instrumental_mixup_rate
        
        self.inst_list = []
        self.pair_list = []
        self.curr_list = []
        
        if not is_validation:
            if vocal_path is not None:
                self.vocal_list = [os.path.join(vocal_path, f) for f in os.listdir(vocal_path) if os.path.isfile(os.path.join(vocal_path, f))]

            if pair_path is not None:
                self.pair_list = [os.path.join(pair_path, f) for f in os.listdir(pair_path) if os.path.isfile(os.path.join(pair_path, f))]

            self.inst_list = [os.path.join(inst_a_path, f) for f in os.listdir(inst_a_path) if os.path.isfile(os.path.join(inst_a_path, f))]
            
            if inst_b_path is not None:
                self.inst_list.extend([os.path.join(inst_b_path, f) for f in os.listdir(inst_b_path) if os.path.isfile(os.path.join(inst_b_path, f))])

            self.rebuild()
        else:
            self.curr_list = [os.path.join(inst_a_path, f) for f in os.listdir(inst_a_path) if os.path.isfile(os.path.join(inst_a_path, f))]

    def rebuild(self):
        self.curr_list = []

        if not math.isnan(self.fake_data_prob):
            for _ in range(self.epoch_size if self.epoch_size is not None else (len(self.pair_list) + len(self.inst_list))):
                if np.random.uniform() < self.fake_data_prob:
                    idx = np.random.randint(len(self.inst_list))
                    self.curr_list.append(self.inst_list[idx])
                else:
                    idx = np.random.randint(len(self.pair_list))
                    self.curr_list.append(self.pair_list[idx])
        else:
            self.curr_list.extend(self.pair_list)
            self.curr_list.extend(self.inst_list)

    def __len__(self):
        return len(self.curr_list)

    def __getitem__(self, idx):
        path = self.curr_list[idx]
        data = np.load(str(path))
        aug = "Y" not in data.files

        X, Xc = data['X'], data['c']
        Y = X if aug else data['Y']

        if not self.is_validation:
            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]

            if aug or np.random.uniform() < 0.05:
                if np.random.uniform() < self.instrumental_mixup_rate:
                    path2 = self.curr_list[np.random.randint(len(self.curr_list))]
                    data2 = np.load(str(path2))
                    X2, X2c = data2['X'], data2['c']
                    Y2 = X2 if "Y" not in data2.files else data2['Y']
                    a = np.random.beta(0.4, 0.4)
                    X2c = (a * X2c) + ((1-a) * Xc)
                    Y = (a * Y2) + ((1-a) * Y)
                    Xc = np.max([Xc, X2c, np.abs(Y).max()])

                V, Vc = self._get_vocals()
                X = Y + V
                c = np.max([Xc, Vc, np.abs(X).max()])
            else:
                if np.random.uniform() < 0.33:
                    V, Vc = self._get_vocals()
                    X = X + (V * np.random.beta(0.4, 1))
                    c = np.max([Xc, Vc, np.abs(X).max()])
                else:
                    c = Xc  
                
            if np.random.uniform() < 0.02:
                X = Y
                c = Xc
        else:
            c = Xc

        return np.abs(X) / c, np.abs(Y) / c

    def _get_vocals(self, recurse_prob=None):
        recurse_prob = self.vocal_recurse_prob if recurse_prob is None else recurse_prob
        idx = np.random.randint(len(self.vocal_list))        
        path = self.vocal_list[idx]
        data = np.load(str(path))
        V, Vc = data['X'], data['c']

        if np.random.uniform() < 0.5:
            V = V[::-1]

        if np.random.uniform() < self.vocal_pan_prob:
            if np.random.uniform() < 0.5:
                V[0] = V[0] * np.random.beta(0.4,1)
            else:
                V[1] = V[1] * np.random.beta(0.4,1)

        if np.random.uniform() < self.vocal_noise_prob:
            noise = np.random.beta(1, 1, size=(V.shape[0], V.shape[1], V.shape[2])).astype('f')
            V = ((1-self.vocal_noise_magnitude) * V) + (self.vocal_noise_magnitude * noise * V)

        if np.random.uniform() < recurse_prob:
            V2, Vc2 = self._get_vocals(recurse_prob=recurse_prob * self.vocal_recurse_prob_decay)
            a = np.random.beta(0.4, 0.4)
            Vc = (a * Vc2) + ((1-a) * Vc)
            V = (a * V2) + ((1-a) * V)

        return V, np.max([Vc, np.abs(V).max()])

class VocalRemoverTrainingSet(torch.utils.data.Dataset):

    def __init__(self, training_set, cropsize, reduction_rate, reduction_weight, mixup_rate, mixup_alpha):
        self.training_set = training_set
        self.cropsize = cropsize
        self.reduction_rate = reduction_rate
        self.reduction_weight = reduction_weight
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.training_set)

    def do_crop(self, X_path, y_path):
        X_mmap = np.load(X_path, mmap_mode='r')
        y_mmap = np.load(y_path, mmap_mode='r')

        start = np.random.randint(0, X_mmap.shape[2] - self.cropsize)
        end = start + self.cropsize

        X_crop = np.array(X_mmap[:, :, start:end], copy=True)
        y_crop = np.array(y_mmap[:, :, start:end], copy=True)

        return X_crop, y_crop

    def do_aug(self, X, y):
        if np.random.uniform() < self.reduction_rate:
            y = spec_utils.aggressively_remove_vocal(X, y, self.reduction_weight)

        if np.random.uniform() < 0.5:
            # swap channel
            X = X[::-1].copy()
            y = y[::-1].copy()

        if np.random.uniform() < 0.01:
            # inst
            X = y.copy()

        # if np.random.uniform() < 0.01:
        #     # mono
        #     X[:] = X.mean(axis=0, keepdims=True)
        #     y[:] = y.mean(axis=0, keepdims=True)

        return X, y

    def do_mixup(self, X, y):
        idx = np.random.randint(0, len(self))
        X_path, y_path, coef = self.training_set[idx]

        X_i, y_i = self.do_crop(X_path, y_path)
        X_i /= coef
        y_i /= coef

        X_i, y_i = self.do_aug(X_i, y_i)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        X = lam * X + (1 - lam) * X_i
        y = lam * y + (1 - lam) * y_i

        return X, y

    def __getitem__(self, idx):
        X_path, y_path, coef = self.training_set[idx]

        X, y = self.do_crop(X_path, y_path)
        X /= coef
        y /= coef

        X, y = self.do_aug(X, y)

        if np.random.uniform() < self.mixup_rate:
            X, y = self.do_mixup(X, y)

        X_mag = np.abs(X)
        y_mag = np.abs(y)

        return X_mag, y_mag


class VocalRemoverValidationSet(torch.utils.data.Dataset):

    def __init__(self, patch_list):
        self.patch_list = patch_list

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        path = self.patch_list[idx]
        data = np.load(path)

        X, y = data['X'], data['y']

        X_mag = np.abs(X)
        y_mag = np.abs(y)

        return X_mag, y_mag


def make_pair(mix_dir, inst_dir, voxaug=False):
    input_exts = ['.wav', '.m4a', '.mp3', '.mp4', '.flac']

    y_list = sorted([
        os.path.join(inst_dir, fname)
        for fname in os.listdir(inst_dir)
        if os.path.splitext(fname)[1] in input_exts])

    if not voxaug:    
        X_list = sorted([
        os.path.join(mix_dir, fname)
        for fname in os.listdir(mix_dir)
        if os.path.splitext(fname)[1] in input_exts])

    else:
        X_list = y_list

    filelist = list(zip(X_list, y_list))

    return filelist


def train_val_split(dataset_dir, val_filelist, selected_validation=[], voxaug=False):
    filelist = make_pair(
        os.path.join(dataset_dir, 'mixtures'),
        os.path.join(dataset_dir, 'instruments'),
        voxaug=voxaug)

    train_filelist = list()
    val_filelist = list()

    for i, entry in enumerate(filelist):
        if len(selected_validation) > 0:
            validation_file = False
            for v in selected_validation:
                if entry[0].find(v) != -1:
                    validation_file = True
                    break
            
            if validation_file:
                val_filelist.append(entry)
            else:
                train_filelist.append(entry)
        else:
            train_filelist.append(entry)

    return train_filelist, val_filelist

def train_val_split(dataset_dir, split_mode, val_rate, val_filelist):
    if split_mode == 'random':
        filelist = make_pair(
            os.path.join(dataset_dir, 'mixtures'),
            os.path.join(dataset_dir, 'instruments')
        )

        random.shuffle(filelist)

        if len(val_filelist) == 0:
            val_size = int(len(filelist) * val_rate)
            train_filelist = filelist[:-val_size]
            val_filelist = filelist[-val_size:]
        else:
            train_filelist = [
                pair for pair in filelist
                if list(pair) not in val_filelist
            ]
    elif split_mode == 'subdirs':
        if len(val_filelist) != 0:
            raise ValueError('`val_filelist` option is not available with `subdirs` mode')

        train_filelist = make_pair(
            os.path.join(dataset_dir, 'training/mixtures'),
            os.path.join(dataset_dir, 'training/instruments')
        )

        val_filelist = make_pair(
            os.path.join(dataset_dir, 'validation/mixtures'),
            os.path.join(dataset_dir, 'validation/instruments')
        )

    return train_filelist, val_filelist


def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def make_training_set(filelist, sr, hop_length, n_fft):
    ret = []
    for X_path, y_path in tqdm(filelist):
        X, y, X_cache_path, y_cache_path = spec_utils.cache_or_load(
            X_path, y_path, sr, hop_length, n_fft
        )
        coef = np.max([np.abs(X).max(), np.abs(y).max()])
        ret.append([X_cache_path, y_cache_path, coef])

    return ret


def make_validation_set(filelist, cropsize, sr, hop_length, n_fft, offset):
    patch_list = []
    patch_dir = 'cs{}_sr{}_hl{}_nf{}_of{}'.format(cropsize, sr, hop_length, n_fft, offset)
    os.makedirs(patch_dir, exist_ok=True)

    for X_path, y_path in tqdm(filelist):
        basename = os.path.splitext(os.path.basename(X_path))[0]

        X, y, _, _ = spec_utils.cache_or_load(X_path, y_path, sr, hop_length, n_fft)
        coef = np.max([np.abs(X).max(), np.abs(y).max()])
        X, y = X / coef, y / coef

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode='constant')

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(patch_dir, '{}_p{}.npz'.format(basename, j))
            start = j * roi_size
            if not os.path.exists(outpath):
                np.savez(
                    outpath,
                    X=X_pad[:, :, start:start + cropsize],
                    y=y_pad[:, :, start:start + cropsize]
                )
            patch_list.append(outpath)

    return patch_list


def get_oracle_data(X, y, oracle_loss, oracle_rate, oracle_drop_rate):
    k = int(len(X) * oracle_rate * (1 / (1 - oracle_drop_rate)))
    n = int(len(X) * oracle_rate)
    indices = np.argsort(oracle_loss)[::-1][:k]
    indices = np.random.choice(indices, n, replace=False)
    oracle_X = X[indices].copy()
    oracle_y = y[indices].copy()

    return oracle_X, oracle_y, indices

def make_vocal_stems(dataset, cropsize=1024, sr=44100, hop_length=512, n_fft=1024, offset=0, root=''):
    input_exts = ['.wav', '.m4a', '.mp3', '.mp4', '.flac']

    filelist = sorted([
        os.path.join(dataset, fname)
        for fname in os.listdir(dataset)
        if os.path.splitext(fname)[1] in input_exts])

    for i, X_path in enumerate(tqdm(filelist)):
        basename = os.path.splitext(os.path.basename(X_path))[0]
        xw, _ = spec_utils.load_wave(X_path, X_path, sr)

        patch_dir = '{}cs{}_sr{}_hl{}_nf{}_of{}{}'.format(root, cropsize, sr, hop_length, n_fft, 0, "_VOCALS")
        os.makedirs(patch_dir, exist_ok=True)

        X, _ = spec_utils.to_spec(xw, xw, hop_length=hop_length, n_fft=n_fft)
        coef = np.abs(xw).max()

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(patch_dir, '{}_p{}.npz'.format(basename, j))

            start = j * roi_size
            if not os.path.exists(outpath) and coef != 0:
                xp = X_pad[:, :, start:start + cropsize]
                xc = np.abs(xp).mean()

                if xc > 0:
                    np.savez(
                        outpath,
                        X=xp,
                        c=coef)

def make_dataset(filelist, cropsize, sr, hop_length, n_fft, offset=0, is_validation=False, suffix='', root=''):
    patch_list = []    
    patch_dir = f'{root}cs{cropsize}_sr{sr}_hl{hop_length}_nf{n_fft}_of{offset}{suffix}'
    os.makedirs(patch_dir, exist_ok=True)

    for X_path, Y_path in tqdm(filelist):
        basename = os.path.splitext(os.path.basename(X_path))[0]

        voxaug = X_path == Y_path

        X, Y = spec_utils.load(X_path, Y_path, sr, hop_length, n_fft)

        coef = np.max([np.abs(X).max(), np.abs(Y).max()])

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
        Y_pad = np.pad(Y, ((0, 0), (0, 0), (l, r)), mode='constant')

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(patch_dir, '{}_p{}.npz'.format(basename, j))
            patch_list.append(outpath)

            start = j * roi_size
            if not os.path.exists(outpath):
                if voxaug:
                    np.savez(
                        outpath,
                        X=X_pad[:, :, start:start + cropsize],
                        c=coef.item())
                else:
                    np.savez(
                        outpath,
                        X=X_pad[:, :, start:start + cropsize],
                        Y=Y_pad[:, :, start:start + cropsize],
                        c=coef.item())

    return VocalRemoverValidationSet(patch_list, is_validation=is_validation)


if __name__ == "__main__":
    import sys
    import utils

    mix_dir = sys.argv[1]
    inst_dir = sys.argv[2]
    outdir = sys.argv[3]

    os.makedirs(outdir, exist_ok=True)

    filelist = make_pair(mix_dir, inst_dir)
    for mix_path, inst_path in tqdm(filelist):
        mix_basename = os.path.splitext(os.path.basename(mix_path))[0]

        X_spec, y_spec, _, _ = spec_utils.cache_or_load(
            mix_path, inst_path, 44100, 1024, 2048
        )

        X_mag = np.abs(X_spec)
        y_mag = np.abs(y_spec)
        v_mag = X_mag - y_mag
        v_mag *= v_mag > y_mag

        outpath = '{}/{}_Vocal.jpg'.format(outdir, mix_basename)
        v_image = spec_utils.spectrogram_to_image(v_mag)
        utils.imwrite(outpath, v_image)