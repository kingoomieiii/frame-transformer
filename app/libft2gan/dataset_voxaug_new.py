import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import librosa

from libft2gan.dataset_utils import to_wave, from_wave

import pedalboard

def create_peak_filter(center_frequency, q_value, gain_db):
    low_shelf_filter = pedalboard.LowShelfFilter()
    low_shelf_filter.cutoff_frequency_hz = center_frequency / q_value
    low_shelf_filter.gain_db = gain_db

    high_shelf_filter = pedalboard.HighShelfFilter()
    high_shelf_filter.cutoff_frequency_hz = center_frequency * q_value
    high_shelf_filter.gain_db = -gain_db

    peak_filter_board = pedalboard.Pedalboard([low_shelf_filter, high_shelf_filter])
    return peak_filter_board

class VoxAugDataset(torch.utils.data.Dataset):
    def __init__(self, instrumental_lib=[], vocal_lib=[], is_validation=False, n_fft=2048, hop_length=1024, cropsize=256, sr=44100, seed=0, inst_rate=0.01, data_limit=None, predict_vocals=False, time_scaling=True, vocal_threshold=0.001, vout_bands=4, predict_phase=False, n_mels=256):
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
        self.n_mels = n_mels

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
            
        if not is_validation and len(vocal_lib) != 0:
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

        if np.random.uniform() < 0.04:
            if np.random.uniform() < 0.5:
                V[0] = 0
            else:
                V[1] = 0

        W = to_wave(V, n_fft=self.n_fft, hop_length=self.hop_length)
        
        augmentations = [
            (0.2, pedalboard.Compressor(threshold_db=np.random.uniform(-30,-10), ratio=np.random.uniform(1.5, 10.0), attack_ms=np.random.uniform(1,50), release_ms=np.random.uniform(50,500))),
            (0.2, pedalboard.Distortion(drive_db=np.random.uniform(0,15))),
            (0.1, pedalboard.HighpassFilter(cutoff_frequency_hz=np.random.uniform(0,1000))),
            (0.2, pedalboard.LowpassFilter(cutoff_frequency_hz=np.random.uniform(2000,10000))),
            (0.1, pedalboard.HighShelfFilter(cutoff_frequency_hz=np.random.uniform(1000, 16000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5, 2) )),
            (0.1, pedalboard.LowShelfFilter(cutoff_frequency_hz=np.random.uniform(1, 1000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5, 2) )),
            (0.4, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(25,500), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.4, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(300,1200), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.4, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(1000,4000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.4, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(4000,12000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.2, pedalboard.Limiter(threshold_db=np.random.uniform(-12,-3), release_ms=np.random.uniform(50,200))),
            (0.2, pedalboard.NoiseGate(threshold_db=np.random.uniform(-100,-20), ratio=np.random.uniform(1,10), attack_ms=np.random.uniform(0.1, 10), release_ms=np.random.uniform(20, 200))),
            (0.2, pedalboard.PitchShift(np.random.uniform(-12,12))),
            (0.2, pedalboard.MP3Compressor(vbr_quality=np.random.uniform(1,6))),
            (0.2, pedalboard.Invert()),
            (0.2, pedalboard.Resample())
        ] 

        random.shuffle(augmentations)

        for p, aug in augmentations:
            if self.random.uniform(0,1) < p:
                W = np.clip(aug.process(W, sample_rate=self.sr), -1, 1)
                
        if self.random.uniform(0,1) < 0.5:
            W = W[::-1]

        VL = librosa.stft(W[0], n_fft=self.n_fft, hop_length=self.hop_length)
        VR = librosa.stft(W[1], n_fft=self.n_fft, hop_length=self.hop_length)
        V = np.stack([VL, VR], axis=0)

        VP = V[:, :-1, :]
        VP = (np.abs(VP) / Vc).reshape((VP.shape[0], self.vout_bands, VP.shape[1] // self.vout_bands, VP.shape[2]))
        VP = VP.mean(axis=2)
        VP = np.where(VP > self.vocal_threshold, 1, 0)

        return V, VP

    def _augment_instruments(self, X):
        if X.shape[2] > self.cropsize:
            start = self.random.randint(0, X.shape[2] - self.cropsize - 1)
            X = X[:, :, start:start+self.cropsize]
            I = X

        if np.random.uniform() < 0.04:
            if np.random.uniform() < 0.5:
                X[0] = 0
            else:
                X[1] = 0
            
        W = to_wave(X, n_fft=self.n_fft, hop_length=self.hop_length)

        augmentations = [
            (0.1, pedalboard.Compressor(threshold_db=np.random.uniform(-30,-10), ratio=np.random.uniform(1.5, 10.0), attack_ms=np.random.uniform(1,50), release_ms=np.random.uniform(50,500))),
            (0.1, pedalboard.Distortion(drive_db=np.random.uniform(0,15))),
            (0.1, pedalboard.HighpassFilter(cutoff_frequency_hz=np.random.uniform(0,1000))),
            (0.1, pedalboard.LowpassFilter(cutoff_frequency_hz=np.random.uniform(2000,10000))),
            (0.1, pedalboard.HighShelfFilter(cutoff_frequency_hz=np.random.uniform(1000, 16000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5, 2) )),
            (0.1, pedalboard.LowShelfFilter(cutoff_frequency_hz=np.random.uniform(1, 1000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5, 2) )),
            (0.2, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(25,500), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.2, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(300,1200), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.2, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(1000,4000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.2, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(4000,12000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.1, pedalboard.Limiter(threshold_db=np.random.uniform(-12,-3), release_ms=np.random.uniform(50,200))),
            (0.1, pedalboard.NoiseGate(threshold_db=np.random.uniform(-100,-20), ratio=np.random.uniform(1,10), attack_ms=np.random.uniform(0.1, 10), release_ms=np.random.uniform(20, 200))),
            (0.2, pedalboard.PitchShift(np.random.uniform(-6,6))),
        ] 

        random.shuffle(augmentations)

        for p, aug in augmentations:
            if self.random.uniform(0,1) < p:
                W = np.clip(aug.process(W, sample_rate=self.sr), -1, 1)

        if self.random.uniform(0,1) < 0.5:
            W = W[::-1]

        XL = librosa.stft(W[0], n_fft=self.n_fft, hop_length=self.hop_length)
        XR = librosa.stft(W[1], n_fft=self.n_fft, hop_length=self.hop_length)
        X = np.stack([XL, XR], axis=0)

        return X
    
    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])
        data = np.load(path, allow_pickle=True)
        aug = 'Y' not in data.files

        X, c = data['X'], data['c']
        Y = X if aug else data['Y']
        V, VP = None, np.zeros((X.shape[0], self.vout_bands, X.shape[2]))

        if not self.is_validation:
            Y = self._augment_instruments(Y)
            V, VP = self._get_vocals(idx)
            X = Y + V
            c = np.max([c, np.abs(X).max()])
        elif X.shape[2] > self.cropsize:
            start = self.random.randint(0, X.shape[2] - self.cropsize - 1)
            X = X[:, :, start:start+self.cropsize]
            Y = Y[:, :, start:start+self.cropsize]

        XML = librosa.feature.melspectrogram(S=np.abs(X[0]), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        XMR = librosa.feature.melspectrogram(S=np.abs(X[1]), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        XM = np.stack([XML, XMR], axis=0)
        XM = XM / np.max([c, XM.max()])

        YML = librosa.feature.melspectrogram(S=np.abs(Y[0]), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        YMR = librosa.feature.melspectrogram(S=np.abs(Y[1]), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        YM = np.stack([YML, YMR], axis=0)
        YM = YM / np.max([c, YM.max()])

        XP = (np.angle(X)) / (np.pi)
        YP = (np.angle(Y)) / (np.pi)
        XS = np.abs(X) / c
        YS = np.abs(Y) / c

        return XS.astype(np.float32), XP.astype(np.float32), XM.astype(np.float32), YS.astype(np.float32), YP.astype(np.float32), YM.astype(np.float32), VP.astype(np.float32)