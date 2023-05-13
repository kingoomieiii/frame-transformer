import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import librosa

import pedalboard

def normalize_waveform(W, W2=None):
    if W2 is not None:
        normalized_waveform = W / np.max([1, np.abs(W).max(), np.abs(W2).max()])
    else:
        normalized_waveform = W / np.max([1, np.abs(W).max()])

    return normalized_waveform

def quantize_waveform(waveform, num_levels=256):
    quantized_waveform = np.round(waveform * (num_levels - 1)).astype(np.int)
    return quantized_waveform

def one_hot_encode_waveform(quantized_waveform, num_levels=256):
    one_hot_waveform = np.eye(num_levels)[quantized_waveform]
    return one_hot_waveform

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

        # self.curr_list = self.curr_list[:16]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.curr_list)

    def _get_vocals(self, idx):
        path = str(self.vocal_list[(self.epoch + idx) % len(self.vocal_list)])
        vdata = np.load(path, allow_pickle=True)
            
        W, Vc = vdata['XW'][:2], vdata['c']
        
        if (W.shape[1] // self.hop_length) > self.cropsize:
            start = self.random.randint(0, (W.shape[1] // self.hop_length) - self.cropsize - 1)
            ws = start * self.hop_length
            we = (start + self.cropsize) * self.hop_length
            W = W[:, ws:we]

        augmentations = [
            (0.2, pedalboard.Compressor(threshold_db=np.random.uniform(-30,-10), ratio=np.random.uniform(1.5, 10.0), attack_ms=np.random.uniform(1,50), release_ms=np.random.uniform(50,500))),
            (0.2, pedalboard.Distortion(drive_db=np.random.uniform(0,15))),
            (0.1, pedalboard.HighpassFilter(cutoff_frequency_hz=np.random.uniform(0,1000))),
            (0.2, pedalboard.LowpassFilter(cutoff_frequency_hz=np.random.uniform(2000,10000))),
            (0.1, pedalboard.HighShelfFilter(cutoff_frequency_hz=np.random.uniform(1000, 16000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5, 2) )),
            (0.1, pedalboard.LowShelfFilter(cutoff_frequency_hz=np.random.uniform(1, 1000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5, 2) )),
            (0.25, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(25,500), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.25, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(300,1200), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.25, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(1000,4000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0.25, pedalboard.PeakFilter(cutoff_frequency_hz=np.random.uniform(4000,12000), gain_db=np.random.uniform(-6,6), q=np.random.uniform(0.5,2))),
            (0., pedalboard.Limiter(threshold_db=np.random.uniform(-12,-3), release_ms=np.random.uniform(50,200))),
            (0.2, pedalboard.NoiseGate(threshold_db=np.random.uniform(-100,-20), ratio=np.random.uniform(1,10), attack_ms=np.random.uniform(0.1, 10), release_ms=np.random.uniform(20, 200))),
            (0.2, pedalboard.PitchShift(np.random.uniform(-12,12))),
            (0.2, pedalboard.MP3Compressor(vbr_quality=np.random.uniform(1,6))),
            (0.2, pedalboard.Invert())
        ] 

        random.shuffle(augmentations)

        for p, aug in augmentations:
            if self.random.uniform(0,1) < p:
                W = aug.process(W, sample_rate=self.sr)
                W = normalize_waveform(W)
                
        if np.random.uniform() < 0.04:
            if np.random.uniform() < 0.5:
                W[0] = 0
            else:
                W[1] = 0

        if self.random.uniform(0,1) < 0.5:
            W = W[::-1]

        # VL = librosa.stft(W[0], n_fft=self.n_fft, hop_length=self.hop_length)
        # VR = librosa.stft(W[1], n_fft=self.n_fft, hop_length=self.hop_length)
        # V = np.stack([VL, VR], axis=0)

        # VP = V[:, :-1, :]
        # VP = (np.abs(VP) / Vc).reshape((VP.shape[0], self.vout_bands, VP.shape[1] // self.vout_bands, VP.shape[2]))
        # VP = VP.mean(axis=2)
        # VP = np.where(VP > self.vocal_threshold, 1, 0)

        # VPmax = np.max(VP, axis=1)
        # WP = np.zeros((W.shape[0], W.shape[1]))
        # for n in range(VPmax.shape[1]):
        #     start = n * self.hop_length
        #     end = start + self.hop_length
        #     WP[:, start:end] = VPmax[:, n, None]

        return W#, WP, VP

    def _augment_instruments(self, W):
        if (W.shape[1] // self.hop_length) > self.cropsize:
            start = self.random.randint(0, (W.shape[1] // self.hop_length) - self.cropsize - 1)
            ws = start * self.hop_length
            we = (start + self.cropsize) * self.hop_length
            W = W[:, ws:we]

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
            (0.2, pedalboard.Invert()),
            (0.1, pedalboard.Limiter(threshold_db=np.random.uniform(-12,-3), release_ms=np.random.uniform(50,200))),
            (0.1, pedalboard.NoiseGate(threshold_db=np.random.uniform(-100,-20), ratio=np.random.uniform(1,10), attack_ms=np.random.uniform(0.1, 10), release_ms=np.random.uniform(20, 200))),
            (0.2, pedalboard.PitchShift(np.random.uniform(-4,4))),
            (0.2, pedalboard.MP3Compressor(vbr_quality=np.random.uniform(1,6))),
        ] 

        random.shuffle(augmentations)

        for p, aug in augmentations:
            if self.random.uniform(0,1) < p:
                W = aug.process(W, sample_rate=self.sr)
                W = np.clip(W, -1, 1)

        if np.random.uniform() < 0.04:
            if np.random.uniform() < 0.5:
                W[0] = 0
            else:
                W[1] = 0

        if self.random.uniform(0,1) < 0.5:
            W = W[::-1]

        return W
    
    def __getitem__(self, idx):
        path = str(self.curr_list[idx % len(self.curr_list)])
        data = np.load(path, allow_pickle=True)
        aug = 'Y' not in data.files

        XW, c = data['XW'][:2], data['c']
        YW  = XW if aug else data['YW'][:2]
        VS, VP, WP = None, np.zeros((XW.shape[0], self.vout_bands, ((XW.shape[1] // self.hop_length)))), np.zeros((XW.shape[0], XW.shape[1]))

        #cw = np.abs(XW).max()

        if not self.is_validation:
            YW = self._augment_instruments(XW)
            VW = self._get_vocals(idx)
            XW = normalize_waveform(normalize_waveform(YW) + normalize_waveform(VW))
            
        elif self.is_validation:
            if (XW.shape[1] // self.hop_length) > self.cropsize:
                start = self.random.randint(0, (XW.shape[1] // self.hop_length) - self.cropsize - 1)
                ws = start * self.hop_length
                we = (start + self.cropsize) * self.hop_length
                XW = XW[:, ws:we]
                YW = YW[:, ws:we]

        YW = normalize_waveform(YW)
        
        # XL = librosa.stft(XW[0], n_fft=self.n_fft, hop_length=self.hop_length)
        # XR = librosa.stft(XW[1], n_fft=self.n_fft, hop_length=self.hop_length)
        # XS = np.stack([XL, XR], axis=0)

        # XML = librosa.feature.melspectrogram(S=np.abs(XS[0]), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        # XMR = librosa.feature.melspectrogram(S=np.abs(XS[1]), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        # XM = np.stack([XML, XMR], axis=0)
        # XM = XM / np.max([c, XM.max()])
        
        # YL = librosa.stft(YW[0], n_fft=self.n_fft, hop_length=self.hop_length)
        # YR = librosa.stft(YW[1], n_fft=self.n_fft, hop_length=self.hop_length)
        # YS = np.stack([YL, YR], axis=0)

        # c = np.max([c, np.abs(XS).max(), np.abs(YS).max()])
        # XP = (np.angle(XS) + np.pi) / (2 * np.pi)
        # YP = (np.angle(YS) + np.pi) / (2 * np.pi)
        # XS = (np.abs(XS) / c)
        # YS = (np.abs(YS) / c)
        #XS = np.concatenate((XS, XP), axis=0)
        # XS = np.concatenate((XS, XP), axis=0)
        # YS = np.concatenate((YS, YP), axis=0)
        # XW = (XW + 1) * 0.5
        # YW = (YW + 1) * 0.5

        return XW.astype(np.float32), YW.astype(np.float32), c.astype(np.float32)
        