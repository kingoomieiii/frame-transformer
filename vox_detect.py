import argparse
import math
import os
import shutil
import warnings
import librosa
import numpy as np
import soundfile as sf
import torch
import json
import time

import shlex, subprocess

from tqdm import tqdm
from frame_transformer_thin import FrameTransformer

from torch.nn import functional as F

from lib import dataset
from lib import spec_utils
from lib import utils
from lib import nets

from vocal_detector import VocalDetector

import ffmpeg

class Separator(object):

    def __init__(self, corrector, model, device, batchsize, cropsize, n_fft, postprocess=False):
        self.corrector = corrector
        self.model = model
        self.offset = 0
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess
        self.n_fft = n_fft

    def _separate(self, X_mag_pad, cropsize=None, padding=None):
        X_dataset = []
        cropsize = self.cropsize if cropsize is None else cropsize
        padding = cropsize // 2 if padding is None else padding
        patches = X_mag_pad.shape[2] // cropsize
        X_mag_pad = np.pad(X_mag_pad, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        for i in range(patches):
            start = (i * cropsize) + padding
            X_mag_crop = X_mag_pad[:, :, (start - padding):(start + cropsize + padding)]
            X_dataset.append(X_mag_crop)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in range(0, patches, self.batchsize):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(np.asarray(X_batch)).to(self.device)

                with torch.cuda.amp.autocast_mode.autocast():
                    pred = self.model(X_batch)
                
                if padding > 0:
                    pred = pred[:, :, :, (padding):-(padding)]

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        #mask = np.pad(mask, ((0,0), (0,1), (0, 0)))

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase, include_phase=False):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask[:2] * X_mag * np.exp(1.j * ((((mask[2:] * 2) - 1) * np.pi) if include_phase else X_phase))
        v_spec = (1 - mask[:2]) * X_mag * np.exp(1.j * X_phase)
        m_spec = mask * 255

        return y_spec, v_spec, m_spec

    def separate(self, X_spec, padding=None, include_phase=False):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, _ = dataset.make_padding(n_frame, self.cropsize, 0)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        if include_phase:
            X_phase_pad = ((X_phase / np.pi) + 1) * 0.5
            X_phase_pad = np.pad(X_phase, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad = np.concatenate((X_mag_pad, X_phase_pad), axis=0)

        mask = self._separate(X_mag_pad, self.cropsize, padding)

        mask = mask[:, :, :n_frame]
        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase, include_phase)

        return y_spec, v_spec, m_spec, mask

    def separate_tta(self, X_spec, cropsizes=[64, 128, 256, 512, 1024], paddings=[128, 256, 512, 1024, 2048]):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        X_mag_pad1 = X_mag / X_mag.max()

        mask = np.zeros_like(X_mag)

        for idx in range(len(paddings)):
            pad_l, pad_r, _ = dataset.make_padding(n_frame, paddings[idx], 0)
            X_mag_pad2 = np.pad(X_mag_pad1, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            mask += self._separate(X_mag_pad2, cropsizes[idx], paddings[idx])[:, :, :n_frame]

        mask = mask / len(paddings)

        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec, m_spec

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_corrector', type=str, default="J://models/local.9.model.pth")
    p.add_argument('--pretrained_model', '-P', type=str, default='baseline.pth')
    p.add_argument('--vocal_detector', type=str, default="voxdetector.pth")
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', type=str, default="G://dataset//novx")
    p.add_argument('--num_res_encoders', type=int, default=4)
    p.add_argument('--num_res_decoders', type=int, default=4)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=16)
    p.add_argument('--cropsize', '-c', type=int, default=128)
    p.add_argument('--padding', type=int, default=64)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--create_webm', action='store_true')
    p.add_argument('--num_encoders', type=int, default=2)
    p.add_argument('--num_decoders', type=int, default=13)
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--cropsizes', type=str, default='128,256,512,1024')
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--num_transformer_blocks', type=int, default=2)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--flag', type=str, default='false')
    p.add_argument('--name', type=str, default=None)

    p.add_argument('--include_phase', type=str, default='false')
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_res_blocks', type=int, default=1)
    p.add_argument('--feedforward_expansion', type=int, default=24)
    p.add_argument('--dropout', type=float, default=0.1)

    args = p.parse_args()

    args.flag = str.lower(args.flag) == 'true'
    args.include_phase = str.lower(args.include_phase) == 'true'
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]

    print('loading model...', end=' ')
    device = torch.device('cpu')

    model = nets.CascadedNet(args.n_fft, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

    detector = VocalDetector(latent_features=1024, num_layers=24)
    detector.load_state_dict(torch.load(args.vocal_detector, map_location=device))
    detector.eval()

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
        detector.to(device)
    print('done')

    output_folder = os.path.join(args.output, '')

    process = None
    queue = []

    if os.path.isdir(args.input):
        args.input = os.path.join(args.input, '').replace("/", "\\")
        d = os.path.basename(os.path.dirname(args.input))
        output_folder = os.path.join(output_folder, d)
        print(output_folder)
        extensions = ['wav', 'm4a', 'mp3', 'mp4', 'flac']
        cover_ext = ["jpg", "png", "bmp"]

        cover = ""

        if output_folder != '' and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = []
        d = os.listdir(args.input)
        for f in d:
            ext = f[::-1].split('.')[0][::-1]

            if ext in extensions:
                files.append(os.path.join(args.input, f))

            if ext in cover_ext:
                cover = os.path.join(args.input, f)

        curr_min = 0
        curr_max = 0
        curr_avg = 0

        values = []

        idx = 0
        for file in tqdm(files):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X, sr = librosa.load(
                    file, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
            basename = os.path.splitext(os.path.basename(file))[0]

            if X.ndim == 1:
                X = np.asarray([X, X])

            X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)

            sp = Separator(None, model, device, args.batchsize, args.cropsize, args.n_fft,   args.postprocess)

            if args.tta:
                y_spec, v_spec, m_spec = sp.separate_tta(X_spec)
            else:
                y_spec, v_spec, m_spec, mask = sp.separate(X_spec, padding=args.padding, include_phase=args.include_phase)

            v = np.abs(v_spec)
            v = v / v.max()
            vmin = v.min()
            vmean = v.mean()
            vvar = v.var()
            vmed = np.median(v)
            mmin = mask.min()
            mmean = mask.mean()
            mmax = mask.max()
            med = np.median(mask)

            d = [
                vmin,
                vmean,
                vvar,
                vmed,
                mmin,
                mmean,
                mmax,
                med
            ]


            values = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(0)
            f = torch.sigmoid(detector(values))[0].item()

            if f > 0.5:
                print(f'{file} min={mask.min()} avg={mask.mean()} max={mask.max()} f={f}')

            del X_spec
            del X

        print(f'min={curr_min / len(files)}, avg={curr_avg / len(files)}, max={curr_max / len(files)}')

    else:
        print('loading wave source...', end=' ')
        X, sr = librosa.load(
            args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
        basename = os.path.splitext(os.path.basename(args.input))[0]
        print('done')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        print('stft of wave source...', end=' ')
        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        print('done')

        sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

        if args.tta:
            y_spec, v_spec, m_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec, m_spec = sp.separate(X_spec, padding=args.padding)

        print('inverse stft of instruments...', end=' ')
        wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
        print('done')
        sf.write('{}_Instruments.wav'.format(basename), wave.T, sr)

        print('inverse stft of vocals...', end=' ')
        c = np.abs(y_spec).max()
        v_spec = c.copy()
        v_spec.real = (v_spec.real / c + np.random.normal(size=v_spec.shape)) * c
        wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
        print('done')
        sf.write('{}_Vocals.wav'.format(basename), wave.T, sr)

        if args.output_image:
            image = spec_utils.spectrogram_to_image(y_spec)
            utils.imwrite('{}_Instruments.jpg'.format(basename), image)

            image = spec_utils.spectrogram_to_image(v_spec)
            utils.imwrite('{}_Vocals.jpg'.format(basename), image)

if __name__ == '__main__':
    main()