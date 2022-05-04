import argparse
import os

import shutil
import math
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils
from lib import utils

import torch.nn.functional as F

import json

from lib.frame_transformer import FrameTransformer

class Separator(object):

    def __init__(self, model, device, batchsize, cropsize, postprocess=False):
        self.model = model
        self.offset = 0
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _separate_sliding(self, X_mag_pad, roi_size):
        X_dataset = []
        X_dataset_prev = []
        X_dataset_next = []

        patches = (X_mag_pad.shape[2] - 2 * self.offset) // roi_size

        X_mag_prev = np.pad(X_mag_pad, ((0,0), (0,0), (self.cropsize//2,0)))[:,:,:-(self.cropsize //2)]
        X_mag_next = np.pad(X_mag_pad, ((0,0), (0,0), (0,self.cropsize//2)))[:,:,(self.cropsize // 2):]

        for i in range(patches):
            start = i * roi_size
            X_mag_crop = X_mag_pad[:, :, start:start + self.cropsize]
            X_prev_crop = X_mag_prev[:, :, start:start + self.cropsize]
            X_next_crop = X_mag_next[:, :, start:start + self.cropsize]
            X_dataset.append(X_mag_crop)
            X_dataset_prev.append(X_prev_crop)
            X_dataset_next.append(X_next_crop)

        X_dataset = np.asarray(X_dataset)
        X_dataset_prev = np.asarray(X_dataset_prev)
        X_dataset_next = np.asarray(X_dataset_next)

        self.model.eval()
        with torch.no_grad():
            mask = []
            mask_prev = []
            mask_next = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch_prev = X_dataset_prev[i: i + self.batchsize]
                X_batch_next = X_dataset_next[i: i + self.batchsize]
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)
                X_batch_prev = torch.from_numpy(X_batch_prev).to(self.device)
                X_batch_next = torch.from_numpy(X_batch_next).to(self.device)

                pred = self.model(X_batch)
                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

                pred_prev = self.model(X_batch_prev)
                pred_prev = pred_prev.detach().cpu().numpy()
                pred_prev = np.concatenate(pred_prev, axis=2)
                mask_prev.append(pred_prev)

                pred_next = self.model(X_batch_next)
                pred_next = pred_next.detach().cpu().numpy()
                pred_next = np.concatenate(pred_next, axis=2)
                mask_next.append(pred_next)

            mask = np.concatenate(mask, axis=2)
            mask_prev = np.concatenate(mask_prev, axis=2)
            mask_next = np.concatenate(mask_next, axis=2)

        mask_prev = np.pad(mask_prev[:, :, (self.cropsize // 2):], ((0,0), (0,0), (0,self.cropsize // 2)))
        mask_next = np.pad(mask_next[:, :, :-(self.cropsize // 2)], ((0,0), (0,0), ((self.cropsize // 2), 0)))
        
        mask_slice = mask[:, :, self.cropsize//2:-self.cropsize//2]
        prev_slice = mask_prev[:, :, self.cropsize//2:-self.cropsize//2]
        next_slice = mask_next[:, :, self.cropsize//2:-self.cropsize//2]

        avg_slice = (mask_slice + prev_slice + next_slice) / 3.0

        mask_prev[:, :, -(self.cropsize // 2):] = mask[:, :, -(self.cropsize // 2):]
        end = (mask[:, :, -(self.cropsize // 2):] + mask_next[:, :, -(self.cropsize // 2):]) / 2

        mask_next[:, :, :self.cropsize // 2] = mask[:, :, :self.cropsize // 2]
        beg = (mask[:, :, :self.cropsize // 2] + mask_prev[:, :, :self.cropsize // 2]) / 2

        mask = np.concatenate((beg, avg_slice, end), axis=2)

        return mask

    def _separate(self, X_mag_pad, roi_size):
        X_dataset = []
        patches = (X_mag_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_mag_crop = X_mag_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_mag_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                pred = self.model(X_batch)

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

        return y_spec, v_spec

    def separate(self, X_spec, sliding=False):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        if not sliding:
            mask = self._separate(X_mag_pad, roi_size)
        else:
            mask = self._separate_sliding(X_mag_pad, roi_size)

        mask = mask[:, :, :n_frame]
        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask_tta = self._separate(X_mag_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/model_iter0.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', type=str, default="")
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=8)
    p.add_argument('--cropsize', '-c', type=int, default=1024)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_encoders', type=int, default=2)
    p.add_argument('--num_decoders', type=int, default=2)
    p.add_argument('--num_bands', type=int, default=8)
    p.add_argument('--feedforward_dim', type=int, default=3072)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--sliding_tta', '-st', action='store_true')
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')    
    model = FrameTransformer(channels=args.channels, n_fft=args.n_fft, num_encoders=args.num_encoders, num_decoders=args.num_decoders, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    if str.endswith(args.input, '.json'):
        convert = None
        copy = None
        output_folder = None

        with open(args.input, 'r', encoding='utf8') as f:
            obj = json.load(f)
            convert = obj.get('convert')
            copy = obj.get('copy')
            output_folder = obj.get('output')
            
        output_folder = '' if output_folder is None else output_folder
        convert = [] if convert is None else convert
        copy = [] if copy is None else copy

        if args.output != '':
            output_folder = f'{args.output}/{output_folder}/'

        if output_folder != '' and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in tqdm(copy):
            basename = os.path.splitext(os.path.basename(file))[0]
            shutil.copyfile(file, '{}{}_Instruments.wav'.format(output_folder, basename))

        for file in tqdm(convert):
            print('loading wave source...', end=' ')
            X, sr = librosa.load(
                file, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
            basename = os.path.splitext(os.path.basename(file))[0]
            print('done')

            if X.ndim == 1:
                X = np.asarray([X, X])

            print('stft of wave source...', end=' ')
            X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
            print('done')

            sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

            if args.tta and not args.sliding_tta:
                y_spec, v_spec = sp.separate_tta(X_spec)
            else:
                y_spec, v_spec = sp.separate(X_spec, False)

            print('inverse stft of instruments...', end=' ')
            wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
            print('done')
            sf.write('{}{}_Instruments.wav'.format(output_folder, basename), wave.T, sr)

            print('inverse stft of vocals...', end=' ')
            wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
            print('done')
            sf.write('{}{}_Vocals.wav'.format(output_folder, basename), wave.T, sr)

            if args.output_image:
                image = spec_utils.spectrogram_to_image(y_spec)
                utils.imwrite('{}{}_Instruments.jpg'.format(output_folder, basename), image)
                image = spec_utils.spectrogram_to_image(v_spec)
                utils.imwrite('{}{}_Vocals.jpg'.format(output_folder, basename), image)
            
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
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        print('inverse stft of instruments...', end=' ')
        wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
        print('done')
        sf.write('{}_Instruments.wav'.format(basename), wave.T, sr)

        print('inverse stft of vocals...', end=' ')
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
