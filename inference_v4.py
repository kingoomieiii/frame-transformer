import argparse
import os
import shutil
import librosa
import numpy as np
import soundfile as sf
import torch
import json

from tqdm import tqdm
from frame_transformer_v4 import FrameTransformer

from lib import dataset
from lib import spec_utils
from lib import utils
from lib import nets

class Separator(object):

    def __init__(self, corrector, model, device, batchsize, cropsize, postprocess=False):
        self.corrector = corrector
        self.model = model
        self.offset = 0
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

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
        self.corrector.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(np.asarray(X_batch)).to(self.device)[:, :, :1024]

                pred = X_batch * torch.sigmoid(self.model(X_batch))
                
                if padding > 0:
                    pred = pred[:, :, :, (padding):-(padding)]

                pred = self.corrector(pred)

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        mask = np.pad(mask, ((0,0), (0,1), (0, 0)))

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase, c):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * c * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        m_spec = mask * 255

        return y_spec, v_spec, m_spec

    def separate(self, X_spec, padding=None):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, _ = dataset.make_padding(n_frame, self.cropsize, 0)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        c = X_mag_pad.max()
        X_mag_pad /= c

        mask = self._separate(X_mag_pad, self.cropsize, padding)

        mask = mask[:, :, :n_frame]
        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase, c)

        return y_spec, v_spec, m_spec

    def separate_tta(self, X_spec, cropsize=256, paddings=[0, 256, 512]):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, _ = dataset.make_padding(n_frame, cropsize, 0)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = np.zeros_like(X_mag)

        for padding in paddings:
            mask += self._separate(X_mag_pad, cropsize, padding)[:, :, :n_frame]

        mask = mask / len(paddings)

        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec, m_spec

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='J://models/local.49.remover.pth')
    p.add_argument('--pretrained_corrector', type=str, default="J://models/local.5.corrector.pth")
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', type=str, default="")
    p.add_argument('--num_res_encoders', type=int, default=4)
    p.add_argument('--num_res_decoders', type=int, default=4)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=1)
    p.add_argument('--cropsize', '-c', type=int, default=512)
    p.add_argument('--padding', type=int, default=1024)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--num_encoders', type=int, default=2)
    p.add_argument('--num_decoders', type=int, default=13)
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--cropsizes', type=str, default='128,256,512,1024')
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--num_transformer_blocks', type=int, default=2)
    p.add_argument('--bias', type=str, default='true')

    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--num_res_blocks', type=int, default=1)
    p.add_argument('--feedforward_expansion', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)

    args = p.parse_args()

    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]

    print('loading model...', end=' ')
    device = torch.device('cpu')  
    #model = FramePrimer2(channels=args.channels, feedforward_dim=args.feedforward_dim, n_fft=args.n_fft, dropout=0, num_res_blocks=args.num_res_blocks)
    
    #model = FrameTransformer(channels=args.channels, n_fft=args.n_fft, dropout=args.dropout, expansion=4)
    model = FrameTransformer(channels=args.channels, n_fft=args.n_fft, dropout=args.dropout, expansion=args.feedforward_expansion, num_heads=args.num_heads)
    corrector = nets.CascadedNet(args.n_fft, 32, 256)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    corrector.load_state_dict(torch.load(args.pretrained_corrector, map_location=device))

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
        corrector.to(device)
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
            shutil.copyfile(file, '{}/{}_Instruments.wav'.format(output_folder, basename))

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

            sp = Separator(corrector, model, device, args.batchsize, args.cropsize, args.postprocess)

            if args.tta:
                y_spec, v_spec, m_spec = sp.separate_tta(X_spec, cropsize=args.cropsize)
            else:
                y_spec, v_spec, m_spec = sp.separate(X_spec, padding=args.padding)

            print('inverse stft of instruments...', end=' ')
            wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
            print('done')
            sf.write('{}/{}_Instruments.wav'.format(output_folder, basename), wave.T, sr)

            print('inverse stft of vocals...', end=' ')

            wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
            print('done')
            sf.write('{}/{}_Vocals.wav'.format(output_folder, basename), wave.T, sr)

            if args.output_image:
                try:
                    image = spec_utils.spectrogram_to_image(y_spec)
                    utils.imwrite('{}{}_Instruments.jpg'.format(output_folder, basename), image)
                    image = spec_utils.spectrogram_to_image(v_spec)
                    utils.imwrite('{}{}_Vocals.jpg'.format(output_folder, basename), image)
                    image = np.uint8(m_spec)
                    image = np.pad(image, ((0,1), (0,0), (0,0)))
                    image = image.transpose(1, 2, 0)
                    utils.imwrite('{}{}_Mask.jpg'.format(output_folder, basename), image)
                except:
                    pass
            
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
            y_spec, v_spec, m_spec = sp.separate_tta(X_spec, cropsizes=args.cropsizes)
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
