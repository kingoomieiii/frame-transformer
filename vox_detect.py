import argparse
import numpy as np
import torch
from libft.dataset_detection import VoxDetectDataset
from lib import dataset
from lib import spec_utils
from lib import nets

import torch.utils.data

from vocal_detector import VocalDetector

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
    p.add_argument('--pretrained_model', '-P', type=str, default='baseline.pth')
    p.add_argument('--vocal_detector', type=str, default="voxdetector.pth")
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', type=str, default="G://dataset//novx")
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=16)
    p.add_argument('--cropsize', '-c', type=int, default=128)
    p.add_argument('--padding', type=int, default=64)

    p.add_argument('--instrumental_lib', type=str, default="C://cs2048_sr44100_hl1024_nf2048_of0|D://cs2048_sr44100_hl1024_nf2048_of0|F://cs2048_sr44100_hl1024_nf2048_of0|H://cs2048_sr44100_hl1024_nf2048_of0")
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_res_blocks', type=int, default=1)
    p.add_argument('--feedforward_expansion', type=int, default=24)
    p.add_argument('--dropout', type=float, default=0.1)

    args = p.parse_args()

    args.flag = str.lower(args.flag) == 'true'
    args.include_phase = str.lower(args.include_phase) == 'true'
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    args.instrumental_lib = [p for p in args.instrumental_lib.split('|')]

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

    dataset = VoxDetectDataset(
        path=args.instrumental_lib,
        vocal_path=[],
        is_validation=False,
        inst_rate=1
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=4,
        shuffle=False
    )

    model.eval()

    for X, s in dataloader:
        X = X.to(device)
        
        with torch.cuda.amp.autocast_mode.autocast():
            mask = model(X)

        v = X * (1 - mask)

        vmin = v.min()
        vmean = v.mean()
        vvar = v.var()
        vmed = torch.median(v)
        mmin = mask.min()
        mmean = mask.mean()
        mmax = mask.max()
        med = torch.median(mask)

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
            print(f'{s} min={mask.min()} avg={mask.mean()} max={mask.max()} f={f}')

if __name__ == '__main__':
    main()
