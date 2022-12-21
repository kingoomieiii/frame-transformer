import argparse
import logging
import math
import os
import random
import soundfile as sf

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import wandb

from tqdm import tqdm

from inference_thin import Separator
from dataset_voxfit import VoxAugDataset
from frame_transformer_thin import FrameTransformer
from torch.nn import functional as F
from lib import spec_utils

from lib.lr_scheduler_linear_warmup import LinearWarmupScheduler
from lib.lr_scheduler_polynomial_decay import PolynomialDecayScheduler

def setup_logger(name, logfile='LOGFILENAME.log', out_dir='logs'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(f'{out_dir}/{logfile}', encoding='utf8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def init_epoch(dataloader, model, device):
    model.train()
    model.zero_grad()

    for itr, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)[:, :, :model.max_bin]
        y_batch = y_batch.to(device)[:, :, :model.max_bin]

        with torch.cuda.amp.autocast_mode.autocast():
            pred = torch.sigmoid(model(X_batch))

        break

def train_epoch(dataloader, model, device, optimizer, accumulation_steps, progress_bar, lr_warmup=None, grad_scaler=None, use_wandb=True, step=0, include_phase=False, model_dir="", save_every=20000):
    model.train()
    mag_loss = 0
    batch_mag_loss = 0    
    crit = nn.L1Loss()
    batch_loss = 0
    batches = 0
    model.zero_grad()

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (X, Y) in enumerate(pbar):
        for _ in range(64):
            X = X.to(device)[:, :, :model.max_bin]
            Y = Y.to(device)[:, :, :model.max_bin]

            with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
                pred = torch.sigmoid(model(X))
                
            l1_mag = crit(X[:, :2] * pred[:, :2], Y[:, :2]) / accumulation_steps

            batch_mag_loss = batch_mag_loss + l1_mag
            accum_loss = l1_mag

            if torch.logical_or(accum_loss.isnan(), accum_loss.isinf()):
                print('nan training loss; aborting')
                quit()

            if grad_scaler is not None:
                grad_scaler.scale(accum_loss).backward()
            else:
                accum_loss.backward()

            if (itr + 1) % accumulation_steps == 0:
                if progress_bar:
                    pbar.set_description(f'{step}: {str(batch_mag_loss.item())}')

                if use_wandb:
                    wandb.log({
                        'loss': batch_loss.item()
                    })

                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

                step = step + 1
                
                if lr_warmup is not None:
                    lr_warmup.step()

                model.zero_grad()
                batches = batches + 1
                mag_loss = mag_loss + batch_mag_loss.item()
                batch_mag_loss = 0

                if batches % save_every == 0:
                    model_path = f'{model_dir}models/remover.{step}.tmp.pth'
                    torch.save(model.state_dict(), model_path)

    return mag_loss / batches

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--checkpoint', type=str, default="H://models/local.87")
    p.add_argument('--mixed_precision', type=str, default='true')

    p.add_argument('--warmup_steps', type=int, default=0)
    p.add_argument('--learning_rate', '-l', type=float, default=1e-3)
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-8)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=1)

    p.add_argument('--song_dir', type=str, default='C://stg')
    p.add_argument('--out_dir', type=str, default='D://stg')
    p.add_argument('--song', type=str, default='')
    p.add_argument('--num_passes', type=int, default=128)
    p.add_argument('--model_dir', type=str, default='H://')

    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--cropsize', type=int, default=3072)
    
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--feedforward_expansion', type=int, default=24)
    p.add_argument('--num_heads', type=int, default=8)
    
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw', 'sgd', 'radam', 'rmsprop'], default='adam')
    p.add_argument('--amsgrad', type=str, default='false')
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--llrd', type=str, default='false')
    p.add_argument('--lock', type=str, default='false')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--wandb', type=str, default='false')
    p.add_argument('--wandb_project', type=str, default='VOCAL-REMOVER')
    p.add_argument('--wandb_entity', type=str, default='carperbr')
    p.add_argument('--wandb_run_id', type=str, default=None)
    p.add_argument('--prefetch_factor', type=int, default=2)
    args = p.parse_args()

    args.amsgrad = str.lower(args.amsgrad) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.wandb = str.lower(args.wandb) == 'true'

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, id=args.wandb_run_id, resume="must" if args.wandb_run_id is not None else None)

    print(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    train_dataset = VoxAugDataset(
        song=args.song,
        dir=args.song_dir,
        mul=1,
        cropsize=args.cropsize
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    model = FrameTransformer(in_channels=2, out_channels=2, channels=args.channels, expansion=args.feedforward_expansion, n_fft=args.n_fft, dropout=0, num_heads=args.num_heads)

    model.train()
    mag_loss = 0
    batch_mag_loss = 0    
    crit = nn.L1Loss()
    batch_loss = 0
    batches = 0
    model.zero_grad()

    pbar = tqdm(train_dataloader)

    X, sr = librosa.load(train_dataset.X_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
    Yr, _ = librosa.load(train_dataset.Y_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

    if X.ndim == 1:
        X = np.asarray([X, X])

    if Yr.ndim == 1:
        Yr = np.asarray([Yr, Yr])
    
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    y_mags = []

    for itr, (X2, Y2) in enumerate(pbar):
        groups = [
            { "params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.learning_rate }
        ]
            
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'# {wandb.run.name if args.wandb else ""}; num params: {params}')    
        
        optimizer = torch.optim.Adam(
            groups,
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay
        )

        grad_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None

        if torch.cuda.is_available() and args.gpu >= 0:
            device = torch.device('cuda:{}'.format(args.gpu))
            model.to(device)

        step = 0
        warmup_steps = args.warmup_steps
        decay_steps = args.num_passes - warmup_steps

        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=0),
            PolynomialDecayScheduler(optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=decay_steps, start_step=warmup_steps, current_step=step)
        ])

        if args.checkpoint is not None:
            model.load_state_dict(torch.load(f'{args.checkpoint}.model.pth', map_location=device))

        X2 = X2.to(device)[:, :, :model.max_bin]
        Y2 = Y2.to(device)[:, :, :model.max_bin]

        for i in range(args.num_passes):
            with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
                pred = torch.sigmoid(model(X2))
                
            l1_mag = crit(X2 * pred, Y2)

            if grad_scaler is not None:
                grad_scaler.scale(l1_mag).backward()
            else:
                l1_mag.backward()

            pbar.set_description(f'{step}: {str(l1_mag.item())}')

            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            step = step + 1
            
            if scheduler is not None:
                scheduler.step()

            model.zero_grad()
            batches = batches + 1
            mag_loss = mag_loss + l1_mag.item()
  
        pred = (X2 * pred).detach().cpu().numpy()
        y_mags.append(pred)
        

    Y_mag = np.squeeze(np.concatenate(y_mags, axis=3), axis=0)
    print(f"# Separating tracks")

    basename = os.path.splitext(os.path.basename(train_dataset.X_path))[0]
    separator = Separator(None, model, device, args.batch_size, args.cropsize, n_fft=args.n_fft)
    
    X, sr = librosa.load(train_dataset.X_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
    Yr, _ = librosa.load(train_dataset.Y_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

    if X.ndim == 1:
        X = np.asarray([X, X])

    if Yr.ndim == 1:
        Yr = np.asarray([Yr, Yr])
    
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    Y_spec = spec_utils.wave_to_spectrogram(Yr, args.hop_length, args.n_fft)
    Y_spec, V_spec, _ = separator.process(X_spec, Y_mag) # .separate(X_spec)

    os.makedirs(os.path.join(args.song_dir, 'separations'), exist_ok=True)

    wave = spec_utils.spectrogram_to_wave(Y_spec, hop_length=args.hop_length)
    sf.write(os.path.join(args.song_dir, "separations", f"{basename}.instruments.wav"), wave.T, sr)

    wave = spec_utils.spectrogram_to_wave(V_spec, hop_length=args.hop_length)
    sf.write(os.path.join(args.song_dir, "separations", f"{basename}.vocals.wav"), wave.T, sr)

if __name__ == '__main__':
    main()