import argparse
from datetime import datetime
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import clip_grad_norm_

from lib import dataset
from lib import spec_utils
from tqdm import tqdm

from lib.frame_transformer_ar import FrameTransformer
from lib.lr_scheduler_linear_warmup import LinearWarmupScheduler
from lib.lr_scheduler_polynomial_decay import PolynomialDecayScheduler

import torch.nn.functional as F

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

def mixup(X, Y, alpha=1):
    indices = torch.randperm(X.size(0))
    X2 = X[indices]
    Y2 = Y[indices]
    alpha = np.full((X.shape[0]), fill_value=alpha)
    lam = torch.FloatTensor(np.random.beta(alpha, alpha))
    inv_lam = torch.ones_like(lam) - lam
    lam = lam.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(X.device)
    inv_lam = inv_lam.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(X.device)
    X = X * lam + X2 * inv_lam
    Y = Y * lam + Y2 * inv_lam
    return X, Y

def train_epoch(dataloader, model, device, optimizer, accumulation_steps, grad_scaler, progress_bar, mixup_rate, mixup_alpha, lr_warmup=None):
    model.train()
    sum_loss = 0
    batch_loss = 0
    crit = nn.L1Loss()

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (src, tgt) in enumerate(pbar):
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            pred = F.relu6(model(src)) / 6

        loss = crit(pred, tgt)
        accum_loss = loss / accumulation_steps
        batch_loss = batch_loss + accum_loss

        if grad_scaler is not None:
            grad_scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            if progress_bar:
                pbar.set_description(str(batch_loss.item()))

            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 0.5)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            if lr_warmup is not None:
                lr_warmup.step()

            model.zero_grad()
            batch_loss = 0

        sum_loss += loss.item() * len(src)

    # the rest batch
    if (itr + 1) % accumulation_steps != 0:
        grad_scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 0.5)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        model.zero_grad()

    return sum_loss / len(dataloader.dataset)

def validate_epoch(dataloader, model, device, grad_scaler, include_phase=False):
    model.eval()
    sum_loss = 0
    crit = nn.L1Loss()

    with torch.no_grad():
        for src, tgt in tqdm(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
                h = F.relu6(model(src))/6.0
 
            loss = crit(h, tgt)
    
            if torch.logical_or(loss.isnan(), loss.isinf()):
                print('non-finite or nan validation loss; aborting')
                quit()
            else:
                sum_loss += loss.item() * len(src)

    return sum_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='')
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_encoders', type=int, default=2)
    p.add_argument('--num_decoders', type=int, default=2)
    p.add_argument('--num_bands', type=int, default=8)
    p.add_argument('--feedforward_dim', type=int, default=3072)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--amsgrad', type=str, default='false')
    p.add_argument('--vocal_recurse_prob', type=float, default=0.5)
    p.add_argument('--vocal_recurse_prob_decay', type=float, default=0.5)
    p.add_argument('--vocal_noise_prob', type=float, default=0.5)
    p.add_argument('--vocal_noise_magnitude', type=float, default=0.5)
    p.add_argument('--vocal_pan_prob', type=float, default=0.5)
    p.add_argument('--batchsize', '-B', type=int, default=1)
    p.add_argument('--accumulation_steps', '-A', type=int, default=4)
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--dataset', '-d', required=False)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--learning_rate', '-l', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw'], default='adam')
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-7)
    #p.add_argument('--lr_scheduler_warmup_steps', '-LW', type=int, default=32000) # controlled by args.warmup_epoch now
    #p.add_argument('--lr_scheduler_decay_steps', type=int, default=128000) # controlled by args.epoch now
    p.add_argument('--lr_scheduler_decay_power', type=float, default=1.0)
    p.add_argument('--lr_scheduler_current_step', type=int, default=0)
    p.add_argument('--cropsize', '-C', type=int, default=1024)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.2)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=4)
    p.add_argument('--val_cropsize', '-c', type=int, default=1024)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--warmup_epoch', type=int, default=4)
    p.add_argument('--epoch', '-E', type=int, default=32)
    p.add_argument('--epoch_size', type=int, default=None)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--fake_data_prob', type=float, default=math.nan)
    p.add_argument('--mixup_rate', '-M', type=float, default=0.5)
    p.add_argument('--mixup_alpha', '-a', type=float, default=0.4)
    p.add_argument('--phase_in', type=str, default='false')
    p.add_argument('--phase_out', type=str, default='false')
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--pretrained_model_scheduler', type=str, default=None)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--mixed_precision', type=str, default='true')
    p.add_argument('--force_voxaug', type=str, default='false')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--model_dir', type=str, default='E://')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--dropout', type=float, default=0.4)
    args = p.parse_args()

    args.amsgrad = str.lower(args.amsgrad) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.bias = str.lower(args.bias) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.phase_in = str.lower(args.phase_in) == 'true'
    args.phase_out = str.lower(args.phase_out) == 'true'
    args.force_voxaug = str.lower(args.force_voxaug) == 'true'

    logger.info(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    train_dataset = dataset.VocalAutoregressiveDataset(
        path="C://cs2048_sr44100_hl1024_nf2048_of0",
        extra_path="G://cs2048_sr44100_hl1024_nf2048_of0",
        mix_path="C://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
        vocal_path="G://cs2048_sr44100_hl1024_nf2048_of0_VOCALS",
        is_validation=False,
        epoch_size=args.epoch_size,
        cropsize=args.cropsize,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha,
        pair_mul=1
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataset = dataset.VocalAutoregressiveDataset(
        path="G://cs2048_sr44100_hl1024_nf2048_of0_VALIDATION",
        is_validation=True,
        epoch_size=args.epoch_size,
        cropsize=args.cropsize,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    val_filelist = []
    if args.val_filelist is not None:
        with open(args.val_filelist, 'r', encoding='utf8') as f:
            val_filelist = json.load(f)

    if args.debug:
        logger.info('### DEBUG MODE')
    elif args.val_filelist is None and args.split_mode == 'random':
        with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        logger.info('{} {} {}'.format(i + 1, os.path.basename(X_fname), os.path.basename(y_fname)))

    device = torch.device('cpu')
    model = FrameTransformer(channels=args.channels, n_fft=args.n_fft, num_encoders=args.num_encoders, num_decoders=args.num_decoders, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize, out_activate=None, encoder_only=True)

    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)    
    
    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'# num params: {params}')
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay
        )

    steps = len(train_dataset) // (args.batchsize * args.accumulation_steps)
    warmup_steps = steps * args.warmup_epoch
    decay_steps = steps * args.epoch - warmup_steps

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=args.lr_scheduler_current_step),
        PolynomialDecayScheduler(optimizer, base_lr=args.learning_rate, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=decay_steps, start_step=warmup_steps, current_step=args.lr_scheduler_current_step)
    ])

    if args.pretrained_model_scheduler is not None:
        scheduler.load_state_dict(torch.load(args.pretrained_model_scheduler))

    log = []
    best_loss = np.inf
    for epoch in range(args.epoch):
        train_dataset.rebuild()

        logger.info('# epoch {}'.format(epoch))
        train_loss = train_epoch(train_dataloader, model, device, optimizer, args.accumulation_steps, grad_scaler, args.progress_bar, args.mixup_rate, args.mixup_alpha, lr_warmup=scheduler)
        val_loss_mag = validate_epoch(val_dataloader, model, device, grad_scaler)

        logger.info(
            '  * training loss = {:.6f}, validation loss mag = {:.6f}'
            .format(train_loss, val_loss_mag)
        )

        if (val_loss_mag) < best_loss or args.save_all:
            if (val_loss_mag) < best_loss:
                best_loss = val_loss_mag
                logger.info('  * best validation loss')

            model_path = f'{args.model_dir}models/model_iter{epoch}.pth'
            scheduler_path = f'{args.model_dir}models/scheduler_iter{epoch}.pth'
            torch.save(model.state_dict(), model_path)
            torch.save(scheduler.state_dict(), scheduler_path)

        log.append([1, val_loss_mag])
        with open('loss_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(log, f, ensure_ascii=False)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)