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
from frame_primer.dataset_denoising import DenoisingDataset
from frame_transformer import FrameTransformer

from lib import dataset
from tqdm import tqdm

from lib.lr_scheduler_linear_warmup import LinearWarmupScheduler
from lib.lr_scheduler_polynomial_decay import PolynomialDecayScheduler

import wandb
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

def train_epoch(dataloader, model, device, optimizer, accumulation_steps, grad_scaler, progress_bar, lr_warmup=None, step=0, batch_growth_start=1, batch_growth_target=256, batch_growth_duration=4, epoch=0, use_wandb=True):
    model.train()

    sum_mask_loss = 0
    batch_loss = 0
    batches = 0
    mask_crit = nn.L1Loss()

    i = 0
    skipped = 0

    model.zero_grad()

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for i, (X, eps) in enumerate(pbar):
        X = X.to(device)[:, :, :model.max_bin]
        eps = eps.to(device)[:, :, :model.max_bin]

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            pred = model(X)

        loss = mask_crit(pred, eps) / accumulation_steps

        if torch.logical_or(loss.isnan(), loss.isinf()):
            print('non-finite loss')

        batch_loss = batch_loss + loss

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (i + 1) % accumulation_steps == 0:
            if progress_bar:
                pbar.set_description(f'{str(batch_loss.item())}')

            if use_wandb:
                wandb.log({
                    'loss': batch_loss.item()
                })

            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            if lr_warmup is not None:
                lr_warmup.step()

            batches = batches + 1
            sum_mask_loss += batch_loss.item() * len(X) * accumulation_steps
            model.zero_grad()
            batch_loss = 0

    return sum_mask_loss / batches

def validate_epoch(dataloader, model, device, grad_scaler, reconstruction_loss_type=""):
    model.eval()
    sum_noise_loss = 0
    mask_crit = nn.L1Loss()

    with torch.no_grad():
        for X, eps in tqdm(dataloader):
            X = X.to(device)[:, :, :model.max_bin]
            eps = eps.to(device)[:, :, :model.max_bin]

            with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
                pred = model(X)

            noise_loss = mask_crit((pred + 1) * 0.5, (eps + 1) * 0.5)
    
            if torch.logical_or(noise_loss.isnan(), noise_loss.isinf()):
                print('non-finite or nan validation loss') 
            
            sum_noise_loss += noise_loss.item() * len(X)

    return sum_noise_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()

    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--max_cropsize', type=int, default=2048)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--mixup_rate', '-M', type=float, default=0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=0.4)
    p.add_argument('--mixed_precision', type=str, default='false')

    p.add_argument('--channels', type=int, default=2)
    p.add_argument('--channel_scale', type=int, default=1)
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--feedforward_expansion', type=int, default=4)
    p.add_argument('--num_res_blocks', type=int, default=3)
    p.add_argument('--num_transformer_encoders', type=int, default=2)    
    p.add_argument('--num_transformer_decoders', type=int, default=2)    
    p.add_argument('--num_bands', type=int, default=8)
    p.add_argument('--feedforward_dim', type=int, default=4096)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--dropout', type=float, default=0.1)

    p.add_argument('--cropsizes', type=str, default='256,512')
    p.add_argument('--epochs', type=str, default='30,50')
    p.add_argument('--epoch_sizes', type=str, default='None,None')
    p.add_argument('--batch_sizes', type=str, default='2,1')
    p.add_argument('--accumulation_steps', '-A', type=str, default='4,2')

    p.add_argument('--amsgrad', type=str, default='false')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--learning_rate', '-l', type=float, default=2.5317e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw'], default='adamw')
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-7)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=1.0)
    p.add_argument('--lr_scheduler_current_step', type=int, default=0)
    p.add_argument('--cropsize', '-C', type=int, default=512)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.2)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=1)
    p.add_argument('--val_cropsize', '-c', type=int, default=1024)
    p.add_argument('--num_workers', '-w', type=int, default=2)
    p.add_argument('--curr_warmup_epoch', type=int, default=0) # 633000
    p.add_argument('--token_warmup_epoch', type=int, default=4)
    p.add_argument('--warmup_epoch', type=int, default=1)
    p.add_argument('--warmup_steps', type=int, default=90000)
    p.add_argument('--decay_steps', type=int, default=1150000)
    p.add_argument('--epoch', '-E', type=int, default=30)
    p.add_argument('--epoch_size', type=float, default=None)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--force_voxaug', type=str, default='false')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--model_dir', type=str, default='J://')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--wandb_project', type=str, default='frame-transformer-pretraining')
    p.add_argument('--wandb_entity', type=str, default='carperbr')
    p.add_argument('--wandb_run_id', type=str, default=None)
    p.add_argument('--wandb', type=str, default='true')
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--sigma', type=float, default=0.5)
    args = p.parse_args()

    args.amsgrad = str.lower(args.amsgrad) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.bias = str.lower(args.bias) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.force_voxaug = str.lower(args.force_voxaug) == 'true'
    args.wandb = str.lower(args.wandb) == 'true'
    args.epochs = [int(epoch) for i, epoch in enumerate(args.epochs.split(','))]
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    args.batch_sizes = [int(batch_size) for batch_size in args.batch_sizes.split(',')]
    args.accumulation_steps = [int(steps) for steps in args.accumulation_steps.split(',')]
    
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, id=args.wandb_run_id, resume="must" if args.wandb_run_id is not None else None)

    logger.info(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    train_dataset = DenoisingDataset(
        path=[
            "C://cs2048_sr44100_hl1024_nf2048_of0",
            "D://cs2048_sr44100_hl1024_nf2048_of0",
            "D://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "F://cs2048_sr44100_hl1024_nf2048_of0",
            "F://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "H://cs2048_sr44100_hl1024_nf2048_of0",
            "H://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "J://cs2048_sr44100_hl1024_nf2048_of0",
            "J://cs2048_sr44100_hl1024_nf2048_of0_PAIRS",
            "J://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "K://cs2048_sr44100_hl1024_nf2048_of0",
            "K://cs2048_sr44100_hl1024_nf2048_of0_PAIRS",
            "K://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
        ],
        epoch_size=args.epoch_size,
        cropsize=args.cropsize
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')

    #model = FramePrimer(channels=args.channels, depth=args.depth, num_transformer_encoders=args.num_transformer_encoders, num_transformer_decoders=args.num_transformer_decoders, n_fft=args.n_fft, cropsize=args.cropsize, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias)
    #model = FramePrimer(channels=args.channels, scale_factor=args.channel_scale, feedforward_dim=args.feedforward_dim, depth=args.depth, num_transformer_encoders=args.num_transformer_encoders, num_transformer_decoders=args.num_transformer_decoders, n_fft=args.n_fft, cropsize=args.max_cropsize, num_bands=args.num_bands, bias=args.bias, dropout=args.dropout, num_res_blocks=args.num_res_blocks)
    #model = FramePrimer2(channels=args.channels, scale_factor=args.channel_scale, feedforward_expansion=args.feedforward_expansion, depth=args.depth, num_transformer_blocks=args.num_transformer_encoders, n_fft=args.n_fft, cropsize=args.max_cropsize, num_bands=args.num_bands, bias=args.bias, dropout=args.dropout, num_res_blocks=args.num_res_blocks)
   
    model = FrameTransformer(channels=args.channels, n_fft=args.n_fft, dropout=args.dropout, expansion=args.feedforward_expansion)

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

    steps = len(train_dataset) // (args.batch_sizes[0] * args.accumulation_steps[0])
    warmup_steps = steps * args.warmup_epoch
    decay_steps = steps * args.epochs[-1] + warmup_steps

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=(steps * args.curr_warmup_epoch)),
        PolynomialDecayScheduler(optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=decay_steps, start_step=warmup_steps, current_step=(steps * args.curr_warmup_epoch))
    ])
    
    val_dataset = None
    curr_idx = 0

    model.train()

    log = []
    best_loss = np.inf
    for epoch in range(args.curr_warmup_epoch, args.epochs[-1] + args.epoch):
        train_dataset.rebuild()
        
        if epoch > args.epochs[curr_idx] or val_dataset is None:
            for i,e in enumerate(args.epochs):
                if epoch > e:
                    print(curr_idx)
                    print(args.curr_warmup_epoch)
                    print(e)
                    curr_idx = i + 1
            
            curr_idx = min(curr_idx, len(args.cropsizes) - 1)
            cropsize = args.cropsizes[curr_idx]
            batch_size = args.batch_sizes[curr_idx]
            accum_steps = args.accumulation_steps[curr_idx]
            print(f'increasing cropsize to {cropsize}, batch size to {batch_size}, accum steps to {accum_steps}')

            train_dataset.cropsize = cropsize
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor
            )
            
            val_dataset = DenoisingDataset(
                path=[f"C://cs{cropsize}_sr44100_hl1024_nf2048_of0_VALIDATION"],
                gamma=args.gamma,
                sigma=args.sigma,
                cropsize=cropsize,
                is_validation=True
            )

            val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2
            )
        
        logger.info('# epoch {}'.format(epoch))
        train_loss = train_epoch(train_dataloader, model, device, optimizer, accum_steps, grad_scaler, args.progress_bar, lr_warmup=scheduler, epoch=epoch, use_wandb=args.wandb)
        val_loss = validate_epoch(val_dataloader, model, device, grad_scaler)

        if args.wandb:
            wandb.log({
                'train_loss': train_loss,
                'validation_loss': val_loss,
            })

        logger.info(
            '  * validation loss = {:.6f}'
            .format(val_loss)
        )

        if val_loss < best_loss or args.save_all:
            if (val_loss) < best_loss:
                best_loss = val_loss
                logger.info('  * best validation loss')

            model_path = f'{args.model_dir}models/pre_iter{epoch}.pth'
            torch.save(model.state_dict(), model_path)

        log.append([train_loss, val_loss])
        with open('loss_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(log, f, ensure_ascii=False)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)