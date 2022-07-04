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

from lib import dataset
from tqdm import tqdm

from frame_primer.frame_primer import FramePrimer
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

def train_epoch(dataloader, model, device, optimizer, accumulation_steps, grad_scaler, progress_bar, lr_warmup=None, step=0, batch_growth_start=1, batch_growth_target=256, batch_growth_duration=4, epoch=0):
    model.train()

    sum_mask_loss = 0
    batch_loss = 0
    batches = 0
    mask_crit = nn.L1Loss()

    i = 0
    skipped = 0

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for src, tgt, num_indices, indices in pbar:
        src = src.to(device)[:, :, :model.max_bin]
        tgt = tgt.to(device)[:, :, :model.max_bin]
        num_indices = num_indices.to(device)
        indices = indices.to(device)

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            pred = src * model(src)

        x_batch = None
        y_batch = None
        for n in range(src.shape[0]):
            idx_count_itm = num_indices[n]
            indices_itm = indices[n]

            for idx in range(idx_count_itm):
                start = indices_itm[idx]
                unmasked = pred[None, n, :, :, start:start+dataloader.dataset.token_size]
                target = tgt[None, n, :, :, start:start+dataloader.dataset.token_size]
                x_batch = torch.cat((x_batch, unmasked), dim=0) if x_batch is not None else unmasked
                y_batch = torch.cat((y_batch, target), dim=0) if y_batch is not None else target

        loss = mask_crit(x_batch, y_batch) / accumulation_steps

        if torch.logical_or(loss.isnan(), loss.isinf()):
            print('non-finite loss, skipping batch')
            model.zero_grad()
            i = 0
            skipped = skipped + accumulation_steps
        else:
            batch_loss = batch_loss + loss

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if progress_bar:
                    pbar.set_description(f'{str(batch_loss.item())}')

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
                sum_mask_loss += batch_loss.item()
                model.zero_grad()
                batch_loss = 0
                i = 0
            else:
                i = i + 1

    return sum_mask_loss / batches

def validate_epoch(dataloader, model, device, grad_scaler, reconstruction_loss_type=""):
    model.eval()

    sum_mask_loss = 0
    sum_token_loss = 0

    mask_crit = nn.L1Loss()

    with torch.no_grad():
        for src, tgt, num_indices, indices in tqdm(dataloader):
            src = src.to(device)[:, :, :model.max_bin]
            tgt = tgt.to(device)[:, :, :model.max_bin]
            num_indices = num_indices.to(device)
            indices = indices.to(device)

            with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
                pred = src * model(src)

            x_batch = None
            y_batch = None
            for n in range(src.shape[0]):
                idx_count_itm = num_indices[n]
                indices_itm = indices[n]

                for idx in range(idx_count_itm):
                    start = indices_itm[idx]
                    unmasked = pred[None, n, :, :, start:start+dataloader.dataset.token_size]
                    target = tgt[None, n, :, :, start:start+dataloader.dataset.token_size]
                    x_batch = torch.cat((x_batch, unmasked), dim=0) if x_batch is not None else unmasked
                    y_batch = torch.cat((y_batch, target), dim=0) if y_batch is not None else target

            token_loss = mask_crit(x_batch, y_batch)
            mask_loss = mask_crit(pred, tgt)
    
            if torch.logical_or(mask_loss.isnan(), mask_loss.isinf()):
                print('non-finite or nan validation loss') 
            
            sum_mask_loss += mask_loss.item() * len(src)
            sum_token_loss += token_loss.item() * len(src)

    return sum_mask_loss / len(dataloader.dataset), sum_token_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generator_type', type=str.lower, choices=['primer', 'evprimer', 'unet', 'vanilla', 'vqfpu'])
    p.add_argument('--id', type=str, default='')
    p.add_argument('--channels', type=int, default=2)
    p.add_argument('--depth', type=int, default=5)
    p.add_argument('--num_transformer_blocks', type=int, default=2)
    p.add_argument('--num_bands', type=int, default=16)
    p.add_argument('--feedforward_dim', type=int, default=4096)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--amsgrad', type=str, default='false')
    p.add_argument('--batchsize', '-B', type=int, default=1)
    p.add_argument('--accumulation_steps', '-A', type=int, default=8)
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--dataset', '-d', required=False)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--learning_rate', '-l', type=float, default=1e-4)
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
    p.add_argument('--curr_warmup_epoch', type=int, default=0)
    p.add_argument('--token_warmup_epoch', type=int, default=4)
    p.add_argument('--warmup_epoch', type=int, default=1)
    p.add_argument('--epoch', '-E', type=int, default=30)
    p.add_argument('--epoch_size', type=int, default=None)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--mixup_rate', '-M', type=float, default=0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=0.4)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--pretrained_model_scheduler', type=str, default=None)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--mixed_precision', type=str, default='false')
    p.add_argument('--force_voxaug', type=str, default='false')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--model_dir', type=str, default='G://')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--token_size', type=int, default=64)
    p.add_argument('--mask_rate', type=float, default=0.2)
    p.add_argument('--next_frame_chunk_size', type=int, default=0)
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--wandb_project', type=str, default='frame-transformer-pretraining')
    p.add_argument('--wandb_entity', type=str, default='carperbr')
    p.add_argument('--wandb_run_id', type=str, default=None)
    args = p.parse_args()

    args.amsgrad = str.lower(args.amsgrad) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.bias = str.lower(args.bias) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.force_voxaug = str.lower(args.force_voxaug) == 'true'

    config = {
        "token_size": args.token_size,
        "mask_rate": args.mask_rate,
        "seed": args.seed,
        "mixup_rate": args.mixup_rate,
        "mixup_alpha": args.mixup_alpha,
        "learning_rate": args.learning_rate,
        "batchsize": args.batchsize,
        "accumulation_steps": args.accumulation_steps,
        "num_bands": args.num_bands,
        "num_transformer_blocks": args.num_transformer_blocks,
        "feedforward_dim": args.feedforward_dim,
        "channels": args.channels,
        "bias": args.bias,
        "dropout": args.dropout,
        "amsgrad": args.amsgrad,
        "optimizer": args.optimizer,
        "cropsize": args.cropsize,
        "curr_warmup_epoch": args.curr_warmup_epoch,
        "token_warmup_epoch": args.token_warmup_epoch,
        "warmup_epoch": args.warmup_epoch,
        "epoch": args.epoch,
        "pretrained_model": args.pretrained_model,
        "mixed_precision": args.mixed_precision,
    }

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config, id=args.wandb_run_id, resume="must" if args.wandb_run_id is not None else None)

    logger.info(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    train_dataset = dataset.MaskedPretrainingDataset(
        path=[
            "C://cs2048_sr44100_hl1024_nf2048_of0",
            "D://cs2048_sr44100_hl1024_nf2048_of0",
            "F://cs2048_sr44100_hl1024_nf2048_of0",
            "D://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "C://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "F://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "H://cs2048_sr44100_hl1024_nf2048_of0_MIXES"
        ],
        is_validation=False,
        epoch_size=args.epoch_size,
        cropsize=args.cropsize,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha,
        mask_rate=args.mask_rate,
        next_frame_chunk_size=args.next_frame_chunk_size,
        token_size=args.token_size
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )

    num_tokens = (args.cropsize + args.next_frame_chunk_size) // args.token_size

    mask_indices = []
    for token in range(num_tokens):
        if np.random.uniform() < args.mask_rate:
            mask_indices.append(token)

    print(f'mask indices={mask_indices}')
    
    val_dataset = dataset.MaskedPretrainingDataset(
        path=["C://cs2048_sr44100_hl1024_nf2048_of0_VALIDATION"],
        is_validation=True,
        cropsize=args.cropsize,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha,
        mask_rate=0,
        next_frame_chunk_size=0,
        mask_indices=mask_indices
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
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

    model = FramePrimer(channels=args.channels, depth=args.depth, num_transformer_encoders=0, num_transformer_decoders=args.num_transformer_blocks, n_fft=args.n_fft, cropsize=args.cropsize, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias)

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
    decay_steps = steps * args.epoch + warmup_steps
    token_steps = steps * args.token_warmup_epoch

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=(steps * args.curr_warmup_epoch)),
        PolynomialDecayScheduler(optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=decay_steps, start_step=warmup_steps, current_step=(steps * args.curr_warmup_epoch))
    ])

    train_dataset.warmup_steps = token_steps

    if args.pretrained_model_scheduler is not None:
        scheduler.load_state_dict(torch.load(args.pretrained_model_scheduler))

    log = []
    best_loss = np.inf
    for epoch in range(args.epoch):
        train_dataset.rebuild()
        
        logger.info('# epoch {}'.format(epoch))
        train_loss_mask = train_epoch(train_dataloader, model, device, optimizer, args.accumulation_steps, grad_scaler, args.progress_bar, lr_warmup=scheduler, epoch=epoch)
        val_loss_mask, val_token = validate_epoch(val_dataloader, model, device, grad_scaler)

        wandb.log({
            'validation_full_loss': val_loss_mask,
            'validation_token_loss': val_token
        })

        logger.info(
            '  * validation loss mask = {:.6f}, validation loss token = {:.6f}'
            .format(val_loss_mask, val_token)
        )

        if (val_token) < best_loss or args.save_all:
            if (val_token) < best_loss:
                best_loss = val_token
                logger.info('  * best validation loss')

            model_path = f'{args.model_dir}models/pre_iter{epoch}.pth'
            torch.save(model.state_dict(), model_path)

        log.append([train_loss_mask, val_token])
        with open('loss_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(log, f, ensure_ascii=False)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)