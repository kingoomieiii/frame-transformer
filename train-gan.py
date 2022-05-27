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
from lib.frame_primer import FramePrimer, FramePrimerCritic

from lib.frame_transformer import FrameTransformer, FrameTransformerCritic

from lib.frame_transformer_unet import FrameTransformerUnet
from lib.conv_discriminator import ConvDiscriminator
from lib.frame_transformer_discriminator import FrameTransformerDiscriminator as FrameTransformerUnetDiscriminator
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

def train_epoch(dataloader, modeler, critic, device, optimizer, critic_optimizer, grad_scaler, critic_scaler, progress_bar, lr_warmup=None, lr_warmup_critic=None, lambda_l1=100, lambda_gen=2.0, lambda_critic=4.0, modeler_adversarial_start=1024):
    modeler.train()

    sum_mask_loss = 0
    sum_nxt_loss = 0
    sum_gen_loss = 0
    sum_critic_loss = 0

    mask_crit = nn.L1Loss()
    bce_crit = nn.BCEWithLogitsLoss()

    critic_loss = 0

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (src, tgt, is_next, starts) in enumerate(pbar):
        src = src.to(device)
        tgt = tgt.to(device)
        is_next = is_next.to(device)

        if len(is_next.shape) == 1:
            is_next = is_next.unsqueeze(-1)
        
        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            mask = modeler(src)
            real = critic(tgt)
            fake = critic(src * mask.detach())
            detection_truth = torch.zeros_like(fake)
    
            for start in starts:
                detection_truth[:, :, :, start:start+dataloader.dataset.token_size] = 1
            detection_truth[:, -dataloader.dataset.next_frame_chunk_size-dataloader.dataset.separator_size:-dataloader.dataset.next_frame_chunk_size+dataloader.dataset.separator_size] = 0

            fake_detection_loss = bce_crit(fake, detection_truth)
            real_detection_loss = bce_crit(real, torch.zeros_like(real))
            critic_loss = (real_detection_loss + fake_detection_loss) / 2

        critic.zero_grad()
        critic_scaler.scale(critic_loss).backward()
        critic_scaler.unscale_(critic_optimizer)
        clip_grad_norm_(critic.parameters(), 0.5)
        critic_scaler.step(critic_optimizer)
        critic_scaler.update()

        if lr_warmup_critic is not None:
            lr_warmup_critic.step()

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            token_loss = mask_crit(src * mask, tgt)
            fake = critic(src * mask)

            fake_loss = None
            for start in starts:
                segment = fake[:, :, :, start:start+dataloader.dataset.token_size]
                l = bce_crit(segment, torch.zeros_like(segment))
                fake_loss = fake_loss + l if fake_loss is not None else l

            fake_loss = fake_loss / len(starts)

            loss = token_loss * lambda_l1 + lambda_gen * fake_loss

        modeler.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        clip_grad_norm_(modeler.parameters(), 0.5)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if lr_warmup is not None:
            lr_warmup.step()

        if progress_bar:
            pbar.set_description(f'{str(token_loss.item())}|{str(fake_loss.item())}|{str(critic_loss.item())}')

        sum_mask_loss += token_loss.item() * len(src)
        sum_critic_loss += critic_loss.item() * len(src)
        sum_gen_loss += loss.item() * len(src)

    return sum_mask_loss / len(dataloader.dataset), sum_nxt_loss / len(dataloader.dataset), sum_gen_loss / len(dataloader.dataset), sum_critic_loss / len(dataloader.dataset)

def validate_epoch(dataloader, model, device, grad_scaler):
    model.eval()

    sum_mask_loss = 0
    sum_nxt_loss = 0

    mask_crit = nn.L1Loss()

    with torch.no_grad():
        for src, tgt, is_next, tokens in tqdm(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            is_next = is_next.to(device)

            with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
                pred, _ = model(src)
 
            mask_loss = mask_crit(src * pred, tgt)
            loss = mask_loss
    
            if torch.logical_or(loss.isnan(), loss.isinf()):
                print('non-finite or nan validation loss; aborting')
                quit()
            else:
                sum_mask_loss += mask_loss.item() * len(src)

    return sum_mask_loss / len(dataloader.dataset), sum_nxt_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='')
    p.add_argument('--gen_type', type=str.lower, choices=['primer', 'unet', 'vanilla'])
    p.add_argument('--critic_type', type=str.lower, choices=['primer', 'conv', 'unet', 'vanilla'])
    p.add_argument('--curr_warmup_epoch', type=int, default=0)
    p.add_argument('--warmup_epoch', type=int, default=1)
    p.add_argument('--epoch', '-E', type=int, default=30)
    p.add_argument('--lambda_l1', type=float, default=100)
    p.add_argument('--lambda_gen', type=float, default=1.0)
    p.add_argument('--lambda_critic', type=float, default=1.0)
    p.add_argument('--epoch_size', type=int, default=None)
    p.add_argument('--channels', type=int, default=2)
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--num_transformer_blocks', type=int, default=2)
    p.add_argument('--num_bands', type=int, default=8)
    p.add_argument('--feedforward_dim', type=int, default=4096)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--amsgrad', type=str, default='false')
    p.add_argument('--batchsize', '-B', type=int, default=1)
    p.add_argument('--accumulation_steps', '-A', type=int, default=4)
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--dataset', '-d', required=False)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--learning_rate', '-l', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw', 'rmsprop'], default='adam')
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-8)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=1.0)
    p.add_argument('--lr_scheduler_current_step', type=int, default=0)
    p.add_argument('--cropsize', '-C', type=int, default=512)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.2)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=1)
    p.add_argument('--val_cropsize', '-c', type=int, default=1024)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--token_warmup_epoch', type=int, default=4)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--mixup_rate', '-M', type=float, default=0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=0.4)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--pretrained_critic', type=str, default=None)
    p.add_argument('--pretrained_model_scheduler', type=str, default=None)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--mixed_precision', type=str, default='true')
    p.add_argument('--force_voxaug', type=str, default='false')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--model_dir', type=str, default='G://')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--token_size', type=int, default=16)
    p.add_argument('--mask_rate', type=float, default=0.2)
    p.add_argument('--next_frame_chunk_size', type=int, default=512)
    p.add_argument('--prefetch_factor', type=int, default=2)
    args = p.parse_args()

    args.amsgrad = str.lower(args.amsgrad) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.bias = str.lower(args.bias) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.force_voxaug = str.lower(args.force_voxaug) == 'true'

    logger.info(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    train_dataset = dataset.MaskedPretrainingDataset(
        path="C://cs2048_sr44100_hl1024_nf2048_of0",
        extra_path="D://cs2048_sr44100_hl1024_nf2048_of0",
        mix_path=[
            "D://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "C://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "F://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "H://cs2048_sr44100_hl1024_nf2048_of0_MIXES"],
        is_validation=False,
        epoch_size=args.epoch_size,
        cropsize=args.cropsize,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha,
        pair_mul=1,
        mask_rate=args.mask_rate,
        next_frame_chunk_size=args.next_frame_chunk_size,
        token_size=args.token_size,
        num_steps=0
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    val_dataset = dataset.MaskedPretrainingDataset(
        path="C://cs2048_sr44100_hl1024_nf2048_of0_VALIDATION",
        is_validation=True,
        epoch_size=args.epoch_size,
        cropsize=args.cropsize,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha,
        mask_rate=args.mask_rate,
        next_frame_chunk_size=args.next_frame_chunk_size
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batchsize,
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

    if args.gen_type == 'unet':
        modeler = FrameTransformerUnet(channels=args.channels, n_fft=args.n_fft, depth=args.depth, num_transformer_blocks=args.num_transformer_blocks, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize + args.next_frame_chunk_size)
    elif args.gen_type == 'primer':
        modeler = FramePrimer(channels=args.channels, n_fft=args.n_fft, num_transformer_blocks=args.num_transformer_blocks, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize + args.next_frame_chunk_size, dropout=args.dropout)
    else:
        modeler = FrameTransformer(channels=args.channels, n_fft=args.n_fft, num_transformer_blocks=args.num_transformer_blocks, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize + args.next_frame_chunk_size, dropout=args.dropout)

    if args.critic_type == 'conv':
        critic = ConvDiscriminator(channels=args.channels, n_fft=args.n_fft, depth=args.depth)
    elif args.critic_type == 'primer':
        critic = FramePrimerCritic(channels=args.channels, n_fft=args.n_fft, num_transformer_blocks=args.num_transformer_blocks, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize + args.next_frame_chunk_size, dropout=args.dropout)
    elif args.critic_type == 'unet':
        critic = FrameTransformerUnetDiscriminator(channels=args.channels, n_fft=args.n_fft, depth=args.depth, num_transformer_blocks=args.num_transformer_blocks, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize + args.next_frame_chunk_size)
    elif args.critic_type == 'vanilla':
        critic = FrameTransformerCritic(channels=args.channels, n_fft=args.n_fft, num_transformer_blocks=args.num_transformer_blocks, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize + args.next_frame_chunk_size, dropout=args.dropout)

    if args.pretrained_model is not None:
        modeler.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if args.pretrained_critic is not None:
        critic.load_state_dict(torch.load(args.pretrained_critic, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        modeler.to(device)
        critic.to(device)
    
    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    critic_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    
    model_parameters = filter(lambda p: p.requires_grad, modeler.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'# num params: {params}')
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, modeler.parameters()),
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.999)
        )

        critic_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, critic.parameters()),
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.999)
        )
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, modeler.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        critic_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, modeler.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, modeler.parameters()),
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.999)
        )

        critic_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, critic.parameters()),
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.999)
        )

    steps = len(train_dataset) // args.batchsize
    warmup_steps = steps * args.warmup_epoch
    decay_steps = steps * args.epoch + warmup_steps
    token_steps = steps * args.token_warmup_epoch

    modeler_scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=(steps * args.curr_warmup_epoch)),
        PolynomialDecayScheduler(optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=decay_steps, start_step=warmup_steps, current_step=(steps * args.curr_warmup_epoch))
    ])

    critic_scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(critic_optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=(steps * args.curr_warmup_epoch)),
        PolynomialDecayScheduler(critic_optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=decay_steps, start_step=warmup_steps, current_step=(steps * args.curr_warmup_epoch))
    ])

    train_dataset.warmup_steps = token_steps

    log = []
    best_loss = np.inf
    for epoch in range(args.epoch):
        train_dataset.rebuild()

        logger.info('# epoch {}'.format(epoch))
        train_loss_mask, modeler_loss, critic_loss = train_epoch(train_dataloader, modeler, critic, device, optimizer, critic_optimizer, grad_scaler, critic_scaler, args.progress_bar, lr_warmup=modeler_scheduler, lr_warmup_critic=critic_scheduler, lambda_l1=args.lambda_l1, lambda_gen=args.lambda_gen, lambda_critic=args.lambda_critic)

        val_loss_mask1 = validate_epoch(val_dataloader, modeler, device, grad_scaler)
        val_loss_mask2 = validate_epoch(val_dataloader, modeler, device, grad_scaler)
        val_loss_mask3 = validate_epoch(val_dataloader, modeler, device, grad_scaler)
        val_loss_mask4 = validate_epoch(val_dataloader, modeler, device, grad_scaler)

        val_loss_mask = (val_loss_mask1 + val_loss_mask2 + val_loss_mask3 + val_loss_mask4) / 4

        logger.info(
            '  * training loss mask = {:.6f}, train loss modeler = {:6f}, train loss critic = {:6f}'
            .format(train_loss_mask, modeler_loss, critic_loss)
        )

        logger.info(
            '  * validation loss mask = {:.6f}'
            .format(val_loss_mask1)
        )

        logger.info(
            '  * validation loss mask = {:.6f}'
            .format(val_loss_mask2)
        )

        logger.info(
            '  * validation loss mask = {:.6f}'
            .format(val_loss_mask3)
        )

        logger.info(
            '  * validation loss mask = {:.6f}'
            .format(val_loss_mask4)
        )

        if (val_loss_mask) < best_loss or args.save_all:
            if (val_loss_mask) < best_loss:
                best_loss = val_loss_mask
                logger.info('  * best validation loss')

            model_path = f'{args.model_dir}models/model_iter{epoch}.modeler.pth'
            critic_path = f'{args.model_dir}models/model_iter{epoch}.critic.pth'
            torch.save(modeler.state_dict(), model_path)
            torch.save(critic.state_dict(), critic_path)

        log.append([train_loss_mask, val_loss_mask])
        with open('loss_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(log, f, ensure_ascii=False)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)