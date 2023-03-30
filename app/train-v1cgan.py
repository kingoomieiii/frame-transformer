import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed
import torch.utils.data
import wandb

from tqdm import tqdm

from libft2gan.dataset_voxaug import VoxAugDataset
from libft2gan.frame_transformer import FrameTransformerGenerator, FrameTransformerDiscriminator
from libft2gan.lr_scheduler_linear_warmup import LinearWarmupScheduler
from libft2gan.lr_scheduler_polynomial_decay import PolynomialDecayScheduler

from torch.nn import functional as F

def train_epoch(dataloader, generator, discriminator, device, optimizer_gen, optimizer_disc, accumulation_steps, progress_bar, scheduler_gen=None, scheduler_disc=None, grad_scaler_gen=None, grad_scaler_disc=None, step=0, lam=100):
    gen_loss = 0
    batch_gen_loss = 0
    batch_gen_loss_l1 = 0
    batch_gen_loss_gan = 0

    disc_loss = 0
    batch_disc_loss = 0
    batch_disc_fake_loss = 0
    batch_disc_real_loss = 0

    batches = 0
    
    generator.train()
    discriminator.train()

    generator.zero_grad()
    discriminator.zero_grad()
    
    bce_loss = nn.BCEWithLogitsLoss()
    
    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (X, Y) in enumerate(pbar):
        X = X.to(device)[:, :, :generator.max_bin]
        Y = Y.to(device)[:, :, :generator.max_bin]
        
        discriminator.zero_grad()

        with torch.cuda.amp.autocast_mode.autocast():
            M = generator(X)
            M = torch.sigmoid(M)
            Z = X * M
            pavg = torch.mean(M)
            pmin = torch.min(M)
            disc_fake = discriminator(torch.cat((X, Z.detach()), dim=1))
            disc_real = discriminator(torch.cat((X, Y), dim=1))            
            d_real_loss = bce_loss(disc_real, torch.ones_like(disc_real))
            d_fake_loss = bce_loss(disc_fake, torch.zeros_like(disc_fake))

            d_loss = ((d_real_loss + d_fake_loss) * 0.5)
            batch_disc_fake_loss = batch_disc_fake_loss + d_fake_loss
            batch_disc_real_loss = batch_disc_real_loss + d_real_loss
            batch_disc_loss = batch_disc_loss + d_loss
            grad_scaler_disc.scale(d_loss).backward()
            
        grad_scaler_disc.unscale_(optimizer_disc)
        torch.nn.utils.clip_grad.clip_grad_norm_(discriminator.parameters(), 0.5)
        grad_scaler_disc.step(optimizer_disc)
        grad_scaler_disc.update()

        if itr % accumulation_steps == 0:
            optimizer_gen.zero_grad()

        with torch.cuda.amp.autocast_mode.autocast():
            M = generator(X)
            M = torch.sigmoid(M)
            Z = X * M
            disc_fake = discriminator(torch.cat((X, Z.detach()), dim=1))
            g_gan_loss = bce_loss(disc_fake, torch.ones_like(disc_fake))
            g_l1_loss = F.l1_loss(Z, Y)
            g_loss = g_gan_loss + lam * g_l1_loss
            grad_scaler_gen.scale(g_loss).backward()

        batch_gen_loss_l1 = batch_gen_loss_l1 + g_l1_loss
        batch_gen_loss_gan = batch_gen_loss_gan + g_gan_loss
        batch_gen_loss = batch_gen_loss + g_loss

        if (itr + 1) % accumulation_steps == 0:
            if progress_bar:
                pbar.set_description(f'{step}: {str((batch_gen_loss_l1 / accumulation_steps).item())}||{str((batch_gen_loss_gan / accumulation_steps).item())}||{str((batch_disc_fake_loss / accumulation_steps).item())}||{str((batch_disc_real_loss / accumulation_steps).item())}||{pavg.item()}||{pmin.item()}')

            grad_scaler_gen.unscale_(optimizer_gen)
            torch.nn.utils.clip_grad.clip_grad_norm_(generator.parameters(), 0.5)
            grad_scaler_gen.step(optimizer_gen)
            grad_scaler_gen.update()
            disc_loss = disc_loss + batch_disc_loss.item()
            gen_loss = gen_loss + batch_gen_loss_l1.item()
            batch_gen_loss = 0
            batch_gen_loss_l1 = 0
            batch_gen_loss_gan = 0
            batch_disc_loss = 0
            batch_disc_real_loss = 0
            batch_disc_fake_loss = 0
            step = step + 1
            batches = batches + 1
            
            if scheduler_gen is not None:                
                scheduler_gen.step()
            
            if scheduler_disc is not None:                
                scheduler_disc.step()

    return gen_loss / batches, disc_loss / batches,  step

def validate_epoch(dataloader, model, device):
    model.train()
    crit = nn.L1Loss()

    mag_loss = 0

    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)[:, :, :model.max_bin]
            Y = Y.to(device)[:, :, :model.max_bin]

            with torch.cuda.amp.autocast_mode.autocast():
                pred = torch.sigmoid(model(X))

            l1_mag = crit(X[:, :2] * pred[:, :2], Y[:, :2])
            loss = l1_mag

            if torch.logical_or(loss.isnan(), loss.isinf()):
                print('nan validation loss; aborting')
                quit()
            else:
                mag_loss += l1_mag.item() * len(X)

    return mag_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()

    p.add_argument('--distributed', type=str, default='false')

    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--checkpoint_gen', type=str, default=None)
    p.add_argument('--checkpoint_disc', type=str, default=None)
    p.add_argument('--mixed_precision', type=str, default='true')
    p.add_argument('--learning_rate', '-l', type=float, default=1e-4)
    p.add_argument('--lam', type=float, default=100)
    
    p.add_argument('--model_dir', type=str, default='/media/ben/internal-nvme-b')
    p.add_argument('--instrumental_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0|/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0")
    p.add_argument('--vocal_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0_VOCALS")
    p.add_argument('--validation_lib', type=str, default="/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VALIDATION")

    p.add_argument('--curr_step', type=int, default=0)
    p.add_argument('--curr_epoch', type=int, default=0)
    p.add_argument('--warmup_steps', type=int, default=8000)
    p.add_argument('--decay_steps', type=int, default=1000000)
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-12)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=0.1)
    p.add_argument('--lr_verbosity', type=int, default=1000)
    
    p.add_argument('--channels', type=int, default=16)
    p.add_argument('--expansion', type=int, default=10240)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    
    p.add_argument('--stages', type=str, default='900000,1108000')
    p.add_argument('--cropsizes', type=str, default='256,512')
    p.add_argument('--batch_sizes', type=str, default='2,1')
    p.add_argument('--accumulation_steps', '-A', type=str, default='4,8')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw', 'sgd', 'radam', 'rmsprop'], default='adam')
    p.add_argument('--prefetch_factor', type=int, default=4)
    p.add_argument('--num_workers', '-w', type=int, default=2)
    p.add_argument('--epoch', '-E', type=int, default=40)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.stages = [int(s) for i, s in enumerate(args.stages.split(','))]
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    args.batch_sizes = [int(batch_size) for batch_size in args.batch_sizes.split(',')]
    args.accumulation_steps = [int(steps) for steps in args.accumulation_steps.split(',')]
    args.instrumental_lib = [p for p in args.instrumental_lib.split('|')]
    args.vocal_lib = [p for p in args.vocal_lib.split('|')]
    args.distributed = str.lower(args.distributed) == 'true'

    args.model_dir = os.path.join(args.model_dir, "")

    print(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')

    train_dataset = VoxAugDataset(
        path=args.instrumental_lib,
        vocal_path=args.vocal_lib,
        is_validation=False,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )
    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset)

    val_dataset = VoxAugDataset(
        path=[args.validation_lib],
        vocal_path=None,
        cropsize=args.cropsizes[0],
        is_validation=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    generator = FrameTransformerGenerator(in_channels=2, out_channels=2, channels=args.channels, expansion=args.expansion, n_fft=args.n_fft, dropout=args.dropout, num_heads=args.num_heads)
    discriminator = FrameTransformerDiscriminator(in_channels=4, channels=args.channels, expansion=args.expansion, n_fft=args.n_fft, dropout=args.dropout, num_heads=args.num_heads)
    
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        generator.to(device)
        discriminator.to(device)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
        discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])

    if args.checkpoint_gen is not None:
        generator.load_state_dict(torch.load(f'{args.checkpoint_gen}.model.pth', map_location=device))

    if args.checkpoint_disc is not None:
        discriminator.load_state_dict(torch.load(f'{args.checkpoint_disc}.model.pth', map_location=device))
        
    model_parameters = filter(lambda p: p.requires_grad, generator.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'# num params: {params}')    
    
    optimizer_gen = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    optimizer_disc = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    grad_scaler_gen = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    grad_scaler_disc = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    
    stage = 0
    step = args.curr_step
    epoch = args.curr_epoch

    scheduler_gen = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer_gen, target_lr=args.learning_rate, num_steps=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity),
        PolynomialDecayScheduler(optimizer_gen, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=args.decay_steps, start_step=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity)
    ])

    scheduler_disc = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer_disc, target_lr=args.learning_rate, num_steps=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity),
        PolynomialDecayScheduler(optimizer_disc, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=args.decay_steps, start_step=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity)
    ])

    _ = validate_epoch(val_dataloader, generator, device)

    best_loss = float('inf')
    while step < args.stages[-1]:
        if best_loss == float('inf') or step >= args.stages[stage]:
            for idx in range(len(args.stages)):
                if step >= args.stages[idx]:
                    stage = idx + 1
       
            cropsize = args.cropsizes[stage]
            batch_size = args.batch_sizes[stage]
            accum_steps = args.accumulation_steps[stage]
            print(f'setting cropsize to {cropsize}, batch size to {batch_size}, accum steps to {accum_steps}')


            val_dataset.cropsize = cropsize
            val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )
            train_dataset.data_limit = 16

            train_dataset.cropsize = cropsize

            if args.distributed:
                train_dataloader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    shuffle=False,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor
                )
            else:
                train_dataloader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor
                )

        print('# epoch {}'.format(epoch))
        train_dataloader.dataset.set_epoch(epoch)
        train_loss_mag, disc_loss, step = train_epoch(train_dataloader, generator, discriminator, device, optimizer_gen=optimizer_gen, optimizer_disc=optimizer_disc, accumulation_steps=accum_steps, progress_bar=args.progress_bar, scheduler_gen=scheduler_gen, scheduler_disc=scheduler_disc, grad_scaler_gen=grad_scaler_gen, grad_scaler_disc=grad_scaler_disc, step=step, lam=args.lam)
        val_loss_mag = validate_epoch(val_dataloader, generator, device)

        print(
            '  * training l1 loss = {:.6f}, training disc loss = {:6f}, validation loss = {:.6f}'
            .format(train_loss_mag, disc_loss, val_loss_mag)
        )

        if val_loss_mag < best_loss:
            best_loss = val_loss_mag
            print('  * best validation loss')

        model_path = f'{args.model_dir}models/local.{epoch}'
        torch.save(generator.state_dict(), f'{model_path}.modelt.pth')

        epoch += 1

if __name__ == '__main__':
    main()