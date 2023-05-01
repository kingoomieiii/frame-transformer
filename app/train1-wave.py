import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed

from tqdm import tqdm
import wandb

from libft2gan.dataset_voxaug3 import VoxAugDataset
from libft2gan.specwave_transformer8 import SpecWaveTransformer
from libft2gan.lr_scheduler_linear_warmup import LinearWarmupScheduler
from libft2gan.lr_scheduler_polynomial_decay import PolynomialDecayScheduler
from libft2gan.signal_loss import sdr_loss

from torch.nn import functional as F

import torchaudio
import torchaudio.transforms as T

def halve_tensor(X):
    X1 = X[:, :, :, (X.shape[3] // 2):]
    X2 = X[:, :, :, :(X.shape[3] // 2)]
    X = torch.cat((X1, X2), dim=0)
    return X

# def to_spec(W):
#     SL = torch.stft(W[0].float(), n_fft=2048, hop_length=1024, normalized=True, return_complex=True)
#     SR = torch.stft(W[1].float(), n_fft=2048, hop_length=1024, normalized=True, return_complex=True)
#     S = torch.cat((SL.unsqueeze(1), SR.unsqueeze(1)), dim=1)
#     return S

def train_epoch(dataloader, model, device, optimizer, accumulation_steps, progress_bar, lr_warmup=None, grad_scaler=None, step=0, max_bin=0, use_wandb=False, predict_mask=True, predict_phase=False):
    model.train()

    batch_loss = 0
    mag_loss = 0
    batches = 0
    
    model.zero_grad()
    torch.cuda.empty_cache()

    to_spec = T.Spectrogram(n_fft=2048, hop_length=1024, power=None, return_complex=True).to(device)
    to_mel = T.MelScale(n_mels=128, sample_rate=44100, n_stft=1025).to(device)

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (XW, YW, c) in enumerate(pbar):
        XW = XW.to(device)
        YW = YW.to(device)
        c = c.to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            YS = to_spec(YW)
            YP = ((torch.angle(YS) + torch.pi) / (2 * torch.pi))[:, :, :-1]
            YS = (torch.abs(YS) / c)[:, :, :-1]
        
        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            PW, PS, PP = model(XW, c)

        phase_loss = (1 - torch.mean(torch.cos(PP - YP))) / accumulation_steps
        spectral_loss = F.l1_loss(PS, YS) / accumulation_steps
        wave_loss = F.l1_loss(PW, YW) / accumulation_steps
        accum_loss = wave_loss + spectral_loss + phase_loss
        batch_loss = batch_loss + wave_loss.item()

        if torch.logical_or(accum_loss.isnan(), accum_loss.isinf()):
            print('nan training loss; aborting')
            quit()

        if grad_scaler is not None:
            grad_scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            if progress_bar:
                pbar.set_description(f'{step}: wave={wave_loss.item()}, mag={spectral_loss.item()}, phase={phase_loss.item()}')
            
            if use_wandb:
                wandb.log({
                    'spec_loss': wave_loss.item(),
                    'wave_loss': wave_loss.item(),
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
            mag_loss = mag_loss + batch_loss
            batch_loss = 0

    return mag_loss / batches, step

def validate_epoch(dataloader, model, device, max_bin=0, use_wandb=False, predict_mask=True, predict_phase=False):
    model.train()
    crit = nn.L1Loss()

    sum_spec = 0
    sum_wave = 0

    sum_loss = 0
    batches = 0

    torch.cuda.empty_cache()

    to_spec = T.Spectrogram(n_fft=2048, hop_length=1024, power=None, return_complex=True).to(device)
    to_mel = T.MelScale(n_mels=128, sample_rate=44100, n_stft=1025).to(device)

    with torch.no_grad():
        for itr, (XW, YW, c) in enumerate(dataloader):
            XW = XW.to(device)
            YW = YW.to(device)
            c = c.to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            YS = to_spec(YW)
            YP = ((torch.angle(YS) + torch.pi) / (2 * torch.pi))[:, :, :-1]
            YS = (torch.abs(YS) / c)[:, :, :-1]
            
            with torch.cuda.amp.autocast_mode.autocast():
                PW, PS, PP = model(XW, c)

            mag_loss = F.l1_loss(PS, YS)
            wave_loss = F.l1_loss(PW, YW)
            
            if use_wandb:
                wandb.log({
                    'wave_loss_val': wave_loss.item()
                })

            loss = wave_loss

            if torch.logical_or(loss.isnan(), loss.isinf()):
                print('nan validation loss; aborting')
                quit()
            else:
                sum_spec += mag_loss.item()
                sum_wave += wave_loss.item()
                batches += 1

    return sum_wave / batches, sum_spec / batches

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--pretrained_checkpoint', type=str, default=None)#"H://models/local.0.pre.pth")
    p.add_argument('--checkpoint', type=str, default=None)#"H://models/local.12.pre.pth")
    p.add_argument('--mixed_precision', type=str, default='true')
    p.add_argument('--learning_rate', '-l', type=float, default=1e-4)
    p.add_argument('--lam', type=float, default=100)
    p.add_argument('--distributed', type=str, default="false")
    p.add_argument('--world_rank', type=int, default=0)

    p.add_argument('--predict_mask', type=str, default='true')
    p.add_argument('--predict_phase', type=str, default='false')

    p.add_argument('--model_dir', type=str, default='H://')
    p.add_argument('--instrumental_lib', type=str, default="C://cs2048_sr44100_hl1024_nf2048_of0|D://cs2048_sr44100_hl1024_nf2048_of0|F://cs2048_sr44100_hl1024_nf2048_of0|H://cs2048_sr44100_hl1024_nf2048_of0")
    p.add_argument('--vocal_lib', type=str, default="C://cs2048_sr44100_hl1024_nf2048_of0_VOCALS|D://cs2048_sr44100_hl1024_nf2048_of0_VOCALS")
    p.add_argument('--validation_lib', type=str, default="C://cs2048_sr44100_hl1024_nf2048_of0_VALIDATION")
    
    # p.add_argument('--model_dir', type=str, default='/media/ben/internal-nvme-b')
    # p.add_argument('--instrumental_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0|/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0")
    # p.add_argument('--vocal_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0_VOCALS")
    # p.add_argument('--validation_lib', type=str, default="/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VALIDATION")

    p.add_argument('--curr_step', type=int, default=0)
    p.add_argument('--curr_epoch', type=int, default=0)
    p.add_argument('--warmup_steps', type=int, default=12000)
    p.add_argument('--decay_steps', type=int, default=1000000)
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-12)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=0.1)
    p.add_argument('--lr_verbosity', type=int, default=1000)
     
    p.add_argument('--num_bridge_layers', type=int, default=8)
    p.add_argument('--num_attention_maps', type=int, default=3)
    p.add_argument('--wave_channels', type=int, default=4)
    p.add_argument('--frame_channels', type=int, default=8)
    p.add_argument('--wave_expansion', type=int, default=4)
    p.add_argument('--frame_expansion', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    
    p.add_argument('--stages', type=str, default='900000,1108000')
    p.add_argument('--cropsizes', type=str, default='256,512')
    p.add_argument('--batch_sizes', type=str, default='2,1')
    p.add_argument('--accumulation_steps', '-A', type=str, default='4,8')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw', 'sgd', 'radam', 'rmsprop'], default='adam')
    p.add_argument('--prefetch_factor', type=int, default=4)
    p.add_argument('--num_workers', '-w', type=int, default=8)
    p.add_argument('--epoch', '-E', type=int, default=40)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--wandb', type=str, default='false')
    p.add_argument('--wandb_project', type=str, default='VOCAL-REMOVER')
    p.add_argument('--wandb_entity', type=str, default='carperbr')
    p.add_argument('--wandb_run_id', type=str, default=None)

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
    args.predict_phase = str.lower(args.predict_phase) == 'true'
    args.predict_mask = str.lower(args.predict_mask) == 'true'
    args.wandb = str.lower(args.wandb) == 'true'

    args.model_dir = os.path.join(args.model_dir, "")

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, id=args.wandb_run_id, resume="must" if args.wandb_run_id is not None else None)

    print(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')

    train_dataset = VoxAugDataset(
        instrumental_lib=args.instrumental_lib,
        vocal_lib=args.vocal_lib,
        is_validation=False,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        predict_phase=args.predict_phase
    )

    train_sampler = torch.utils.data.DistributedSampler(train_dataset) if args.distributed else None

    val_dataset = VoxAugDataset(
        instrumental_lib=[args.validation_lib],
        vocal_lib=None,
        is_validation=True,
        predict_phase=args.predict_phase
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

    generator = SpecWaveTransformer(wave_in_channels=2, frame_in_channels=2, wave_out_channels=2, frame_out_channels=2, wave_channels=args.wave_channels, frame_channels=args.frame_channels, dropout=args.dropout, n_fft=args.n_fft, wave_heads=args.num_heads, wave_expansion=args.wave_expansion, frame_heads=args.num_heads, frame_expansion=args.frame_expansion, num_attention_maps=args.num_attention_maps)

    # generator = FrameWaveTransformer(wave_in_channels=2, frame_in_channels=4, wave_out_channels=2, frame_out_channels=2, wave_channels=args.wave_channels, frame_channels=args.frame_channels, dropout=args.dropout, n_fft=args.n_fft, wave_transformer_layers=args.num_bridge_layers, wave_heads=args.num_heads, wave_expansion=args.wave_expansion, frame_heads=args.num_heads, frame_expansion=args.frame_expansion, num_attention_maps=args.num_attention_maps)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        generator.to(device)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])

    if args.checkpoint is not None:
        generator.load_state_dict(torch.load(f'{args.checkpoint}', map_location=device))
    elif args.pretrained_checkpoint is not None:
        generator.load_state_dict(torch.load(f'{args.pretrained_checkpoint}', map_location=device))

    model_parameters = filter(lambda p: p.requires_grad, generator.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'# num params: {params}')

    optimizer_gen = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=args.learning_rate
    )

    grad_scaler_gen = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    
    stage = 0
    step = args.curr_step
    epoch = args.curr_epoch

    scheduler_gen = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer_gen, target_lr=args.learning_rate, num_steps=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity),
        PolynomialDecayScheduler(optimizer_gen, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=args.decay_steps, start_step=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity)
    ])

    val_dataset.cropsize = 2048
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # wave, spec = validate_epoch(val_dataloader, generator, device, max_bin=args.n_fft // 2, predict_mask=args.predict_mask, predict_phase=args.predict_phase)

    best_loss = float('inf')
    while step < args.stages[-1]:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if best_loss == float('inf') or step >= args.stages[stage]:
            for idx in range(len(args.stages)):
                if step >= args.stages[idx]:
                    stage = idx + 1
       
            cropsize = args.cropsizes[stage]
            batch_size = args.batch_sizes[stage]
            accum_steps = args.accumulation_steps[stage]
            print(f'setting cropsize to {cropsize}, batch size to {batch_size}, accum steps to {accum_steps}')

            train_dataset.cropsize = cropsize
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False if args.distributed else True,
                sampler=train_sampler if args.distributed else None,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=True
            )

        print('# epoch {}'.format(epoch))
        train_dataloader.dataset.set_epoch(epoch)
        train_loss_mag, step = train_epoch(train_dataloader, generator, device, optimizer=optimizer_gen, accumulation_steps=accum_steps, progress_bar=args.progress_bar, lr_warmup=scheduler_gen, grad_scaler=grad_scaler_gen, step=step, max_bin=args.n_fft // 2, use_wandb=args.wandb, predict_mask=args.predict_mask, predict_phase=args.predict_phase)
        wave, spec = validate_epoch(val_dataloader, generator, device, max_bin=args.n_fft // 2, predict_mask=args.predict_mask, predict_phase=args.predict_phase)

        print(
            '  * training l1 loss = {:.6f}, wave_loss = {:6f} spec loss = {:.6f}'
            .format(train_loss_mag, wave, spec)
        )
        
        if spec < best_loss:
            best_loss = spec
            print('  * best validation loss')

        if args.world_rank == 0:
            model_path = f'{args.model_dir}models/local.{epoch}'
            torch.save(generator.state_dict(), f'{model_path}.stg1.{"phase" if args.predict_phase else "mag"}.pth')
        epoch += 1

    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()