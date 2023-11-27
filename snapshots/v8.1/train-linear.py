import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.utils.data
import torchaudio.transforms as A
import wandb

from tqdm import tqdm

from libft2gan.dataset_voxaug import VoxAugDataset
from libft2gan.frame_transformer5 import FrameTransformer
from libft2gan.signal_loss import sdr_loss

import torchaudio.transforms as T

from torch.nn import functional as F
from libft2.lr_scheduler_linear_warmup import LinearWarmupScheduler
from libft2.lr_scheduler_polynomial_decay import PolynomialDecayScheduler

def apply_mixup(X, Y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    b = X.shape[0]
    idx = torch.randperm(b)

    XM = lam * X + (1 - lam) * X[idx, :]
    YM = lam * Y + (1 - lam) * Y[idx, :]

    return XM, YM

def train_epoch(dataloader, model, device, optimizer, accumulation_steps, progress_bar, lr_warmup=None, grad_scaler=None, use_wandb=True, step=0, model_dir="", save_every=20000, max_bin=0):
    model.train()

    batch_bce_loss = 0

    mag_loss = 0
    batch_mag_loss = 0

    batch_loss = 0
    batches = 0
    
    model.zero_grad()
    torch.cuda.empty_cache()

    to_mel = T.MelScale(n_mels=128, sample_rate=44100, n_stft=1024).to(device)

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (X, Y, VP) in enumerate(pbar):
        X = X.to(device)[:, :, :max_bin]
        Y = Y.to(device)[:, :, :max_bin]
        #X, Y = apply_mixup(X, Y)

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            pred = torch.sigmoid(model(X))
            pavg = torch.mean(pred)
            pmin = torch.min(pred)
            pred = X * pred
                
        pred_m = to_mel(pred)
        y_m = to_mel(Y)
        mae_loss = F.l1_loss(pred, Y) / accumulation_steps
        mel_loss = F.l1_loss(pred_m, y_m) / accumulation_steps
        sdr = sdr_loss(pred, Y) / accumulation_steps
        accum_loss = mae_loss
        batch_mag_loss = batch_mag_loss + mae_loss
        batch_bce_loss = batch_bce_loss + mel_loss

        if torch.logical_or(accum_loss.isnan(), accum_loss.isinf()):
            print('nan training loss; aborting')
            quit()

        if grad_scaler is not None:
            grad_scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            if progress_bar:
                pbar.set_description(f'{step}: {str(batch_mag_loss.item())}|{str(batch_bce_loss.item())}|{pavg.item()}|{pmin.item()}')

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
            batch_bce_loss = 0

            # if batches % save_every == 0:
            #     model_path = f'{model_dir}models/remover.{step}.tmp.pth'
            #     torch.save(model.state_dict(), model_path)

    return mag_loss / batches, step

def validate_epoch(dataloader, model, device, max_bin=0):
    model.eval()
    crit = nn.L1Loss()

    mag_loss = 0
    mel_loss = 0

    torch.cuda.empty_cache()
    to_mel = T.MelScale(n_mels=128, sample_rate=44100, n_stft=1024).to(device)

    with torch.no_grad():
        for X, Y, _ in dataloader:
            X = X.to(device)[:, :, :max_bin]
            Y = Y.to(device)[:, :, :max_bin]

            with torch.cuda.amp.autocast_mode.autocast():
                pred = torch.sigmoid(model(X))
                pred = X * pred

            pred_m = to_mel(pred)
            y_m = to_mel(Y)
            l1_mel = F.l1_loss(pred_m, y_m)
            l1_mag = crit(pred, Y)

            if torch.logical_or(l1_mag.isnan(), l1_mag.isinf()):
                print('nan validation loss; aborting')
            else:
                mag_loss += l1_mag.item() * len(X)
                mel_loss += l1_mel.item() * len(X)

    return mag_loss / len(dataloader.dataset), mel_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()

    p.add_argument('--distributed', type=str, default='false')
    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', '-s', type=int, default=56)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--checkpoint', type=str, default='/media/ben/internal-nvme-b/models/local.27.553587.voxaug.linear1.pth') # 9
    p.add_argument('--mixed_precision', type=str, default='true')
    p.add_argument('--use_swa', type=str, default="false")
    p.add_argument('--learning_rate', '-l', type=float, default=1e-5)
    p.add_argument('--learning_rate_swa', type=float, default=1e-4)
    
    # p.add_argument('--model_dir', type=str, default='/media/ben/internal-nvme-b')
    # p.add_argument('--instrumental_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0|/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0")
    # p.add_argument('--vocal_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0_VOCALS")
    # p.add_argument('--validation_lib', type=str, default="/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VALIDATION")
    
    p.add_argument('--model_dir', type=str, default='/media/ben/internal-nvme-b')
    # p.add_argument('--instrumental_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0|/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0|/media/ben/Evo 870 SATA 2/cs2048_sr44100_hl1024_nf2048_of0_PAIRS")
    p.add_argument('--instrumental_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0|/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0")#|/media/ben/Evo 870 SATA 2/chaos-app/pt1|/media/ben/Evo 870 SATA 2/chaos-app/pt2")
    p.add_argument('--vocal_lib', type=str, default="/home/ben/cs2048_sr44100_hl1024_nf2048_of0_VOCALS|/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VOCALS")
    p.add_argument('--validation_lib', type=str, default="/media/ben/internal-nvme-b/cs2048_sr44100_hl1024_nf2048_of0_VALIDATION")

    p.add_argument('--curr_step', type=int, default=553587) # 197708
    p.add_argument('--curr_epoch', type=int, default=28) # 10
    p.add_argument('--warmup_steps', type=int, default=553587)
    p.add_argument('--decay_steps', type=int, default=2000000)
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-10)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=0.5)
    p.add_argument('--lr_verbosity', type=int, default=1000)
    
    p.add_argument('--num_attention_maps', type=int, default=1)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_bridge_layers', type=int, default=4)
    p.add_argument('--latent_expansion', type=int, default=4)
    p.add_argument('--expansion', type=float, default=2.2)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    
    p.add_argument('--stages', type=str, default='2000000')
    p.add_argument('--cropsizes', type=str, default='256')
    p.add_argument('--batch_sizes', type=str, default='1')
    p.add_argument('--accumulation_steps', '-A', type=str, default='8')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw', 'sgd', 'radam', 'rmsprop'], default='adam')
    p.add_argument('--amsgrad', type=str, default='false')
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--num_workers', '-w', type=int, default=8)
    p.add_argument('--epoch', '-E', type=int, default=40)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--llrd', type=str, default='false')
    p.add_argument('--lock', type=str, default='false')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--wandb', type=str, default='false')
    p.add_argument('--wandb_project', type=str, default='VOCAL-REMOVER')
    p.add_argument('--wandb_entity', type=str, default='carperbr')
    p.add_argument('--wandb_run_id', type=str, default=None)
    p.add_argument('--cropsize', type=int, default=0)
    args = p.parse_args()

    args.amsgrad = str.lower(args.amsgrad) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.llrd = str.lower(args.llrd) == 'true'
    args.lock = str.lower(args.lock) == 'true'
    args.wandb = str.lower(args.wandb) == 'true'
    args.use_swa = str.lower(args.use_swa) == 'true'
    args.stages = [int(s) for i, s in enumerate(args.stages.split(','))]
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    args.batch_sizes = [int(batch_size) for batch_size in args.batch_sizes.split(',')]
    args.accumulation_steps = [int(steps) for steps in args.accumulation_steps.split(',')]
    args.instrumental_lib = [p for p in args.instrumental_lib.split('|')]
    args.vocal_lib = [p for p in args.vocal_lib.split('|')]
    args.distributed = str.lower(args.distributed) == 'true'

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
        hop_length=args.hop_length
    )

    train_sampler = torch.utils.data.DistributedSampler(train_dataset) if args.distributed else None

    val_dataset = VoxAugDataset(
        instrumental_lib=[args.validation_lib],
        vocal_lib=None,
        cropsize=2048,
        is_validation=True
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    model = FrameTransformer(in_channels=2, out_channels=2, channels=args.channels, expansion=args.expansion, n_fft=args.n_fft, dropout=args.dropout, num_heads=args.num_heads, num_attention_maps=args.num_attention_maps)
    
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(f'{args.checkpoint}', map_location=device))

    groups = [
        { "params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.learning_rate }
    ]
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'# {wandb.run.name if args.wandb else ""}; num params: {params}')    
    
    optimizer = torch.optim.AdamW(
        groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    
    stage = 0
    step = args.curr_step
    epoch = args.curr_epoch

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity),
        PolynomialDecayScheduler(optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=args.decay_steps, start_step=args.warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity)
    ])        

    val_dataset.cropsize = 2048
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    #val_loss_mag, val_loss_mel = validate_epoch(val_dataloader, model, device, max_bin=args.n_fft // 2)

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

            train_dataset.cropsize = cropsize

            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor
            )

        print('# epoch {}'.format(epoch))
        train_dataloader.dataset.set_epoch(epoch)
        train_loss_mag, step = train_epoch(train_dataloader, model, device, optimizer, accum_steps, args.progress_bar, lr_warmup=scheduler, grad_scaler=grad_scaler, use_wandb=args.wandb, step=step, model_dir=args.model_dir, max_bin=args.n_fft // 2)
        val_loss_mag, val_loss_mel = validate_epoch(val_dataloader, model, device, max_bin=args.n_fft // 2)

        if args.wandb:
            wandb.log({
                'train_loss_mag': train_loss_mag,
                'val_loss_mag': val_loss_mag,
            })

        print(
            '  * training loss = {:.6f}, validation loss (lin) = {:.6f}, validation loss (mel) = {:.6f}'
            .format(train_loss_mag, val_loss_mag, val_loss_mel)
        )

        if val_loss_mag < best_loss:
            best_loss = val_loss_mag
            print('  * best validation loss')

        model_path = f'{args.model_dir}models/{wandb.run.name if args.wandb else "local"}.{epoch}'
        torch.save(model.state_dict(), f'{model_path}.{step}.voxaug.linear1.pth')

        epoch += 1

if __name__ == '__main__':
    main()