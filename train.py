import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import wandb

from tqdm import tqdm

from dataset_voxaug2 import VoxAugDataset
from frame_transformer import FrameTransformer
from torch.nn import functional as F

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
    
    phase_loss = 0
    batch_phase_loss = 0

    sum_loss = 0
    crit = nn.L1Loss()
    batch_loss = 0
    batches = 0
    
    model.zero_grad()

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (X, Y) in enumerate(pbar):
        X = X.to(device)[:, :, :model.max_bin*2]
        Y = Y.to(device)[:, :, :model.max_bin*2]

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            X2 = F.interpolate(X, scale_factor=(0.5,1), mode='nearest')
            pred = torch.sigmoid(model(X2))
            pred = F.interpolate(pred, scale_factor=(2,1), mode='bicubic', align_corners=True)
            
        l1_mag = crit(X[:, :2] * pred[:, :2], Y[:, :2]) / accumulation_steps
        l1_phase = crit(pred[:, 2:], Y[:, 2:]) / accumulation_steps if include_phase else torch.zeros_like(l1_mag)

        batch_mag_loss = batch_mag_loss + l1_mag
        batch_phase_loss = batch_phase_loss + l1_phase

        accum_loss = l1_mag * 10 + l1_phase

        if torch.logical_or(accum_loss.isnan(), accum_loss.isinf()):
            print('nan training loss; aborting')
            quit()

        if grad_scaler is not None:
            grad_scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            if progress_bar:
                pbar.set_description(f'{step}: {str(batch_mag_loss.item())}|{str(batch_phase_loss.item())}')

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
            phase_loss = phase_loss + batch_phase_loss.item()
            batch_mag_loss = 0
            batch_phase_loss = 0

            if batches % save_every == 0:
                model_path = f'{model_dir}models/remover.{step}.tmp.pth'
                torch.save(model.state_dict(), model_path)

    return mag_loss / batches, phase_loss / batches, step

def validate_epoch(dataloader, model, device, include_phase=False):
    model.eval()
    crit = nn.L1Loss()

    mag_loss = 0
    phase_loss = 0

    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)[:, :, :model.max_bin*2]
            Y = Y.to(device)[:, :, :model.max_bin*2]

            X2 = F.interpolate(X, scale_factor=(0.5,1), mode='nearest')
            pred = torch.sigmoid(model(X2))
            pred = F.interpolate(pred, scale_factor=(2,1), mode='bicubic', align_corners=True)

            l1_mag = crit(X[:, :2] * pred[:, :2], Y[:, :2])
            l1_phase = crit(pred[:, 2:], Y[:, 2:]) if include_phase else torch.zeros_like(l1_mag)
            loss = l1_mag + l1_phase

            if torch.logical_or(loss.isnan(), loss.isinf()):
                print('nan validation loss; aborting')
                quit()
            else:
                mag_loss += l1_mag.item() * len(X)
                phase_loss += l1_phase.item() * len(X)

    return mag_loss / len(dataloader.dataset), phase_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--mixed_precision', type=str, default='true') # seems to encounter NaN loss after a few hours when using mixed precision.

    p.add_argument('--curr_step', type=int, default=0)
    p.add_argument('--curr_epoch', type=int, default=0)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--decay_steps', type=int, default=1000000)
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-12)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=0.1)
    p.add_argument('--lr_verbosity', type=int, default=1000)
    
    p.add_argument('--include_phase', type=str, default='false')
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--feedforward_expansion', type=int, default=5)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)

    p.add_argument('--cropsizes', type=str, default='256')
    p.add_argument('--steps', type=str, default='1000000')
    p.add_argument('--epochs', type=str, default='58')
    p.add_argument('--batch_sizes', type=str, default='4')
    p.add_argument('--accumulation_steps', '-A', type=str, default='2')
    p.add_argument('--force_voxaug', type=str, default='true')

    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'adamw', 'sgd', 'radam', 'rmsprop'], default='adam')
    p.add_argument('--amsgrad', type=str, default='false')
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--epoch', '-E', type=int, default=40)
    p.add_argument('--epoch_size', type=int, default=None)
    p.add_argument('--learning_rate', '-l', type=float, default=1e-4)
    p.add_argument('--progress_bar', '-pb', type=str, default='true')
    p.add_argument('--save_all', type=str, default='true')
    p.add_argument('--model_dir', type=str, default='J://')
    p.add_argument('--llrd', type=str, default='false')
    p.add_argument('--lock', type=str, default='false')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--wandb', type=str, default='true')
    p.add_argument('--wandb_project', type=str, default='VOCAL-REMOVER')
    p.add_argument('--wandb_entity', type=str, default='carperbr')
    p.add_argument('--wandb_run_id', type=str, default=None)
    p.add_argument('--prefetch_factor', type=int, default=2)
    p.add_argument('--cropsize', type=int, default=0)
    args = p.parse_args()

    args.include_phase = str.lower(args.include_phase) == 'true'
    args.amsgrad = str.lower(args.amsgrad) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.save_all = str.lower(args.save_all) == 'true'
    args.force_voxaug = str.lower(args.force_voxaug) == 'true'
    args.llrd = str.lower(args.llrd) == 'true'
    args.lock = str.lower(args.lock) == 'true'
    args.wandb = str.lower(args.wandb) == 'true'
    args.steps = [int(s) for i, s in enumerate(args.steps.split(','))]
    total_steps = sum(args.steps)
    args.epochs = [int(epoch) for i, epoch in enumerate(args.epochs.split(','))]
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    args.batch_sizes = [int(batch_size) for batch_size in args.batch_sizes.split(',')]
    args.accumulation_steps = [int(steps) for steps in args.accumulation_steps.split(',')]

    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, id=args.wandb_run_id, resume="must" if args.wandb_run_id is not None else None)

    print(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    train_dataset = VoxAugDataset(
        path=[
            "C://cs256_sr44100_hl1024_nf2048_of0",
            "D://cs256_sr44100_hl1024_nf2048_of0",
            "F://cs256_sr44100_hl1024_nf2048_of0",
            "H://cs256_sr44100_hl1024_nf2048_of0",
        ],
        vocal_path=[
            "D://cs256_sr44100_hl1024_nf2048_of0_VOCALS"
        ],
        is_validation=False
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    model = FrameTransformer(in_channels=4 if args.include_phase else 2, out_channels=4 if args.include_phase else 2, channels=args.channels, expansion=args.feedforward_expansion, n_fft=args.n_fft // 2, dropout=args.dropout, num_heads=args.num_heads)
    
    groups = [
        { "params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.learning_rate }
    ]
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'# {wandb.run.name if args.wandb else ""}; num params: {params}')    
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            groups,
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            groups,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.learning_rate,
            amsgrad=args.amsgrad,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'radam':
        optimizer = torch.optim.RAdam(
            groups,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(groups, lr=args.learning_rate)

    val_dataset = None
    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    curr_idx = 0
    step = args.curr_step
    curr_epoch = args.curr_epoch
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    checkpoint = None
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(f'{args.checkpoint}.model.pth', map_location=device))
        # optimizer.load_state_dict(torch.load(f'{args.checkpoint}.opt.pth')) # unfortunately PyTorch has a bug that causes memory to increase drastically if you try to load optimizer from checkpoint. Pretty shocked they'd leave such a glaring bug for so long...

        if grad_scaler is not None:
            grad_scaler.load_state_dict(torch.load(f'{args.checkpoint}.scaler.pth', map_location=device))
    
    steps = len(train_dataset) // (args.batch_sizes[0] * args.accumulation_steps[0])
    warmup_steps = args.warmup_steps
    num_epochs = args.decay_steps // steps

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity),
        PolynomialDecayScheduler(optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=args.decay_steps, start_step=warmup_steps, current_step=step, verbose_skip_steps=args.lr_verbosity)
    ])

    print(f'# {num_epochs} epochs')
    best_loss = np.inf
    for epoch in range(curr_epoch, num_epochs):
        if epoch > args.epochs[curr_idx] or val_dataset is None:
            for i,e in enumerate(args.epochs):
                if epoch > e:
                    curr_idx = i + 1
            
            curr_idx = min(curr_idx, len(args.cropsizes) - 1)
            cropsize = args.cropsizes[curr_idx]
            batch_size = args.batch_sizes[curr_idx]
            accum_steps = args.accumulation_steps[curr_idx]
            print(f'setting cropsize to {cropsize}, batch size to {batch_size}, accum steps to {accum_steps}')

            train_dataset.cropsize = cropsize
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor
            )

            val_dataset = VoxAugDataset(
                path=[f"C://cs{cropsize}_sr44100_hl{args.hop_length}_nf{args.n_fft}_of0_VALIDATION"],
                vocal_path=None,
                is_validation=True
            )

            val_dataloader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )

            val_dataset2 = VoxAugDataset(
                path=[f"C://cs{cropsize}_sr44100_hl{args.hop_length}_nf{args.n_fft}_of0_VALIDATION"],
                vocal_path=None,
                is_validation=True
            )

            val_dataloader2 = torch.utils.data.DataLoader(
                dataset=val_dataset2,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )

        print('# epoch {}'.format(epoch))

        train_dataloader.dataset.set_epoch(epoch)
        train_loss_mag, train_loss_phase, step = train_epoch(train_dataloader, model, device, optimizer, accum_steps, args.progress_bar, lr_warmup=scheduler, grad_scaler=grad_scaler, use_wandb=args.wandb, step=step, include_phase=args.include_phase, model_dir=args.model_dir)
        val_loss_mag, val_loss_phase = validate_epoch(val_dataloader, model, device, include_phase=args.include_phase)
        val_loss_mag2, _ = validate_epoch(val_dataloader2, model, device, include_phase=args.include_phase)

        if args.wandb:
            wandb.log({
                'train_loss_mag': train_loss_mag,
                'train_loss_phase': train_loss_phase,
                'val_loss_mag': val_loss_mag,
                'val_loss_phase': val_loss_phase
            })

        print(
            '  * training loss = {:.6f}, validation loss = {:.6f}, validation loss (clipped) = {:.6f}, phase = {:.6f}'
            .format(train_loss_mag, val_loss_mag, val_loss_mag2, val_loss_phase)
        )

        if val_loss_mag2 < best_loss:
            best_loss = val_loss_mag2
            print('  * best validation loss')

        model_path = f'{args.model_dir}models/{wandb.run.name if args.wandb else "local"}.{epoch}'

        torch.save(model.state_dict(), f'{model_path}.model.pth')
        
        if grad_scaler is not None:
            torch.save(grad_scaler.state_dict(), f'{model_path}.scaler.pth')


if __name__ == '__main__':
    main()