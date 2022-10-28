import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import wandb

from tqdm import tqdm

from dataset_voxpred import VoxAugDataset
from frame_transformer_v4 import FrameTransformer

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

def train_epoch(dataloader, model, device, optimizer, accumulation_steps, progress_bar, lr_warmup=None, grad_scaler=None, use_wandb=True, step=0):
    model.train()
    sum_loss = 0
    crit = nn.L1Loss()
    batch_loss = 0
    batches = 0
    
    model.zero_grad()

    pbar = tqdm(dataloader) if progress_bar else dataloader
    for itr, (X, V) in enumerate(pbar):
        X = X.to(device)[:, :, :model.max_bin]
        V = V.to(device)[:, :, :model.max_bin]

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            v_pred = torch.sigmoid(model(X))

        l1_loss = crit(v_pred, V) / accumulation_steps

        batch_loss = batch_loss + l1_loss
        accum_loss = l1_loss

        if torch.logical_or(accum_loss.isnan(), accum_loss.isinf()):
            print('nan training loss; aborting')
            quit()

        if grad_scaler is not None:
            grad_scaler.scale(accum_loss).backward()
        else:
            accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            if progress_bar:
                pbar.set_description(f'{step}: {str(batch_loss.item())}')

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
            sum_loss = sum_loss + batch_loss.item()
            batch_loss = 0

        #sum_loss += accum_loss.item() * len(X_batch) * accumulation_steps

    # # the rest batch
    # if (itr + 1) % accumulation_steps != 0:
    #     # grad_scaler.unscale_(optimizer)
    #     # clip_grad_norm_(model.parameters(), 0.5)
    #     optimizer.step()
    #     model.zero_grad()

    return sum_loss / batches, step

def validate_epoch(dataloader, model, device):
    model.eval()
    crit = nn.L1Loss()
    sum_loss = 0

    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)[:, :, :model.max_bin]
            Y = Y.to(device)[:, :, :model.max_bin]

            v_pred = torch.sigmoid(model(X))
            mag_loss = crit(X - v_pred, Y)

            if torch.logical_or(mag_loss.isnan(), mag_loss.isinf()):
                print('nan validation loss; aborting')
                quit()
            else:
                sum_loss += mag_loss.item() * len(X)

    return sum_loss / len(dataloader.dataset)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', '-s', type=int, default=51)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--mixed_precision', type=str, default='true') # seems to encounter NaN loss after a few hours when using mixed precision.

    p.add_argument('--curr_epoch', type=int, default=0)
    p.add_argument('--warmup_steps', type=int, default=2)
    p.add_argument('--decay_steps', type=int, default=750000)
    p.add_argument('--curr_step', type=int, default=0)
    p.add_argument('--lr_scheduler_decay_target', type=int, default=1e-12)
    p.add_argument('--lr_scheduler_decay_power', type=float, default=1.0)
    p.add_argument('--lr_verbosity', type=int, default=1000)

    p.add_argument('--vocal_noise_rate', type=float, default=1.0)
    
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--feedforward_expansion', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)

    p.add_argument('--cropsizes', type=str, default='256,512')
    p.add_argument('--steps', type=str, default='400000,500000')
    p.add_argument('--epochs', type=str, default='58,60')
    p.add_argument('--batch_sizes', type=str, default='3,1')
    p.add_argument('--accumulation_steps', '-A', type=str, default='3,8')
    p.add_argument('--force_voxaug', type=str, default='false')

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
            # "C://cs2048_sr44100_hl1024_nf2048_of0",
            # "D://cs2048_sr44100_hl1024_nf2048_of0",
            # "F://cs2048_sr44100_hl1024_nf2048_of0",
            # "H://cs2048_sr44100_hl1024_nf2048_of0",
            # "J://cs2048_sr44100_hl1024_nf2048_of0",
            # "K://cs2048_sr44100_hl1024_nf2048_of0",
        ],
        pair_path=[
            "F://cs2048_sr44100_hl1024_nf2048_of0_PAIRS",
            "K://cs2048_sr44100_hl1024_nf2048_of0_PAIRS",
            "J://cs2048_sr44100_hl1024_nf2048_of0_PAIRS",
        ],
        vocal_path=[
            "F://cs2048_sr44100_hl1024_nf2048_of0_VOCALS",
            "K://cs2048_sr44100_hl1024_nf2048_of0_VOCALS",
        ],
        is_validation=False,
        epoch_size=args.epoch_size,
        force_voxaug=args.force_voxaug,
        pair_mul=1,
        noise_rate=args.vocal_noise_rate
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')
    model = FrameTransformer(in_channels=2, channels=args.channels, expansion=args.feedforward_expansion, n_fft=args.n_fft, dropout=args.dropout, num_heads=args.num_heads)

    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

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
    
    steps = len(train_dataset) // (args.batch_sizes[0] * args.accumulation_steps[0])
    warmup_steps = args.warmup_steps
    num_epochs = args.decay_steps // steps

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=warmup_steps, current_step=args.curr_step, verbose_skip_steps=args.lr_verbosity),
        PolynomialDecayScheduler(optimizer, target=args.lr_scheduler_decay_target, power=args.lr_scheduler_decay_power, num_decay_steps=args.decay_steps, start_step=warmup_steps, current_step=args.curr_step, verbose_skip_steps=args.lr_verbosity)
    ])

    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    val_dataset = None
    curr_idx = 0
    step = args.curr_step

    print(f'{num_epochs} epochs')

    best_loss = np.inf
    for epoch in range(args.curr_epoch, num_epochs):
        train_dataset.rebuild()

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
                path=[f"C://cs256_sr44100_hl1024_nf2048_of0_VALIDATION"],
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
                path=[f"C://cs256_sr44100_hl1024_nf2048_of0_VALIDATION"],
                vocal_path=None,
                is_validation=True,
                clip_validation=True
            )

            val_dataloader2 = torch.utils.data.DataLoader(
                dataset=val_dataset2,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )

        print('# epoch {}'.format(epoch))

        train_loss, step = train_epoch(train_dataloader, model, device, optimizer, accum_steps, args.progress_bar, lr_warmup=scheduler, grad_scaler=grad_scaler, use_wandb=args.wandb, step=step)
        val_loss = validate_epoch(val_dataloader, model, device)
        val_loss2 = validate_epoch(val_dataloader2, model, device)

        if args.wandb:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
            })

        print(
            '  * training loss = {:.6f}, validation loss = {:.6f}, validation loss (clipped) = {:.6f}'
            .format(train_loss, val_loss, val_loss2)
        )

        if (val_loss2) < best_loss:
            best_loss = val_loss2
            print('  * best validation loss')

        model_path = f'{args.model_dir}models/{wandb.run.name if args.wandb else "local"}.{epoch}.remover.pth'
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()