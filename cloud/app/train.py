import argparse
import datetime
from io import BytesIO
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from cloud.app.lib.frame_transformer import FrameTransformer
from cloud.app.lib.dataset import VocalRemoverCloudDataset
from cloud.app.lib.warmup_lr import WarmupLR
import multiprocessing
from google.cloud import storage
from tqdm import tqdm
import os
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

import torch.distributed as dist
import torch.multiprocessing as mp

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
    for itr, (X_batch, y_batch) in enumerate(pbar):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if np.random.uniform() < mixup_rate:
            X_batch, y_batch = mixup(X_batch, y_batch, mixup_alpha)

        with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
            pred = model(X_batch)

        loss = crit(pred * X_batch, y_batch)
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

        sum_loss += loss.item() * len(X_batch)

    # the rest batch
    if (itr + 1) % accumulation_steps != 0:
        grad_scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 0.5)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        model.zero_grad()

    return sum_loss / len(dataloader.dataset)

def validate_epoch(dataloader, model, device, grad_scaler):
    model.eval()
    sum_loss = 0
    crit = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            with torch.cuda.amp.autocast_mode.autocast(enabled=grad_scaler is not None):
                pred = model(X_batch)

            loss = crit(X_batch * pred, y_batch)
    
            if torch.logical_or(loss.isnan(), loss.isinf()):
                print('non-finite or nan validation loss; aborting')
                quit()
            else:
                sum_loss += loss.item() * len(X_batch)

    return sum_loss / len(dataloader.dataset)

def node_main(idx, args):
    rank = args.nr * args.gpus + idx

    def print_master(str):
        if rank == 0:
            print(str)

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    print_master(f'WORLD SIZE: {args.world_size}')
    print_master(f'CURRENT RANK: {rank}')

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    train_dataset = VocalRemoverCloudDataset(dataset=args.train_dataset, vocal_dataset=args.vocal_dataset, num_training_items=args.num_training_items)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True
    )

    val_dataset = VocalRemoverCloudDataset(dataset=args.validation_dataset, vocal_dataset=args.vocal_dataset, num_training_items=args.num_training_items)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(f'cuda:{idx}')
    model = FrameTransformer(channels=args.channels, n_fft=args.n_fft, num_decoders=args.num_decoders, num_bands=args.num_bands, feedforward_dim=args.feedforward_dim, bias=args.bias, cropsize=args.cropsize).to(device)
    model = DistributedDataParallel(model, device_ids=[idx])
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print_master(f'Param count: {params}')

    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler() if args.mixed_precision else None
    
    client = storage.Client()
    bucket = client.bucket('bc-vocal-remover')

    if bucket.get_blob('models/') == None:
        blob = bucket.blob('models/')
        blob.upload_from_string("")

    if not os.path.exists('models'):
        os.makedirs('models')

    if args.checkpoint is not None:
        blob = bucket.get_blob(f'models/{args.checkpoint}.pth')
        
        if blob != None:
            blob_data = blob.download_as_bytes()
            bytes = BytesIO(blob_data)
            print(f'{rank} loading checkpoint')
            model.load_state_dict(torch.load(bytes))
            blob = bucket.blob('models/')
            print(f'{rank} loaded checkpoint')
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        amsgrad=args.amsgrad
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        threshold=1e-6,
        min_lr=args.lr_min,
        verbose=True
    )

    lr_warmup = WarmupLR(optimizer, target_lr=args.learning_rate, num_steps=args.lr_warmup_steps, current_step=args.lr_warmup_current_step, verbose=True) if args.lr_warmup_steps > 0 else None

    for epoch in range(args.epochs):
        model.train()                

        pb = tqdm(train_dataloader) if args.progress_bar and rank == 0 else train_dataloader
        for X, Y in pb:
            X = X.to(device)
            Y = Y.to(device)            
            train_loss = train_epoch(train_dataloader, model, device, optimizer, args.accumulation_steps, grad_scaler, args.progress_bar, args.mixup_rate, args.mixup_alpha, lr_warmup=lr_warmup)
            val_loss = validate_epoch(val_dataloader, model, device, grad_scaler)

        scheduler.step(val_loss)

        if rank == 0:
            model_path = f'{args.job_name}.i{epoch}.pth'
            torch.save(model.module.state_dict(), f'models/{model_path}')
            model_blob = bucket.blob(f'models/{model_path}')
            model_blob.upload_from_filename(f'models/{model_path}')

        print_master(f'Epoch {epoch} loss: total={train_loss}')
        print_master(f'Epoch {epoch} validation loss: {val_loss}')
        print_master("")
    
    print_master('destroying group')
    torch.distributed.destroy_process_group()
    print('Training session complete!')
            
def main():
    p = argparse.ArgumentParser('Vocal Remover')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--job_name', type=str, default='')
    p.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_encoders', type=int, default=0)
    p.add_argument('--num_decoders', type=int, default=4)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_bands', type=int, default=8)
    p.add_argument('--feedforward_dim', type=int, default=2048)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--vanilla', type=str, default='false')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_warmup_steps', type=int, default=4)
    p.add_argument('--lr_warmup_iter', type=int, default=0)
    p.add_argument('--validation_dataset', type=str, default='cs2048_sr44100_hl1024_nf2048_of0/')
    p.add_argument('--train_dataset', type=str, default='cs2048_sr44100_hl1024_nf2048_of0/')
    p.add_argument('--vocal_dataset', type=str, default='cs2048_sr44100_hl1024_nf2048_of0_VOCALS/')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--num_training_items', type=int, default=None)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--nodes', type=int, default=1)
    p.add_argument('--n_fft', type=int, default=2048)
    p.add_argument('--cropsize', type=int, default=256)
    p.add_argument('--mixed_precision', type=str, default='true')
    p.add_argument('--progress_bar', type=str, default='true')
    args = p.parse_args()

    args.bias = str.lower(args.bias) == 'true'
    args.vanilla = str.lower(args.vanilla) == 'true'
    args.mixed_precision = str.lower(args.mixed_precision) == 'true'
    args.progress_bar = str.lower(args.progress_bar) == 'true'
    args.world_size = args.gpus * args.nodes
    args.nr = int(os.environ['RANK'])

    mp.spawn(node_main, args=(args,), nprocs=args.gpus, join=True)

if __name__ == '__main__':
    main()