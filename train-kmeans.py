import argparse
from datetime import datetime
import json
import logging
import math
import os
import random

import numpy as np
import torch
import torch.utils.data

from tqdm import tqdm
from lib.dataset_kmeans import KMeansPreprocessingDataset

import torch.nn.functional as F
from lib.kmeans import Kmeans

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=str, default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_clusters', type=int, default=128)
    p.add_argument('--num_init_samples', type=int, default=32)
    p.add_argument('--n_fft', type=int, default=2048)
    p.add_argument('--batchsize', '-B', type=int, default=10)
    p.add_argument('--epoch', '-E', type=int, default=30)
    p.add_argument('--cropsize', '-C', type=int, default=512)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--model_dir', type=str, default='G://')
    p.add_argument('--token_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--prefetch_factor', type=int, default=2)
    args = p.parse_args()

    logger.info(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

    init_dataset = KMeansPreprocessingDataset(
        path="C://cs2048_sr44100_hl1024_nf2048_of0",
        extra_path="D://cs2048_sr44100_hl1024_nf2048_of0",
        mix_path=[
            "D://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "C://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "F://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "H://cs2048_sr44100_hl1024_nf2048_of0_MIXES"],
        cropsize=args.token_size,
        token_size=args.token_size
    )

    init_dataloader = torch.utils.data.DataLoader(
        dataset=init_dataset,
        batch_size=args.num_clusters,
        shuffle=True,
        num_workers=12,
        prefetch_factor=2
    )

    max_bin = args.n_fft // 2
    num_clusters = args.num_clusters

    device = torch.device('cuda')
    kmeans = Kmeans(num_clusters, num_features=2*args.token_size*max_bin).to(device)
    kmeans.initialize(init_dataloader, device, num_init_samples=args.num_init_samples)

    del init_dataloader
    del init_dataset

    train_dataset = KMeansPreprocessingDataset(
        path="C://cs2048_sr44100_hl1024_nf2048_of0",
        extra_path="D://cs2048_sr44100_hl1024_nf2048_of0",
        mix_path=[
            "D://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "C://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "F://cs2048_sr44100_hl1024_nf2048_of0_MIXES",
            "H://cs2048_sr44100_hl1024_nf2048_of0_MIXES"],
        cropsize=args.cropsize,
        token_size=args.token_size
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=16
    )

    for epoch in range(args.epoch):
        sum_point_loss = 0
        sum_delta_loss = 0

        i = 0
        pb = tqdm(train_dataloader)
        for X in pb:
            i += 1
            with torch.no_grad():
                X = X.to(device)
                X = X[:, :, :, :, :max_bin]
                N,T,W,C,H = X.shape
                X = X.reshape(N*T,W*C*H)

                with torch.cuda.amp.autocast_mode.autocast():
                    center_delta, point_loss = kmeans(X)
                    sum_point_loss += point_loss
                    sum_delta_loss += center_delta

            pb.set_description_str(f'delta={center_delta.item()}, point={point_loss.item()}')

        cluster_path = f'{args.model_dir}models/model_iter{epoch}.cluster.pth'
        torch.save(kmeans.state_dict(), cluster_path)
        logger.info('# epoch {} complete; point loss = {}'.format(epoch, sum_point_loss / len(train_dataloader)))

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)