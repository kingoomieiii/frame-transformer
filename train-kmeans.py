import argparse
from datetime import datetime
import json
import logging
import math
import os
import random

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from lib.dataset_kmeans import KMeansPreprocessingDataset

import torch.nn.functional as F

from kmeans_pytorch import kmeans

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
    p.add_argument('--n_fft', type=int, default=2048)
    p.add_argument('--batchsize', '-B', type=int, default=6)
    p.add_argument('--epoch', '-E', type=int, default=30)
    p.add_argument('--cropsize', '-C', type=int, default=512)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--model_dir', type=str, default='G://')
    p.add_argument('--token_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--prefetch_factor', type=int, default=4)
    args = p.parse_args()

    logger.info(args)

    random.seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 1)

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
        prefetch_factor=args.prefetch_factor
    )

    max_bin = args.n_fft // 2
    num_clusters = 128

    first = True
    centers = []

    device = torch.device('cuda')

    for epoch in range(args.epoch):
        train_dataset.rebuild()

        for itr, X in enumerate(tqdm(train_dataloader)):
            X = X.to(device)
            X = X[:, :, :, :, :max_bin]
            N,T,W,C,H = X.shape
            X = X.reshape(N*T,W*C*H)
            B = X[torch.randperm(X.shape[0])]

            with torch.cuda.amp.autocast_mode.autocast():
                _, centers = kmeans(B, 150, cluster_centers=centers, tol=0.0, device=device, iter_limit=20 if first else 1, tqdm_flag=False)

            first = False

        logger.info('# epoch {}'.format(epoch))

        cluster_path = f'{args.model_dir}models/model_iter{epoch}.cluster.npz'
        np.savez(cluster_path, centers=centers)

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)