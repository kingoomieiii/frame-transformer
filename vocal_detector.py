import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import torch.utils.data

from lib.lr_scheduler_linear_warmup import LinearWarmupScheduler
from lib.lr_scheduler_polynomial_decay import PolynomialDecayScheduler

class ValuesDataset(torch.utils.data.Dataset):
    def __init__(self, value_arrays=[]):
        self.items = []

        for arr in value_arrays:
            for value in arr:
                self.items.append(value)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        values = self.items[idx]
        vmin = values[0]
        vmean = values[1]
        vvar = values[2]
        vmed = values[3]
        mmin = values[4]
        mmean = values[5]
        mmax = values[6]
        med = values[7]
        flagged = values[8]
        
        return np.array([vmin, vmean, vvar, vmed, mmin, mmean, mmax, med]).astype(np.float32), np.float32(flagged)

class ResBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.InstanceNorm1d(1)
        self.linear1 = nn.Linear(in_features, in_features * 4)
        self.linear2 = nn.Linear(in_features * 4, in_features)
        
    def __call__(self, x):
        h = self.norm(x)
        h = self.linear2(torch.relu(self.linear1(h)) ** 2)
        return x + self.dropout(h)

class VocalDetector(nn.Module):
    def __init__(self, in_features=8, latent_features=128, num_layers=16):
        super().__init__()

        self.in_project = nn.Linear(in_features, latent_features)        
        self.layers = nn.Sequential(*[ResBlock(latent_features) for _ in range(num_layers)])
        self.out = nn.Sequential(
            nn.InstanceNorm1d(1),
            nn.Linear(latent_features, 1))

    def __call__(self, x):
        h = self.in_project(x).unsqueeze(1)
        h = self.layers(h)
        return self.out(h).squeeze(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--data', type=str, default='v0.npz,i0.npz')
    p.add_argument('--batchsize', type=int, default=8)
    p.add_argument('--epochs', type=int, default=128)
    p.add_argument('--latent_features', type=int, default=1024)
    p.add_argument('--num_layers', type=int, default=24)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    args = p.parse_args()

    args.data = [data for data in args.data.split(',')]

    data = [np.load(d)['values'] for d in args.data]

    dataset = ValuesDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    device = torch.device('cuda')
    model = VocalDetector(in_features=8, latent_features=args.latent_features, num_layers=args.num_layers)
    model = model.to(device)
    crit = nn.BCEWithLogitsLoss()

    groups = [
        { "params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.learning_rate },
    ]

    optimizer = torch.optim.Adam(
        groups,
        lr=args.learning_rate
    )

    best_loss = float('inf')

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        LinearWarmupScheduler(optimizer, target_lr=args.learning_rate, num_steps=500, verbose_skip_steps=100),
        PolynomialDecayScheduler(optimizer, target=1e-12, power=1, num_decay_steps=2000, start_step=500, verbose_skip_steps=100)
    ])

    step = 0
    for epoch in range(100000):
        model.zero_grad()

        epoch_loss = 0

        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X).squeeze(-1)
            loss = crit(pred, Y)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            epoch_loss = epoch_loss + loss.item() * args.batchsize
            scheduler.step()
            step = step + 1

        epoch_loss = epoch_loss / len(dataset)

        if epoch_loss < best_loss:
            print(f'epoch loss: {epoch_loss} *best*')
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'voxdetector.pth')
        else:
            print(f'epoch loss: {epoch_loss}')

        if step > 10000:
            quit()

if __name__ == '__main__':
    main()