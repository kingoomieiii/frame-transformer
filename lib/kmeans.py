import torch
import torch.nn as nn
import numpy as np
from lib.dataset_kmeans import KMeansPreprocessingDataset

class Kmeans(nn.Module):
    def __init__(self, num_clusters, num_features, momentum=0.5, n_fft=2048, learning_rate=1e-4, betas=(0.9, 0.999, 1, 1), omega=0.01):
        super(Kmeans, self).__init__()

        self.num_clusters = num_clusters
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.betas = betas
        self.omega = omega
        self.centroids = None
        self.energy = torch.inf
        self.momentum = momentum
        self.max_bin = n_fft // 2
        self.centroids = torch.randn(num_clusters, num_features)
        self.m = torch.zeros(num_clusters)
        self.v = torch.zeros(num_clusters)
        self.step = 0
        
    def initialize(self, dataloader: KMeansPreprocessingDataset, device=None, num_init_samples=8):
        with torch.no_grad():
            centroids = None

            for i, B in enumerate(dataloader):
                B = B.to(device)
                B = B[:, :, :, :, :self.max_bin]
                N,T,W,C,H = B.shape
                B = B.reshape(N*T,W*C*H)

                centroids = B if centroids is None else centroids + B

                if i == num_init_samples-1:
                    break

            centroids = centroids / num_init_samples
            self.centroids = centroids.clone()

    def __call__(self, x):
        if self.centroids.device != x.device:
            self.centroids = self.centroids.to(x.device)
            self.m = self.m.to(x.device)
            self.v = self.v.to(x.device)
            self.betas = self.betas.to(x.device)

        self.step = self.step + 1
        dist = torch.sum(torch.square(torch.sub(x.unsqueeze(0), self.centroids.unsqueeze(1))), dim=2)
        nearest = torch.argmin(dist, dim=0)
        point_loss = torch.sum(dist[:, nearest]) / x.shape[0]
        centroid_loss = 1 / torch.sum(torch.square(torch.sub(self.centroids.unsqueeze(0), self.centroids.unsqueeze(1))))

        sum_delta = 0
        delta_k = 0
        for K in range(self.num_clusters):
            idx, = torch.where(nearest == K)

            if len(idx) > 0:
                delta_k += 1
                old_center = self.centroids[K]
                new_center = torch.sum(x[idx, :], dim=0) / len(idx)
                delta = torch.sum(torch.square(torch.sub(new_center, old_center)))
                sum_delta = sum_delta + delta
                m = self.betas[0] * self.m[K] + (1 - self.betas[0]) * (delta + self.betas[2] * point_loss + self.betas[3] * centroid_loss)
                v = self.betas[1] * self.v[K] + (1 - self.betas[1]) * torch.pow((delta + self.betas[2] * point_loss + self.betas[3] * centroid_loss), 2)
                mh = m / (1 - self.betas[0] ** self.step)
                vh = v / (1 - self.betas[1] ** self.step)
                next_center = old_center - (self.learning_rate * mh / (torch.sqrt(vh) + 1e-8))
                self.centroids[K] = next_center
                self.m[K] = m
                self.v[K] = v

        return sum_delta / delta_k, point_loss, centroid_loss