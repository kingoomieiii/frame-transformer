import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from lib.dataset_kmeans import KMeansPreprocessingDataset

class KmeansClustering(nn.Module):
    def __init__(self, num_clusters, num_features, momentum=0.5, n_fft=2048):
        super(KmeansClustering, self).__init__()

        self.num_clusters = num_clusters
        self.num_features = num_features
        self.centroids = None
        self.energy = torch.inf
        self.momentum = momentum
        self.max_bin = n_fft // 2
        self.centroids = torch.randn(num_clusters, num_features)
        
    def initialize(self, dataloader: KMeansPreprocessingDataset, device=None, num_init_samples=8):
        with torch.no_grad():
            centroids = None

            for i, B in enumerate(tqdm(dataloader)):
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
        
        with torch.no_grad():
            sum_delta = 0
            delta_k = 0
            dist = torch.sum(torch.square(torch.sub(x.unsqueeze(0), self.centroids.unsqueeze(1))), dim=2)
            nearest = torch.argmin(dist, dim=0)
            avg_dist = torch.sum(dist[:, nearest]) / x.shape[0]

            for K in range(self.num_clusters):
                idx, = torch.where(nearest == K)

                if len(idx) > 0:
                    delta_k += 1
                    old_center = self.centroids[K]
                    new_center = torch.sum(x[idx, :], dim=0) / len(idx)
                    delta = torch.sum(torch.square(torch.sub(old_center, new_center)))
                    sum_delta = sum_delta + delta
                    self.centroids[K] = new_center

            return sum_delta / delta_k, avg_dist