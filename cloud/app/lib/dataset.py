import random
import numpy as np
import torch
import torch.utils.data

from io import BytesIO
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('bc-vocal-remover')

class VocalRemoverCloudDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vocal_dataset, num_training_items=None, force_voxaug=True, is_validation=False):
        self.num_training_items = num_training_items
        self.force_voxaug = force_voxaug
        self.is_validation = is_validation

        blobs = list(client.list_blobs(bucket, prefix=dataset))

        patch_list = []
        for blob in blobs:
            patch_list.append(blob.name)

        vocal_blobs = list(client.list_blobs(bucket, prefix=vocal_dataset))

        vocal_list = []
        for blob in vocal_blobs:
            vocal_list.append(blob.name)

        self.full_list = patch_list
        self.patch_list = patch_list
        self.vocal_list = vocal_list

        self.reset()

    def reset(self):
        if self.num_training_items is not None:
            random.shuffle(self.full_list)
            self.patch_list = self.full_list[:self.num_training_items]

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        path = self.patch_list[idx]
        blob = bucket.get_blob(path)
        blob_data = blob.download_as_bytes()         
        resource = BytesIO(blob_data)
        data = np.load(resource)

        aug = 'Y' not in data.files
        X, Xc = data['X'], data['c']
        Y = X if aug else data['Y']

        if not self.is_validation:
            if self.slide:
                start = np.random.randint(0, X.shape[2] - self.cropsize)
                stop = start + self.cropsize
                X = X[:,:,start:stop]
                Y = Y[:,:,start:stop]

            if aug and np.random.uniform() > 0.02:
                V, Vc = self._get_vocals()
                X = Y + V
                c = np.max([Xc, Vc, np.abs(X).max()])
            else:
                if np.random.uniform() < 0.25:
                    V, Vc = self._get_vocals()
                    a = np.random.beta(1, 1)
                    X = X + (V * a)
                
                c = np.max([Xc, np.abs(X).max()])

            if np.random.uniform() < 0.5:
                X = X[::-1]
                Y = Y[::-1]

            if np.random.uniform() < 0.025:
                X = Y
                c = Xc
        else:
            c = Xc

        return np.abs(X) / c, np.abs(Y) / c

    def _get_vocals(self):
        vidx = np.random.randint(len(self.vocal_list))            
        vpath = self.vocal_list[vidx]
        vblob = bucket.get_blob(vpath)
        vblob_data = vblob.download_as_bytes()
        vres = BytesIO(vblob_data)
        vdata = np.load(vres)
        V, Vc = vdata['X'], vdata['c']

        if np.random.uniform() < 0.5:
            V = V[::-1]

        if np.random.uniform() < 0.025:
            if np.random.uniform() < 0.5:
                V[0] = V[0] * 0
            else:
                V[1] = V[1] * 0

        if self.slide:
            start = np.random.randint(0, V.shape[2] - self.cropsize)
            stop = start + self.cropsize
            V = V[:,:,start:stop]

        if np.random.uniform() < 0.5:
            vidx2 = np.random.randint(len(self.vocal_list))                
            vpath2 = self.vocal_list[vidx2]
            vblob2 = bucket.get_blob(vpath2)
            vblob_data2 = vblob2.download_as_bytes()
            vres2 = BytesIO(vblob_data2)
            vdata2 = np.load(vres2)
            V2, Vc2 = vdata2['X'], vdata2['c']

            if np.random.uniform() < 0.5:
                V2 = V2[::-1]

            if np.random.uniform() < 0.025:
                if np.random.uniform() < 0.5:
                    V2[0] = V2[0] * 0
                else:
                    V2[1] = V2[1] * 0

            a = np.random.beta(1, 1)
            inv = 1 - a

            if self.slide:
                start = np.random.randint(0, V2.shape[2] - self.cropsize)
                stop = start + self.cropsize
                V2 = V2[:,:,start:stop]

            Vc = (Vc * a) + (Vc2 * inv)
            V = (V * a) + (V2 * inv)

        return V, Vc