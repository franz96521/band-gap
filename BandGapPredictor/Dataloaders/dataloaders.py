from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import pytorch_lightning as pl



class LoadMolecules(Dataset):
    def __init__(self, molecules_root, band_root, max_samples=100):
        self.root = molecules_root
        self.files = list(os.listdir(molecules_root))
        if max_samples:
            self.files = self.files[:min(max_samples, len(self.files))]
        self.bandgap = np.loadtxt(
            f'{band_root}/bandgaps.csv', dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(f'{self.root}/{self.files[idx]}').astype(np.float32)
        data = np.transpose(data, (3, 0, 1, 2))

        return data, self.bandgap[idx]


class InMemoryDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size, num_workers=0):
        super().__init__()
        self.data = data
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_size =int( len(data)*0.7)
        self.val_size = int(len(data)*.2)
        self.test_size = len(data) - self.train_size - self.val_size
        self.train_data, self.val_data, self.test_data = None, None, None

    def setup(self, stage=None):
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
            self.data, [self.train_size, self.val_size,   self.test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False,  num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers )