import pytorch_lightning as pl
import torch
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset

def swiss2d_data(n=100000):
    x, _ = make_swiss_roll(n, noise=0.5)
    x = x[:, [0, 2]]
    return x.astype('float32')

class SwissRollDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_data = swiss2d_data(100000)
            train_tensor = torch.tensor(train_data)
            self.train_dataset = TensorDataset(train_tensor)

            val_data = swiss2d_data(1000)
            val_tensor = torch.tensor(val_data)
            self.val_dataset = TensorDataset(val_tensor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
