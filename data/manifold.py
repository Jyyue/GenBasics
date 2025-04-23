import torch
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset

def swiss2d_data(n=100000):
    x, _ = make_swiss_roll(n, noise=0.5)
    x = x[:, [0, 2]]
    return x.astype('float32')

class SwissRoll_DataModule:
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        self.mu = None
        self.sigma = None

    def get_train_loader(self):
        train_data = swiss2d_data(100000)
        train_tensor = torch.tensor(train_data)
        train_dataset = TensorDataset(train_tensor)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        val_data = swiss2d_data(1000)
        val_tensor = torch.tensor(val_data)
        val_dataset = TensorDataset(val_tensor)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
    
# TODO: support more that swiss2d ... 