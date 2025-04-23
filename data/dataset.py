from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# think about batch size of validation set as a memory trick

class DataModule:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def get_train_loader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        val_data = datasets.MNIST('.', train=False, download=True, transform=transform)
        return DataLoader(val_data, batch_size=self.batch_size)
