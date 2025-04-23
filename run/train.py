from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ..arc.vae import SimpleNN
from ..data.manifold_lighting import SwissRollDataModule

# 数据加载
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建模型
model = SimpleNN()

# 创建训练器并训练模型
trainer = Trainer(max_epochs=5)
trainer.fit(model, train_loader)
