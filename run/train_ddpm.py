import torch
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from ..data.utils import save_training_plot, savefig
from ..data.manifold import SwissRoll_DataModule

sys.path.append(os.path.abspath(os.path.dirname("/Users/jiayi/Developer/MLPW/")))

def save_multi_scatter_2d(data: np.ndarray) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            axs[i, j].scatter(data[i * 3 + j, :, 0], data[i * 3 + j, :, 1])
    plt.title("Q1 Samples")
    plt.savefig("results/q1_samples.png")

def alpha(t):
    return torch.cos(torch.pi/2 * t)

def sigma(t):
    return torch.sin(torch.pi/2 * t)

def yita(t, tm1):
  return sigma(tm1)/sigma(t)/np.sqrt(1-alpha(t)**2/alpha(tm1)**2)


def is_power_2(i):
    return (i > 0) and (i & (i - 1)) == 0
'''
def is_power_2(i):
    for k in range(10):
      if i == 1:
        return True
      else:
        res = i%2
        if res == 0:
          i = i/2
        else:
          return False
'''
def lr_lambda(step):
    warmup_steps = 100
    if step < warmup_steps:
        return float(step) / warmup_steps  # 线性warmup
    else:
        # 余弦衰减
        cos_decayed = 0.5 * (1 + np.cos((step - warmup_steps) / (9700 - warmup_steps) * 3.141592653589793))
        return cos_decayed

def noiser(data):
    ''' add noise to data with shape [batch_size, dim]'''
    batch_size = data.shape[0]
    t = torch.rand((batch_size, 1)) # (batchsize, 1)
    noise = torch.randn_like(data)  # (batch_size, dim) * (batchsize, 1) -> (batch_size, dim)
    noised_data = alpha(t) * data + sigma(t) * noise # (batch_size, dim)
    return noised_data, noise, t

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        #self.fc1 = nn.Linear(input_size, 64)
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64)
        #self.fc4 = nn.Linear(64, 64)
        #self.fc5 = nn.Linear(64, output_size)
        #self.relu = nn.ReLU()
        self.nh = 64
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(3, self.nh),
            torch.nn.ReLU(), 
            torch.nn.Linear(self.nh, self.nh),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nh, self.nh),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nh, self.nh),
            torch.nn.ReLU(),
            torch.nn.Linear(self.nh, 2))

    def forward(self, x):
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.relu(self.fc3(x))
        #x = self.relu(self.fc4(x))
        #x = self.fc5(x)  # 输出层不使用激活函数（根据具体任务可以选择）
        x = self.layer(x)
        return x


def compute_statistics(data_loader):
    all_data = []
    for batch in data_loader:
        all_data.append(batch[0])  # 获取数据部分
    all_data = torch.cat(all_data)  # 合并所有数据
    mu = all_data.mean(dim=0)
    sigma = all_data.std(dim=0)
    return mu, sigma


def train_model(model, data_module, epochs=100, initial_lr = 1e-3):
    train_loader = data_module.get_train_loader()
    val_loader = data_module.get_val_loader()

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    #scheduler = LambdaLR(optimizer, lr_lambda)

    train_losses = []
    test_losses = []

    mu, gamma = compute_statistics(train_loader)

    for epoch in range(epochs + 1):
      model.train()
      for train_batch in train_loader:
        # nronalize data
        train_batch = (train_batch[0] - mu) / gamma
        x_noised, noise, t = noiser(train_batch)
        noise_pred = model(torch.cat((x_noised, t), dim = 1))
        train_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        #if epoch == 30:
        #    print("-----------------")
        #    print("train: ", train_batch[-1])
        #    print("t: ", t[-1])
        ##    print("x_noised: ", x_noised[-1])
        #    print("input: ", torch.cat((x_noised, t), dim = 1)[-1])
        #    print("noise: ", noise[-1])
        #    print("noise_pred: ", noise_pred[-1])
        #train_loss = torch.mean((noise_pred - noise) ** 2) # mse per dim in full bath_size, 2

        # save train loss
        #train_losses.append(train_loss.detach().numpy()) # loss at epoch i, when i = 0, model is init model

        # update model
        #optimizer.zero_grad()
        #train_loss.backward()
        #optimizer.step()
        #scheduler.step()
        #torch.cuda.empty_cache()
        #gc.collect()
      #test_loss = 0
      model.eval()
      with torch.no_grad():
        for val_batch in val_loader:
          val_batch = (val_batch[0] - mu) / gamma
          x_noised, noise, t = noiser(val_batch)
          noise_pred = model(torch.cat((x_noised, t), dim = 1))
          test_loss += torch.mean((noise_pred - noise) ** 2)
        test_losses.append(test_loss.detach().numpy())
      #gc.collect()
      print("epoch ", epoch," finished, test loss ", test_loss)
    num_steps = 512
    ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1) # from noise(t~1) to initial image(t~0)
    x_t = torch.randn(2000, 2) # start from time t, recond result from 1 st step of diffusion (that is time )
    all_samples = []
    for i in range(num_steps):
      t = torch.full((2000, 1), ts[i])
      tm1 = torch.full((2000, 1), ts[i + 1])
      eps_hat = model(torch.cat((x_t, t),dim=1))
      alpha_t = alpha(t)**2/alpha(tm1)**2
      beta_t = 1-alpha_t
      x_t = 1/torch.sqrt(alpha_t) *(x_t - (1-alpha_t)/sigma(t)* eps_hat) + torch.sqrt(beta_t) * torch.randn((2000, 2))
      #x_t = alpha(tm1)*((x_t-sigma(t)*eps_hat)/alpha(t)) + torch.sqrt(torch.maximum(sigma(tm1)**2 - yita(t, tm1)**2, torch.tensor([0])))* eps_hat + yita(t, tm1) * torch.randn((2000, 2))
      # normalize back
      if is_power_2(i+1):
        all_samples.append(x_t * gamma + mu)
    return  train_losses, test_losses, torch.stack(all_samples).detach().numpy()

    #def sampeling(model):
    #diffusion_step = np.power(2, np.linspace(0, 9, 9)).astype(int) # np.array([1, 2, 4, ..., 256])
    num_steps = 512
    ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1) # from noise(t~1) to initial image(t~0)
    x_t = torch.randn(2000, 2) # start from time t, recond result from 1 st step of diffusion (that is time )
    all_samples = []
    for i in range(num_steps):
      t = torch.full((2000, 1), ts[i])
      tm1 = torch.full((2000, 1), ts[i + 1])
      eps_hat = model(torch.cat((x_t, t),dim=1))
      x_t = alpha(tm1)*((x_t-sigma(t)*eps_hat)/alpha(t)) + torch.sqrt(torch.maximum(sigma(tm1)**2 - yita(t, tm1)**2, torch.tensor([0])))* eps_hat + yita(t, tm1) * torch.randn((2000, 2))
      # normalize back
      if is_power_2(i+1):
        all_samples.append(x_t * gamma + mu)

if __name__ == "__main__":
    data_module = SwissRoll_DataModule(batch_size=1024)
    model = MLP(3, 2)
    train_losses, test_losses, samples  = train_model(model, data_module, epochs=100)
    #samples = sampeling(model)
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q1 Train Plot",
        f"results/q1_train_plot.png"
    )
    save_multi_scatter_2d(samples)
