import torch
import torch.nn as nn
import lightning as pl
import numpy as np
import torch.nn.functional as F

from MLPW.arc.utils.mlp import MLP

"""
Models
"""
class DSM(pl.LightningModule):
    """
    Denoising Score Matching with single noise scale
    """
    def __init__(self, sigma = 0.5, x_dim = 2):
        super(DSM, self).__init__()
        self.sigma = sigma
        self.net = MLP(x_dim, x_dim)
    def forward(self, x):
        return self.net(x)
    def training_step(self, batch, batch_idx):
        x = batch[0]
        eps = torch.randn_like(x)
        x_noised = x + self.sigma * eps
        score = self(x_noised)
        loss = 0.5 * F.mse_loss(self.sigma*score, -eps) # to avoid devide bu small number
        #self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        # pass
        return 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

class DSM_discreteT_1(pl.LightningModule):
    """
    Denoising Score Matching with multi-discrete scale and Simple Time Embedding
    """
    def __init__(self, noise_levels=None, sigma_min=0.1, sigma_max=10, num_scales=10, x_dim=2, t_dim=1):
        super().__init__()
        if noise_levels is None:
            noise_levels = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_scales))
        self.sigmas = torch.tensor(noise_levels, dtype=torch.float32)
        self.model = MLP(x_dim+t_dim, x_dim)

    def forward(self, x, t_index):
        return self.model(torch.cat((x, t_index.float()),dim=-1))
    def training_step(self, batch, batch_idx):
        x = batch[0]
        B = x.shape[0]
        device = x.device

        t_index = torch.randint(0, len(self.sigmas), (B, 1), device=device) 
        sigma = self.sigmas.to(device)[t_index.squeeze()].view(-1, 1)

        eps = torch.randn_like(x)
        x_noised = x + sigma * eps
        score = self(x_noised, t_index)
        loss = 0.5 * F.mse_loss(score * sigma, -eps)

        #self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class TimeEmbeddedMLP(nn.Module):
    def __init__(self, x_dim, t_dim, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, t_dim)
        )
        self.net = MLP(x_dim + t_dim, x_dim)

    def forward(self, x, t_index):
        # t_index: (B, 1)
        t_emb = self.embedding(t_index.float())  # [B, t_dim]
        x = torch.cat([x, t_emb], dim=-1)
        return self.net(x)

class DSM_discreteT_2(pl.LightningModule):
    def __init__(self, noise_levels=None, sigma_min=0.1, sigma_max=10, num_scales=10, x_dim=2, t_dim=16):
        super().__init__()
        if noise_levels is None:
            noise_levels = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_scales))
        self.sigmas = torch.tensor(noise_levels, dtype=torch.float32)
        self.model = TimeEmbeddedMLP(x_dim, t_dim)

    def forward(self, x, t_index):
        return self.model(x, t_index)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        B = x.shape[0]
        device = x.device

        t_index = torch.randint(0, len(self.sigmas), (B, 1), device=device) 
        sigma = self.sigmas.to(device)[t_index.squeeze()].view(-1, 1)

        eps = torch.randn_like(x)
        x_noised = x + sigma * eps
        score = self(x_noised, t_index)
        loss = 0.5 * F.mse_loss(score * sigma, -eps)

        #self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class DSM_discreteT_2_broken_loss(pl.LightningModule):
    def __init__(self, noise_levels=None, sigma_min=0.1, sigma_max=10, num_scales=10, x_dim=2, t_dim=16):
        super().__init__()
        if noise_levels is None:
            noise_levels = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_scales))
        self.sigmas = torch.tensor(noise_levels, dtype=torch.float32)
        self.model = TimeEmbeddedMLP(x_dim, t_dim)

    def forward(self, x, t_index):
        return self.model(x, t_index)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        B = x.shape[0]
        device = x.device

        t_index = torch.randint(0, len(self.sigmas), (B, 1), device=device) 
        sigma = self.sigmas.to(device)[t_index.squeeze()].view(-1, 1)

        eps = torch.randn_like(x)
        x_noised = x + sigma * eps
        score = self(x_noised, t_index)
        loss = 0.5 * F.mse_loss(score, -eps/sigma)

        #self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class DSM_continous_2(pl.LightningModule):
    """
    Denoising Score Matching with multi-continous scale and Complex Time Embedding
    """
    def __init__(self, noise_levels=None, sigma_min=0.1, sigma_max=10, x_dim=2, t_dim=16):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.model = TimeEmbeddedMLP(x_dim, t_dim)

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        B = x.shape[0]
        device = x.device

        t = torch.rand((B, 1), device=device) 
        sigma = torch.exp(t * (self.sigma_max - self.sigma_min) + self.sigma_min).to(device)

        eps = torch.randn_like(x)
        x_noised = x + sigma * eps
        score = self(x_noised, t)
        loss = 0.5 * F.mse_loss(score * sigma, -eps)

        #self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

"""
Samplers
"""

@torch.no_grad()
def langevin_sampling(model, n_samples = 400, T=1000, eta=0.1, device='cpu'):
    x = (torch.rand(n_samples, 2, device=device) - 0.5) * 30
    for _ in range(T):
        noise = torch.randn_like(x)
        grad = model(x) # minus grad (score)
        x = x + eta * grad + torch.sqrt(torch.tensor(2 * eta)) * noise
    return x

@torch.no_grad()
def annealed_langevin_sampling(model, n_samples=400, T=100, eta=0.1, device='cpu'):
    sigmas = model.sigmas.to(device)
    x = (torch.rand(n_samples, 2, device=device) - 0.5) * 30

    for t in reversed(range(len(sigmas))):
        sigma = sigmas[t].item()
        for _ in range(T):
            noise = torch.randn_like(x)
            t_index = torch.full((x.size(0), 1), t, dtype=torch.long, device=device)
            grad = model(x, t_index)
            x = x + eta* grad + torch.sqrt(torch.tensor(2 * eta)) * noise
    return x

@torch.no_grad()
def annealed_langevin_sampling_sigma(model, n_samples=400, T=100, eta=0.1, device='cpu'):
    sigmas = model.sigmas.to(device)
    x = (torch.rand(n_samples, 2, device=device) - 0.5) * 30

    for t in reversed(range(len(sigmas))):
        sigma = sigmas[t].item()
        for _ in range(T):
            noise = torch.randn_like(x)
            t_index = torch.full((x.size(0), 1), t, dtype=torch.long, device=device)
            grad = model(x, t_index)
            step_size = eta * (sigma ** 2)
            x = x + step_size * grad + torch.sqrt(torch.tensor(2.0 * step_size)) * noise
    return x