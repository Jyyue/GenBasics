import lightning as pl
import torch
import torch.nn as nn
from MLPW.arc.utils.mlp import MLP

    
class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(2, 4)
        self.decoder = MLP(2, 2)
    
    def training_step(self, batch, batch_id):
        x = batch[0]
        result = self.encoder(x)
        mu, log_sigma = result[:, 0:2], result[:, 2:4]
        z = mu + torch.randn_like(mu)*log_sigma.exp()
        x_hat = self.decoder(z)
        RC = -torch.nn.functional.mse_loss(x_hat, x)
        KL = torch.mean(0.5* torch.sum(-1 - log_sigma + mu ** 2 + log_sigma.exp(), dim=1), dim=0)
        loss = -RC + KL
        self.log("train_loss", loss)
        return loss # - log likelihood


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
