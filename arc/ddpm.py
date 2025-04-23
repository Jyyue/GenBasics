import torch
from torch import optim, nn
import lightning as pl
from MLPW.arc.utils.mlp import MLP

# define any number of nn.Modules (or use your current ones)

# TODO: solve CPU/GPU device coherence

def noiser(x0):
    eps = torch.randn_like(x0[0])
    t = torch.rand((x0[0].shape[0],1)).to(x0[0].device)
    xt = torch.cos(torch.pi/2 *t)*x0[0] + torch.sin(torch.pi/2 *t)*eps
    return eps, t, xt

# define the LightningModule
class DDPM(pl.LightningModule):
    def __init__(self, din=3, dout=2):
        super().__init__()
        self.decoder = MLP(din, dout)

    def training_step(self, batch, batch_idx):
        eps, t, xt = noiser(batch) # batch is x0, clean input
        eps_hat = self.decoder(torch.cat((xt, t), dim=1))
        loss = nn.functional.mse_loss(eps, eps_hat)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




