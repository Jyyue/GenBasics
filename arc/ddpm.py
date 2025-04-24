import torch
from torch import optim, nn
import lightning as pl
from MLPW.arc.utils.mlp import MLP
import numpy as np
from tqdm import tqdm

def noiser(x0):
    eps = torch.randn_like(x0[0])
    #eps = torch.randn_like(x0[0]) - 0.5) * 30
    t = torch.rand((x0[0].shape[0],1)).to(x0[0].device)
    # \sqrt{avg(alpha(t))} = cos(\pi/2 * t)
    xt = torch.cos(torch.pi/2 *t)*x0[0] + torch.sin(torch.pi/2 *t)*eps
    return eps, t, xt

class DDPM(pl.LightningModule):
    def __init__(self, dim_t=1, dim_x=2, noise_schedule=None):
        super().__init__()
        self.net = MLP(dim_t+dim_x, dim_x)
    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=-1))

    def training_step(self, batch, batch_idx):
        eps, t, xt = noiser(batch) # batch is x0, clean input
        eps_hat = self(xt, t)
        loss = nn.functional.mse_loss(eps, eps_hat)
        #self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
def markov_chain_sampling(model, num_steps=1024):
    """
    Markov Chain Sampling
    """
    ts = np.linspace(1 - 1e-4, 1e-4, num_steps + 1) # from noise(t~1) to initial image(t~0)
    xt = torch.randn(2000, 2) # start from time t, recond result from 1 st step of diffusion (that is time )
    all_samples = []
    def is_power_of_4(i):
        return (i > 0) and (i & (i - 1)) == 0 and (i % 3 == 1)
    for i in tqdm(range(num_steps), desc="Processing"):
        t = torch.full((2000, 1), ts[i])
        t_1 = torch.full((2000, 1), ts[i+1])
        eps_pred = model(xt, t)
        #print(xt[0], noise_pred[0])
        alpha_ave_t = torch.cos(torch.pi/2 * t)**2
        alpha_ave_t_1 = torch.cos(torch.pi/2 * t_1)**2
        alpha_t = alpha_ave_t/alpha_ave_t_1
        mu_t_pred = (xt - (1-alpha_t)/(1-alpha_ave_t).sqrt()*eps_pred)/alpha_t.sqrt()
        beta_t = 1 - (alpha_ave_t / alpha_ave_t_1)
        beta_wave = (1 - alpha_ave_t_1) / (1 - alpha_ave_t) * beta_t
        xt = mu_t_pred + beta_wave*torch.randn_like(xt)
        #if i==15 or i==63 or i==255 or i==511 or i==1023:
        #    all_samples.append(xt.detach().numpy())
    return xt#all_samples




