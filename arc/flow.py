import lightning as pl
import torch
import torch.nn as nn
from MLPW.arc.utils.mlp import MLP

"""

"""
class FlowOneStep(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.s1 = MLP(1, 1)
        self.t1 = MLP(1, 1)
        self.s2 = MLP(1, 1)
        self.t2 = MLP(1, 1)

    def forward(self, x):
        # 这里定义的是公式里f^-1的形式！也就是, x = f(z), z = f^-1(x)
        # x to z
        x1, x2 = x[:, 0], x[:, 1]

        # Layer 1
        s1 = self.s1(x1.unsqueeze(-1)).squeeze(-1)
        t1 = self.t1(x1.unsqueeze(-1)).squeeze(-1)
        x2 = x2 * torch.exp(s1) + t1
        log_det = s1

        # Layer 2
        s2 = self.s2(x2.unsqueeze(-1)).squeeze(-1)
        t2 = self.t2(x2.unsqueeze(-1)).squeeze(-1)
        x1 = x1 * torch.exp(s2) + t2
        log_det += s2

        x_out = torch.stack([x1, x2], dim=1)
        return x_out, log_det

    def inverse(self, x):
        # z to x
        x1, x2 = x[:, 0], x[:, 1]

        # Inverse Layer 2
        s2 = self.s2(x2.unsqueeze(-1)).squeeze(-1)
        t2 = self.t2(x2.unsqueeze(-1)).squeeze(-1)
        x1 = (x1 - t2) * torch.exp(-s2)

        # Inverse Layer 1
        s1 = self.s1(x1.unsqueeze(-1)).squeeze(-1)
        t1 = self.t1(x1.unsqueeze(-1)).squeeze(-1)
        x2 = (x2 - t1) * torch.exp(-s1)

        x_out = torch.stack([x1, x2], dim=1)
        return x_out

"""
flow model
"""
class FlowOneStep_chunk(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.s1 = MLP(1, 1)
        self.t1 = MLP(1, 1)
        self.s2 = MLP(1, 1)
        self.t2 = MLP(1, 1)


class flow(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        for i in range(3):
            self.layer.append(FlowOneStep())
    
    def training_step(self, batch, batch_id):
        x = batch[0] # 1024, 2
        sum_log_det = 0
        for layer in self.layer:
            x, log_det = layer(x)
            sum_log_det += log_det
        # final layer x has Normal distribution
        log_pz = -0.5*torch.sum(x**2, dim=1)
        #print(log_pz.shape, sum_log_det.shape)
        loss = torch.mean(- log_pz - sum_log_det) # 因为这里是从x到z的过程，所以loss是负的
        self.log("train_loss", loss)
        print(loss)
        return loss # - log p(f-1(x)) + log det J(x)
    
    def generation(self, z):
        # z to x
        for layer in reversed(self.layer):
            z = layer.inverse(z)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
