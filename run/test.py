from lightning import Trainer
from torch.utils.data import DataLoader

from MLPW.data.manifold import SwissRoll_DataModule

train_loader = SwissRoll_DataModule().get_train_loader()