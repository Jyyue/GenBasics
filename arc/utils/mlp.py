import torch

class MLP(torch.nn.Module):
    '''
    (batch_size, din) -> (batch_size, dout)
    '''
    def __init__(self, din, dout, nh=164, depth=3):
        super().__init__()
        self.din = din
        self.dout = dout
        self.nh = nh
        self.depth = depth
        self.layers = []
        layers = [torch.nn.Linear(din, nh), torch.nn.ReLU()]
        for i in range(depth-2):
            layers += [torch.nn.Linear(nh, nh), torch.nn.ReLU()]
        layers += [torch.nn.Linear(nh, dout)]
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)