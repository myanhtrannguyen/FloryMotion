import torch
from .u2net import U2NETP

class U2NetSmall(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = U2NETP(3, 1)

    def forward(self, x):
        outputs = self.model(x)
        return outputs