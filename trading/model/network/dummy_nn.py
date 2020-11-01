import torch
from torch import nn


class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "Dummy Network"

    def __str__(self):
        return self.model_name

    def forward(self, x):
        return range(100)
