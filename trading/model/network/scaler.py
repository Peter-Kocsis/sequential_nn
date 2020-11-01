import torch
from torch import nn
import numpy as np

eps = 1e-4


class MaxScaler(nn.Module):
    def __init__(self, scaler_max=[1], **kwargs):
        super().__init__()
        self.device = "cpu"
        self.max = torch.tensor(scaler_max, requires_grad=False)

    def update_params(self, x):
        self.max = torch.max(x, 0)[0]

    def forward(self, x):
        return x.to(self.device) / self.max

    def inverse(self, y, index=None):
        if index is not None:
            return y.to(self.device) * self.max[index]
        return y.to(self.device) * self.max


class MinMaxScaler:
    """
    Transforms each channel to the range [0, 1].
    """

    def __init__(self):
        self.min = None
        self.max = None

    @property
    def scale(self):
        return 1.0 / (self.max - self.min)

    def fit(self, data):
        self.max = data.max(dim=1, keepdim=True)[0] + eps
        self.min = data.min(dim=1, keepdim=True)[0]

    def forward(self, input):
        self.fit(input)
        output = input.sub(self.min).mul(self.scale)
        return output

    def inverse(self, output):
        input = output.div(self.scale).add(self.min)
        return input
