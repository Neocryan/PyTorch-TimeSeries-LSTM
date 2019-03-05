import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class BaseDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, *input):
        pass
