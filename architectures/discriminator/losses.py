from torch import nn
import torch
from torch.functional import Tensor


class LSDiscriminatorialLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_real: Tensor, y_fake: Tensor):
        loss = 0.5 * torch.mean((y_real - 1) ** 2) + \
               0.5 * torch.mean((y_fake + 1) ** 2)
        return loss


class LSGeneratorLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_fake: Tensor):
        loss = 0.5 * torch.mean((y_fake - 1) ** 2)
        return loss