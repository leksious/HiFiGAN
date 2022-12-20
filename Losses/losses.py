import torch
from torch import Tensor
from torch.nn import L1Loss, MSELoss
import torch.nn as nn


class FeatureLoss(L1Loss):
    def forward(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += super().forward(rl, gl)
        return loss


class LossGenerator(MSELoss):
    def forward(self, discr, _):
        loss = 0
        for batch in discr:
            loss += ((batch - 1) ** 2).mean()
        return loss


class MelLoss(L1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super().forward(input, target)
        return loss


class DiscriminatorLoss(MSELoss):
    def forward(self, reals, gens):
        loss = 0

        for real, gen in zip(reals, gens):
            loss += torch.mean((real - 1) ** 2 + gen ** 2)
        return loss