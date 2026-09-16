import torch
import torch.nn as nn


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class InverseLog(nn.Module):
    def __init__(self):
        super(InverseLog, self).__init__()

    def forward(self, x):
        return torch.sign(x) * (torch.expm1(torch.abs(x)))


class MogeActivation(nn.Module):
    def __init__(self):
        super(MogeActivation, self).__init__()

    def forward(self, ret):
        assert ret.shape[-3] == 3
        xy, z = ret.split([2, 1], dim=-3)
        z = torch.exp(z)
        points = torch.cat([xy * z, z], dim=-3)
        return points

class PointmapExp(nn.Module):
    def __init__(self):
        super(PointmapExp, self).__init__()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xy, z = points.split([2, 1], dim=1)
        z = torch.exp(z)
        points = torch.cat([xy * z, z], dim=1)
        return points
