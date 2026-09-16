import torch
import torch.nn as nn


class BaseQuery(nn.Module):
    def __init__(self, uv=None, xyz=None, batch_uv=None, full_uv=False, width=None, height=None):
        super().__init__()

        self.uv = uv
        self.xyz = xyz
        self.batch_uv = batch_uv

        self.full_uv = full_uv
        self.width = width
        self.height = height

    @property
    def query_nums(self):
        return self.uv.shape[-2]
