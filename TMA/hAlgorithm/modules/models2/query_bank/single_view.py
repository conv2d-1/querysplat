import os
import sys

sys.path.append(os.getcwd())

import math
import random

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.query_bank.query import BaseQuery
from hAlgorithm.utils import cuda_timing_context, instantiate_from_config


class QueryBank5(nn.Module):
    """full_ratio, support normal loss"""
    def __init__(self, train_sampler=None, with_edge_mask=False, noise=None, noise_ratio=None, offset=0.5, timing=False, full_ratio=0):
        super().__init__()

        self.grid_mem = dict()

        self.with_edge_mask = with_edge_mask
        self.offset = offset
        self.noise = noise
        self.noise_ratio = noise_ratio
        self.full_ratio = full_ratio
        self.timing = timing

        self.train_sampler = instantiate_from_config(train_sampler)

    def create_uv_grid(
        self,
        width: int,
        height: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        u = torch.arange(width, dtype=dtype, device=device)
        v = torch.arange(height, dtype=dtype, device=device)

        # Create 2D meshgrid (width x height) and stack into UV
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        uv_grid = torch.stack((uu, vv), dim=-1) + self.offset

        return uv_grid

    def get_noise(self, shape, device):
        """
        Generate random noise in range [-noise, noise] and ensure uv_grid stays within [0, 1]

        Args:
            shape: Shape of the tensor to generate noise for

        Returns:
            Noise tensor with same shape as input
        """
        if self.noise is None or self.noise <= 0:
            return torch.zeros(shape, device=device)

        # Generate uniform random noise in range [-noise, noise]
        noise_tensor = torch.empty(shape, device=device).uniform_(-self.noise, self.noise)

        return noise_tensor

    def forward(self, image, edge_mask=None, meta_data=None, **kwargs):

        with torch.no_grad():
            sub_pixel_scale = meta_data["sub_pixel_scale"]

            if sub_pixel_scale == -1:
                width = int(meta_data["origin_width"][0])
                height = int(meta_data["origin_height"][0])
            elif sub_pixel_scale == 1:
                width = int(meta_data["input_width"][0])
                height = int(meta_data["input_height"][0])
            else:
                width = int(meta_data["origin_width"][0] / sub_pixel_scale)
                height = int(meta_data["origin_height"][0] / sub_pixel_scale)

            with cuda_timing_context("uv_grid", self.timing):
                mem_key = (height, width)
                if mem_key in self.grid_mem and self.grid_mem[mem_key].device == image.device:
                    uv_grid = self.grid_mem[mem_key].clone()
                    uv_grid = uv_grid.to(image.dtype)
                else:
                    uv_grid = self.create_uv_grid(width, height, dtype=image.dtype, device=image.device)
                    self.grid_mem[mem_key] = uv_grid.clone()

            full_uv = True
            sampling = True
            if self.full_ratio is not None and self.full_ratio > 0:
                if random.random() <= self.full_ratio:
                    sampling = False

            if self.training and sampling and self.train_sampler is not None:
                with cuda_timing_context("uv_grid train_sampler", self.timing):

                    if self.train_sampler.num_samples >= width * height:
                        full_uv = True
                        sampling = False
                    else:
                        if self.with_edge_mask and edge_mask is not None:
                            edge_mask = edge_mask.view(-1, *edge_mask.shape[-2:])
                            edge_mask = edge_mask.sum(0) >= 1

                            edge_uv_grid = uv_grid[edge_mask]
                            if edge_uv_grid.shape[0] >= self.train_sampler.num_samples:
                                uv_grid = self.train_sampler(edge_uv_grid)
                            else:
                                other_uv_grid = uv_grid[~edge_mask]
                                other_uv_grid = self.train_sampler(other_uv_grid, num_samples=self.train_sampler.num_samples - edge_uv_grid.shape[0])
                                uv_grid = torch.cat([edge_uv_grid, other_uv_grid], dim=0)
                        else:
                            uv_grid = self.train_sampler(uv_grid.reshape(-1, 2))
                        
                        full_uv = False

            if self.training and sampling and self.noise is not None:
                if self.noise_ratio is None or random.random() <= self.noise_ratio:
                    uv_grid += self.get_noise(uv_grid.shape, device=uv_grid.device)
                    full_uv = False

            uv_grid[..., 0] /= (width - 1)
            uv_grid[..., 1] /= (height - 1)
            uv_grid = torch.clamp(uv_grid, min=0.0, max=1.0)

            with cuda_timing_context("query", self.timing):
                query = BaseQuery(uv=uv_grid.reshape(-1, 2), full_uv=full_uv, width=width, height=height)

        return query


if __name__ == "__main__":

    image = torch.ones([1, 3, 32, 64]).cuda()

    query_bank = QueryBank5(
        train_sampler=dict(
            type="hAlgorithm.modules.models2.query_bank.sampler.RandomSampler",
            num_samples=10,
        )
    )
    query = query_bank(image)
    print(query)
