import numpy as np
import torch
import torch.nn.functional as F


def nei_delta(input, pad=2):
    if not type(input) is torch.Tensor:
        input = torch.from_numpy(input.astype(np.float32))
    if len(input.shape) < 3:
        input = input[:, :, None]
    h, w, c = input.shape
    # reshape
    input = input.permute(2, 0, 1)[None]
    input = F.pad(input, pad=(pad, pad, pad, pad), mode="replicate")
    kernel = 2 * pad + 1
    input = F.unfold(input, [kernel, kernel], padding=0)
    input = input.reshape(c, -1, h, w).permute(2, 3, 0, 1).squeeze()  # hw(3)*25
    return torch.amax(input, dim=-1), torch.amin(input, dim=-1), input


def edge_filter(metric_dpt, valid_mask, times=0.1):
    _min, _max = torch.quantile(
        metric_dpt[valid_mask],
        torch.tensor([0.05, 0.95]),
    )
    _range = _max - _min
    nei_max, nei_min, _ = nei_delta(metric_dpt)
    delta = nei_max - nei_min
    edge = delta > times * _range
    edge = (
        F.max_pool2d(edge[None, None].float(), kernel_size=3, stride=1, padding=1).squeeze(0).bool()
    )
    edge = edge & valid_mask
    return edge
