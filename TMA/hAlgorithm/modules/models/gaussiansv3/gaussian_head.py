from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from pytorch3d import ops
from simple_knn._C import distCUDA2
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn.pool import fps
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.voxel_grid import voxel_grid
from torch_geometric.utils import scatter

from hAlgorithm.modules.models.gaussiansv3.gaussian_adapter import GaussianAdapter
from hAlgorithm.modules.utils.gaussians.projection import get_world_rays, sample_image_grid
from hAlgorithm.modules.utils.normal import get_surface_normalv2


def scatter_weighted_mean(
    features: Tensor,
    weights: Tensor,
    cluster: Tensor,
    weights_cluster: Tensor,
    dim: int,
) -> Tensor:
    """_summary_

    Args:
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        cluster (LongTensor): [N] IDs of each point (clusters.max() should be <= N, or you'll OOM)
        weights_cluster (Tensor): [N,] aggregated weights of each cluster, used to normalize
        dim (int): Dimension along which to do the reduction -- should be 0

    Returns:
        Tensor: Agggregated features, weighted by weights and normalized by weights_cluster
    """
    assert dim == 0, "Dim != 0 not yet implemented"
    feature_cluster = scatter(features * weights[:, None], cluster, dim=dim, reduce="sum")
    feature_cluster = feature_cluster / weights_cluster[:, None]
    return feature_cluster


def voxelize(
    pos: Tensor,
    voxel_size: float,
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, Tensor]] = None,
    end: Optional[Union[float, Tensor]] = None,
) -> Tuple[Tensor]:
    """Returns voxel indices and packed (consecutive) indices for points

    Args:
        pos (Tensor): [N, 3] locations
        voxel_size (float): Size (resolution) of each voxel in the grid
        batch (Optional[Tensor], optional): Batch index of each point in pos. Defaults to None.
        start (Optional[Union[float, Tensor]], optional): Mins along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from min values in pos.
        end (Optional[Union[float, Tensor]], optional):  Maxes along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from max values in pos.
    Returns:
        voxel_idx (LongTensor): Idx of each point's voxel coordinate. E.g. [0, 0, 4, 3, 3, 4]
        cluster_consecutive_idx (LongTensor): Packed idx -- contiguous in cluster ID. E.g. [0, 0, 2, 1, 1, 2]
        batch_sample: See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html
    """
    voxel_cluster = voxel_grid(pos=pos, batch=batch, size=voxel_size, start=start, end=end)
    cluster_consecutive_idx, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm] if batch is not None else None
    cluster_idx = voxel_cluster
    return cluster_idx, cluster_consecutive_idx, batch_sample


def reduce_pointcloud(
    voxel_cluster: Tensor,
    pos: Tensor,
    features: Tensor,
    weights: Optional[Tensor] = None,
    feature_reduce: str = "mean",
) -> Tuple[Tensor]:
    """Pools values within each voxel

    Args:
        voxel_cluster (LongTensor): [N] IDs of each point
        pos (Tensor): [N, 3] position of each point
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        rgbs (Optional[Tensor], optional): [N, 3] colors of each point. Defaults to None.
        feature_reduce (str, optional): Feature reduction method. Defaults to 'mean'.

    Raises:
        NotImplementedError: if unknown reduction method

    Returns:
        pos_cluster (Tensor): weighted average position within each voxel
        feature_cluster (Tensor): aggregated feature of each voxel
        weights_cluster (Tensor): aggregated weights of each voxel
        rgb_cluster (Tensor): colors of each voxel
    """
    if weights is None:
        weights = torch.ones_like(pos[..., 0])
    weights_cluster = scatter(weights, voxel_cluster, dim=0, reduce="sum")

    pos_cluster = scatter_weighted_mean(pos, weights, voxel_cluster, weights_cluster, dim=0)

    if feature_reduce == "mean":
        feature_cluster = scatter_weighted_mean(
            features, weights, voxel_cluster, weights_cluster, dim=0
        )
    elif feature_reduce == "max":
        feature_cluster = scatter(features, voxel_cluster, dim=0, reduce="max")
    elif feature_reduce == "sum":
        feature_cluster = scatter(features * weights[:, None], voxel_cluster, dim=0, reduce="sum")
    else:
        raise NotImplementedError(f"Unknown feature reduction method {feature_reduce}")

    return pos_cluster, feature_cluster


def farthest_point_sampling(xyz, npoints):
    """
    Inputs:
    ----------
    xyz     : torch.Tensor
              (B, N, 3) tensor where N > npoints
    npoints : int32
              number of features in the sampled set

    Outputs:
    -------
    out     : torch.Tensor
              (B, npoints, 3) tensor containing the set
    """

    input_device = xyz.device
    device = input_device
    # xyz = xyz.to(device=device)

    _xyz = xyz.view(-1, xyz.shape[-1])
    batch = torch.kron(torch.arange(start=0, end=xyz.size(0)), torch.ones(xyz.size(1)))
    batch = batch.long().to(device=device)
    ratio = npoints / xyz.size(-2)
    index = fps(_xyz, batch, ratio=ratio, random_start=True)
    _xyz_out = _xyz[index]
    out = torch.reshape(_xyz_out, (xyz.size(0), -1, 3))
    out = out.to(input_device)
    return out


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvBNReLU1D, self).__init__()
        self.act = nn.ReLU(True)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            # self.act
        )

    def forward(self, x):
        return self.net(x)


class PointFeatureAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, fps_num=20_000, knn_k=32, voxel_size=0.1):
        super().__init__()
        self.fps_num = fps_num
        self.k = knn_k
        self.out_channels = out_channels
        self.k_offset = 1
        self.voxel_size = voxel_size

        self.embedding = ConvBNReLU1D(3, 32, bias=True)
        # in_channels += 32  # xyz embedding
        # in_channels += 1
        self.offset_mlp = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(
                in_channels=in_channels, out_channels=3 * self.k_offset, kernel_size=1, bias=True
            ),
        )
        # self.mlp = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=True),
        #     nn.GELU(),
        #     nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=True),
        #     # nn.BatchNorm1d(in_channels),
        #     nn.GELU(),
        #     nn.Conv1d(in_channels, out_channels*2, kernel_size=1, bias=True),
        #     # nn.BatchNorm1d(in_channels),
        #     nn.GELU(),
        #     nn.Conv1d(out_channels*2, out_channels*self.k_offset, kernel_size=1, bias=True),
        # )
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward_offset(self, xyz, feat, grad):
        """
        xyz: [b, n, 3]
        feat: [b, c, n]
        grad: [b, c, n]
        """
        breakpoint()
        b = feat.shape[0]
        device = xyz.device

        # sample xyz with grad
        high_freq_xyz = []
        high_freq_feat = []
        for i in range(b):
            idx = torch.multinomial(grad, 20000)
            high_freq_xyz.append(xyz[i][idx, :].unsqueeze(0))
            high_freq_feat.append(feat[i][:, idx].unsqueeze(0))

        high_freq_xyz = torch.cat(high_freq_xyz, dim=0)  # [b, 20000, 3]
        high_freq_feat = torch.cat(high_freq_feat, dim=0)  # [b, c, 20000]

        breakpoint()

        _xyz = xyz.view(-1, xyz.shape[-1])
        _feat = feat.transpose(2, 1).reshape(-1, feat.shape[1])
        batch = torch.kron(torch.arange(start=0, end=xyz.size(0)), torch.ones(xyz.size(1)))
        batch = batch.long().to(device=device)
        # https://github.com/ok-robot/ok-robot/blob/174c742b6a1866a2da8f75a89e2c544bb77ab5f9/ok-robot-navigation/voxel_map/voxel.py#L168
        cluster_voxel_idx, cluster_consecutive_idx, _ = voxelize(_xyz, voxel_size=0.01, batch=batch)
        xyz, feat = reduce_pointcloud(
            cluster_consecutive_idx, pos=_xyz, features=_feat, feature_reduce="max"
        )

        feat = feat.transpose(0, 1)
        feat = feat.reshape(b, *feat.shape)  # [b. c, n]

        feat = torch.cat([feat, high_freq_feat], dim=2)

        rgb = feat[:, -3:, :]
        rgb = (rgb + 1) * 0.5  # [b, 3, n]

        xyz = xyz.reshape(b, *xyz.shape)  # [b, n, 3]
        xyz = torch.cat([xyz, high_freq_xyz], dim=1)

        xyz_feat = self.embedding(xyz.transpose(2, 1))  # [b, c, n]
        offset = self.offset_mlp(torch.cat([feat, xyz_feat], dim=1))  # [b, 15, n]
        feat = self.mlp(torch.cat([feat, xyz_feat], dim=1))

        # xyz = xyz.unsqueeze(2) + offset.reshape(b, 5, 3, -1) * 0.1
        xyz = xyz.unsqueeze(2) + offset.transpose(2, 1).reshape(b, -1, self.k_offset, 3) * 0.2
        rgb = rgb.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k_offset, 1)

        xyz = xyz.reshape(b, -1, 3)
        rgb = rgb.reshape(b, -1, 3).transpose(2, 1)

        feat = feat.transpose(2, 1).reshape(b, -1, self.k_offset, self.out_channels)
        feat = feat.reshape(b, -1, self.out_channels).transpose(2, 1)

        return xyz, feat, rgb

    def xyz_normalize(self, xyz):
        xyz_mean = xyz.mean(dim=1, keepdim=True)
        xyz_norm = torch.quantile(xyz.norm(dim=2), 0.9, dim=1) * 0.5
        xyz = (xyz - xyz_mean) / xyz_norm
        return xyz, xyz_mean, xyz_norm

    def forward(self, xyz, feat):
        """
        xyz: [b, n, 3]
        feat: [b, c, n]
        """
        # return self.forward_offset(xyz, feat)
        # b, n = feat.shape[0], feat.shape[2]
        # device = xyz.device

        b = feat.shape[0]
        device = xyz.device

        xyz, xyz_mean, xyz_norm = self.xyz_normalize(xyz)

        _xyz = xyz.view(-1, xyz.shape[-1])
        _feat = feat.transpose(2, 1).reshape(-1, feat.shape[1])
        batch = torch.kron(torch.arange(start=0, end=xyz.size(0)), torch.ones(xyz.size(1)))
        batch = batch.long().to(device=device)
        # # compute the voxel_size based on the norm of the xyz
        with torch.no_grad():
            cluster_voxel_idx, cluster_consecutive_idx, _ = voxelize(
                _xyz, voxel_size=self.voxel_size, batch=batch
            )
        xyz, feat = reduce_pointcloud(
            cluster_consecutive_idx, pos=_xyz, features=_feat, feature_reduce="mean"
        )

        feat = feat.transpose(0, 1)
        feat = feat.reshape(b, *feat.shape)  # [b. c, n]

        rgb = feat[:, -3:, :]
        rgb = (rgb + 1) * 0.5

        xyz = xyz.reshape(b, *xyz.shape)

        # xyz_feat = self.embedding(xyz.transpose(2, 1))  # [b, c, n]
        # feat = self.mlp(torch.cat([feat, xyz_feat], dim=1))
        feat = self.mlp(feat)

        xyz = (xyz * xyz_norm) + xyz_mean

        return xyz, feat, rgb


class GaussianHead(nn.Module):
    def __init__(self, feat_dim, downsample=False, cfg=None):
        super().__init__()

        self.gaussian_adapter = GaussianAdapter(cfg)

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        num_gaussian_parameters += 1  # predict opacity
        num_gaussian_parameters += 1  # predict mask
        num_gaussian_parameters += 3  # prpedict xyz offset
        in_channels = feat_dim + 3  # rgb input
        # self.head = nn.Sequential(
        #     nn.Conv2d(
        #       in_channels, in_channels, 1, 1, 0),
        #     nn.GELU(),
        #     nn.Conv2d(
        #         in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode='replicate'),
        #     nn.GELU(),
        #     nn.Conv2d(num_gaussian_parameters * 2,
        #                 num_gaussian_parameters, 1, 1, 0, bias=True)
        # )
        self.downsample = downsample
        self.feature_aggregator = PointFeatureAggregator(in_channels, 60, 200, voxel_size=0.005)

        self.scharr_x = nn.Parameter(
            torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .repeat(1, 3, 1, 1),
            requires_grad=False,
        )
        self.scharr_y = nn.Parameter(
            torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .repeat(1, 3, 1, 1),
            requires_grad=False,
        )

    def compute_rgb_grad(self, rgb):
        """
        rgb: [B, V, C, H, W]
        """
        b, v, _, h, w = rgb.shape
        rgb = rearrange(rgb, "b v c h w -> (b v) c h w")
        rgb_x = F.conv2d(rgb, self.scharr_x, padding=1)
        rgb_y = F.conv2d(rgb, self.scharr_y, padding=1)
        rgb_grad = torch.sqrt(rgb_x**2 + rgb_y**2 + 1e-6)
        return rearrange(rgb_grad, "(b v) c h w -> b v c h w", b=b, v=v)

    def depth_to_xyz(self, depth, intrinsics, extrinsics):
        """
        Convert depth to xyz in the world
        depth: [B, V, 1, H, W]
        intrinsics: [B, V, 3, 3]
        extrinsics: [B, V, 4, 4]  camera to world
        """
        b, v, _, h, w = depth.shape

        extrinsics = rearrange(extrinsics, "b v i j -> b v () () () i j")
        intrinsics = rearrange(intrinsics, "b v i j -> b v () () () i j")

        depth = rearrange(depth, "b v 1 h w -> b v (h w) () ()", b=b, v=v)

        xy_ray, _ = sample_image_grid((h, w), depth.device)
        xy_ray = rearrange(xy_ray, "h w xy -> () () (h w) () () xy")
        xy_ray = xy_ray.expand(b, v, -1, -1, -1, -1)

        origins, directions = get_world_rays(xy_ray, extrinsics, intrinsics, with_normalize=False)

        xyz = origins + directions * depth[..., None]  # [B, V, H*W, 1, 1, 3]

        # with torch.no_grad():
        #     normal, _ = get_surface_normalv2(rearrange(xyz, "b v (h w) () () c -> (b v) h w c", h=h, w=w))  # [bv c h w]
        #     normal = rearrange(normal, "(b v) c h w -> b v (h w) () () c", b=b, v=v)  # [B, V, H*W, 1, 1, 3]

        return xyz

    def forward(self, images, features, depth, extrinsics, intrinsics, **kwargs):
        """
        images: [B, F*V, C, H, W]
        features: [B, F*V, C, H, W]
        depth: [B, F*V, 1, H, W]
        extrinsics: [B, F*V, 4, 4]
        intrinsics: [B, F*V, 3, 3]
        """
        b, v, _, h, w = images.shape

        # grad = self.compute_rgb_grad(images)  # [B, F*V, C, H, W]
        depth = depth.detach()
        xyz = self.depth_to_xyz(depth, intrinsics, extrinsics)  # [B, V, H*W, 1, 1, 3]

        features = torch.cat([features, images], dim=2)
        features = rearrange(features, "b v c h w -> b c (v h w)")
        xyz = rearrange(xyz, "b v n 1 1 c -> b (v n) c")
        # normal = rearrange(normal, "b v n 1 1 c -> b c (v n)")
        # grad = rearrange(grad, "b v c h w -> b c (v h w)")

        # features = torch.cat([features, normal], dim=1)  # [B, C+3, V*H*W]

        xyz, gaussians, rgb = self.feature_aggregator(xyz, features)  # [b, c, n]
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy()[0])
        # pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy()[0].T)
        # o3d.io.write_point_cloud("test.ply", pcd)
        # breakpoint()

        gaussians = self.gaussian_adapter(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            xyz,
            rearrange(rgb, "b rgb n -> b n () rgb"),
            rearrange(gaussians, "b c n-> b n () c"),
            # normal=rearrange(normal, "b c n -> b n () c"),
        )

        return gaussians
