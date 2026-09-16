import gc
import logging
import traceback
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_to_matrix
from simple_knn._C import distCUDA2
from torch import nn

from .types import RGB2SH, build_rotation


class Gaussians(nn.Module):
    def __init__(
        self,
        sh_degree=3,
        pretrain_ply=None,
        geo_ply=None,
        pcd=None,
        load_gs_scales=False,
        load_gs_rotations=False,
        load_gs_opacities=False,
        pre_filtering=True,
    ):
        super().__init__()

        self.batch_idx = 0
        self.sh_degree = sh_degree
        # TODO: delete active_sh_degree
        self.active_sh_degree = sh_degree
        self._means = torch.empty(0)
        # self._harmonics = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None

        self.grad_screenspace_points = torch.empty(0)

        # 梯度累计
        self.xyz_gradient_accum = torch.empty(0)
        # 累计次数
        self.denom = torch.empty(0)
        # 最大的高斯半径
        self.max_radii2D = torch.empty(0)
        self.percent_dense = 0.001  # 0.01

        # prune percent
        self.prune_percent = 0.5

        # self.scaling_activation =  lambda x: torch.clamp(torch.exp(x), max=0.5)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: torch.log(x / (1 - x))
        self.rotation_activation = torch.nn.functional.normalize

        self.load_gs_scales = load_gs_scales
        self.load_gs_rotations = load_gs_rotations
        self.load_gs_opacities = load_gs_opacities
        self.pre_filtering = pre_filtering

        # 解析 ply 文件 并对高斯附初值
        if pretrain_ply is not None:
            self.load_ply(pretrain_ply, load_rest=False, load_for_test=True)
            logging.info(f"Gaussians load_ply from {pretrain_ply}!!")
        elif geo_ply is not None:
            self.load_geo_ply(geo_ply, load_rest=False)
            logging.info(f"Gaussians load_geo_ply from {geo_ply}!!")
        elif pcd is not None:
            self.create_from_pcd(pcd)
            logging.info(f"Gaussians create_from_pcd from {pcd}!!")

    @property
    def batch_size(self):
        return self._means.shape[0] if self._means is not None else 0

    def construct_list_of_attributes(self, save_rest=True):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3 * 1):
            l.append("f_dc_{}".format(i))
        if save_rest:
            for i in range(3 * ((self.sh_degree + 1) ** 2 - 1)):
                l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(3):
            l.append("scale_{}".format(i))
        for i in range(4):
            l.append("rot_{}".format(i))
        return l

    def export_ply(self, path=None, save_rest=True):  # save_rest = False
        xyz = self._means[0].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # f_dc: [B, n, d_sh**2, 3] -> [n, 3, 1] GaussianV2输出是[[n, 1, 3]] GaussianV2输出是[[n, 3, 1]]
        f_dc = (
            self._features_dc[0]
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )  # 3
        # f_rest：[B, n, d_sh**2, 3] -> [n, 3, 15]
        if save_rest:
            f_rest = (
                self._features_rest[0]
                .detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )  # 45
        # import pdb;pdb.set_trace()
        opacities = self._opacity[0].detach().cpu().numpy()
        scale = self._scaling[0].detach().cpu().numpy()
        rotation = self._rotation[0].detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes(save_rest)
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if save_rest:
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
        else:
            attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)

        # import pdb; pdb.set_trace()
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        if path is None:
            return PlyData([el])

        PlyData([el]).write(path)
        logging.info(f"Saved ply to {path}")

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity(), torch.ones_like(self.get_opacity()) * 0.01)
        ).unsqueeze(0)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, load_rest=True, load_for_test=False):
        plydata = None
        if isinstance(path, str):
            plydata = PlyData.read(path)
        else:
            logging.info(f"load_ply from PlyData object")
            plydata = path
        mean = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]  # (n,1)

        features_dc = np.zeros((mean.shape[0], 1, 3))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        if load_rest:
            extra_f_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * ((self.sh_degree + 1) ** 2 - 1)
            features_extra = np.zeros((mean.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

            if load_for_test:
                features_extra = features_extra.reshape(
                    features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1
                )
                features_extra = np.transpose(features_extra, (0, 2, 1))
            else:
                features_extra = features_extra.reshape(
                    features_extra.shape[0], (self.sh_degree + 1) ** 2 - 1, 3
                )
        else:
            features_extra = np.zeros((mean.shape[0], 3 * ((self.sh_degree + 1) ** 2 - 1)))
            features_extra = features_extra.reshape(
                features_extra.shape[0], (self.sh_degree + 1) ** 2 - 1, 3
            )

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((mean.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((mean.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        device = "cuda"
        mean_torch = torch.tensor(mean[None, ...], dtype=torch.float, device=device)
        features_dc_torch = torch.tensor(features_dc[None, ...], dtype=torch.float, device=device)
        features_extra_torch = torch.tensor(
            features_extra[None, ...], dtype=torch.float, device=device
        )
        opacities_torch = torch.tensor(opacities[None, ...], dtype=torch.float, device=device)
        scales_torch = torch.tensor(scales[None, ...], dtype=torch.float, device=device)
        rots_torch = torch.tensor(rots[None, ...], dtype=torch.float, device=device)

        if self.pre_filtering:
            with torch.no_grad():
                op = self.opacity_activation(opacities_torch[0, :, 0])
                scale = self.scaling_activation(scales_torch[0])
                volume = scale.prod(dim=-1)
                importance = op * volume

                topk = int(0.90 * importance.shape[0])
                _, keep_indices = torch.topk(importance, topk, largest=True, sorted=False)

        else:
            keep_indices = torch.arange(mean.shape[0], device=device)

        # 重置初值
        if not self.load_gs_scales:
            logging.debug(f"Gaussians load_ply: not load_scale, using distCUDA2 to compute scales")
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(mean).float().cuda()), 0.0000001)
            scales_torch = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3).unsqueeze(0)

        if not self.load_gs_rotations:
            logging.debug(
                f"Gaussians load_ply: not load_rot, initializing with identity quaternion"
            )
            rots = torch.zeros((mean.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            rots_torch = rots.unsqueeze(0)

        if not self.load_gs_opacities:
            logging.debug(f"Gaussians load_ply: not load_opacity, initializing with 0.1")
            opacities_torch = self.inverse_opacity_activation(
                0.1 * torch.ones((mean.shape[0], 1), dtype=torch.float, device="cuda")
            ).unsqueeze(0)

        # === 根据 keep_indices 筛选所有属性 ===
        self._means = nn.Parameter(mean_torch[:, keep_indices, :].contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(
            features_dc_torch[:, keep_indices, :, :].contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features_extra_torch[:, keep_indices, :, :].contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            opacities_torch[:, keep_indices, :].contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            scales_torch[:, keep_indices, :].contiguous().requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            rots_torch[:, keep_indices, :].contiguous().requires_grad_(True)
        )

        # 初始化训练中的缓存变量
        N_kept = keep_indices.shape[0]
        self.xyz_gradient_accum = torch.zeros((1, N_kept, 1), device=device)
        self.denom = torch.zeros((1, N_kept, 1), device=device)
        self.max_radii2D = torch.zeros((1, N_kept), device=device)

        self.active_sh_degree = self.sh_degree

    def load_geo_ply(self, path, load_rest=True):
        plydata = PlyData.read(path)
        mean = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        features_dc = np.zeros((mean.shape[0], 1, 3))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        if load_rest:
            extra_f_names = [
                p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * ((self.sh_degree + 1) ** 2 - 1)
            features_extra = np.zeros((mean.shape[0], 3 * ((self.sh_degree + 1) ** 2 - 1)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            features_extra = np.zeros((mean.shape[0], 3 * ((self.sh_degree + 1) ** 2 - 1)))
        features_extra = features_extra.reshape(
            features_extra.shape[0], (self.sh_degree + 1) ** 2 - 1, 3
        )

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(mean).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3).unsqueeze(0)

        rots = torch.zeros((mean.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        rots = rots.unsqueeze(0)

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((mean.shape[0], 1), dtype=torch.float, device="cuda")
        ).unsqueeze(0)

        self._means = nn.Parameter(
            torch.tensor(
                np.expand_dims(mean, axis=0), dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(np.expand_dims(features_dc, axis=0), dtype=torch.float, device="cuda")
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(np.expand_dims(features_extra, axis=0), dtype=torch.float, device="cuda")
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        self.xyz_gradient_accum = torch.zeros((1, self.get_xyz().shape[0], 1), device="cuda")
        self.denom = torch.zeros((1, self.get_xyz().shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((1, self.get_xyz().shape[0]), device="cuda")

    def create_from_pcd(self, ply_path):
        """
        从点云创建高斯模型
        Args:
            ply_path: 点云文件路径
        """
        # 读取点云文件
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            if len(pcd.points) == 0:
                raise ValueError(f"点云文件 {ply_path} 中没有点")

            # 如果点云没有颜色，添加默认颜色
            if not pcd.has_colors():
                default_color = np.ones((len(pcd.points), 3)) * 0.5  # 默认灰色
                pcd.colors = o3d.utility.Vector3dVector(default_color)

            # 转换点云数据到CUDA tensor
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            # 转换为CUDA tensor
            fused_point_cloud = torch.tensor(points).float().cuda()
            fused_color = RGB2SH(torch.tensor(colors).float().cuda())

            # 初始化特征
            features_dc = torch.zeros((fused_point_cloud.shape[0], 1, 3), device="cuda")
            features_dc[:, 0, :] = fused_color

            # 初始化额外特征
            features_extra = torch.zeros(
                (fused_point_cloud.shape[0], (self.sh_degree + 1) ** 2 - 1, 3), device="cuda"
            )

            # 计算点之间的距离用于初始化缩放
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3).unsqueeze(0)

            # 初始化旋转（默认为单位四元数）
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1  # 实部为1，虚部为0的单位四元数
            rots = rots.unsqueeze(0)

            # 初始化不透明度
            opacities = self.inverse_opacity_activation(
                0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            ).unsqueeze(0)

            # 设置模型参数
            self._means = nn.Parameter(fused_point_cloud.unsqueeze(0).requires_grad_(True))
            self._features_dc = nn.Parameter(
                features_dc.unsqueeze(0).contiguous().requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                features_extra.unsqueeze(0).contiguous().requires_grad_(True)
            )
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))

            # 初始化梯度累积和最大半径
            self.xyz_gradient_accum = torch.zeros((1, self.get_xyz().shape[0], 1), device="cuda")
            self.denom = torch.zeros((1, self.get_xyz().shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((1, self.get_xyz().shape[0]), device="cuda")

            print(f"从点云文件 {ply_path} 初始化了 {len(points)} 个高斯点")

        except Exception as e:
            traceback.print_exc()
            logging.error(f"从点云创建高斯模型失败: {e}")
            raise

    # def get_norm_scale(self):
    #     if self.norm_scales is not None:
    #         return self.norm_scales[self.batch_idx]
    #     else:
    #         return None

    def get_xyz(self):
        return self._means[self.batch_idx]

    def get_opacity(self):
        return torch.sigmoid(self._opacity[self.batch_idx])  # [..., None]

    def get_scale(self):
        return self.scaling_activation(self._scaling[self.batch_idx])

    def get_rotation(self):
        return self.rotation_activation(self._rotation[self.batch_idx])
        # return self._rotation[self.batch_idx]

    def get_shs(self):
        features_dc = self._features_dc
        features_rest = self._features_rest  # b,n,k,c
        shs = torch.cat((features_dc, features_rest), dim=2)
        return shs[self.batch_idx]

    def apply_mask(self, mask):
        # return # TODO: handle mask
        bs = mask.shape[0]
        mask = mask.reshape(bs, -1)
        for i in range(bs):
            self.means[i] = self.means[i][mask[i]]
            self.scales[i] = self.scales[i][mask[i]]
            self.rotations[i] = self.rotations[i][mask[i]]
            self.harmonics[i] = self.harmonics[i][mask[i]]
            self.opacities[i] = self.opacities[i][mask[i]]

    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation())

    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scale().min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_normal(self, camera_center):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = camera_center - self.get_xyz()
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global

    def __len__(self):
        return self.batch_size

    def __getitem__(self, batch_idx):
        self.batch_idx = batch_idx
        return self

    def get_valid_ratio(self):
        """
        return the ratio of valid gaussians, 0-1
        """
        # return torch.mean(self.mask.float())
        return 1.0

    def __repr__(self):
        repr_str = self.__class__.__name__ + "(\n"
        attributes = [
            "_means",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
        ]
        for attribute in attributes:
            data = getattr(self, attribute, None)
            if data is not None and data.numel() > 0:
                repr_str += f"\t{attribute}=[min:{data.min().item():.5f}, max:{data.max().item():.5f}, mean:{data.mean().item():.5f}, sum:{data.abs().sum().item():.5f}],\n"
        repr_str += ")"
        return repr_str

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        # import pdb;pdb.set_trace()
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][:, mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:, mask]

                del self.optimizer.state[group["params"][0]]

                original_grad = (
                    group["params"][0].grad.clone() if group["params"][0].grad is not None else None
                )
                group["params"][0] = nn.Parameter(
                    (group["params"][0][:, mask].requires_grad_(True))
                )

                self.optimizer.state[group["params"][0]] = stored_state

                if original_grad is not None:
                    pruned_grad = original_grad[:, mask]
                    group["params"][0].grad = torch.zeros_like(pruned_grad)
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                original_grad = (
                    group["params"][0].grad.clone() if group["params"][0].grad is not None else None
                )
                group["params"][0] = nn.Parameter(group["params"][0][:, mask].requires_grad_(True))
                if original_grad is not None:
                    pruned_grad = original_grad[:, mask]
                    group["params"][0].grad = torch.zeros_like(pruned_grad)  # 这里的梯度要清零
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._means = optimizable_tensors["means"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[:, valid_points_mask]

        self.denom = self.denom[:, valid_points_mask]
        self.max_radii2D = self.max_radii2D[:, valid_points_mask]
        self.tmp_radii = self.tmp_radii[:, valid_points_mask]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:

                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group["params"][0]]

                    original_grad = (
                        group["params"][0].grad.clone()
                        if group["params"][0].grad is not None
                        else None
                    )
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group["params"][0]] = stored_state

                    if original_grad is not None:
                        group["params"][0].grad = torch.zeros_like(original_grad)

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    original_grad = (
                        group["params"][0].grad.clone()
                        if group["params"][0].grad is not None
                        else None
                    )
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    if original_grad is not None:
                        group["params"][0].grad = torch.zeros_like(original_grad)
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        # import pdb;pdb.set_trace()
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=1
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=1
                )

                del self.optimizer.state[group["params"][0]]

                original_grad = (
                    group["params"][0].grad.clone() if group["params"][0].grad is not None else None
                )
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=1).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                if original_grad is not None:
                    expanded_grad = torch.cat(
                        (original_grad, torch.zeros_like(extension_tensor)), dim=1
                    )
                    group["params"][0].grad = torch.zeros_like(expanded_grad)
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                original_grad = (
                    group["params"][0].grad.clone() if group["params"][0].grad is not None else None
                )

                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=1).requires_grad_(True)
                )

                if original_grad is not None:
                    expanded_grad = torch.cat(
                        (original_grad, torch.zeros_like(extension_tensor)), dim=1
                    )
                    group["params"][0].grad = torch.zeros_like(expanded_grad)

                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_tmp_radii,
    ):
        d = {
            "means": new_xyz,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._means = optimizable_tensors["means"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii), dim=1)

        self.xyz_gradient_accum = torch.zeros((1, self.get_xyz().shape[0], 1), device="cuda")
        self.denom = torch.zeros((1, self.get_xyz().shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((1, self.get_xyz().shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz().shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((1, n_init_points), device="cuda")
        padded_grad[:, : grads.shape[1]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # logging.info(f"split gaussians grad is: {padded_grad[selected_pts_mask]}")

        selected_pts_mask = torch.logical_and(
            selected_pts_mask.squeeze(0),
            torch.max(self.get_scale(), dim=1).values > self.percent_dense * scene_extent,
        )  # 0.01*1.0 大于0.01的scale删除
        selected_pts_mask = selected_pts_mask.unsqueeze(0)

        logging.info(f"split gaussians number is: {selected_pts_mask.sum()}")
        # import pdb;pdb.set_trace()
        stds = self.get_scale()[selected_pts_mask[0]].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask, ...]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz()[
            selected_pts_mask[0]
        ].repeat(N, 1).unsqueeze(0)
        new_scaling = self.scaling_inverse_activation(
            self.get_scale()[selected_pts_mask[0]].repeat(N, 1) / (0.8 * N)
        ).unsqueeze(0)
        new_rotation = self._rotation[selected_pts_mask, ...].repeat(N, 1).unsqueeze(0)
        new_features_dc = (
            self._features_dc[selected_pts_mask, ...].repeat(N, 1, 1).unsqueeze(0)
        )  #  改成horomonics
        new_features_rest = self._features_rest[selected_pts_mask, ...].repeat(N, 1, 1).unsqueeze(0)
        new_opacity = self._opacity[selected_pts_mask, ...].repeat(N, 1).unsqueeze(0)
        new_tmp_radii = self.tmp_radii[selected_pts_mask, ...].repeat(N).unsqueeze(0)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros((1, N * selected_pts_mask.sum()), device="cuda", dtype=bool),
            ),
            dim=1,
        )
        self.prune_points(prune_filter.squeeze(0))

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )  # 1,N

        # logging.info(f"clone gaussians bigger grad is: {grads[selected_pts_mask]}")

        selected_pts_mask = torch.logical_and(
            selected_pts_mask.squeeze(0),
            torch.max(self.get_scale(), dim=1).values <= self.percent_dense * scene_extent,
        )  # 0.01*scene_extent
        selected_pts_mask = selected_pts_mask.unsqueeze(0)

        logging.info(f"clone gaussians num is: {selected_pts_mask.sum()}")
        if selected_pts_mask.sum() > 0:
            new_xyz = self._means[selected_pts_mask].unsqueeze(0)
            new_features_dc = self._features_dc[selected_pts_mask].unsqueeze(0)
            new_features_rest = self._features_rest[selected_pts_mask].unsqueeze(0)
            new_opacities = self._opacity[selected_pts_mask].unsqueeze(0)
            new_scaling = self._scaling[selected_pts_mask].unsqueeze(0)
            new_rotation = self._rotation[selected_pts_mask].unsqueeze(0)

            new_tmp_radii = self.tmp_radii[selected_pts_mask].unsqueeze(0)

            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_opacities,
                new_scaling,
                new_rotation,
                new_tmp_radii,
            )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity() < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D.squeeze() > max_screen_size
            big_points_ws = self.get_scale().max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
            logging.info(f"big_points vs: {big_points_vs.sum()}, ws: {big_points_ws.sum()}")

        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def pruning_points_opacity(self, radii, min_opacity=0.005, max_screen_size=20, extent=1.0):
        self.tmp_radii = radii
        prune_mask = (self.get_opacity() < min_opacity).squeeze()

        logging.info(f"opacity prune gaussians number is: {prune_mask.sum()}")
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scale().max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        logging.info(f"scale and opacity prune gaussians number is: {prune_mask.sum()}")
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter[0], :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def prune_after_render(self, viewpoint_stack, render=None):
        imp_list = torch.zeros(self.get_xyz().shape[0], device="cuda").unsqueeze(0)
        near = torch.ones((1, 1), device="cuda") * 0.1
        far = torch.ones((1, 1), device="cuda") * 100

        for viewpoint_cam in viewpoint_stack:
            extrinsics = viewpoint_cam["extrinsics"].clone()
            extrinsics = extrinsics.inverse()
            (h, w) = viewpoint_cam["hw"]
            intrinsics = viewpoint_cam["intrinsics"].clone()
            intrinsics[..., 0, :] /= w
            intrinsics[..., 1, :] /= h

            imp_list = torch.maximum(
                imp_list,
                render(
                    self,
                    extrinsics.to("cuda"),
                    intrinsics.to("cuda"),
                    near=near,
                    far=far,
                    image_shape=viewpoint_cam["hw"],
                    depth_mode="depth",
                    mw_score=True,
                ).important_score[0],
            )
            gc.collect()
        self.prune_gaussians(self.prune_percent, imp_list[0])

    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask.unsqueeze(0))
