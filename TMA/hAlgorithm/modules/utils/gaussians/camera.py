import torch
from torch import nn

from hAlgorithm.modules.utils.gaussians.graphics_utils import (
    focal2fov,
    getProjectionMatrixCenterShift,
    mat_to_quat_trans,
    quat_trans_to_mat,
)


class Camera(nn.Module):
    def __init__(self, T_w2c, K, img_h, img_w, device="cuda"):
        super(Camera, self).__init__()
        self.device = device
        self.T_w2c = T_w2c.clone().to(device)
        self.K = K.clone().to(device)

        self.img_h = torch.tensor(img_h).to(device)
        self.img_w = torch.tensor(img_w).to(device)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.fovX = focal2fov(self.fx, self.img_w)
        self.fovY = focal2fov(self.fy, self.img_h)

        self.zfar = torch.tensor(100.0).to(device)  # 100.0
        self.znear = torch.tensor(0.01).to(device)  # 0.01

        w2c_quat, w2c_trans = mat_to_quat_trans(self.T_w2c)
        self.w2c_quat = nn.Parameter(w2c_quat.to(device))
        self.w2c_trans = nn.Parameter(w2c_trans.to(device))

        self.projection_matrix = (
            getProjectionMatrixCenterShift(
                znear=self.znear,
                zfar=self.zfar,
                cx=self.cx,
                cy=self.cy,
                fl_x=self.fx,
                fl_y=self.fy,
                w=self.img_w,
                h=self.img_h,
            )
            .transpose(0, 1)
            .to(device)
        )

    @property
    def world_view_transform(self):
        w2c_quat = self.w2c_quat / torch.norm(self.w2c_quat)
        return quat_trans_to_mat(w2c_quat, self.w2c_trans).transpose(0, 1).to(self.device)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

    def get_calib_matrix_nerf(self):
        intrinsic_matrix = (
            torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
            .float()
            .to(self.device)
        )
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous()
        return intrinsic_matrix, extrinsic_matrix


class CameraList(nn.Module):
    def __init__(self):
        super(CameraList, self).__init__()
        self.cameras = nn.ModuleDict()

    def append(self, datas, extrinsics_name, intrinsics_name, device=None):
        if isinstance(datas, list):
            for i, data_item in enumerate(datas):
                self.append_batch(data_item, extrinsics_name=extrinsics_name, intrinsics_name=intrinsics_name, device=device)
        elif isinstance(datas, dict):
            self.append_dict(datas, device=device)
        else:
            raise NotImplementedError

    @staticmethod
    def get_camera_name(frame_id, view_id):
        return f"{int(frame_id)} + {int(view_id)}"

    @staticmethod
    def camera_name_to_id(name):
        frame_id, view_id = name.strip().split(" + ")
        return int(frame_id), int(view_id)

    def append_batch(self, data_batch, extrinsics_name, intrinsics_name, device=None):
        meta_data = data_batch["meta_data"]
        frame_id = meta_data["frame_id"]
        view_id = meta_data["view_id"]

        img_h, img_w = data_batch["image"].shape[1:]

        extrinsics = data_batch[extrinsics_name]
        intrinsics = data_batch[intrinsics_name]

        camera = Camera(extrinsics, intrinsics, img_h, img_w, device=device)

        name = self.get_camera_name(frame_id=frame_id, view_id=view_id)
        self.cameras[name] = camera
    
    def append_dict(self, data_dict, device=None):
        # exp: {"frame_id_0": {"view_id_0": "intrinsics":[], "extrinsics":[]}}
        for frame_key, views in data_dict.items():
            frame_id = int(frame_key.replace("frame_id_", ""))
            for view_key, data in views.items():
                view_id = int(view_key.replace("view_id_", ""))
                intrinsics = torch.tensor(data["intrinsics"]).float()
                extrinsics = torch.tensor(data["extrinsics"]).float()
                img_h, img_w = int(data["img_h"]), int(data["img_w"])
                camera = Camera(extrinsics, intrinsics, img_h, img_w, device=device)
                name = self.get_camera_name(frame_id=frame_id, view_id=view_id)
                self.cameras[name] = camera

    def get_camera(self, frame_id, view_id):
        name = self.get_camera_name(frame_id=frame_id, view_id=view_id)
        if name in self.cameras:
            return self.cameras[name]
        else:
            return None
