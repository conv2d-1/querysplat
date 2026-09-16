import torch

def fov_encoding_to_intri(fov, image_size_hw):
    fov_h = fov[..., 0]
    fov_w = fov[..., 1]

    H, W = image_size_hw
    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)
    intrinsics = fov.new_zeros(fov.shape[:2] + (3, 3))
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1

    return intrinsics

def intri_to_fov(intrinsics, image_size_hw):
    # intrinsics: BxSx3x3
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return fov_w, fov_h


def pi3_pose_fov_to_extri_intri(
    pose, # w2c
    fov=None,
    image_size_hw=None,
    pose_encoding_type="pi3",
    translation_scale=None,
    normalize_cameras=True,
):
    if pose_encoding_type in ["pi3", "pi3_fov"]:
        if normalize_cameras:
            # NOTE: trans to first camera
            w2c_pred = pose
            base_c2w_pred = w2c_pred[:, 0:1].inverse()
            extrinsics = w2c_pred @ base_c2w_pred

        # TODO
        if translation_scale is not None:
            extrinsics[..., :3, 3] *= translation_scale[..., 0, 0]
        
    else:
        raise NotImplementedError

    if pose_encoding_type == "pi3_fov":
        fov_h = fov[..., 0]
        fov_w = fov[..., 1]

        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)
        intrinsics = pose.new_zeros(pose.shape[:2] + (3, 3))
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        intrinsics = None

    return extrinsics, intrinsics
