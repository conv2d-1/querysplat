import math

import torch
from einops import pack

from hAlgorithm.modules.utils.gaussians.camera_trajectory import (
    interpolate_intrinsics,
    interpolate_poses_spline,
)


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.norm(v), min=eps)


def _look_at_c2w(position: torch.Tensor, target: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    forward = _safe_normalize(target - position)
    right = torch.cross(up_hint, forward, dim=0)
    if torch.norm(right) < 1e-6:
        right = torch.cross(
            torch.tensor([1.0, 0.0, 0.0], device=position.device, dtype=position.dtype),
            forward,
            dim=0,
        )
    right = _safe_normalize(right)
    up = _safe_normalize(torch.cross(forward, right, dim=0))

    c2w = torch.eye(4, device=position.device, dtype=position.dtype)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def _build_orbit_poses(
    base_c2w: torch.Tensor,
    target: torch.Tensor,
    num_frames: int,
    radius: float,
    vertical: float,
    forward_amp: float,
) -> list:
    base_pos = base_c2w[:3, 3]
    right = base_c2w[:3, 0]
    up = base_c2w[:3, 1]
    forward = base_c2w[:3, 2]

    poses = []
    n = max(2, int(num_frames))
    for i in range(n):
        theta = 2.0 * math.pi * float(i) / float(n)
        offset = right * (radius * math.sin(theta)) + up * (vertical * math.sin(2.0 * theta)) + forward * (forward_amp * 0.5 * (1.0 - math.cos(theta)))
        pos = base_pos + offset
        poses.append(_look_at_c2w(pos, target, up))
    return poses


def _build_swing_poses(
    base_c2w: torch.Tensor,
    num_frames: int,
    radius: float,
    forward_amp: float,
) -> list:
    base_pos = base_c2w[:3, 3]
    right = base_c2w[:3, 0]
    forward = base_c2w[:3, 2]

    key_offsets = [
        torch.zeros(3, device=base_pos.device, dtype=base_pos.dtype),
        -right * radius,
        right * radius,
        forward * forward_amp,
        torch.zeros(3, device=base_pos.device, dtype=base_pos.dtype),
    ]

    poses = []
    seg_frames = max(1, int(num_frames) // (len(key_offsets) - 1))
    for seg in range(len(key_offsets) - 1):
        p0 = base_pos + key_offsets[seg]
        p1 = base_pos + key_offsets[seg + 1]
        for i in range(seg_frames):
            alpha = 1.0 if seg_frames == 1 else float(i) / float(seg_frames - 1)
            pos = (1.0 - alpha) * p0 + alpha * p1
            pose = base_c2w.clone()
            pose[:3, 3] = pos
            if seg > 0 and i == 0:
                continue
            poses.append(pose)
    return poses


def render_video_generic(
    render_fn,
    gaussians,
    trajectory_fn,
    h,
    w,
    n_interp: int = 30,
    loop_reverse: bool = True,
) -> None:
    from hAlgorithm.modules.pipelines2.utils.visualize_mv import apply_color_map

    c2w, intrinsics = trajectory_fn(n_interp)
    w2c = c2w.inverse()

    def depth_map(result):
        # result: (n, H, W)
        if result[result > 0].numel() == 0:
            near = 0
        else:
            near = result[result > 0][:16_000_000].quantile(0.01)
        far = result.view(-1)[:16_000_000].quantile(0.99)
        result = 1 - (result - near) / (far - near)
        result = result.unsqueeze(1)  # (n, 1, H, W) for apply_color_map
        return apply_color_map(result, "turbo")  # returns (n, 3, H, W)

    b, n = w2c.shape[:2]

    rgb_list = []
    depth_list = []
    for i in range(n):
        render_rgb, render_depth = render_fn(
            gaussians=gaussians,
            intrinsics=intrinsics[:, i],
            w2c=w2c[:, i],
            image_shape=(h, w),
        )
        if render_rgb is None:
            return None
        rgb_list.append(render_rgb)
        depth_list.append(render_depth)

    render_rgb = torch.stack(rgb_list, dim=1)  # (b, n, C, H, W)
    render_depth = torch.stack(depth_list, dim=1)  # (b, n, H, W)

    depth_color = depth_map(render_depth[0].detach())
    rgb = render_rgb[0]

    video = torch.cat([rgb, depth_color], dim=3).permute(0, 2, 3, 1)

    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()

    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* h w c")[0]

    return video


def render_video_interpolation(
    render_fn,
    gaussians,
    w2c,
    intrinsics,
    h,
    w,
    trajectory="spline",
    n_interp=24,
    loop=False,
    loop_reverse=False,
    radius=0.5,
    vertical=0.1,
    forward_amp=0.2,
    device="cpu",
):
    """Render video with camera trajectory interpolation.

    Args:
        trajectory: "spline" (interpolate between input poses),
                   "swing" (left-right-forward motion),
                   "orbit" (orbit around scene center)
    """
    intrinsics = intrinsics.clone()
    c2w = w2c.clone().inverse()  # w2c -> c2w for interpolation

    if trajectory == "spline" and c2w.shape[1] == 1:
        return None

    def trajectory_fn_spline(n_interp):
        if loop:
            c2w_ = torch.cat([c2w, c2w[0:, 0:1]], dim=1)
        else:
            c2w_ = c2w
        b, v, _, _ = c2w_.shape
        c2w_target = interpolate_poses_spline(c2w_.reshape(b * v, 4, 4)[:, :3].cpu(), n_interp)
        c2w_target = c2w_target.reshape(b, -1, 4, 4).to(device).float()

        num_frames = c2w_target.shape[1]
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)

        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[..., 0, :] /= w
        intrinsics_normalized[..., 1, :] /= h
        intrinsics_target = interpolate_intrinsics(
            intrinsics_normalized[0, 0],
            intrinsics_normalized[0, -1],
            t,
        )
        intrinsics_target[..., 0, :] *= w
        intrinsics_target[..., 1, :] *= h
        intrinsics_target = intrinsics_target[None]
        return c2w_target, intrinsics_target

    def trajectory_fn_swing(n_interp):
        base_c2w = c2w[0, 0]  # use first camera as base
        poses = _build_swing_poses(base_c2w, n_interp, radius, forward_amp)
        c2w_target = torch.stack(poses, dim=0).unsqueeze(0).to(device).float()  # (1, N, 4, 4)

        intrinsics_target = intrinsics[0, 0:1].expand(c2w_target.shape[1], -1, -1).clone()  # (N, 3, 3)
        intrinsics_target = intrinsics_target[None]  # (1, N, 3, 3)
        return c2w_target, intrinsics_target

    def trajectory_fn_orbit(n_interp):
        base_c2w = c2w[0, 0]  # use first camera as base

        base_pos = base_c2w[:3, 3]
        base_forward = base_c2w[:3, 2]

        # Compute target as a point along camera's forward direction
        # Use median depth of gaussians as the distance
        means = gaussians["means"].reshape(-1, 3)
        cam_to_points = means - base_pos
        depths = (cam_to_points * base_forward).sum(dim=-1)  # project to forward direction
        median_depth = depths[depths > 0].median() if (depths > 0).any() else depths.abs().median()
        target = base_pos + base_forward * median_depth

        # means = gaussians["means"]
        # target = means.mean(dim=(0, 1))  # scene center

        poses = _build_orbit_poses(base_c2w, target, n_interp, radius, vertical, forward_amp)
        c2w_target = torch.stack(poses, dim=0).unsqueeze(0).to(device).float()  # (1, N, 4, 4)

        intrinsics_target = intrinsics[0, 0:1].expand(c2w_target.shape[1], -1, -1).clone()  # (N, 3, 3)
        intrinsics_target = intrinsics_target[None]  # (1, N, 3, 3)
        return c2w_target, intrinsics_target

    if trajectory == "swing":
        trajectory_fn = trajectory_fn_swing
    elif trajectory == "orbit":
        trajectory_fn = trajectory_fn_orbit
    else:
        trajectory_fn = trajectory_fn_spline

    return render_video_generic(render_fn, gaussians, trajectory_fn, h, w, n_interp=n_interp, loop_reverse=loop_reverse)
