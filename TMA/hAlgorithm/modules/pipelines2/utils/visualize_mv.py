import json
import logging
import os

import cv2
import numpy as np
import torch
from einops import pack

from hAlgorithm.datasets_mv.track.vggt_track import visualize_tracks_on_images
from hAlgorithm.modules.pipelines2.utils.points_filter import filtered_points_with_confidence
from hAlgorithm.modules.pipelines2.utils.visualize import (
    save_depth_map,
    save_error,
    save_image,
    save_point_cloud,
    save_video,
    save_warp,
)
from hAlgorithm.modules.utils.gaussians.camera_trajectory import (
    interpolate_intrinsics,
    interpolate_poses_spline,
)
from hAlgorithm.utils import (
    apply_color_map,
    grid_images,
)
from hAlgorithm.modules.utils.ray import debug_visualize_rays_comparison_numpy


def vis_images(cfg, mv_outputs, gt_out_dir, data_idx, frame_num, view_num):
    if gt_out_dir is not None and mv_outputs[0].rgb is not None:
        save_path = os.path.join(gt_out_dir, f"merge_rgb_{data_idx:06d}.jpg")
        if not os.path.exists(save_path):
            os.makedirs(gt_out_dir, exist_ok=True)
            images = [(outputs.rgb[:, :, ::-1] * 255).astype(np.uint8) for outputs in mv_outputs]
            grid_images(
                save_path=save_path,
                images=images,
                col=min(4, len(images)) if frame_num == 1 or view_num == 1 else view_num,
            )
            logging.info(f"save images: {save_path}")

            if cfg["save_everything"]:
                for i, image in enumerate(images):
                    save_path = os.path.join(gt_out_dir, "images", f"rgb_{data_idx:06d}_{i:03d}.jpg")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, image)


def vis_render(cfg, mv_outputs, gs_out_dir, data_idx):
    render_out_dir = os.path.join(gs_out_dir, "grid_images")
    if cfg["save_render_results"] and mv_outputs[0].render_rgb is not None:
        os.makedirs(render_out_dir, exist_ok=True)
        save_path = os.path.join(render_out_dir, f"render_rgb_{data_idx:06d}.jpg")
        render_rgb = [(np.clip(outputs.render_rgb[..., ::-1], 0.0, 1.0) * 255).astype(np.uint8) for outputs in mv_outputs]
        grid_images(save_path, images=render_rgb, col=min(4, len(render_rgb)))

    if cfg["save_render_results"] and mv_outputs[0].render_depth is not None:
        os.makedirs(render_out_dir, exist_ok=True)
        save_path = os.path.join(render_out_dir, f"render_dpt_{data_idx:06d}.jpg")

        def colorize(depth):
            min_val, max_val = depth.min(), depth.max()
            depth_norm = 1 - (depth - min_val) / (max_val - min_val + 1e-9)
            depth_colored = apply_color_map(depth_norm, "turbo")
            return (depth_colored.clip(0, 1) * 255).astype(np.uint8)

        render_depth = [colorize(outputs.render_depth) for outputs in mv_outputs]
        grid_images(save_path, images=render_depth, col=min(4, len(render_depth)))


def render_video_generic(
    cfg,
    model,
    gaussians,
    trajectory_fn,
    h,
    w,
    n_interp: int = 30,
    loop_reverse: bool = True,
) -> None:

    extrinsics, intrinsics = trajectory_fn(n_interp)

    def depth_map(result):
        if result[result > 0].numel() == 0:
            near = 0
        else:
            near = result[result > 0][:16_000_000].quantile(0.01)
        far = result.view(-1)[:16_000_000].quantile(0.99)
        result = 1 - (result - near) / (far - near)
        return apply_color_map(result, "turbo")

    outputs = model(
        rgb=None,
        gaussians=gaussians,
        c2w=extrinsics,
        intrinsics=intrinsics,
        image_shape=(h, w),
        depth_mode="depth",
        only_rendering=True,
    )

    depth_color = depth_map(outputs["render_depth"][0].detach())
    rgb = outputs["render_rgb"][0]

    if "render_normal" in outputs:
        normal = (outputs["render_normal"][0] + 1) * 0.5
        video = torch.cat([rgb, depth_color, normal], dim=3).permute(0, 2, 3, 1)
    else:
        video = torch.cat([rgb, depth_color], dim=3).permute(0, 2, 3, 1)

    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()

    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* h w c")[0]

    return video


def render_video_interpolation(cfg, model, gaussians, extrinsics, intrinsics, h, w, loop=False, loop_reverse=False, device="cpu"):

    intrinsics = intrinsics.clone()
    extrinsics_inv = extrinsics.clone().inverse()

    if extrinsics_inv.shape[1] == 1:
        return None

    def trajectory_fn(n_interp):
        if loop:
            extrinsics_inv_ = torch.cat([extrinsics_inv, extrinsics_inv[0:, 0:1]], dim=1)
        extrinsics_inv_ = extrinsics_inv
        b, v, _, _ = extrinsics_inv_.shape
        extrinsics_inv_target = interpolate_poses_spline(extrinsics_inv_.reshape(b * v, 4, 4)[:, :3].cpu(), n_interp)
        extrinsics_inv_target = extrinsics_inv_target.reshape(b, -1, 4, 4).to(device).float()

        num_frames = b * extrinsics_inv_target.shape[1]
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)

        intrinsics[..., 0, :] /= w
        intrinsics[..., 1, :] /= h
        intrinsics_target = interpolate_intrinsics(
            intrinsics[0, 0],
            intrinsics[0, -1],
            t,
        )
        intrinsics_target = intrinsics_target[None]
        return extrinsics_inv_target, intrinsics_target

    return render_video_generic(cfg, model, gaussians, trajectory_fn, h, w, n_interp=24, loop_reverse=loop_reverse)


def vis_render_video(cfg, model, mv_outputs, gs_out_dir, data_idx, frame_num, view_num, device="cpu"):
    os.makedirs(gs_out_dir, exist_ok=True)
    gaussians = mv_outputs[0].gaussians
    save_path = os.path.join(gs_out_dir, f"gaussians_{data_idx:06d}.ply")
    gaussians.export_ply(save_path)

    if cfg["save_render_video"]:
        if mv_outputs[0].extrinsics is None or mv_outputs[0].intrinsics is None or cfg["render_video_with_pred_camera"]:
            extrinsics = torch.stack([torch.from_numpy(outputs.extrinsics_pred) for outputs in mv_outputs], dim=0)
            intrinsics = torch.stack([torch.from_numpy(outputs.intrinsics_pred) for outputs in mv_outputs], dim=0)
        else:
            extrinsics = torch.stack([torch.from_numpy(outputs.extrinsics) for outputs in mv_outputs], dim=0)
            intrinsics = torch.stack([torch.from_numpy(outputs.intrinsics) for outputs in mv_outputs], dim=0)

        if cfg["save_render_video_with_normalize_c2w"]:
            scale = torch.tensor([outputs.prompt_scale for outputs in mv_outputs]).unsqueeze(-1)
            extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / scale

        extrinsics = extrinsics.reshape(-1, frame_num * view_num, 4, 4).float().to(device)
        intrinsics = intrinsics.reshape(-1, frame_num * view_num, 3, 3).float().to(device)

        render_video = render_video_interpolation(
            cfg=cfg,
            model=model,
            gaussians=gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            h=mv_outputs[0].pointmap_h,
            w=mv_outputs[0].pointmap_w,
            device=device,
        )
        save_video(render_video, gs_out_dir, "video", data_idx, info=True)

def sample_points(points, colors, ratio=None):
    if ratio is not None:
        N = points.shape[0]
        n_samples = int(N * ratio)
        indices = np.random.choice(N, size=n_samples, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    return points, colors

def vis_glb_results(cfg, mv_outputs, glb_out_dir, data_idx, frame_num, view_num, gt_out_dir=None):
    glb_points_max_ratio = cfg.get("glb_points_max_ratio", None)

    if gt_out_dir is not None and mv_outputs[0].pointmap_gt_global is not None:
        pointmap_gt_global = [outputs.pointmap_gt_global.reshape(-1, 3).copy() for outputs in mv_outputs]
        merge_pointmap_gt_global = np.concatenate(pointmap_gt_global, axis=0)
    else:
        pointmap_gt_global = merge_pointmap_gt_global = None

    if mv_outputs[0].glb_mv_pointmap is not None:
        glb_mv_pointmap = [outputs.glb_mv_pointmap.reshape(-1, 3).copy() for outputs in mv_outputs]
        merge_glb_mv_pointmap = np.concatenate(glb_mv_pointmap, axis=0)
    else:
        glb_mv_pointmap = merge_glb_mv_pointmap = None

    if mv_outputs[0].local2glb_pointmap is not None:
        local2glb_pointmap = [outputs.local2glb_pointmap.reshape(-1, 3).copy() for outputs in mv_outputs]
        merge_local2glb_pointmap = np.concatenate(local2glb_pointmap, axis=0)
    else:
        local2glb_pointmap = merge_local2glb_pointmap = None

    if mv_outputs[0].pointmap_color is not None:
        pointmap_color = [outputs.pointmap_color.reshape(-1, 3).copy() for outputs in mv_outputs]
        merge_pointmap_color = np.concatenate(pointmap_color, axis=0)
    else:
        pointmap_color = merge_pointmap_color = None

    if merge_pointmap_gt_global is not None:
        chech_path = os.path.join(gt_out_dir, f"glb_points_gt_{data_idx:06d}.ply")
        if not os.path.exists(chech_path):
            os.makedirs(gt_out_dir, exist_ok=True)
            points, colors = sample_points(merge_pointmap_gt_global, merge_pointmap_color, glb_points_max_ratio)
            save_point_cloud(
                points,
                colors,
                gt_out_dir,
                "glb_points_gt",
                data_idx,
                info=True,
            )

    if merge_glb_mv_pointmap is not None:
        os.makedirs(glb_out_dir, exist_ok=True)
        points, colors = sample_points(merge_glb_mv_pointmap, merge_pointmap_color, glb_points_max_ratio)
        save_point_cloud(
            points,
            colors,
            glb_out_dir,
            "glb_points",
            data_idx,
            info=False,
        )
        if cfg["save_filtered_results"] and mv_outputs[0].glb_mv_confidence is not None:
            filtered_points = []
            filtered_points_color = []
            for idx, outputs in enumerate(mv_outputs):
                points = glb_mv_pointmap[idx]
                colors = pointmap_color[idx] if pointmap_color is not None else None
                points, colors = filtered_points_with_confidence(
                    points=points,
                    confidence=outputs.glb_mv_confidence.copy(),
                    h=outputs.pointmap_h,
                    w=outputs.pointmap_w,
                    output_conf_ratio=cfg["output_conf_ratio"],
                    colors=colors,
                )
                filtered_points.append(points)
                if colors is not None:
                    filtered_points_color.append(colors)

            filtered_points = np.concatenate(filtered_points, axis=0)
            if len(filtered_points_color) > 0:
                filtered_points_color = np.concatenate(filtered_points_color, axis=0)
            else:
                filtered_points_color = None
            points, colors = sample_points(filtered_points, filtered_points_color, glb_points_max_ratio)
            save_point_cloud(
                points,
                colors,
                glb_out_dir,
                f"filtered_glb_points",
                data_idx,
                info=False,
            )

    if cfg["save_local2glb_results"] and merge_local2glb_pointmap is not None:
        os.makedirs(glb_out_dir, exist_ok=True)
        points, colors = sample_points(merge_local2glb_pointmap, merge_pointmap_color, glb_points_max_ratio)
        save_point_cloud(
            points,
            colors,
            glb_out_dir,
            f"lcl2glb_points",
            data_idx,
            info=False,
        )
        if cfg["save_filtered_results"] and mv_outputs[0].confidence is not None:
            filtered_points = []
            filtered_points_color = []
            for idx, outputs in enumerate(mv_outputs):
                points = local2glb_pointmap[idx]
                colors = pointmap_color[idx] if pointmap_color is not None else None
                points, colors = filtered_points_with_confidence(
                    points=points,
                    confidence=outputs.local2glb_confidence.copy(),
                    h=outputs.pointmap_h,
                    w=outputs.pointmap_w,
                    output_conf_ratio=cfg["output_conf_ratio"],
                    colors=colors,
                )
                filtered_points.append(points)
                if colors is not None:
                    filtered_points_color.append(colors)

            filtered_points = np.concatenate(filtered_points, axis=0)
            if len(filtered_points_color) > 0:
                filtered_points_color = np.concatenate(filtered_points_color, axis=0)
            else:
                filtered_points_color = None
            points, colors = sample_points(filtered_points, filtered_points_color, glb_points_max_ratio)
            save_point_cloud(
                points,
                colors,
                glb_out_dir,
                f"filtered_lcl2glb_points",
                data_idx,
                info=False,
            )

    if cfg["save_glb_sf_results"] or cfg["save_glb2local_results"]:
        curr_glb_out_dir = os.path.join(glb_out_dir, f"{data_idx:06d}")
        os.makedirs(curr_glb_out_dir, exist_ok=True)
        for frame_index in range(frame_num):
            for view_index in range(view_num):
                prefix = f"f{frame_index:03d}_v{view_index:03d}_"
                index = frame_index * view_num + view_index
                # prefix = f"idx{index:04d}_"
                colors = pointmap_color[index] if pointmap_color is not None else None

                if cfg["save_glb_sf_results"] and glb_mv_pointmap is not None:
                    save_point_cloud(
                        glb_mv_pointmap[index],
                        colors,
                        curr_glb_out_dir,
                        f"{prefix}glb_points",
                        info=False,
                    )
                if cfg["save_glb_sf_results"] and pointmap_gt_global is not None:
                    chech_path = os.path.join(gt_out_dir, f"{prefix}glb_points_gt.ply")
                    if not os.path.exists(chech_path):
                        curr_gt_out_dir = os.path.join(gt_out_dir, f"{data_idx:06d}")
                        os.makedirs(curr_gt_out_dir, exist_ok=True)
                        save_point_cloud(
                            pointmap_gt_global[index],
                            colors,
                            curr_gt_out_dir,
                            f"{prefix}glb_points_gt",
                            info=True,
                        )

                if cfg["save_glb2local_results"] and mv_outputs[index].glb2local_pointmap is not None:
                    save_point_cloud(
                        mv_outputs[index].glb2local_pointmap.reshape(-1, 3).copy(),
                        colors,
                        curr_glb_out_dir,
                        f"{prefix}glb2local_points",
                        info=False,
                    )

                if cfg["save_local2glb_results"] and local2glb_pointmap is not None:
                    save_point_cloud(
                        local2glb_pointmap[index],
                        colors,
                        curr_glb_out_dir,
                        f"{prefix}lcl2glb_points",
                        info=False,
                    )


def vis_camera_results(cfg, mv_outputs, camera_out_dir, frame_num, view_num):
    for frame_index in range(frame_num):
        for view_index in range(view_num):
            prefix = f"f{frame_index:03d}_v{view_index:03d}_"
            index = frame_index * view_num + view_index
            # prefix = f"idx{index:04d}_"
            outputs = mv_outputs[index]

            if outputs.intrinsics_pred is not None:
                os.makedirs(camera_out_dir, exist_ok=True)
                data = dict(pred=outputs.intrinsics_pred.tolist())
                if outputs.intrinsics is not None:
                    data["gt"] = outputs.intrinsics.tolist()

                save_path = os.path.join(camera_out_dir, f"{prefix}intris.json")
                with open(save_path, "w") as f:
                    json.dump(data, f, indent=2)

            if outputs.extrinsics_pred is not None:
                os.makedirs(camera_out_dir, exist_ok=True)
                data = dict(pred=outputs.extrinsics_pred.tolist())
                if outputs.extrinsics is not None:
                    data["gt"] = outputs.extrinsics.tolist()

                save_path = os.path.join(camera_out_dir, f"{prefix}extris.json")
                with open(save_path, "w") as f:
                    json.dump(data, f, indent=2)


def vis_local_results_single(cfg, outputs, meta_data, out_dir, gt_out_dir, prefix=""):
    data_idx = meta_data["data_idx"][0]
    if "rgb" in meta_data["data_info"][0]:
        rgb = meta_data["data_info"][0]["rgb"]
        logging.info(f"vis {data_idx}, out_dir: {out_dir}, rgb: {rgb}")
    else:
        logging.info(f"vis {data_idx}, out_dir: {out_dir}")

    # Depth Map Visualization
    if outputs.depth_align is not None:
        save_depth_map(outputs.depth_align.copy(), out_dir, f"{prefix}depth", data_idx, info=False)

    # Point Cloud Visualization
    if outputs.pointmap is not None:
        save_point_cloud(
            outputs.pointmap.copy(),
            (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
            out_dir,
            f"{prefix}point",
            data_idx,
            info=False,
        )

    # Ground Truth Point Cloud Visualization
    if outputs.pointmap_gt is not None and outputs.pointmap is not None:
        pointmap_shape = (outputs.pointmap_h, outputs.pointmap_w)
        pred_depthmap = outputs.pointmap.copy()[:, 2].reshape(pointmap_shape)
        try:
            gt_depthmap = outputs.pointmap_gt.copy()[:, 2].reshape(pointmap_shape)
            save_error(pred_depthmap, gt_depthmap, out_dir, f"{prefix}error", data_idx)
        except:
            logging.warning(f"gt_depthmap {outputs.pointmap_gt.shape}, pointmap_shape {pointmap_shape}")

    # Confidence Visualization
    if outputs.confidence is not None:
        confidence = outputs.confidence.copy()
        save_depth_map(confidence, out_dir, f"{prefix}conf", data_idx, info=False)
        conf_thresh = np.percentile(confidence.reshape(-1), cfg["output_conf_ratio"] * 100)
        confidence_mask = (confidence > conf_thresh).astype(float)
        save_depth_map(confidence_mask, out_dir, f"{prefix}conf_mask", data_idx)

    # invalid mask Visualization
    # if outputs.invalid_mask is not None:
    #     invalid_mask = outputs.invalid_mask.copy()
    #     save_depth_map(invalid_mask, out_dir, f"{prefix}invalid_mask", data_idx, info=False)
    #     invalid_mask_binary = (invalid_mask > 0).astype(float)
    #     save_depth_map(invalid_mask_binary, out_dir, f"{prefix}invalid_mask_binary", data_idx)

    # Filtered Point Cloud Visualization
    if outputs.filtered_pointmap is not None:
        save_point_cloud(
            outputs.filtered_pointmap.copy(),
            (outputs.filtered_pointmap_color.copy() if outputs.filtered_pointmap_color is not None else None),
            out_dir,
            f"{prefix}filtered_point",
            data_idx,
        )

    if outputs.prompt_pointmap is not None:
        if outputs.prompt_h != outputs.pointmap_h or outputs.prompt_w != outputs.pointmap_w:
            if outputs.pointmap_color is not None:
                tmp_color = cv2.resize(
                    outputs.pointmap_color.copy().reshape(outputs.pointmap_h, outputs.pointmap_w, -1),
                    dsize=(outputs.prompt_w, outputs.prompt_h),
                    interpolation=cv2.INTER_LINEAR,
                ).reshape(outputs.prompt_w * outputs.prompt_h, -1)
            else:
                tmp_color = None

            save_point_cloud(
                outputs.prompt_pointmap.copy(),
                tmp_color,
                out_dir,
                f"{prefix}prompt_point",
                data_idx,
            )
        else:
            save_point_cloud(
                outputs.prompt_pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}prompt_point",
                data_idx,
            )

        save_depth_map(
            outputs.prompt_pointmap.copy().reshape(outputs.prompt_h, outputs.prompt_w, 3)[:, :, 2],
            out_dir,
            f"{prefix}prompt_point",
            data_idx,
            info=False,
        )
    
    if outputs.pointmap_gt is not None:
        chech_path = os.path.join(gt_out_dir, f"points_gt_{data_idx:06d}.ply")
        if not os.path.exists(chech_path):
            save_point_cloud(
                outputs.pointmap_gt.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                gt_out_dir,
                "points_gt",
                data_idx,
                info=True,
            )


def vis_local_results(cfg, mv_outputs, mv_out_dir, frame_num, view_num, gt_out_dir, meta_data):
    data_info = meta_data.pop("data_info")
    for frame_index in range(frame_num):
        for view_index in range(view_num):
            prefix = f"f{frame_index:03d}_v{view_index:03d}_"
            index = frame_index * view_num + view_index

            if isinstance(data_info[0], dict):
                meta_data["data_info"] = data_info
            else:
                meta_data["data_info"] = [data_info[0][index]]

            vis_local_results_single(cfg, mv_outputs[index], meta_data, mv_out_dir, gt_out_dir, prefix=prefix)

    meta_data["data_info"] = data_info


def vis_extra_local_results(cfg, mv_outputs, mv_out_dir, frame_num, view_num, meta_data):
    for frame_index in range(frame_num):
        for view_index in range(view_num):
            prefix = f"f{frame_index:03d}_v{view_index:03d}_"
            index = frame_index * view_num + view_index
            data_idx = meta_data["data_idx"][0]
            output = mv_outputs[index]

            if output.normal is not None:
                from hAlgorithm.modules.metrics.normal_eval_metrics import pointmap2normal

                if output.normal_gt is not None:
                    target_normal = output.normal_gt.copy() * [0.5, -0.5, -0.5] + 0.5
                elif output.pointmap_gt is not None:
                    target_points = torch.from_numpy(output.pointmap_gt).reshape([output.pointmap_h, output.pointmap_w, 3])

                    target_normal, normal_masks = pointmap2normal(target_points, torch.from_numpy(output.depth_mask))
                    if normal_masks.sum() > 0:
                        target_normal = target_normal.squeeze(0).permute(1, 2, 0).numpy() * [0.5, -0.5, -0.5] + 0.5
                    else:
                        target_normal = None
                else:
                    target_normal = None

                normal = output.normal.copy() * [0.5, -0.5, -0.5] + 0.5

                if target_normal is not None:
                    diff = np.abs(target_normal - normal).sum(axis=-1).clip(0, 1)
                    diff = (apply_color_map(diff, "turbo") * 255).astype(np.uint8)

                    diff = np.concatenate([(target_normal * 255).astype(np.uint8), (normal * 255).astype(np.uint8), diff], axis=1)
                    save_path = os.path.join(mv_out_dir, f"{prefix}normal_{data_idx:06d}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
                else:
                    save_image(rgb=normal, out_dir=mv_out_dir, file_prefix=f"{prefix}normal", data_idx=data_idx, info=False)

                if output.pointmap is not None:
                    points = torch.from_numpy(output.pointmap).reshape([output.pointmap_h, output.pointmap_w, 3])
                    if output.depth_mask is not None:
                        points_normal, normal_masks = pointmap2normal(points, torch.from_numpy(output.depth_mask))
                    else:
                        points_normal, normal_masks = pointmap2normal(points, torch.ones_like(points[..., 0]).bool())
                    
                    if normal_masks.sum() > 0:
                        points_normal = points_normal.squeeze(0).permute(1, 2, 0).numpy() * [0.5, -0.5, -0.5] + 0.5

                        if target_normal is not None:
                            diff = np.abs(target_normal - points_normal).sum(axis=-1).clip(0, 1)
                            diff = (apply_color_map(diff, "turbo") * 255).astype(np.uint8)
                            diff = np.concatenate([(target_normal * 255).astype(np.uint8), (points_normal * 255).astype(np.uint8), diff], axis=1)
                            save_path = os.path.join(mv_out_dir, f"{prefix}points_normal{data_idx:06d}.jpg")
                            cv2.imwrite(save_path, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
                        else:
                            save_image(rgb=points_normal, out_dir=mv_out_dir, file_prefix=f"{prefix}points_normal", data_idx=data_idx, info=False)

            if output.invalid_mask is not None:
                invalid_mask = output.invalid_mask > 0
                rgb = output.rgb.copy()
                rgb[invalid_mask, :] *= 0.5
                rgb[invalid_mask, 0] += 0.5
                save_image(rgb=rgb, out_dir=mv_out_dir, file_prefix=f"{prefix}invalid_mask", data_idx=data_idx, info=False)

            if output.ray_directions is not None and output.ray_directions_gt is not None:
                ray_directions = output.ray_directions.copy()
                ray_directions_gt = output.ray_directions_gt.copy()
                debug_visualize_rays_comparison_numpy(
                    ray_directions, ray_directions_gt,
                    os.path.join(mv_out_dir, f"{prefix}ray_directions_cmp_{data_idx:06d}.png"),
                )

def vis_track_results(cfg, mv_outputs, track_out_dir, data_idx, gt_out_dir=None):
    if mv_outputs[0].track_pred is not None:
        track_pred = torch.stack([torch.from_numpy(outputs.track_pred) for outputs in mv_outputs], dim=0)
        track_vis_pred = torch.stack([torch.from_numpy(outputs.track_vis_pred) for outputs in mv_outputs], dim=0)

        track_vis_pred = track_vis_pred >= cfg["output_track_vis_thresh"]
        track_confidence_pred = torch.stack([torch.from_numpy(outputs.track_confidence_pred) for outputs in mv_outputs], dim=0)
        track_vis_pred = track_vis_pred & (track_confidence_pred >= cfg["output_track_conf_thresh"])

        pointmap_h = mv_outputs[0].pointmap_h
        pointmap_w = mv_outputs[0].pointmap_w

        image = torch.stack(
            [torch.from_numpy(outputs.pointmap_color[:, [2, 1, 0]].reshape(pointmap_h, pointmap_w, 3)) for outputs in mv_outputs],
            dim=0,
        )
        visualize_tracks_on_images(
            image[None],
            track_pred[None],
            track_vis_mask=track_vis_pred[None],
            out_dir=track_out_dir,
            image_format="HWC",  # "CHW" or "HWC"
            normalize_mode="",
            cmap_name="hsv",
        )

    if gt_out_dir is not None and mv_outputs[0].track_gt is not None:
        os.makedirs(gt_out_dir, exist_ok=True)
        track_gt = torch.stack([torch.from_numpy(outputs.track_gt) for outputs in mv_outputs], dim=0)
        track_vis = torch.stack([torch.from_numpy(outputs.track_vis) for outputs in mv_outputs], dim=0)
        pointmap_h = mv_outputs[0].pointmap_h
        pointmap_w = mv_outputs[0].pointmap_w

        image = torch.stack(
            [torch.from_numpy(outputs.pointmap_color[:, [2, 1, 0]].reshape(pointmap_h, pointmap_w, 3)) for outputs in mv_outputs],
            dim=0,
        )
        visualize_tracks_on_images(
            image[None],
            track_gt[None],
            track_vis_mask=track_vis[None],
            out_dir=gt_out_dir,
            image_format="HWC",  # "CHW" or "HWC"
            normalize_mode="",
            cmap_name="hsv",
            prefix="gt_",
        )


def vis_match_results(cfg, mv_outputs, match_out_dir, data_idx, frame_num, view_num, gt_out_dir=None, overlap_act=False):
    
    for frame_index in range(frame_num):
        for view_index in range(view_num):
            index = frame_index * view_num + view_index
            # gt visualize
            warp_gt_path = os.path.join(gt_out_dir, f"{data_idx:06d}", f"f{frame_index:03d}_v{view_index:03d}")
            
            cur_dir = os.path.join(match_out_dir, f"f{frame_index:03d}_v{view_index:03d}")
            if warp_gt_path is not None and (not os.path.exists(warp_gt_path)):
                visualize_gt = True
            else:
                visualize_gt = False
            match_results = mv_outputs[index].dense_matching
            if match_results is None or len(match_results) == 0:
                continue
            os.makedirs(cur_dir, exist_ok=True)
            for pair_i, outputs in enumerate(match_results):
                image0 = outputs.image0[0]
                image1 = outputs.image1[0]
                warp = outputs.warp
                overlap = outputs.overlap
                warp_coarse = outputs.warp_coarse
                overlap_coarse = outputs.overlap_coarse
                pred_conf = outputs.pred_covariance[..., -1]
                
                warp_gt = outputs.warp_gt
                overlap_gt = outputs.overlap_gt

                if overlap_act and overlap is not None:
                    overlap = overlap.sigmoid()
                if overlap_act and overlap_coarse is not None:
                    overlap_coarse = overlap_coarse.sigmoid()
                
                p1 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
                p2 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                
                # gt
                if visualize_gt:
                    os.makedirs(warp_gt_path, exist_ok=True)
                    save_warp(p1, p2, warp_gt, overlap_gt.float(),
                        warp_gt_path, "warp_gt", pair_i
                    )
                # Prediction
                if warp is not None:
                    save_warp(p1, p2, warp, overlap, cur_dir, "warp", pair_i)
                
                    if pred_conf is not None:
                        overlap = torch.logical_and(overlap, (pred_conf > 0)[..., None]).float()
                        save_warp(p1, p2, warp, overlap, cur_dir, "warp_conf", pair_i)

                if warp_coarse is not None:
                    if overlap is not None:
                        save_warp(p1, p2, warp_coarse, overlap, cur_dir, "coarse_warp", pair_i)
                    else:
                        save_warp(p1, p2, warp_coarse, overlap_coarse, cur_dir, "coarse_warp", pair_i)
