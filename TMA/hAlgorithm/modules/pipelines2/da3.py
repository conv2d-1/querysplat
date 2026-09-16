import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines2.base import Pipeline
from hAlgorithm.modules.pipelines2.utils.outputs import ReconstructOutput
from hAlgorithm.modules.pipelines2.utils.save_outputs import save_mv_outputs
from hAlgorithm.modules.pipelines2.utils.visualize_mv import (
    vis_camera_results,
    vis_extra_local_results,
    vis_glb_results,
    vis_images,
    vis_local_results,
    vis_render,
    vis_render_video,
    vis_track_results,
)
from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame, get_rays_in_world_frame, recover_pinhole_intrinsics_from_ray_directions
from hAlgorithm.utils import instantiate_from_config


class DepthAnything3Pipeline(Pipeline):
    """Pipeline for Multi-View Feed-forward Reconstruction."""

    def __init__(
        self,
        # inputs name
        intrinsics_name=None,
        extrinsics_name=None,
        scale_name=None,
        prompt_depth_name=None,
        target_local_depth_name=None,
        target_global_points_name=None,
        target_depth_mask_name=None,
        target_invalid_mask_name=None,
        align_name=None,
        save_output_cfg=None,
        forward_with_camera=False,
        **kwargs,
    ):
        super(DepthAnything3Pipeline, self).__init__(**kwargs)

        # inputs name
        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.scale_name = scale_name
        self.prompt_depth_name = prompt_depth_name
        self.target_local_depth_name = target_local_depth_name
        self.target_global_points_name = target_global_points_name
        self.target_depth_mask_name = target_depth_mask_name
        self.target_invalid_mask_name = target_invalid_mask_name
        self.align_name = align_name
        self.forward_with_camera = forward_with_camera

        self.save_output_cfg = dict(
            save_everything=False,
            save_output_conf=True,
            save_gaussians=True,
            save_render_results=True,
            save_render_video=False,
            save_render_video_with_normalize_c2w=True,
            save_glb_results=True,
            save_glb_sf_results=False,
            save_glb2local_results=False,
            save_local2glb_results=True,
            save_cameras=True,
            save_local_results=False,
            save_extra_local_results=False,
            save_track_results=True,
            save_filtered_results=True,
            save_normal=True,
            save_normal_vis=False,
            save_output_only_local_glb=False,
            save_name_match_rgb=True,
            output_normalize_cameras=True,
            output_match_input_res=True,
            output_conf_ratio=0.2,
            output_intrinsics_from_ray=False,
            output_colmap_format=False,
            gt_out_dir="gt",
        )
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)

        if self.save_output_cfg["save_everything"]:
            for key in self.save_output_cfg:
                if isinstance(self.save_output_cfg[key], bool) and key.startswith("save_"):
                    if key in ["save_output_only_local_glb"]:
                        self.save_output_cfg[key] = False
                    else:
                        self.save_output_cfg[key] = True
    
    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        if ckpt_path is not None:
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state_dict is not None:
            res = self.load_state_dict(state_dict, strict=False)
            logging.info(f"Model parameters are loaded from {ckpt_path}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")


    def get_inputs(self, batch):
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter

        image = batch["image"].to(device=self.device)

        scale = intrinsics = ray_directions = extrinsics = ray_world = prompt_depth = None

        # if self.scale_name is not None and self.scale_name in batch:
        #     scale = batch[self.scale_name].to(self.device)[..., None, None, None]

        if self.intrinsics_name is not None and self.intrinsics_name in batch:
            intrinsics = batch[self.intrinsics_name].to(device=self.device)

        if self.extrinsics_name is not None and self.extrinsics_name in batch :
            extrinsics = batch[self.extrinsics_name].to(device=self.device)
            if scale is not None:
                extrinsics[..., :3, 3] = self.normalize(extrinsics[..., :3, 3], scale[..., 0, 0])

        target_local_depth = target_global_points = target_depth_mask = None

        if self.target_local_depth_name is not None and self.target_local_depth_name in batch:
            target_local_depth = batch[self.target_local_depth_name].to(device=self.device)
            if scale is not None:
                target_local_depth = self.normalize(target_local_depth, scale)

        if self.target_global_points_name is not None and self.target_global_points_name in batch:
            target_global_points = batch[self.target_global_points_name].to(device=self.device)
            if scale is not None:
                target_global_points = self.normalize(target_global_points, scale)

        if self.target_depth_mask_name is not None and self.target_depth_mask_name in batch:
            target_depth_mask = batch[self.target_depth_mask_name].to(self.device)

        target_normal = target_normal_mask = target_motion_mask = target_invalid_mask = None

        if self.target_invalid_mask_name is not None and self.target_invalid_mask_name in batch:
            target_invalid_mask = batch[self.target_invalid_mask_name].to(self.device)

        image_show = align_data = None

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                image_show = image_show.float().numpy()

            if self.save_output_cfg["output_match_input_res"] and self.align_name in batch:
                align_data = batch[self.align_name]

        return (
            name,
            total_iter,
            meta_data,
            image,
            intrinsics,
            extrinsics,
            scale,
            prompt_depth,
            target_local_depth,
            target_global_points,
            target_depth_mask,
            target_normal,
            target_normal_mask,
            target_motion_mask,
            target_invalid_mask,
            image_show,
            align_data,
            ray_directions,
            ray_world,
        )

    def train_step(self, batch):
        return None, None

    def postprocess(
        self,
        pred_local_points=None,
        pred_local_conf=None,
        pred_local_normal=None,
        pred_local_invalid_mask=None,
        pred_global_points=None,
        pred_global_conf=None,
        pred_extrinsics=None,
        pred_intrinsics=None,
        image=None,
        image_show=None,
        scale=None,
        prompt_depth=None,
        target_local_depth=None,
        target_global_points=None,
        target_depth_mask=None,
        target_normal=None,
        target_normal_mask=None,
        intrinsics=None,
        extrinsics=None,
        align_data=None,
    ):
        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        if image_show is not None and pred_local_points is not None:
            points_colors = image_show[0].transpose(1, 2, 0)
        else:
            points_colors = image[0].cpu().float().numpy().transpose(1, 2, 0)
            points_colors = (points_colors + 1) * 0.5 * 255

        if pred_local_points is not None:
            points_h, points_w = pred_local_points.shape[-2:]

            if points_h != points_colors.shape[0] or points_w != points_colors.shape[1]:
                points_colors = cv2.resize(points_colors, dsize=(points_w, points_h), interpolation=cv2.INTER_LINEAR)
            points_colors = points_colors.reshape(-1, 3)

            pred_local_points = pred_local_points[0].cpu().float().numpy()
            pred_local_depth = pred_local_points[-1].clip(1e-3)
            pred_local_points = pred_local_points.reshape(3, -1).transpose(1, 0)

        else:
            pred_local_depth = None
            points_h, points_w = points_colors.shape[:2]
            points_colors = points_colors.reshape(-1, 3)

        if pred_local_points is not None and pred_local_conf is not None:
            pred_local_conf = pred_local_conf[0, 0].cpu().float().numpy()
            conf_thresh = np.percentile(pred_local_conf.reshape(-1), self.save_output_cfg["output_conf_ratio"] * 100)
            filtered_pred_local_points = pred_local_points[pred_local_conf.reshape(-1) > conf_thresh]
            filtered_points_colors = points_colors[pred_local_conf.reshape(-1) > conf_thresh]
        else:
            filtered_pred_local_points = filtered_points_colors = None

        if self.save_output_cfg["output_match_input_res"] and pred_local_depth is not None and align_data is not None:
            align_data = align_data[0].numpy()
            h, w = align_data.shape[:2]
            pred_local_depth = cv2.resize(pred_local_depth, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

            if pred_local_conf is not None:
                pred_local_conf = cv2.resize(pred_local_conf, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        if pred_local_normal is not None:
            pred_local_normal = pred_local_normal[0].cpu().float().numpy().transpose(1, 2, 0)

        if pred_local_invalid_mask is not None:
            pred_local_invalid_mask = pred_local_invalid_mask[0, 0].cpu().float().numpy()

        if pred_global_points is not None:
            pred_global_points = pred_global_points[0].detach().cpu().float().numpy().transpose(1, 2, 0)

        # Add multi-view confidence predictions if available
        if pred_global_conf is not None:
            pred_global_conf = pred_global_conf[0, 0].detach().cpu().float().numpy()

        # Add predicted extrinsics if available
        if pred_extrinsics is not None:
            pred_extrinsics = pred_extrinsics[0].detach().cpu().float().numpy()

        # Add predicted intrinsics if available
        if pred_intrinsics is not None:
            pred_intrinsics = pred_intrinsics[0].detach().cpu().float().numpy()

        if pred_local_points is not None and pred_extrinsics is not None and pred_intrinsics is not None:
            local2glb_points = (
                np.linalg.inv(pred_extrinsics)
                @ np.concatenate(
                    [pred_local_points, np.ones([pred_local_points.shape[0], 1])],
                    axis=-1,
                    dtype=np.float32,
                ).T
            ).T[:, :3]
        else:
            local2glb_points = None

        if scale is not None:
            scale = float(scale.squeeze().cpu())

        if prompt_depth is not None:
            prompt_h, prompt_w = prompt_depth.shape[-2:]
            prompt_depth = prompt_depth[0].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
        else:
            prompt_h = prompt_w = None

        if target_local_depth is not None:
            target_local_depth = target_local_depth[0].cpu().float().numpy().reshape(3, -1).transpose(1, 0)

        if target_global_points is not None:
            target_global_points = target_global_points[0].cpu().float().numpy().reshape(3, -1).transpose(1, 0)

        if target_depth_mask is not None:
            target_depth_mask = target_depth_mask[0].cpu().numpy()

        if intrinsics is not None:
            intrinsics = intrinsics[0].cpu().numpy()

        if extrinsics is not None:
            extrinsics = extrinsics[0].cpu().numpy()

        return ReconstructOutput(
            pointmap_color=points_colors,
            pointmap_h=points_h,
            pointmap_w=points_w,
            pointmap=pred_local_points,
            depth_align=pred_local_depth,
            confidence=pred_local_conf,
            local2glb_pointmap=local2glb_points,
            local2glb_confidence=pred_local_conf,
            filtered_pointmap=filtered_pred_local_points,
            filtered_pointmap_color=filtered_points_colors,
            glb_mv_pointmap=pred_global_points,
            glb_mv_confidence=pred_global_conf,
            extrinsics_pred=pred_extrinsics,
            intrinsics_pred=pred_intrinsics,
            prompt_scale=scale,
            prompt_pointmap=prompt_depth,
            prompt_h=prompt_h,
            prompt_w=prompt_w,
            pointmap_gt=target_local_depth,
            pointmap_gt_global=target_global_points,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            normal=pred_local_normal,
            normal_gt=target_normal,
            normal_mask=target_normal_mask,
            depth_mask=target_depth_mask,
            invalid_mask=pred_local_invalid_mask,
        )

    @torch.no_grad()
    def infer(self, **batch):
        self.eval()

        (
            name,
            total_iter,
            meta_data,
            image,
            intrinsics,
            extrinsics,
            scale,
            prompt_depth,
            target_local_depth,
            target_global_points,
            target_depth_mask,
            target_normal,
            target_normal_mask,
            target_motion_mask,
            target_invalid_mask,
            image_show,
            align_data,
            ray_directions,
            ray_world,
        ) = self.get_inputs(batch)

        if "use_amp" in batch and batch["use_amp"]:
            with torch.autocast("cuda", enabled=bool(batch["use_amp"]), dtype=batch["amp_dtype"]):
                results = self.model(
                    image,
                    intrinsics=intrinsics if self.forward_with_camera else None,
                    extrinsics=extrinsics if self.forward_with_camera else None,
                )
        else:
            results = self.model(
                image,
                intrinsics=intrinsics if self.forward_with_camera else None,
                extrinsics=extrinsics if self.forward_with_camera else None,
            )

        # Extract predictions from the model outputs
        pred_local_depth = pred_local_conf = pred_extrinsics = pred_intrinsics = pred_local_invalid_mask = None
        if "depth" in results:
            pred_local_depth = results.get("depth").unsqueeze(2)
        if "depth_conf" in results:
            pred_local_conf = results.get("depth_conf").unsqueeze(2)
        pred_extrinsics = results.get("extrinsics", None)
        pred_intrinsics = results.get("intrinsics", None)
        if "sky" in results:
            # magic number from DA3 code
            pred_local_invalid_mask = (results["sky"] > 0.3).unsqueeze(2)

        if pred_local_depth is not None:
            from hAlgorithm.modules.models2.external.depth_anything_3.utils.alignment import least_squares_scale_scalar
            scale_factor = least_squares_scale_scalar(target_local_depth[:, :, -1][target_depth_mask.squeeze(2)], pred_local_depth[:, :, -1][target_depth_mask.squeeze(2)])
            pred_local_depth *= scale_factor

        if pred_extrinsics is not None:
            pred_extrinsics[:, :, :3, 3] *= scale_factor

            B, V = pred_extrinsics.shape[:2]
            pred_extrinsics = torch.cat([pred_extrinsics, pred_extrinsics.new_zeros(B, V, 1, 4)], dim=-2)
            pred_extrinsics[:, :, 3, 3] = 1

        gaussians = results.get("ffgs_gaussians")
        render_rgb = results.get("ffgs_render_rgb")
        render_depth = results.get("ffgs_render_depth")
        # render_normal = results.get("ffgs_render_normal")
        if pred_local_depth is None:
            pred_local_depth = results.get("ffgs_depth")
            pred_local_conf = results.get("ffgs_confidence")

        if prompt_depth is not None and scale is not None:
            prompt_depth = self.denormalize(prompt_depth, scale=scale)
        if target_local_depth is not None and scale is not None:
            target_local_depth = self.denormalize(target_local_depth, scale=scale)
        if target_global_points is not None and scale is not None:
            target_global_points = self.denormalize(target_global_points, scale=scale)
        if extrinsics is not None and scale is not None:
            extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], scale=scale[..., 0, 0])

        if pred_local_depth is not None:
            if pred_local_depth.shape[-3] == 1:
                if pred_intrinsics is not None:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=pred_intrinsics)
                else:
                    pred_local_depth = self.depth_to_points(pred_local_depth, K=intrinsics)

            if scale is not None:
                pred_local_depth = self.denormalize(pred_local_depth, scale=scale)

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        def get_single_view_data(data, index):
            return data[:, index] if data is not None else None

        mv_outputs = []
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi
                single = self.postprocess(
                    pred_local_points=get_single_view_data(pred_local_depth, index),
                    pred_local_conf=get_single_view_data(pred_local_conf, index),
                    pred_extrinsics=get_single_view_data(pred_extrinsics, index),
                    pred_intrinsics=get_single_view_data(pred_intrinsics, index),
                    pred_local_invalid_mask=get_single_view_data(pred_local_invalid_mask, index),
                    image=get_single_view_data(image, index),
                    image_show=get_single_view_data(image_show, index),
                    scale=get_single_view_data(scale, index),
                    prompt_depth=get_single_view_data(prompt_depth, index),
                    target_local_depth=get_single_view_data(target_local_depth, index),
                    target_global_points=get_single_view_data(target_global_points, index),
                    target_depth_mask=get_single_view_data(target_depth_mask, index),
                    target_normal=get_single_view_data(target_normal, index),
                    target_normal_mask=get_single_view_data(target_normal_mask, index),
                    intrinsics=get_single_view_data(intrinsics, index),
                    extrinsics=get_single_view_data(extrinsics, index),
                    align_data=get_single_view_data(align_data, 0),
                )

                # Add Gaussian representation if available
                # Gaussian is shared across views, and maybe across frames in the future
                if gaussians is not None and index == 0:
                    single.gaussians = gaussians

                single.rgb = (image[0, index].permute(1, 2, 0).cpu().numpy() + 1) * 0.5

                # Add rendered RGB image if available
                if render_rgb is not None:
                    single.render_rgb = render_rgb[0, index].cpu().numpy().transpose(1, 2, 0)

                # Add rendered depth map if available
                if render_depth is not None:
                    single.render_depth = render_depth[0, index].cpu().numpy()

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                mv_outputs.append(single)

        return mv_outputs

    def get_out_dir(self, out_dir, data_idx=None):
        gs_out_dir = os.path.join(out_dir, "gaussians")
        glb_out_dir = os.path.join(out_dir, "glb")

        if data_idx is not None:
            camera_out_dir = os.path.join(out_dir, f"camera/{data_idx:06d}")
            mv_out_dir = os.path.join(out_dir, f"mvdepth/{data_idx:06d}")
            track_out_dir = os.path.join(out_dir, f"track/{data_idx:06d}")
        else:
            camera_out_dir = os.path.join(out_dir, "camera")
            mv_out_dir = os.path.join(out_dir, "mvdepth")
            track_out_dir = os.path.join(out_dir, "track")

        return gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir

    def get_gt_out_dir(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None, None, None

        abs_gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        _, gt_glb_out_dir, _, _, gt_track_out_dir = self.get_out_dir(abs_gt_out_dir, data_idx=None)

        return abs_gt_out_dir, gt_glb_out_dir, gt_track_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        data_idx = meta_data["data_idx"][0]

        gs_out_dir, glb_out_dir, camera_out_dir, mv_out_dir, track_out_dir = self.get_out_dir(out_dir, data_idx=data_idx)
        gt_out_dir, gt_glb_out_dir, gt_track_out_dir = self.get_gt_out_dir(out_dir)

        # Images Visualization
        if gt_out_dir is not None:
            vis_images(cfg=self.save_output_cfg, mv_outputs=outputs_list, gt_out_dir=gt_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num)

        # Local results Visualization
        if self.save_output_cfg["save_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_out_dir, meta_data=meta_data)
        if self.save_output_cfg["save_extra_local_results"]:
            os.makedirs(mv_out_dir, exist_ok=True)
            vis_extra_local_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, mv_out_dir=mv_out_dir, frame_num=frame_num, view_num=view_num, meta_data=meta_data)

        # Global results Visualization
        if self.save_output_cfg["save_glb_results"]:
            vis_glb_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, glb_out_dir=glb_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, gt_out_dir=gt_glb_out_dir)

        # Camera Visualization
        if self.save_output_cfg["save_cameras"]:
            vis_camera_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, camera_out_dir=camera_out_dir, frame_num=frame_num, view_num=view_num)

        # Gaussians Visualization
        if self.save_output_cfg["save_gaussians"] and outputs_list[0].gaussians is not None:
            vis_render(cfg=self.save_output_cfg, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx)
            vis_render_video(cfg=self.save_output_cfg, model=self.model, mv_outputs=outputs_list, gs_out_dir=gs_out_dir, data_idx=data_idx, frame_num=frame_num, view_num=view_num, device=self.device)

        # Track Visualization
        if self.save_output_cfg["save_track_results"] and outputs_list[0].track_pred is not None:
            os.makedirs(track_out_dir, exist_ok=True)
            vis_track_results(cfg=self.save_output_cfg, mv_outputs=outputs_list, track_out_dir=track_out_dir, data_idx=data_idx, gt_out_dir=gt_track_out_dir)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        save_mv_outputs(
            cfg=self.save_output_cfg,
            mv_outputs=outputs,
            meta_data=meta_data,
            out_dir=out_dir,
            output_meta_dict=output_meta_dict,
        )
    
    
    @torch.inference_mode()
    def simple_inference(
        self,
        frame_ids,
        view_ids,
        img_list,
        lidar_list=None,
        conf_list=None,
        invalid_mask_list=None,
        extrinsics_list=None,
        intrinsics_list=None,
        chunk_size=None,
        overlap=0,
        process_res=504,
        output_dir=None,
        save_points=True,
        **kwargs,
    ):
        import cv2
        import numpy as np
        import open3d as o3d
        import torch
        from PIL import Image
        from tqdm import tqdm

        from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve
        from hAlgorithm.modules.pipelines2.utils.save_points import save_confident_pointcloud_batch
        from hAlgorithm.modules.utils.parallel_utils import parallel_execution
        from hAlgorithm.modules.utils.pose_align import align_poses_umeyama

        from .utils.kosmo_util import _resize_longest_side, _resize_ixt, _normalize_image, rotation_matrix_to_quaternion

        w2c_outputs = dict()
        other_outputs = dict()
        ply_path_list = []

        print_progress = chunk_size is not None and chunk_size > 500

        # step1: chunk
        if chunk_size is None:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            if overlap >= chunk_size:
                raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
            if len(img_list) <= chunk_size:
                num_chunks = 1
                chunk_indices = [(0, len(img_list))]
            else:
                step = chunk_size - overlap
                num_chunks = (len(img_list) - overlap + step - 1) // step
                chunk_indices = []
                for i in range(num_chunks):
                    start_idx = i * step
                    end_idx = min(start_idx + chunk_size, len(img_list))

                    if i > 0 and i == num_chunks - 1:
                        cur_chunk_size = end_idx - start_idx
                        if cur_chunk_size < chunk_size // 2:
                            start_idx = end_idx - chunk_size // 2

                    chunk_indices.append((start_idx, end_idx))

        print(f"[DepthAnything3Pipeline] Processing {len(img_list)} images in {num_chunks} chunks of size {chunk_size} with {overlap} overlap")
        print(f"[DepthAnything3Pipeline] output_dir: {output_dir}")

        for chunk_idx in tqdm(range(len(chunk_indices))):
            start_idx, end_idx = chunk_indices[chunk_idx]
            chunk_image_paths = img_list[start_idx:end_idx]
            chunk_view_ids = view_ids[start_idx:end_idx] if isinstance(view_ids, (list, tuple)) else view_ids
            chunk_invalid_mask_paths = invalid_mask_list[start_idx:end_idx] if invalid_mask_list is not None else None
            chunk_extrinsics = extrinsics_list[start_idx:end_idx] if extrinsics_list is not None else None
            chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None

            def process_one(idx):
                image_path = chunk_image_paths[idx]
                invalid_mask = chunk_invalid_mask_paths[idx] if chunk_invalid_mask_paths is not None else None
                extrinsic = chunk_extrinsics[idx] if chunk_extrinsics is not None else None
                intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None

                pil_img = Image.open(image_path).convert("RGB")
                orig_w, orig_h = pil_img.size

                # Boundary resize
                pil_img = _resize_longest_side(pil_img, process_res)
                w, h = pil_img.size
                intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

                # Convert to tensor & normalize
                img_tensor = _normalize_image(pil_img)
                _, H, W = img_tensor.shape

                assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

                if invalid_mask is not None and invalid_mask[0] is not None:
                    invalid_mask = np.load(invalid_mask).astype(np.float32)
                    invalid_mask = cv2.resize(
                        invalid_mask,
                        dsize=(W, H),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    invalid_mask = invalid_mask > 0

                img_show = np.asarray(pil_img)

                return img_show, img_tensor, intrinsic, extrinsic, invalid_mask

            # step2: process
            outputs = parallel_execution(
                list(range(len(chunk_image_paths))),
                action=process_one,
                num_processes=8,
                print_progress=print_progress,
                sequential=False,
                desc=f"read chunk {chunk_idx}",
            )
            images_show, images, intrinsics, extrinsics, invalid_masks = zip(*outputs)

            images_show = np.stack(images_show, axis=0)
            images = torch.stack(images).unsqueeze(0).float()
            extrinsics = np.asarray(extrinsics)[None].astype(np.float32) if extrinsics is not None and extrinsics[0] is not None else None
            intrinsics = np.asarray(intrinsics)[None].astype(np.float32) if intrinsics is not None and intrinsics[0] is not None else None

            ori_extrinsics = extrinsics.copy()
            if extrinsics is not None:
                w2c = extrinsics
                base_c2w_pred = np.linalg.inv(w2c[:, 0:1])
                extrinsics = w2c @ base_c2w_pred

            # step4: inputs to cuda
            images = images.cuda()
            intrinsics = torch.from_numpy(intrinsics).float().cuda()
            extrinsics = torch.from_numpy(extrinsics).float().cuda()

            meta_data = dict(frames=[1], views=[images.shape[1]], input_width=[images.shape[-1]], input_height=[images.shape[-2]])
            meta_data["data_info"] = {"scene": [["kosmo"]]}

            # step4: model infer
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                results = self.model(
                    images,
                    scale=None,
                    # prompt_depth=lidar_depths,
                    intrinsics=intrinsics,
                    ray_directions=None,
                    w2c=extrinsics,
                    ray_world=None,
                    query_points=None,
                    meta_data=meta_data,
                )
                torch.cuda.empty_cache()

            depth = results["depth"].cpu().unsqueeze(2)
            confidence = results["depth_conf"].cpu().unsqueeze(2)
            pred_extrinsics = results["extrinsics"].cpu()
            pred_intrinsics = results.get("intrinsics").cpu()

            B, V, ch = pred_extrinsics.shape[:3]
            if ch == 3:
                pred_extrinsics = torch.cat([pred_extrinsics, pred_extrinsics.new_zeros(B, V, 1, 4)], dim=-2)
                pred_extrinsics[:, :, 3, 3] = 1

            # step5: align
            _, _, scale, aligned_extrinsics = align_poses_umeyama(
                ori_extrinsics[0],
                pred_extrinsics.numpy()[0],
                ransac=False,
                return_aligned=True,
                random_state=42,
            )

            depth *= scale
            points = self.depth_to_points(depth, K=pred_intrinsics, device=depth.device)

            depth = depth[0, :, 0].contiguous().numpy()
            points = points[0].permute(0, 2, 3, 1).contiguous().numpy()

            confidence = confidence[0, :, 0].numpy()

            # step6: save
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)

                if save_points:
                    points = points.reshape(points.shape[0], -1, 3)
                    points = np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=-1)
                    world_points = np.einsum("bij,bkj->bki", np.linalg.inv(aligned_extrinsics), points)[..., :3]
                    if isinstance(view_ids, int):
                        ply_path = os.path.join(output_dir, f"pcd/camera_{view_ids}/{chunk_idx}_pcd.ply")
                    else:
                        ply_path = os.path.join(output_dir, f"pcd/{chunk_idx}_pcd.ply")
                    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

                    conf_threshold = np.percentile(confidence, 30)

                    if invalid_masks is not None and invalid_masks[0] is not None:
                        invalid_masks = np.stack(invalid_masks, axis=0)
                        save_confident_pointcloud_batch(
                            points=world_points.reshape(-1, 3),  # shape: (H, W, 3)
                            colors=images_show.reshape(-1, 3),  # shape: (H, W, 3)
                            confs=confidence.reshape(-1),  # shape: (H, W)
                            output_path=ply_path,
                            conf_threshold=conf_threshold,
                            sample_ratio=0.01,
                            valid_mask=~invalid_masks.reshape(-1),
                        )
                    else:
                        save_confident_pointcloud_batch(
                            points=world_points.reshape(-1, 3),  # shape: (H, W, 3)
                            colors=images_show.reshape(-1, 3),  # shape: (H, W, 3)
                            confs=confidence.reshape(-1),  # shape: (H, W)
                            output_path=ply_path,
                            conf_threshold=conf_threshold,
                            sample_ratio=0.01,
                        )
                    ply_path_list.append(ply_path)

                def process_one(idx):
                    image_basename = os.path.basename(chunk_image_paths[idx])[:-4]
                    view_id = chunk_view_ids[idx] if isinstance(chunk_view_ids, (list, tuple)) else chunk_view_ids
                    cur_depth = depth[idx]
                    cur_conf = confidence[idx]

                    depth_path = os.path.join(output_dir, f"depth/camera_{view_id}/{image_basename}.npy")
                    if not os.path.exists(depth_path):
                        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                        np.save(depth_path, cur_depth.astype(np.float16))

                    conf_path = os.path.join(output_dir, f"conf/camera_{view_id}/{image_basename}.npy")
                    if not os.path.exists(conf_path):
                        os.makedirs(os.path.dirname(conf_path), exist_ok=True)
                        np.save(conf_path, cur_conf.astype(np.float16))

                    return depth_path, conf_path

                assert depth.shape[0] == len(chunk_image_paths)
                outputs = parallel_execution(
                    list(range(depth.shape[0])),
                    action=process_one,
                    num_processes=8,
                    print_progress=print_progress,
                    sequential=False,
                    desc="save chunk",
                )
                depth_paths, conf_paths = zip(*outputs)

                if "depth" not in other_outputs:
                    other_outputs["depth"] = dict()
                if "conf" not in other_outputs:
                    other_outputs["conf"] = dict()

                other_outputs["depth"].update(dict(zip(chunk_image_paths, depth_paths)))
                other_outputs["conf"].update(dict(zip(chunk_image_paths, conf_paths)))
                w2c_outputs.update({image_path: w2c for image_path, w2c in zip(chunk_image_paths, aligned_extrinsics)})

        aligned_extrinsics = np.stack([w2c_outputs[path] for path in img_list], axis=0)
        all_c2w = np.linalg.inv(aligned_extrinsics)

        if isinstance(view_ids, int):
            output_dir = os.path.join(output_dir, f"colmap/sparse/{view_ids}")
        else:
            output_dir = os.path.join(output_dir, f"colmap/sparse/0")
        os.makedirs(output_dir, exist_ok=True)

        # ----------- 生成 merge points ------------
        if len(ply_path_list) > 0:
            pcd = o3d.io.read_point_cloud(ply_path_list[0])
            for ply_path in ply_path_list[1:]:
                pcd += o3d.io.read_point_cloud(ply_path)
            pcd_path = os.path.join(output_dir, "points3D.ply")
            o3d.io.write_point_cloud(pcd_path, pcd)
        else:
            pcd_path = None

        # ----------- 生成 camera points ------------
        ply_path = os.path.join(output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_c2w)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = [255, 0, 0]
            for pose in all_c2w:
                position = pose[:3, 3]
                f.write(f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n")

        print(f"[DepthAnything3Pipeline] Camera poses visualization saved to {ply_path}")

        # ----------- 生成 images.txt ------------
        images_path = os.path.join(output_dir, "images.txt")
        os.makedirs(os.path.dirname(images_path), exist_ok=True)
        with open(images_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for i, (w2c, img_path) in enumerate(zip(aligned_extrinsics, img_list)):
                if pose is None:
                    continue
                R_w2c = w2c[:3, :3]
                t_w2c = w2c[:3, 3]

                # Convert R to quaternion (w, x, y, z)
                q = rotation_matrix_to_quaternion(R_w2c)
                qw, qx, qy, qz = q

                # Image file name (relative or basename)
                img_name = os.path.basename(img_path)

                # CAMERA_ID = i+1 (same as in cameras.txt)
                if isinstance(view_ids, int):
                    f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids)}/{img_name}\n")
                else:
                    f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids[i])}/{img_name}\n")
                f.write("\n")  # 第二行为特征点，留空

        print(f"[DepthAnything3Pipeline] COLMAP images.txt saved to {images_path}")

        return w2c_outputs, pcd_path, other_outputs

    @torch.inference_mode()
    def simple_inference_fake_sv(
        self,
        frame_ids,
        view_ids,
        img_list,
        lidar_list=None,
        conf_list=None,
        invalid_mask_list=None,
        extrinsics_list=None,
        intrinsics_list=None,
        chunk_size=None,
        overlap=0,
        process_res=504,
        output_dir=None,
        save_points=False,
        save_normal=True,
        conf_ratio=None,
        kosmo_mask=None,
        **kwargs,
    ):
        import cv2
        import numpy as np
        import open3d as o3d
        import torch
        from PIL import Image
        from tqdm import tqdm

        from hAlgorithm.modules.utils.parallel_utils import parallel_execution
        from .utils.kosmo_util import _resize_longest_side, _resize_ixt, _normalize_image

        other_outputs = dict()

        print_progress = chunk_size is not None and chunk_size > 500

        assert isinstance(view_ids, int)
        view_id = view_ids

        # step1: chunk
        if chunk_size is None:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            if overlap >= chunk_size:
                raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
            if len(img_list) <= chunk_size:
                num_chunks = 1
                chunk_indices = [(0, len(img_list))]
            else:
                step = chunk_size - overlap
                num_chunks = (len(img_list) - overlap + step - 1) // step
                chunk_indices = []
                for i in range(num_chunks):
                    start_idx = i * step
                    end_idx = min(start_idx + chunk_size, len(img_list))
                    chunk_indices.append((start_idx, end_idx))

        print(f"[DepthAnything3Pipeline] Processing {len(img_list)} images in {num_chunks} chunks of size {chunk_size} with {overlap} overlap")
        print(f"[DepthAnything3Pipeline] output_dir: {output_dir}")

        for chunk_idx in tqdm(range(len(chunk_indices))):
            start_idx, end_idx = chunk_indices[chunk_idx]
            chunk_image_paths = img_list[start_idx:end_idx]
            chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None

            def process_one(idx):
                image_path = chunk_image_paths[idx]
                intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None

                pil_img = Image.open(image_path).convert("RGB")
                orig_w, orig_h = pil_img.size

                # Boundary resize
                pil_img = _resize_longest_side(pil_img, process_res)
                w, h = pil_img.size
                intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

                # Convert to tensor & normalize
                img_tensor = _normalize_image(pil_img)
                _, H, W = img_tensor.shape

                assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

                img_show = np.asarray(pil_img)

                return img_show, img_tensor, intrinsic

            # step2: process
            outputs = parallel_execution(
                list(range(len(chunk_image_paths))),
                action=process_one,
                num_processes=min(8, len(chunk_image_paths)),
                print_progress=print_progress,
                sequential=False,
                desc=f"read chunk {chunk_idx}",
            )
            images_show, images, intrinsics = zip(*outputs)

            images_show = np.stack(images_show, axis=0)
            images = torch.stack(images).unsqueeze(0).float()

            intrinsics = np.asarray(intrinsics)[None]
            intrinsics = torch.from_numpy(intrinsics).float()

            # step4: inputs to cuda
            images = images.cuda()

            meta_data = dict(frames=[1], views=[images.shape[1]], input_width=[images.shape[-1]], input_height=[images.shape[-2]])
            meta_data["data_info"] = {"scene": [["kosmo"]]}

            # step4: model infer
            # import time
            # torch.cuda.synchronize(images.device)
            # start_time = time.time()

            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                results = self.model(
                    images,
                    scale=None,
                    prompt_depth=None,
                    intrinsics=None,
                    ray_directions=None,
                    w2c=None,
                    ray_world=None,
                    query_points=None,
                    meta_data=meta_data,
                )
            # torch.cuda.empty_cache()

            # torch.cuda.synchronize(images.device)
            # end_time = time.time()
            # print(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")

            depth = confidence = invalid_mask = None
            if "depth" in results:
                depth = results["depth"].cpu().unsqueeze(2)
            if "depth_conf" in results:
                confidence = results["depth_conf"].cpu().unsqueeze(2)
            if "sky" in results:
                invalid_mask = results["sky"].cpu().squeeze(1).numpy() > 0.3

            if save_points and depth is not None:
                points = self.depth_to_points(depth, K=intrinsics, device=depth.device)
                points = points[0].permute(0, 2, 3, 1).contiguous().numpy()
                points = points.reshape(-1, *points.shape[-3:])
            
            if depth is not None:
                depth = depth.numpy().reshape(-1, *depth.shape[-2:])
            if confidence is not None:
                confidence = confidence.numpy().reshape(-1, *confidence.shape[-2:])

            # step5: save
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)

                if kosmo_mask is not None:
                    cur_kosmo_mask = kosmo_mask[view_id]
                    if images_show.shape[-2:] != cur_kosmo_mask.shape:
                        cur_kosmo_mask = cv2.resize(cur_kosmo_mask.astype(np.float32), (images_show.shape[-2], images_show.shape[-3])) > 0
                    else:
                        cur_kosmo_mask = cur_kosmo_mask.astype(np.bool)

                    cur_kosmo_mask = cur_kosmo_mask | (images_show[0][..., 0] > 0) | (images_show[0][..., 1] > 0) | (images_show[0][..., 1] > 0)
                else:
                    cur_kosmo_mask = None

                def process_one(idx):
                    depth_path = conf_path = invalid_mask_path = None

                    image_basename = os.path.basename(chunk_image_paths[idx])[:-4]
                    if depth is not None:
                        cur_depth = depth[idx]

                        if confidence is not None:
                            cur_conf = confidence[idx]

                            if conf_ratio is not None:
                                valid_mask = cur_conf >= np.quantile(cur_conf, conf_ratio)
                                cur_depth[valid_mask] = 0

                        if cur_kosmo_mask is not None:
                            cur_depth[~cur_kosmo_mask] = 0

                        depth_path = os.path.join(output_dir, f"depth/camera_{view_id}/{image_basename}.npy")
                        if not os.path.exists(depth_path):
                            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                            np.save(depth_path, cur_depth.astype(np.float16))

                        conf_path = os.path.join(output_dir, f"conf/camera_{view_id}/{image_basename}.npy")
                        if confidence is not None and not os.path.exists(conf_path):
                            os.makedirs(os.path.dirname(conf_path), exist_ok=True)
                            np.save(conf_path, cur_conf.astype(np.float16))

                        if save_points:
                            cur_points = points[idx]
                            cur_images_show = images_show[idx]

                            if cur_kosmo_mask is not None:
                                cur_points *= cur_kosmo_mask[..., None].astype(cur_points.dtype)

                            points_path = os.path.join(output_dir, f"points/camera_{view_id}/{image_basename}.ply")
                            if not os.path.exists(points_path):

                                if conf_ratio is not None:
                                    cur_points = cur_points[valid_mask]
                                    cur_images_show = cur_images_show[valid_mask]

                                os.makedirs(os.path.dirname(points_path), exist_ok=True)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(cur_points.reshape(-1, 3))
                                pcd.colors = o3d.utility.Vector3dVector(cur_images_show.reshape(-1, 3) / 255.0)
                                o3d.io.write_point_cloud(points_path, pcd)
                        
                    if invalid_mask is not None:
                        cur_invalid_mask = invalid_mask[idx].astype(bool)
                        cur_images_show = images_show[idx].astype(float)[:, :, ::-1]

                        if cur_kosmo_mask is not None:
                            cur_invalid_mask[~cur_kosmo_mask] = 0

                        invalid_mask_path = os.path.join(output_dir, f"invalid_mask/camera_{view_id}/{image_basename}.npy")
                        if not os.path.exists(invalid_mask_path):
                            os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
                            np.save(invalid_mask_path, cur_invalid_mask)

                        cur_images_show[cur_invalid_mask, :] *= 0.5
                        cur_images_show[cur_invalid_mask, 2] += 255 * 0.5
                        cur_images_show = cur_images_show.astype(np.uint8)
                        invalid_mask_vis_path = os.path.join(output_dir, f"invalid_mask_vis/camera_{view_id}/{image_basename}.png")
                        if not os.path.exists(invalid_mask_vis_path):
                            os.makedirs(os.path.dirname(invalid_mask_vis_path), exist_ok=True)
                            cv2.imwrite(invalid_mask_vis_path, cur_images_show)
                        
                    return depth_path, conf_path, invalid_mask_path

                outputs = parallel_execution(
                    list(range(images.shape[1])),
                    action=process_one,
                    num_processes=min(8, images.shape[1]),
                    print_progress=print_progress,
                    sequential=False,
                    desc="save chunk",
                )
                depth_paths, conf_paths, invalid_mask_paths = zip(*outputs)

                if depth_paths[0] is not None:
                    if "depth" not in other_outputs:
                        other_outputs["depth"] = dict()
                    other_outputs["depth"].update(dict(zip(chunk_image_paths, depth_paths)))

                if conf_paths[0] is not None:
                    if "conf" not in other_outputs:
                        other_outputs["conf"] = dict()
                    other_outputs["conf"].update(dict(zip(chunk_image_paths, conf_paths)))

                if invalid_mask_paths[0] is not None:
                    if "invalid_mask" not in other_outputs:
                        other_outputs["invalid_mask"] = dict()
                    other_outputs["invalid_mask"].update(dict(zip(chunk_image_paths, invalid_mask_paths)))

        return None, None, other_outputs


