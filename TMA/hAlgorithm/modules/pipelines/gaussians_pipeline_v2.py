import json
import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from einops import pack
from tqdm import tqdm

from hAlgorithm.modules.pipelines.visualize import save_video, save_image
from hAlgorithm.modules.utils.gaussians.camera import Camera, CameraList
from hAlgorithm.modules.utils.gaussians.camera_trajectory import (
    interpolate_intrinsics,
    interpolate_poses_spline,
)
from hAlgorithm.modules.utils.image_utils import get_img_grad_weight
from hAlgorithm.utils import apply_color_map, grid_images, instantiate_from_config

from .outputs import ReconstructOutput
from .prompt_pointmap_pipeline_v2 import PromptPointMapPipeline


class Gaussian_Finetuning_PipelineV2(PromptPointMapPipeline):
    """
    Pipeline for finetuning 3DGS from feedforward GS with input novel images.
    """

    def __init__(
        self,
        lpips_loss=None,
        rgb_l1_loss=None,
        ssim_loss=None,
        render_normal_loss=None,
        render_depth_loss=None,
        render_normal_match_loss=True,
        extrinsics_name="extrinsics_reff",
        depth_loss_end=None,
        normal_loss_end=None,
        normal_match_start=3000,
        depth_loss_step=1,
        object_mask_name=None,
        mesh_voxel_size=0.001,
        mesh_frame_step=1,
        mesh_max_depth=5,
        **kwargs,
    ):
        super(Gaussian_Finetuning_PipelineV2, self).__init__(
            extrinsics_name=extrinsics_name, **kwargs
        )

        self.output_type = ReconstructOutput
        self.extrinsics_c2w = self.model.extrinsics_c2w

        self.object_mask_name = object_mask_name

        self.rgb_l1_loss = instantiate_from_config(rgb_l1_loss)
        self.ssim_loss = instantiate_from_config(ssim_loss)
        self.lpips_loss = instantiate_from_config(lpips_loss)
        self.render_normal_loss = instantiate_from_config(render_normal_loss)
        self.render_depth_loss = instantiate_from_config(render_depth_loss)
        self.render_normal_match_loss = render_normal_match_loss

        self.normal_match_start = normal_match_start
        self.depth_loss_end = depth_loss_end
        self.normal_loss_end = normal_loss_end
        self.depth_loss_step = depth_loss_step

        self.mesh_voxel_size = mesh_voxel_size
        self.mesh_frame_step = mesh_frame_step
        self.mesh_max_depth =  mesh_max_depth

        # NOTE
        self.cameras = CameraList()

    def set_cameras(self, datas, device=None):
        self.cameras.append(
            datas,
            extrinsics_name=self.extrinsics_name,
            intrinsics_name=self.intrinsics_name,
            device=device,
        )

    def get_camera(self, frame_id, view_id):
        return self.cameras.get_camera(frame_id=frame_id, view_id=view_id)

    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state_dict is not None:
            logging.info(f"Model parameters are loaded from {ckpt_path}")
            for key, val in state_dict.items():
                if "gaussian_parameters" in key:
                    base_val = getattr(
                        self.model.gaussian_parameters, key.replace("gaussian_parameters.", "")
                    )
                    new_val = torch.nn.Parameter(
                        val.to(base_val.device, base_val.dtype), requires_grad=True
                    )
                    setattr(
                        self.model.gaussian_parameters,
                        key.replace("gaussian_parameters.", ""),
                        new_val,
                    )
                    logging.info(f"gaussian_parameters {key}, {base_val.shape}->{val.shape}")
                else:
                    setattr(self.model, key, val)
                    logging.info(f"model load {key}, {val}")

    def depth_to_point(self, depth, K, device=None, cache=True):
        if depth.ndim == 4:
            return super().depth_to_point(depth, K, device=device, cache=cache)

        device = device or self.device
        B, N, C, H, W = depth.shape
        cache_key = f"depth_to_point_{H}_{W}"

        if cache_key in self.cache_dict:
            points = self.cache_dict[cache_key]
            if points.device != device:
                points.to(device)
        else:
            grid_x, grid_y = torch.meshgrid(
                torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy"
            )
            points = (
                torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
                .reshape(3, -1)
                .float()
                .to(device)
            )
            if cache:
                self.cache_dict[cache_key] = points
        rays_d = K.inverse() @ points  # (B, 3, HW)
        pts = depth.flatten(3) * rays_d
        depth = pts.reshape(B, N, 3, H, W)
        return depth

    def get_reconstruct_loss(
        self,
        name,
        image,
        depth,
        render_rgb,
        render_depth,
        render_normal=None,
        valid_mask=None,
        rgb_valid_mask=None,
        iterations=None,
    ):
        """
        Computes the reconstruction loss between rendered and target RGB images.

        :param name: Name of the current sample.
        :param image: Target RGB image tensor.
        :param depth: Target depth map tensor.
        :param render_rgb: Rendered RGB image tensor.
        :param render_depth: Rendered depth map tensor.
        :return: Tuple containing total loss and a dictionary of individual losses.
        """
        # rgb = (image.float() + 1) * 0.5
        rgb = image.float()

        if depth is not None:
            depth = depth.permute(0, 1, 3, 4, 2).contiguous()
            depth = depth.reshape(-1, *depth.shape[-3:]).float()

        if render_rgb is not None:
            render_rgb = render_rgb.float()

        if render_depth is not None:
            render_depth = render_depth.permute(0, 1, 3, 4, 2).contiguous()
            render_depth = render_depth.reshape(-1, *render_depth.shape[-3:]).float()

        if render_normal is not None:
            render_normal = render_normal.reshape(-1, *render_normal.shape[-3:]).float()

        if valid_mask is not None:
            valid_mask = valid_mask.squeeze(2).bool()
            valid_mask = valid_mask.reshape(-1, *valid_mask.shape[-2:])
        else:
            valid_mask = render_depth.new_ones(1, *rgb.shape[-2:]).bool()

        loss, loss_dict = 0, dict()

        if self.rgb_l1_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_l1_loss = self.rgb_l1_loss(rgb, render_rgb, mask=rgb_valid_mask, name=name)
            loss += rc_l1_loss
            loss_dict["rc_l1_loss"] = rc_l1_loss

        if self.ssim_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_ssim_loss = self.ssim_loss(rgb, render_rgb, mask=rgb_valid_mask, name=name)
            loss += rc_ssim_loss
            loss_dict["rc_ssim_loss"] = rc_ssim_loss

        if self.lpips_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_lpips_loss = self.lpips_loss(rgb, render_rgb, mask=rgb_valid_mask, name=name)
            loss += rc_lpips_loss
            loss_dict["rc_lpips_loss"] = rc_lpips_loss

        if (
            self.render_normal_loss is not None
            and render_normal is not None
            and render_normal.requires_grad
        ):
            assert depth.shape[-1] == 3
            # depth_norm: [BS, H, W, 3]
            # render_normal: [BS, 3, H, W]
            # valid_mask: [BS, H, W]
            rc_norm_loss = self.render_normal_loss(
                target=depth, prediction=render_normal, mask=valid_mask, name=name
            )
            loss += rc_norm_loss
            loss_dict["rc_norm_loss"] = rc_norm_loss

        if (
            self.render_depth_loss is not None
            and render_depth is not None
            and render_depth.requires_grad
        ):
            if self.depth_loss_end is None or iterations < self.depth_loss_end:
                rc_dpt_loss = self.render_depth_loss(
                    target=depth, prediction=render_depth, mask=valid_mask, name=name
                )
                loss += rc_dpt_loss
                loss_dict["rc_dpt_loss"] = rc_dpt_loss

        return loss, loss_dict

    def train_step(self, batch, iterations):
        """
        Executes a single training step using a batch of data.

        Parameters:
        - batch (dict): A dictionary containing the batch of data, including images, depth information, and masks.
        - optimizer: for prune and clone gaussains
        Returns:
        - loss (Tensor): The total loss for this training step.
        - loss_dict (dict): A dictionary containing individual loss components.
        """
        # Extract metadata from the batch
        meta_data = batch["meta_data"]

        # If the batch does not contain 'frames' or 'views', defer to the superclass implementation
        assert "frames" in meta_data and "views" in meta_data

        self.train()

        with torch.no_grad():
            name = batch["meta_data"]["name"][0]
            image = batch["image"].to(device=self.device, dtype=self.dtype)

            # Unpack necessary inputs from the batch
            target = valid_mask = None
            if self.target_name is not None and self.target_name in batch:
                target = batch[self.target_name].to(device=self.device)
            if self.target_mask_name is not None and self.target_mask_name in batch:
                valid_mask = batch[self.target_mask_name].to(self.device)

        frame_id = meta_data["frame_id"]
        view_id = meta_data["view_id"]
        camera = self.get_camera(frame_id=frame_id, view_id=view_id)
        cameras_batch = [[camera]]

        results = self.model(cameras_batch=cameras_batch)

        # Extract predictions from the model output
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)
        render_normal = results.get("render_normal", None)
        render_depth_normal = results.get("depth_normal", None)

        if target is not None and target.shape[-3] == 1:
            target = self.depth_to_point(target, camera.K)

        if render_depth is not None:
            render_depth = render_depth.unsqueeze(-3)
            render_depth = self.depth_to_point(render_depth, camera.K)

        total_loss, total_loss_dict = 0, dict()
        # Compute reconstruction loss
        loss_rc, loss_dict_rc = self.get_reconstruct_loss(
            name=name,
            image=image,
            depth=target,  #  Target
            render_rgb=render_rgb,
            render_depth=render_depth,
            render_normal=render_normal,
            valid_mask=valid_mask,
            iterations=iterations,
        )
        total_loss += loss_rc
        total_loss_dict.update(loss_dict_rc)

        # 增加normal loss和multi_view loss
        if self.render_normal_match_loss and iterations > self.normal_match_start:
            normal = results.get("render_normal", None)
            depth_normal = results.get("depth_normal", None)
            if normal is not None:
                image_weight = 1.0 - get_img_grad_weight(image[0, 0])
                image_weight = (image_weight).clamp(0, 1).detach() ** 2
                normal_loss = (
                    0.015
                    * (image_weight * (((depth_normal[0, 0] - normal[0, 0])).abs().sum(0))).mean()
                )
                total_loss += normal_loss
                total_loss_dict["rc_normal_match_loss"] = normal_loss

        # 增加多视图一致性监督 TODO 需要该datasetbatch 额外增加一个camera ref

        return total_loss, total_loss_dict, results

    def render_custom(self, w2c, intrinsics, h, w):
        batch_size, views, _, _ = w2c.shape
        cameras_batch = []
        for b in range(batch_size):
            cameras = []
            for v in range(views):
                cameras.append(Camera(w2c[b, v], intrinsics[b, v], h, w))
            cameras_batch.append(cameras)

        # gaussian model to render
        results = self.model(cameras_batch=cameras_batch)
        return results

    @torch.no_grad()
    def infer(self, batch_list, return_cameras=False):
        # Set the model to evaluation mode.
        self.eval()

        render_rgb = []
        render_depth = []
        render_normal = []
        images = []
        meta_data_list = []
        camera_list = []
        object_mask_list = []
        extrinsics_list = []
        intrinsics_list = []

        for batch in batch_list:
            if self.object_mask_name is not None and self.object_mask_name in batch:
                object_mask = batch[self.object_mask_name].squeeze()  # [H,W]
                object_mask_list.append(object_mask)

            images.append(batch["image"])

            # if self.extrinsics_name in batch:
            #     extrinsics_list.append(batch[self.extrinsics_name])
            # if self.intrinsics_name in batch:
            #     intrinsics_list.append(batch[self.intrinsics_name])

            # Extract metadata from the batch input
            meta_data = batch["meta_data"]
            meta_data_list.append(meta_data)

            frame_id = meta_data["frame_id"]
            view_id = meta_data["view_id"]
            cameras = self.get_camera(frame_id=frame_id, view_id=view_id)
            camera_list.append(cameras)

            results = self.model(cameras_batch=[[cameras]], debug=False)

            # gaussians = results.get("gaussians", None)  # Gaussian representation of the scene
            curr_render_rgb = results.get("render_rgb", None)  # Rendered RGB image
            curr_render_depth = results.get("render_depth", None)  # Rendered depth map
            curr_render_normal = results.get("normal", None)

            if curr_render_rgb is not None:
                render_rgb.append(curr_render_rgb.cpu().numpy())
            if curr_render_depth is not None:
                render_depth.append(curr_render_depth.cpu().numpy())
            if curr_render_normal is not None:
                render_normal.append(curr_render_normal.cpu().numpy())

        gaussians = results["gaussians"]

        images = torch.cat(images, dim=1)

        if len(render_rgb) > 0:
            render_rgb = np.concatenate(render_rgb, axis=1)
        else:
            render_rgb = None

        if len(render_depth) > 0:
            render_depth = np.concatenate(render_depth, axis=1)
        else:
            render_depth = None

        if len(render_normal) > 0:
            render_normal = np.concatenate(render_normal, axis=1)
        else:
            render_normal = None

        # Process each frame and view combination
        outputs_list = []
        for index in range(images.shape[1]):
            single = self.output_type(
                gaussians=gaussians if index == 0 else None,
            )

            # Add predicted extrinsics and intrinsics
            meta_data = meta_data_list[index]
            frame_id = meta_data["frame_id"]
            view_id = meta_data["view_id"]

            # if len(extrinsics_list) > 0:
            #     single.extrinsics = extrinsics_list[index].squeeze().numpy()
            # if len(intrinsics_list) > 0:
            #     single.intrinsics = intrinsics_list[index].squeeze().numpy()

            camera = camera_list[index]
            extrinsics_pred = camera.world_view_transform.transpose(0, 1)
            intrinsics_pred = camera.K

            single.extrinsics_pred = extrinsics_pred.detach().cpu().numpy()
            single.intrinsics_pred = intrinsics_pred.detach().cpu().numpy()

            single.rgb = images[0, index].permute(1, 2, 0).numpy()

            if len(object_mask_list) > 0:
                single.object_mask = object_mask_list[index].numpy()

            # Add rendered RGB image if available
            if render_rgb is not None:
                single.render_rgb = render_rgb[0, index].transpose(1, 2, 0)

            # Add rendered depth map if available
            if render_depth is not None:
                single.render_depth = render_depth[0, index]

            if render_normal is not None:
                single.render_normal = render_normal[0, index]

            # Store frame and view indices
            single.frame_index = int(frame_id)
            single.view_index = int(view_id)
            single.total_index = index

            outputs_list.append(single)

        if return_cameras:
            return outputs_list, camera_list
        else:
            return outputs_list

    @torch.no_grad()
    def render_video_generic(
        self,
        trajectory_fn,
        h,
        w,
        n_interp: int = 30,
        loop_reverse: bool = True,
    ) -> None:

        c2w, intrinsics = trajectory_fn(n_interp)

        def depth_map(result):
            try:
                near = result[result > 0][:16_000_000].quantile(0.01)
                far = result.view(-1)[:16_000_000].quantile(0.99)
                result = 1 - (result - near) / (far - near)
            except:
                result[...] = 0
            return apply_color_map(result, "turbo")

        video = []
        for i in tqdm(range(c2w.shape[1]), desc="render video"):
            c2w_single = c2w[:, i : i + 1]
            intrinsics_single = intrinsics[:, i : i + 1]

            intrinsics_single[..., 0, :] *= w
            intrinsics_single[..., 1, :] *= h

            output_prob = self.render_custom(c2w_single.inverse(), intrinsics_single, h, w)

            rgb = output_prob["render_rgb"][0].cpu()
            show = [rgb]
            if output_prob.get("render_depth", None) is not None:
                depth_color = depth_map(output_prob["render_depth"][0].detach()).cpu()
                show.append(depth_color)

            if output_prob.get("render_normal", None) is not None:
                normal = (output_prob["render_normal"][0] + 1).cpu() * 0.5
                show.append(normal)

            show = torch.cat(show, dim=3).permute(0, 2, 3, 1)
            video.append(show)

        video = torch.cat(video, axis=0)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).numpy()

        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* h w c")[0]

        return video

    def render_video_interpolation(
        self, extrinsics, intrinsics, h, w, loop=False, loop_reverse=False
    ):

        intrinsics = intrinsics.clone()
        if not self.extrinsics_c2w:
            c2w = extrinsics.clone().inverse()
        else:
            c2w = extrinsics.clone()

        def trajectory_fn(n_interp):
            if loop:
                c2w_ = torch.cat([c2w, c2w[0:, 0:1]], dim=1)
            c2w_ = c2w
            b, v, _, _ = c2w_.shape
            c2w_target = interpolate_poses_spline(c2w_.reshape(b * v, 4, 4)[:, :3].cpu(), n_interp)
            c2w_target = c2w_target.reshape(b, -1, 4, 4).to(self.device).float()

            num_frames = b * c2w_target.shape[1]
            t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)

            intrinsics[..., 0, :] /= w
            intrinsics[..., 1, :] /= h
            intrinsics_target = interpolate_intrinsics(
                intrinsics[0, 0],
                intrinsics[0, -1],
                t,
            )
            intrinsics_target = intrinsics_target[None]
            return c2w_target, intrinsics_target

        return self.render_video_generic(
            trajectory_fn, h, w, n_interp=12, loop_reverse=loop_reverse
        )

    def vis_camera_results(self, out_dir, outputs_list, **kwargs):
        camera_data = defaultdict(dict)
        for name, camera in self.cameras.cameras.items():
            frame_id, view_id = self.cameras.camera_name_to_id(name)

            intrinsics = camera.K.detach().cpu().numpy().tolist()
            extrinsics = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy().tolist()

            img_h = int(camera.img_h.detach().cpu().numpy().reshape(-1))
            img_w = int(camera.img_w.detach().cpu().numpy().reshape(-1))

            camera_data[f"frame_id_{frame_id}"][f"view_id_{view_id}"] = dict(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                img_h=img_h,
                img_w=img_w,
            )

        save_path = os.path.join(out_dir, "cameras.json")
        with open(save_path, "w") as f:
            json.dump(camera_data, f, indent=2)

        # camera_data = defaultdict(dict)
        # for outputs in outputs_list:
        #     frame_id, view_id = outputs.frame_index, outputs.view_index

        #     intrinsics = outputs.intrinsics.tolist()
        #     extrinsics = outputs.extrinsics.tolist()

        #     camera_data[f"frame_id_{frame_id}"][f"view_id_{view_id}"] = dict(
        #         intrinsics=intrinsics,
        #         extrinsics=extrinsics,
        #     )
        
        # save_path = os.path.join(out_dir, "cameras_ori.json")
        # with open(save_path, "w") as f:
        #     json.dump(camera_data, f, indent=2)

    def visualize(self, outputs_list, meta_data, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        self.vis_camera_results(out_dir, outputs_list)

        if outputs_list[0].rgb is not None:
            save_path = os.path.join(os.path.dirname(os.path.dirname(out_dir)), f"merge_rgb.jpg")
            if os.path.exists(save_path):
                save_path = None
            else:
                logging.info(f"save: {save_path}")

            images = [(outputs.rgb[:, :, ::-1] * 255).astype(np.uint8) for outputs in outputs_list]
            merge_image = grid_images(
                save_path=save_path,
                images=images,
                col=min(4, len(images)),
            )

            if outputs_list[0].object_mask is not None and save_path is not None:
                save_path = os.path.join(
                    os.path.dirname(os.path.dirname(out_dir)), f"merge_mask_rgb.jpg"
                )
                mask_images = [
                    (images[i] * outputs_list[i].object_mask[..., None]).astype(np.uint8)
                    for i in range(len(images))
                ]
                grid_images(
                    save_path=save_path,
                    images=mask_images,
                    col=min(4, len(mask_images)),
                )

        if outputs_list[0].render_rgb is not None:
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"merge_render_rgb.jpg")
            render_rgb = [
                (np.clip(outputs.render_rgb[..., ::-1], 0.0, 1.0) * 255).astype(np.uint8)
                for outputs in outputs_list
            ]

            if outputs_list[0].object_mask is not None:
                mask_save_path = os.path.join(out_dir, f"merge_render_mask_rgb.jpg")
                mask_images = [
                    (render_rgb[i] * outputs_list[i].object_mask[..., None]).astype(np.uint8)
                    for i in range(len(images))
                ]
                grid_images(
                    save_path=mask_save_path,
                    images=mask_images,
                    col=min(4, len(mask_images)),
                )

            if outputs_list[0].rgb is None:
                grid_images(save_path, images=render_rgb, col=min(4, len(render_rgb)))
            else:
                merge_render = grid_images(
                    save_path=None, 
                    images=render_rgb, 
                    col=min(4, len(render_rgb))
                )

                diff = np.abs(merge_image / 255.0 - merge_render / 255.0).mean(axis=-1)
                diff = (apply_color_map(1.0 - diff, "turbo") * 255).astype(np.uint8)

                diff = np.concatenate([merge_render, diff], axis=0)
                cv2.imwrite(save_path, diff)

        if outputs_list[0].render_depth is not None:
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"merge_render_dpt.jpg")

            def colorize(depth):
                min_val, max_val = depth.min(), depth.max()
                depth_norm = 1 - (depth - min_val) / (max_val - min_val + 1e-9)
                depth_colored = apply_color_map(depth_norm, "turbo")
                return (depth_colored.clip(0, 1) * 255).astype(np.uint8)

            render_depth = [colorize(outputs.render_depth) for outputs in outputs_list]
            grid_images(save_path, images=render_depth)

        if outputs_list[0].gaussians is not None:
            gaussians = outputs_list[0].gaussians

            save_path = os.path.join(out_dir, "gaussians.ply")
            gaussians.export_ply().write(save_path)

            # render video
            if outputs_list[0].render_rgb is not None:
                h, w = outputs_list[0].render_rgb.shape[:2]
                extrinsics = (
                    torch.stack(
                        [torch.from_numpy(outputs.extrinsics_pred) for outputs in outputs_list],
                        dim=0,
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                intrinsics = (
                    torch.stack(
                        [torch.from_numpy(outputs.intrinsics_pred) for outputs in outputs_list],
                        dim=0,
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                video = self.render_video_interpolation(extrinsics, intrinsics, h, w)
                save_video(video, out_dir, "video", None, info=True)
    
    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        for i, outputs_sf in enumerate(outputs):
            out_dir = os.path.abspath(out_dir)

            scene = meta_data[i]["scene"][0]
            frame_id = int(meta_data[i]["frame_id"][0])
            view_id = int(meta_data[i]["view_id"][0])

            cur_out_dir = os.path.join(out_dir, scene)
            os.makedirs(cur_out_dir, exist_ok=True)

            render_rgb_path = None
            if outputs_sf.render_rgb is not None:
                render_rgb_path = save_image(
                    outputs_sf.render_rgb.copy(),
                    cur_out_dir,
                    f"render_rgb_{frame_id:06d}_view{view_id:06d}",
                    data_idx=None,
                    info=False,
                )

            camera = self.cameras.get_camera(frame_id=frame_id, view_id=view_id)
            intrinsics = camera.K.detach().cpu().numpy().tolist()
            extrinsics = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy().tolist()

            info = dict(
                view_id=view_id,
                rgb=meta_data[i]["data_info"][0]["rgb"],
                cam_in=[intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]],
                extrinsics=extrinsics,
            )
            if render_rgb_path is not None:
                info["render_rgb"] = render_rgb_path

            if "mf_files" not in output_meta_dict:
                output_meta_dict["mf_files"] = dict()
            if scene not in output_meta_dict["mf_files"]:
                output_meta_dict["mf_files"][scene] = []

            frame_ids = [d["frame_id"] for d in output_meta_dict["mf_files"][scene]]
            if len(frame_ids) == 0:
                index = None
            else:
                if frame_id in set(frame_ids):
                    index = frame_ids.index(frame_id)
                else:
                    index = None

            if index is None:
                output_meta_dict["mf_files"][scene].append(
                    dict(
                        frame_id=frame_id,
                        views=[info],
                    )
                )
            else:
                output_meta_dict["mf_files"][scene][index]["views"].append(info)


    def build_mesh(self, outputs_list, camera_list, out_dir):
        import open3d as o3d

        max_depth = self.mesh_max_depth
        voxel_size = self.mesh_voxel_size

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        camera_list = camera_list[::self.mesh_frame_step]
        outputs_list= outputs_list[::self.mesh_frame_step]
        image_list = [outputs.render_rgb * 255 for outputs in outputs_list]

        if outputs_list[0].object_mask is not None:
            depth_list = []
            for outputs in outputs_list:
                object_mask = outputs.object_mask
                render_depth = outputs.render_depth.copy()
                render_depth[~object_mask] = 0

                mean_depth = render_depth[render_depth > 0].mean()
                max_mean_depth = render_depth[render_depth > mean_depth].mean()
                render_depth[render_depth > max_mean_depth] = 0

                max_depth = max(max_depth, max_mean_depth)

                depth_list.append(render_depth)
        else:
            depth_list = [outputs.render_depth for outputs in outputs_list]

        for image, depth, camera in tqdm(
            zip(image_list, depth_list, camera_list),
            total=len(image_list),
            desc="TSDF Fusion progress",
        ):

            color = o3d.geometry.Image(np.ascontiguousarray(image.astype(np.uint8)))
            depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1000.0,
                depth_trunc=max_depth,
                convert_rgb_to_intensity=False,
            )

            pose = camera.world_view_transform.transpose(0, 1).cpu().numpy()

            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    int(camera.img_w.cpu()),
                    int(camera.img_h.cpu()),
                    float(camera.fx),
                    float(camera.fy),
                    float(camera.cx),
                    float(camera.cy),
                ),
                pose,
            )

        mesh = volume.extract_triangle_mesh()
        save_path = os.path.join(out_dir, "tsdf_fusion.ply")
        o3d.io.write_triangle_mesh(
            save_path,
            mesh,
            write_triangle_uvs=True,
            write_vertex_colors=True,
            write_vertex_normals=True,
        )
        logging.info(f"save, {save_path}")

        def post_process_mesh(mesh, cluster_to_keep=1):
            """
            Post-process a mesh to filter out floaters and disconnected parts
            """
            import copy

            logging.info(
                "post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep)
            )
            mesh_0 = copy.deepcopy(mesh)
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (
                    mesh_0.cluster_connected_triangles()
                )
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
            n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
            mesh_0.remove_unreferenced_vertices()
            mesh_0.remove_degenerate_triangles()
            return mesh_0

        try:
            mesh = post_process_mesh(mesh)
            save_path = os.path.join(out_dir, "tsdf_fusion_post.ply")
            o3d.io.write_triangle_mesh(
                save_path,
                mesh,
                write_triangle_uvs=True,
                write_vertex_colors=True,
                write_vertex_normals=True,
            )
            logging.info(f"save, {save_path}")
        except Exception as e:
            logging.error(e)
