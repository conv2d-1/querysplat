import logging
import os

import numpy as np
import torch
from einops import pack
from tqdm import tqdm

from hAlgorithm.modules.pipelines.visualize import (
    save_depth_map,
    save_image,
    save_video,
)
from hAlgorithm.modules.utils.gaussians.camera_trajectory import (
    interpolate_intrinsics,
    interpolate_poses_spline,
)
from hAlgorithm.modules.utils.image_utils import get_img_grad_weight
from hAlgorithm.utils import apply_color_map, grid_images, instantiate_from_config

from .outputs import ReconstructOutput
from .prompt_pointmap_pipeline_v2 import PromptPointMapPipeline

# from hAlgorithm.utils import cuda_timing_context


class Gaussian_Finetuning_Pipeline(PromptPointMapPipeline):
    """
    Pipeline for finetuning 3DGS from feedforward GS with input novel images.
    """

    def __init__(
        self,
        lpips_loss=None,
        rgb_l1_loss=None,
        ssim_loss=None,
        extrinsics_name="extrinsics_reff",
        normal_match_start=3000,
        **kwargs,
    ):
        super(Gaussian_Finetuning_Pipeline, self).__init__(
            extrinsics_name=extrinsics_name, **kwargs
        )

        self.output_type = ReconstructOutput
        self.extrinsics_c2w = self.model.extrinsics_c2w
        self.normal_match_start = normal_match_start

        self.rgb_l1_loss = instantiate_from_config(rgb_l1_loss)
        self.ssim_loss = instantiate_from_config(ssim_loss)
        self.lpips_loss = instantiate_from_config(lpips_loss)

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

    def get_reconstruct_loss(
        self,
        name,
        image,
        depth,
        render_rgb,
        render_depth,
        valid_mask=None,
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
        if render_rgb is not None:
            render_rgb = render_rgb.float()
        if valid_mask is not None:
            valid_mask = valid_mask.float()
        if render_depth is not None:
            render_depth = render_depth.float()

        loss, loss_dict = 0, dict()

        if self.rgb_l1_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_l1_loss = self.rgb_l1_loss(rgb, render_rgb, mask=None, name=name)
            loss += rc_l1_loss
            loss_dict["rc_l1_loss"] = rc_l1_loss

        if self.ssim_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_ssim_loss = self.ssim_loss(rgb, render_rgb, mask=None, name=name)
            loss += rc_ssim_loss
            loss_dict["rc_ssim_loss"] = rc_ssim_loss

        if self.lpips_loss is not None and render_rgb is not None and render_rgb.requires_grad:
            rc_lpips_loss = self.lpips_loss(rgb, render_rgb, mask=None, name=name)
            loss += rc_lpips_loss
            loss_dict["rc_lpips_loss"] = rc_lpips_loss

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
            target = None
            valid_mask = None
            if self.target_name is not None and self.target_name in batch:
                target = batch[self.target_name].to(device=self.device)
            if self.target_mask_name is not None and self.target_mask_name in batch:
                valid_mask = batch[self.target_mask_name].to(self.device) 

            intrinsics = batch.get(self.intrinsics_name, None)
            extrinsics = batch.get(self.extrinsics_name, None)  # [n, 4, 4]

            if intrinsics is not None:
                intrinsics = intrinsics.to(device=self.device)
            if extrinsics is not None:
                extrinsics = extrinsics.to(device=self.device)

        results = self.model(
            image=image,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

        # Extract predictions from the model output
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)

        total_loss, total_loss_dict = 0, dict()
        # Compute reconstruction loss
        loss_rc, loss_dict_rc = self.get_reconstruct_loss(
            name=name,
            image=image,
            depth=target,  #  Target
            render_rgb=render_rgb,
            render_depth=render_depth,
            valid_mask=valid_mask,
        )
        total_loss += loss_rc
        total_loss_dict.update(loss_dict_rc)

        # scaling_reg_loss 视角culling后的scale算loss
        # gaussians = results.get("gaussians", None)
        # scaling_reg = 0.005*gaussians.get_scaling().prod(dim=1).mean()

        # total_loss += scaling_reg
        # loss_dict_rc["rc_scaling_loss"] = scaling_reg
        # total_loss_dict.update(loss_dict_rc)

        # 增加normal loss和multi_view loss
        if iterations > self.normal_match_start:
            normal = results.get("normal", None)
            depth_normal = results.get("depth_normal", None)
            if normal is not None:
                image_weight = 1.0 - get_img_grad_weight(image[0, 0])
                image_weight = (image_weight).clamp(0, 1).detach() ** 2
                normal_loss = (
                    0.015
                    * (image_weight * (((depth_normal[0, 0] - normal[0, 0])).abs().sum(0))).mean()
                )
                total_loss += normal_loss
                total_loss_dict["rc_normal_loss"] = normal_loss

        # 增加多视图一致性监督 TODO 需要该datasetbatch 额外增加一个camera ref

        return total_loss, total_loss_dict, results

    @torch.no_grad()
    def infer(self, batch_list):
        # Set the model to evaluation mode.
        self.eval()

        intrinsics_list = []
        extrinsics_list = []

        render_rgb = []
        render_depth = []
        render_normal = []
        images = []

        for batch in batch_list:
            # Extract metadata from the batch input
            meta_data = batch["meta_data"]  # 1,1,c,h,w

            # If "frames" or "views" are not in metadata, fallback to the superclass infer method
            assert "frames" in meta_data and "views" in meta_data

            image = batch["image"].to(device=self.device, dtype=self.dtype)
            images.append(batch["image"])

            intrinsics = batch[self.intrinsics_name].to(device=self.device)
            extrinsics = batch[self.extrinsics_name].to(device=self.device)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

            extrinsics = extrinsics.clone()
            if not self.extrinsics_c2w:
                extrinsics = extrinsics.inverse()

            h, w = image.shape[-2:]
            intrinsics = intrinsics.clone()
            intrinsics[..., 0, :] /= w
            intrinsics[..., 1, :] /= h

            # gaussian model to render
            results = self.model(
                hw=(h, w),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                only_rendering=True,
            )
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
        intrinsics = torch.cat(intrinsics_list, dim=1)
        extrinsics = torch.cat(extrinsics_list, dim=1)

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
        for index in range(intrinsics.shape[1]):
            single = self.output_type(
                intrinsics=(
                    intrinsics[:, index].cpu().squeeze(0).numpy()
                    if intrinsics is not None
                    else None
                ),
                extrinsics=(
                    extrinsics[:, index].cpu().squeeze(0).numpy()
                    if extrinsics is not None
                    else None
                ),
                gaussians=gaussians if index == 0 else None,
            )

            single.rgb = images[0, index].permute(1, 2, 0).numpy()

            # Add rendered RGB image if available
            if render_rgb is not None:
                single.render_rgb = render_rgb[0, index].transpose(1, 2, 0)

            # Add rendered depth map if available
            if render_depth is not None:
                single.render_depth = render_depth[0, index]

            if render_normal is not None:
                single.render_normal = render_normal[0, index]

            # Store frame and view indices
            single.frame_index = index
            single.view_index = 0
            single.total_index = index

            outputs_list.append(single)

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

        extrinsics, intrinsics = trajectory_fn(n_interp)

        def depth_map(result):
            try:
                near = result[result > 0][:16_000_000].quantile(0.01)
                far = result.view(-1)[:16_000_000].quantile(0.99)
                result = 1 - (result - near) / (far - near)
            except:
                result[...] = 0
            return apply_color_map(result, "turbo")

        video = []
        for i in tqdm(range(extrinsics.shape[1]), desc="render video"):
            output_prob = self.model(
                extrinsics=extrinsics[:, i : i + 1],
                intrinsics=intrinsics[:, i : i + 1],
                hw=(h, w),
                depth_mode="depth",
                only_rendering=True,
            )
            if output_prob["render_depth"] is not None:
                depth_color = depth_map(output_prob["render_depth"][0].detach()).cpu()
                rgb = output_prob["render_rgb"][0].cpu()
                rgb = torch.cat([rgb, depth_color], dim=3).permute(0, 2, 3, 1)

            else:
                rgb = output_prob["render_rgb"][0].cpu().permute(0, 2, 3, 1)

            video.append(rgb)

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
            extrinsics = extrinsics.clone().inverse()

        def trajectory_fn(n_interp):
            if loop:
                extrinsics_ = torch.cat([extrinsics, extrinsics[0:, 0:1]], dim=1)
            extrinsics_ = extrinsics
            b, v, _, _ = extrinsics_.shape
            extrinsics_target = interpolate_poses_spline(
                extrinsics_.reshape(b * v, 4, 4)[:, :3].cpu(), n_interp
            )
            extrinsics_target = extrinsics_target.reshape(b, -1, 4, 4).to(self.device).float()

            num_frames = b * extrinsics_target.shape[1]
            t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)

            intrinsics[..., 0, :] /= w
            intrinsics[..., 1, :] /= h
            intrinsics_target = interpolate_intrinsics(
                intrinsics[0, 0],
                intrinsics[0, -1],
                t,
            )
            intrinsics_target = intrinsics_target[None]
            return extrinsics_target, intrinsics_target

        return self.render_video_generic(
            trajectory_fn, h, w, n_interp=12, loop_reverse=loop_reverse
        )

    def visualize(self, outputs_list, meta_data, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "000000"), exist_ok=True)

        frame_num = len(outputs_list)
        data_idx = [int(meta_data_i["data_idx"][0]) for meta_data_i in meta_data]

        if outputs_list[0].gaussians is not None:
            gaussians = outputs_list[0].gaussians

            save_path = os.path.join(out_dir, "gaussians.ply")
            gaussians.export_ply().write(save_path)

            # render video
            if outputs_list[0].render_rgb is not None:
                h, w = outputs_list[0].render_rgb.shape[:2]
                extrinsics = (
                    torch.stack(
                        [torch.from_numpy(outputs.extrinsics) for outputs in outputs_list], dim=0
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                intrinsics = (
                    torch.stack(
                        [torch.from_numpy(outputs.intrinsics) for outputs in outputs_list], dim=0
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                video = self.render_video_interpolation(extrinsics, intrinsics, h, w)
                save_video(video, out_dir, "video", None, info=True)

        render_rgb_paths = []
        render_depth_paths = []
        for index in range(frame_num):
            prefix = f"index{index:03d}_"
            outputs = outputs_list[index]

            if outputs.render_rgb is not None:
                save_path = save_image(
                    outputs.render_rgb.copy(),
                    out_dir,
                    f"000000/{prefix}rgb",
                    data_idx[index],
                    info=False,
                )
                render_rgb_paths.append(save_path)

            if outputs.render_depth is not None:
                save_path = save_depth_map(
                    outputs.render_depth.copy(),
                    out_dir,
                    f"000000/{prefix}depth",
                    data_idx[index],
                    info=False,
                )
                render_depth_paths.append(save_path)

        if len(render_rgb_paths) > 0:
            save_path = os.path.join(out_dir, f"merge_render_rgb.jpg")
            grid_images(
                save_path=save_path,
                paths=render_rgb_paths,
                col=min(4, len(render_rgb_paths)),
            )

        if len(render_depth_paths) > 0:
            save_path = os.path.join(out_dir, f"merge_render_depth.jpg")
            grid_images(
                save_path=save_path,
                paths=render_depth_paths,
                col=min(4, len(render_depth_paths)),
            )

        if outputs_list[0].rgb is not None:
            save_path = os.path.join(os.path.dirname(os.path.dirname(out_dir)), f"merge_rgb.jpg")
            if not os.path.exists(save_path):
                images = [
                    (outputs.rgb[:, :, ::-1] * 255).astype(np.uint8) for outputs in outputs_list
                ]
                grid_images(
                    save_path=save_path,
                    images=images,
                    col=min(4, len(images)),
                )
