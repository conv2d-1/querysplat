import os

import torch

from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines.visualize import save_video
from hAlgorithm.modules.utils.gaussians.novel_view_mask import calculate_loss_mask

from .reconstruct_pipeline import ReconstructPipeline


class VGGTReconstructPipeline(ReconstructPipeline):
    """
    Pipeline for reconstructing 3D scenes from input images and depth information.
    """

    def __init__(self, **kwargs):
        super(VGGTReconstructPipeline, self).__init__(**kwargs)

    def train_step(self, batch):
        """
        Executes a single training step using a batch of data.

        Parameters:
        - batch (dict): A dictionary containing the batch of data, including images, depth information, and masks.

        Returns:
        - loss (Tensor): The total loss for this training step.
        - loss_dict (dict): A dictionary containing individual loss components.
        """
        # Extract metadata from the batch
        meta_data = batch["meta_data"]

        # If the batch does not contain 'frames' or 'views', defer to the superclass implementation
        if "frames" not in meta_data and "views" not in meta_data:
            return super().train_step(batch)

        self.train()
        # Unpack necessary inputs from the batch
        (
            name,
            total_iter,
            image,
            _,
            _,
            _,
            _,
            prompt_scale,
            prompt_center,
            prompt_mask,
            _,
            _,
            _,
            valid_mask,
            _,
            _,
        ) = self.get_inputs(batch)

        # Forward pass through the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=None,
            prompt_scale=None,
            prompt_center=None,
            intrinsics=None,
            extrinsics=None,
            novel_intrinsics=None,
            novel_extrinsics=None,
            query_points=None,
            with_freeze=True,
            novel_view_nums=self.novel_view_nums,
        )

        # Extract predictions from the model output
        mv_depth_pred = results.get("mv_depth", None)
        # mv_depth_confidence_pred = results.get("mv_depth_confidence", None)

        pose_enc = results["pose_enc"]
        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding=pose_enc,
            image_size_hw=image.shape[-2:],
            translation_scale=None,
        )

        gaussians = results.get("gaussians", None)
        render_rgb = results.get("render_rgb", None)
        render_depth = results.get("render_depth", None)
        render_alpha = results.get("render_alpha", None)
        # render with a larger focal
        render_large_focal_alpha = results.get("large_focal_alpha", None)
        render_normal = results.get("render_normal", None)
        render_novel_rgb = results.get("render_novel_rgb", None)
        render_novel_depth = results.get("render_novel_depth", None)

        total_loss, total_loss_dict = 0, dict()

        # Compute reconstruction loss
        if render_rgb is not None or render_depth is not None:
            render_depth = render_depth.unsqueeze(2)
            if gaussians.norm_scales is not None:
                render_depth = render_depth * gaussians.norm_scales[:, None, None, None, None]
            render_depth_norm = self.normalize(
                render_depth, scale=prompt_scale, center=prompt_center
            )
            render_depth_norm = self.depth_to_point(render_depth_norm, K=intrinsics)

            rc_loss, rc_loss_dict = self.get_reconstruct_loss(
                name=name,
                image=image,
                depth_norm=mv_depth_pred,  # NOTE
                render_rgb=render_rgb,
                render_depth=render_depth_norm,
                render_alpha=render_alpha,
                render_normal=render_normal,
                gaussians=gaussians,
                valid_mask=valid_mask,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
                intrinsics=intrinsics,
            )
            if render_large_focal_alpha is not None:
                # regularize alpha to encourage gaussian be more compact
                large_focal_alpha_loss = (1 - render_large_focal_alpha).mean() * 0.1
                rc_loss += large_focal_alpha_loss
                rc_loss_dict["rc_lf_alpha_loss"] = large_focal_alpha_loss

            rc_weight = self.task_weight.get("rc", 1.0)
            total_loss += rc_loss * rc_weight
            total_loss_dict["rc"] = rc_loss * rc_weight
            total_loss_dict.update({key: val * rc_weight for key, val in rc_loss_dict.items()})

        if self.novel_view_nums > 0:
            # Split the current batch into two parts: base samples and novel samples
            batch, novel_batch = self.split_novel_batch(batch)
            novel_valid_mask = novel_batch[self.target_mask_name].to(self.device)

            novel_mask = calculate_loss_mask(batch, novel_batch)
            novel_mask = novel_mask.unsqueeze(2)  # [b,v,1,h,w]->[b,v,h,w]
            nrc_loss, nrc_loss_dict = self.get_novel_reconstruct_loss(
                name=name,
                image=novel_batch["image"].to(image.device),
                # depth=novel_batch["depth"].to(image.device),
                depth=None,
                render_rgb=render_novel_rgb,
                render_depth=render_novel_depth,
                novel_mask=novel_mask,
                valid_mask=novel_valid_mask,
            )
            nrc_weight = self.task_weight.get("nrc", 1.0)
            total_loss += nrc_loss * nrc_weight
            total_loss_dict["nrc"] = nrc_loss * nrc_weight
            total_loss_dict.update({key: val * nrc_weight for key, val in nrc_loss_dict.items()})

        return total_loss, total_loss_dict

    def save_render_video(self, outputs_list, gs_out_dir, data_idx, frame_num, view_num):
        gaussians = outputs_list[0].gaussians
        save_path = os.path.join(gs_out_dir, "gaussians.ply")
        gaussians.export_ply().write(save_path)

        if self.render_video_with_pred_camera:
            extrinsics = torch.stack(
                [torch.from_numpy(outputs.extrinsics_pred) for outputs in outputs_list], dim=0
            )
            intrinsics = torch.stack(
                [torch.from_numpy(outputs.intrinsics_pred) for outputs in outputs_list], dim=0
            )
        else:
            extrinsics = torch.stack(
                [torch.from_numpy(outputs.extrinsics) for outputs in outputs_list], dim=0
            )
            intrinsics = torch.stack(
                [torch.from_numpy(outputs.intrinsics) for outputs in outputs_list], dim=0
            )

        prompt_scale = torch.tensor([outputs.prompt_scale for outputs in outputs_list]).float()[
            :, None
        ]
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / prompt_scale

        extrinsics = extrinsics.reshape(-1, frame_num * view_num, 4, 4).float().to(self.device)
        intrinsics = intrinsics.reshape(-1, frame_num * view_num, 3, 3).float().to(self.device)

        render_video = self.render_video_interpolation(
            gaussians,
            extrinsics,
            intrinsics,
            outputs_list[0].pointmap_h,
            outputs_list[0].pointmap_w,
        )
        save_video(render_video, os.path.dirname(gs_out_dir), "video", data_idx, info=True)
