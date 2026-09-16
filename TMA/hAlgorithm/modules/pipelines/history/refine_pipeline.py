import logging
import os

import numpy as np
import torch

from hAlgorithm.modules.models.pi3.utils.pose_enc import pi3_pose_fov_to_extri_intri
from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.utils import instantiate_from_config

from .reconstruct_pipeline import ReconstructPipeline


class RefinePipeline(ReconstructPipeline):
    """
    Pipeline for reconstructing 3D scenes from input images and depth information.
    """

    def __init__(
        self,
        refine_model,
        model_pretrain=None,
        freeze_model=True,
        save_model=False,
        **kwargs,
    ):
        super(RefinePipeline, self).__init__(**kwargs)

        self.model_pretrain = model_pretrain
        self.freeze_model = freeze_model
        self.save_model = save_model

        self.refine_model = instantiate_from_config(refine_model)

        if self.model_pretrain is not None:
            super().load_checkpoint(ckpt_path=self.model_pretrain, state_dict=None)

    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state_dict is not None:
            if len(self.exclude_modules) > 0:  # drop some modules which may have wrong shape
                new_state_dict = dict()
                drop_keys = []
                for key, val in state_dict.items():
                    ignore_val = False
                    for name in self.exclude_modules:
                        if name in key:
                            ignore_val = True
                            break
                    if ignore_val:
                        drop_keys.append(key)
                        continue
                    new_state_dict[key] = val
                res = self.refine_model.load_state_dict(new_state_dict, strict=False)
                logging.info(f"drop keys: {drop_keys}")
            else:
                res = self.refine_model.load_state_dict(state_dict, strict=False)

            logging.info(f"Refine Model parameters are loaded from {ckpt_path}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    def save_checkpoint(self, accelerator, ckpt_dir=None):
        if self.save_model:
            super().save_checkpoint(accelerator=accelerator, ckpt_dir=ckpt_dir)

        refine_model = accelerator.unwrap_model(self.refine_model)
        state_dict = refine_model.state_dict()

        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "refine_ckpt.pth")

            try:
                torch.save(state_dict, ckpt_path)
                logging.info(f"Refine Model is saved to: {ckpt_path}")

            except Exception:
                logging.warning(f"Refine Model is saved error!!")

        return state_dict

    def accelerator_prepare(self, accelerator, optimizer, lr_scheduler, train_dataloader):
        if self.refine_model is None:
            return super().accelerator_prepare(
                accelerator, optimizer, lr_scheduler, train_dataloader
            )

        weight_dtype = torch.float32
        if accelerator.mixed_precision in ["fp16", "fp8"]:
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        logging.info(f"weight_dtype: {weight_dtype}")

        self.cuda(accelerator.device)
        self.device = accelerator.device
        self.dtype = weight_dtype

        if train_dataloader is None:
            self.model = accelerator.prepare(self.model)
            self.refine_model = accelerator.prepare(self.refine_model)
            return None, None, None

        (
            self.model,
            self.refine_model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            self.model,
            self.refine_model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
        return optimizer, train_dataloader, lr_scheduler

    def share_step(self, image, prompt_depth, prompt_scale, intrinsics, extrinsics, meta_data, with_freeze=False):
        # Forward pass through the model
        results = self.refine_model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth,
            prompt_scale=prompt_scale,
            prompt_center=None,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            with_freeze=with_freeze,
        )
        return results

    def train_step(self, batch):
        if self.refine_model is None:
            return super().train_step(batch)
        
        if self.freeze_model:
            with torch.no_grad():
                refine_inputs = super().train_step(batch)[-1]

            total_loss, total_loss_dict = 0, dict()
        else:
            total_loss, total_loss_dict, refine_inputs = super().train_step(batch)

        mv_depth_pred = refine_inputs["mv_depth"]
        # mv_depth_confidence_pred = refine_inputs["mv_depth_confidence"]

        name = refine_inputs["name"]
        total_iter = refine_inputs["total_iter"]
        image = refine_inputs["image"]
        intrinsics = refine_inputs["intrinsics"]
        target_norm = refine_inputs["target_norm"]
        valid_mask = refine_inputs["valid_mask"]
        prompt_scale = refine_inputs["prompt_scale"]

        extrinsics = refine_inputs["extrinsics"]
        # mv_target = refine_inputs["mv_target"]
        meta_data = refine_inputs["meta_data"]

        pose_enc = refine_inputs.get("pose_enc", None)
        if pose_enc is None:
            intrinsics_pred = intrinsics
        else:
            if isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]

            if self.pose_encoding_type == "absT_quaR_FoV":
                _, intrinsics_pred = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc,
                    image_size_hw=image.shape[-2:],
                    translation_scale=None,
                )
                # if self.test_normalize_cameras:
                #     w2c_pred = extrinsics_pred
                #     base_c2w_pred = w2c_pred[:, 0:1].inverse()
                #     extrinsics_pred = w2c_pred @ base_c2w_pred
            elif self.pose_encoding_type == "pi3":
                intrinsics_pred = refine_inputs["intrinsics"]
            elif self.pose_encoding_type == "pi3_fov":
                camera_fov = refine_inputs["camera_fov"]
                _, intrinsics_pred = pi3_pose_fov_to_extri_intri(
                    pose=pose_enc,
                    fov=camera_fov,
                    image_size_hw=image.shape[-2:],
                    translation_scale=None,
                    pose_encoding_type=self.pose_encoding_type,
                    # normalize_cameras=self.test_normalize_cameras,
                )
            else:
                raise NotImplementedError

        # Forward pass through the model
        refine_results = self.share_step(
            image=image,
            prompt_depth=mv_depth_pred,
            prompt_scale=prompt_scale,
            intrinsics=intrinsics_pred,
            extrinsics=extrinsics,
            meta_data=meta_data,
            with_freeze=True,
        )

        refine_depth_pred = refine_results.get("depth", None)
        refine_depth_confidence_pred = refine_results.get("depth_confidence", None)

        # Compute multi-view depth loss if multi-view depthmap or confidence predictions are available
        if (refine_depth_pred is not None and refine_depth_pred.requires_grad) or (
            refine_depth_confidence_pred is not None and refine_depth_confidence_pred.requires_grad
        ):
            # DepthMap trans to PointMap
            if refine_depth_pred.shape[-3] == 1 and not self.depth_only_z:
                refine_depth_pred = self.depth_to_point(refine_depth_pred, K=intrinsics_pred)

            rf_loss, rf_loss_dict = self.get_depth_loss(
                name=name,
                total_iter=total_iter,
                image=image,
                intrinsics=intrinsics_pred,
                prompt_scale=prompt_scale,
                prompt_center=None,
                prompt_mask=None,
                prompt_diffmap=None,
                target_norm=target_norm,
                valid_mask=valid_mask,
                edge_mask=None,
                pointmap_pred=refine_depth_pred,
                confidence_pred=refine_depth_confidence_pred,
            )

            rf_weight = self.task_weight.get("refine", 1.0)
            total_loss += rf_loss * rf_weight
            total_loss_dict["rf"] = 0
            for key, val in rf_loss_dict.items():
                total_loss_dict[f"rf_{key}"] = val * rf_weight
                total_loss_dict["rf"] += val * rf_weight

        return total_loss, total_loss_dict

    @torch.no_grad()
    def infer(self, **batch):
        """
        Executes inference on a given input image and returns the predicted depth map and related outputs.

        Parameters:
        - image (Tensor): The input image tensor.
        - **kwargs: Additional keyword arguments containing depth information and other optional parameters.

        Returns:
        - ReconstructOutput: An object containing the predicted depth map and other related outputs.
        """
        if self.refine_model is None:
            return super().infer(**batch)

        # Extract metadata from the batch input
        meta_data = batch["meta_data"]

        # If "frames" or "views" are not in metadata, fallback to the superclass infer method
        if "frames" not in meta_data and "views" not in meta_data:
            return super().infer(**batch)

        # Set the model to evaluation mode.
        self.eval()

        (
            name,
            total_iter,
            image,
            intrinsics,
            _,
            prompt_depth,
            prompt_depth_norm,
            prompt_scale,
            prompt_center,
            prompt_mask,
            prompt_diffmap,
            target,
            target_norm,
            valid_mask,
            edge_mask,
            image_show,
        ) = self.get_inputs(batch)

        # Handle extrinsics data if available in the batch
        if self.mv_extrinsics_name is not None and self.mv_extrinsics_name in batch:
            extrinsics = batch[self.mv_extrinsics_name].to(device=self.device)
        else:
            extrinsics = None

        # Perform inference using the shared step method of the model
        results = self.model(
            image,
            meta_data=meta_data,
            prompt_depth=prompt_depth_norm,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            render_video_with_pred_camera=self.render_video_with_pred_camera,
        )

        # Extract predictions from the model output
        # Multi-view depth prediction
        mv_depth_pred = results.get("mv_depth", None)
        # Confidence for multi-view depth
        mv_depth_confidence_pred = results.get("mv_depth_confidence", None)
        # Multi-view point cloud map
        mv_pointmap_pred = results.get("mv_pointmap", None)
        # Confidence for multi-view point cloud map
        mv_confidence_pred = results.get("mv_confidence", None)
        # Pose encoding for camera extrinsics/intrinsics
        pose_enc = results.get("pose_enc", None)
        camera_fov = results.get("camera_fov", None)

        # Pose encoding for camera extrinsics/intrinsics
        if pose_enc is not None:
            if isinstance(pose_enc, (list, tuple)):
                pose_enc = pose_enc[-1]
            if self.pose_encoding_type == "absT_quaR_FoV":
                extrinsics_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc,
                    image_size_hw=image.shape[-2:],
                    translation_scale=prompt_scale,
                )
                if self.test_normalize_cameras:
                    w2c_pred = extrinsics_pred
                    base_c2w_pred = w2c_pred[:, 0:1].inverse()
                    extrinsics_pred = w2c_pred @ base_c2w_pred
            elif self.pose_encoding_type == "pi3":
                extrinsics_pred, _ = pi3_pose_fov_to_extri_intri(
                    pose=pose_enc,
                    fov=camera_fov,
                    image_size_hw=image.shape[-2:],
                    translation_scale=prompt_scale,
                    pose_encoding_type=self.pose_encoding_type,
                    normalize_cameras=self.test_normalize_cameras,
                )
                intrinsics_pred = intrinsics
            elif self.pose_encoding_type == "pi3_fov":
                extrinsics_pred, intrinsics_pred = pi3_pose_fov_to_extri_intri(
                    pose=pose_enc,
                    fov=camera_fov,
                    image_size_hw=image.shape[-2:],
                    translation_scale=prompt_scale,
                    pose_encoding_type=self.pose_encoding_type,
                    normalize_cameras=self.test_normalize_cameras,
                )
            else:
                raise NotImplementedError

            if self.output_gt_principal_points:
                intrinsics_pred[..., :2, 2] = intrinsics[..., :2, 2]

        else:
            extrinsics_pred, intrinsics_pred = None, None

        # NOTE:
        # Forward pass through the refine model
        refine_results = self.share_step(
            image=image,
            prompt_depth=mv_depth_pred,
            prompt_scale=prompt_scale,
            intrinsics=intrinsics_pred if intrinsics_pred is not None else intrinsics,
            extrinsics=extrinsics,
            meta_data=meta_data,
            with_freeze=False,
        )

        mv_depth_pred = refine_results.get("depth", None)
        mv_depth_confidence_pred = refine_results.get("depth_confidence", None)

        if self.output_d2p_with_intrinsics_pred and intrinsics_pred is not None:
            K = intrinsics_pred
        else:
            K = intrinsics

        # Convert depth maps to point clouds if necessary
        if mv_depth_pred is not None and mv_depth_pred.shape[-3] == 1:
            mv_depth_pred = self.depth_to_point(mv_depth_pred, K=K)

        if mv_pointmap_pred is not None:
            # assert mv_pointmap_pred.shape[-3] == 3
            if mv_pointmap_pred.shape[-3] == 1:
                mv_pointmap_pred = self.depth_to_point(mv_pointmap_pred, K=K)

            mv_pointmap_pred = self.denormalize(
                mv_pointmap_pred, scale=prompt_scale, center=prompt_center
            )

        output_pointmap_pred = mv_depth_pred
        output_confidence_pred = mv_depth_confidence_pred

        # Prepare alignment ground truth and mask if required
        align_gt = align_mask = None
        if self.match_input_res and self.align_name in batch:
            align_gt = batch[self.align_name]
        if self.match_input_res and self.post_align and self.align_mask_name in batch:
            align_mask = batch[self.align_mask_name]

        # Extract frame and view counts from metadata
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        # Process each frame and view combination
        outputs_list = []
        for fi in range(frame_num):
            for vi in range(view_num):
                # Compute the global index for the current frame-view pair
                index = fi * view_num + vi

                # Post-process the predictions for the current frame-view pair
                single = self.postprocess(
                    intrinsics=intrinsics[:, index] if intrinsics is not None else None,
                    extrinsics=extrinsics[:, index] if extrinsics is not None else None,
                    image=image[:, index] if image is not None else None,
                    target=target[:, index] if target is not None else None,
                    prompt_depth=prompt_depth[:, index] if prompt_depth is not None else None,
                    prompt_scale=prompt_scale[:, index] if prompt_scale is not None else None,
                    prompt_center=(prompt_center[:, index] if prompt_center is not None else None),
                    image_show=image_show[index] if image_show is not None else None,
                    pointmap_pred=(
                        output_pointmap_pred[:, index] if output_pointmap_pred is not None else None
                    ),
                    confidence_pred=(
                        output_confidence_pred[:, index]
                        if output_confidence_pred is not None
                        else None
                    ),
                    gradient_pred=None,
                    prompt_confidence_pred=None,
                    align_gt=align_gt[:, index] if align_gt is not None else None,
                    align_mask=align_mask[:, index] if align_mask is not None else None,
                )

                # Add multi-view point cloud predictions if available
                if mv_pointmap_pred is not None:
                    single.glb_mv_pointmap = (
                        mv_pointmap_pred[0, index].detach().cpu().numpy().transpose(1, 2, 0)
                    )
                    if self.save_glb2local_results and extrinsics is not None:
                        glb2local_pts = self.glb_to_local(
                            mv_pointmap_pred[0:1, index : index + 1].clone(),
                            extrinsics=extrinsics[0:1, index : index + 1],
                            scale=None,
                            center=None,
                        )[0, 0]
                        single.glb2local_pointmap = (
                            glb2local_pts.detach().cpu().numpy().transpose(1, 2, 0)
                        )

                # Add multi-view confidence predictions if available
                if mv_confidence_pred is not None:
                    single.glb_mv_confidence = (
                        mv_confidence_pred[0, index, 0].detach().cpu().numpy()
                    )

                # Add predicted extrinsics if available
                if extrinsics_pred is not None:
                    single.extrinsics_pred = extrinsics_pred[0, index].detach().cpu().numpy()

                # Add predicted intrinsics if available
                if intrinsics_pred is not None:
                    single.intrinsics_pred = intrinsics_pred[0, index].detach().cpu().numpy()

                if (
                    extrinsics_pred is not None
                    and self.output_d2p_with_intrinsics_pred
                    and intrinsics_pred is not None
                ):
                    local_pointmap = single.pointmap
                    local2glb_pointmap = (
                        np.linalg.inv(single.extrinsics_pred)
                        @ np.concatenate(
                            [local_pointmap, np.ones([local_pointmap.shape[0], 1])],
                            axis=-1,
                            dtype=np.float32,
                        ).T
                    )
                    single.local2glb_pointmap = local2glb_pointmap.T[:, :3]
                    single.local2glb_confidence = (
                        mv_depth_confidence_pred[0, index, 0].detach().cpu().numpy()
                    )

                # Store frame and view indices
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                outputs_list.append(single)

        return outputs_list
