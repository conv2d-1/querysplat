import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from hAlgorithm.utils import (
    colorize_depth_maps,
    instantiate_from_config,
)

from .outputs import DepthOutput
from .prompt_pointmap_pipeline_v2 import PromptPointMapPipeline as BasePipeline


class PromptPointMapDistillPipeline(BasePipeline):
    def __init__(
        self,
        model,
        mde_teacher,
        sparse_size=None,
        sparse_nums=None,
        sparse_num_test=None,
        infer_with_teacher=True,
        **kwargs,
    ):
        self.mde_teacher = instantiate_from_config(mde_teacher)
        self.init_teacher = False

        self.infer_with_teacher = infer_with_teacher
        if self.infer_with_teacher:
            assert sparse_size is not None, "sparse_size must be valid when infer_with_teacher is true."
        
        self.sparse_size = sparse_size
        self.sparse_nums_train = self.sparse_nums = sparse_nums
        self.sparse_nums_test = sparse_num_test
        super().__init__(model=model, **kwargs)

    def get_inputs(self, batch, use_teacher_depth=True):
        if use_teacher_depth:
            # Extract and move tensors to the appropriate device and dtype if necessary.
            name = batch["meta_data"]["name"][0]
            total_iter = batch.get("total_iter", None)
            image = batch["image"].to(device=self.device, dtype=self.dtype)

            intrinsics = batch.get(self.intrinsics_name, None)
            extrinsics = batch.get(self.extrinsics_name, None)  # [n, 4, 4]

            if intrinsics is not None:
                intrinsics = intrinsics.to(device=self.device)
            if extrinsics is not None:
                extrinsics = extrinsics.to(device=self.device)

            image_show = batch.get("image_show", None)
            if image_show is not None:
                if image_show.ndim == 4:
                    image_show = image_show.float().squeeze(0).numpy().transpose(1, 2, 0)
                else:
                    image_show = image_show.float().squeeze(0).numpy().transpose(0, 2, 3, 1)
            with torch.no_grad():
                if not self.init_teacher:
                    self.mde_teacher.mde_model = self.mde_teacher.mde_model.to(device=self.device, dtype=self.dtype)
                    self.init_teacher = True
                # with torch.autocast(device_type="cuda", dtype=self.dtype):
                teacher_output = self.mde_teacher(image, return_dict=True)
            target_norm = teacher_output['pointmap'].to(device=self.device, dtype=self.dtype)
            prompt_scale = teacher_output['metric_scale'].to(device=self.device, dtype=self.dtype)[
                    ..., None, None, None
                ]
            target = target_norm * prompt_scale
            valid_mask = torch.logical_and(target[:, -1, ...] > 0.001, target[:, -1, ...] < 200)
            prompt_depth, prompt_mask = self.get_sparse_depth(target, valid_mask)
            prompt_scale = torch.max(prompt_depth[:,-1].flatten(1), dim=1)[0][
                    ..., None, None, None
                ]
            prompt_depth_norm = prompt_depth / prompt_scale
            target_norm = target / prompt_scale
            return (
                name,
                total_iter,
                image,
                intrinsics,
                extrinsics,
                prompt_depth,
                prompt_depth_norm,
                prompt_scale,
                None, # prompt_center,
                prompt_mask,
                None, # prompt_diffmap,
                target,
                target_norm,
                valid_mask,
                None, # edge_mask,
                image_show,
            )
        else:
            return super().get_inputs(batch)

    def get_sparse_depth(self, curr_pointmap: torch.Tensor, valid_mask: torch.Tensor):
        """
        Resize the curr_pointmap to ensure the longest side is max_size while maintaining the aspect ratio,
        and then randomly sample points to zero, keeping only sparse_num valid points.
        
        Args:
            curr_pointmap (torch.Tensor): A tensor of shape (B, 3, H, W) representing point maps.
            valid_mask (torch.Tensor): A tensor of shape (B, H, W) representing valid masks.
            max_size (int): The maximum size for the longest side after resizing.
            sparse_num (int): The number of valid points to keep after sampling.
            
        Returns:
            torch.Tensor: Resized and sampled point map of shape (B, 3, H', W').
            torch.Tensor: Updated valid mask of shape (B, H', W').
        """
        B, _, H, W = curr_pointmap.shape
        max_size = self.sparse_size

        # Determine the new size while maintaining the aspect ratio
        if H > W:
            new_h = max_size
            new_w = int(W * (max_size / H))
        else:
            new_w = max_size
            new_h = int(H * (max_size / W))
        
        # Resize curr_pointmap and valid_mask using nearest neighbor interpolation
        resized_curr_pointmap = TF.resize(curr_pointmap, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
        resized_valid_mask = TF.resize(valid_mask.unsqueeze(1).float(), (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST).squeeze(1)
        
        # Initialize a tensor to store the sampled point map
        sampled_curr_pointmap = resized_curr_pointmap.clone()
        sampled_valid_mask = torch.zeros_like(resized_valid_mask, dtype=torch.bool)

        for b in range(B):
            #sample sparse num
            if isinstance(self.sparse_nums, (list, tuple)):
                min_num, max_num = min(self.sparse_nums), max(self.sparse_nums)
                sparse_num = int(np.random.uniform(min_num, max_num))
            else:
                sparse_num = int(self.sparse_nums)
            valid_indices = torch.nonzero(resized_valid_mask[b], as_tuple=True)
            num_valid_points = len(valid_indices[0])
            random_indices = torch.randperm(num_valid_points)[:sparse_num]
            selected_valid_indices = tuple(idx[random_indices] for idx in valid_indices)
            sampled_valid_mask[b, selected_valid_indices[0], selected_valid_indices[1]] = True
            sampled_curr_pointmap[b, :, ~sampled_valid_mask[b]] = 0

        return sampled_curr_pointmap, sampled_valid_mask

    def train_step(self, batch):
        """
        Executes a single training step using a batch of data.

        Parameters:
        - batch (dict): A dictionary containing the batch of data, including images, depth information, and masks.

        Returns:
        - loss (Tensor): The total loss for this training step.
        - loss_dict (dict): A dictionary containing individual loss components.
        """
        self.train()
        self.sparse_nums = self.sparse_nums_train

        (
            name,
            total_iter,
            image,
            intrinsics,
            extrinsics,
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

        sift_eval_mask = batch.get("sift_eval_mask", None)
        sift_point_mask = batch.get("sift_mask", None)

        inf_mask = batch.get("inf_mask", None)

        results = self.model(
            image, prompt_depth_norm, with_freeze=True, meta_data=batch["meta_data"]
        )
        pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]
        gradient_pred = results.get("gradient", None)
        prompt_confidence_pred = results.get("prompt_confidence", None)
        invalid_mask_pred = results.get("mask", None)
        normal_pred = results.get('normal', None)

        # DepthMap trans to PointMap
        if pointmap_pred.shape[1] == 1:
            pointmap_pred = self.depth_to_point(pointmap_pred, K=intrinsics)

        loss, loss_dict = self.get_loss(
            name=name,
            total_iter=total_iter,
            pointmap_pred=pointmap_pred,
            confidence_pred=confidence_pred,
            target=target_norm,
            intrinsics=intrinsics,
            valid_mask=valid_mask,
            edge_mask=edge_mask,
            prompt_mask=prompt_mask,
            image=image,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            sift_mask=sift_point_mask,
        )

        if self.use_extend_loss:
            loss_extend, loss_dict_extend = self.get_extend_loss(
                name=name,
                pointmap_pred=pointmap_pred,
                invalid_mask_pred=invalid_mask_pred,
                confidence_pred=confidence_pred,
                intrinsics=intrinsics,
                gradient_pred=gradient_pred,
                prompt_confidence_pred=prompt_confidence_pred,
                invalid_mask_target=inf_mask,
                valid_mask=valid_mask,
                depth_target=target_norm,
                prompt_diffmap=prompt_diffmap,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
                normal_pred=normal_pred,
            )
            loss = loss + loss_extend
            loss_dict.update(loss_dict_extend)

        return loss, loss_dict

    @torch.no_grad()
    def infer(self, **batch):
        """
        Executes inference on a given input image and returns the predicted depth map and related outputs.

        Parameters:
        - image (Tensor): The input image tensor.
        - **kwargs: Additional keyword arguments containing depth information and other optional parameters.

        Returns:
        - DepthOutput: An object containing the predicted depth map and other related outputs.
        """

        # Set the model to evaluation mode.
        self.eval()
        self.sparse_nums = self.sparse_nums_test
        (
            name,
            total_iter,
            image,
            intrinsics,
            extrinsics,
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
        ) = self.get_inputs(batch, self.infer_with_teacher)

        sift_eval_mask = batch.get("sift_eval_mask", None)
        sift_point_mask = batch.get("sift_mask", None)

        # Use the shared step method to make predictions based on the input image and depth information.
        results = self.model(image, prompt_depth_norm, meta_data=batch["meta_data"])
        pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]
        gradient_pred = results.get("gradient", None)
        prompt_confidence_pred = results.get("prompt_confidence", None)
        invalid_mask_pred = results.get("mask", None)

        if self.output_ddi:
            tmp_target = batch[self.target_name].to(device=self.device)

            tem_pointmap_pred_ = self.denormalize(
                pointmap_pred, scale=prompt_scale, center=prompt_center
            )
            from hAlgorithm.utils import cuda_timing_context

            with cuda_timing_context("ddi", True):
                depth_ddi = self.ddi(
                    tem_pointmap_pred_[:, -1],
                    prompt_depth[:, -1],
                    prompt_mask[:, 0],
                    prompt_scale,
                    tmp_target[:, -1],
                )
            pointmap_pred[:, -1, :, :] = depth_ddi / prompt_scale

        align_gt = align_mask = None
        if self.match_input_res:
            align_gt = batch[self.align_name]
        if self.match_input_res and self.post_align:
            align_mask = batch[self.align_mask_name]

        return self.postprocess(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image=image,
            target=target,
            prompt_depth=prompt_depth,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            image_show=image_show,
            pointmap_pred=pointmap_pred,
            confidence_pred=confidence_pred,
            invalid_mask_pred=invalid_mask_pred,
            gradient_pred=gradient_pred,
            prompt_confidence_pred=prompt_confidence_pred,
            align_gt=align_gt,
            align_mask=align_mask,
        )


class PromptPointMapTeacherPipeline(PromptPointMapDistillPipeline):
    def __init__(self, model, mde_teacher, **kwargs):
        super().__init__(model, mde_teacher, sparse_size=0, infer_with_teacher=True, **kwargs)
    
    def get_inputs(self, batch, use_teacher_depth=True):
        if use_teacher_depth:
            # Extract and move tensors to the appropriate device and dtype if necessary.
            name = batch["meta_data"]["name"][0]
            total_iter = batch.get("total_iter", None)
            image = batch["image"].to(device=self.device, dtype=self.dtype)

            intrinsics = batch.get(self.intrinsics_name, None)
            extrinsics = batch.get(self.extrinsics_name, None)  # [n, 4, 4]

            if intrinsics is not None:
                intrinsics = intrinsics.to(device=self.device)
            if extrinsics is not None:
                extrinsics = extrinsics.to(device=self.device)

            image_show = batch.get("image_show", None)
            if image_show is not None:
                if image_show.ndim == 4:
                    image_show = image_show.float().squeeze(0).numpy().transpose(1, 2, 0)
                else:
                    image_show = image_show.float().squeeze(0).numpy().transpose(0, 2, 3, 1)
            with torch.no_grad():
                if not self.init_teacher:
                    self.mde_teacher.mde_model = self.mde_teacher.mde_model.to(device=self.device, dtype=self.dtype)
                    self.init_teacher = True
                # with torch.autocast(device_type="cuda", dtype=self.dtype):
                teacher_output = self.mde_teacher(image, return_dict=True)
            target_norm = teacher_output['pointmap'].to(device=self.device, dtype=self.dtype)
            prompt_scale = teacher_output['metric_scale'].to(device=self.device, dtype=self.dtype)[
                    ..., None, None, None
                ]
            target = target_norm * prompt_scale
            valid_mask = torch.logical_and(target[:, -1, ...] > 0.001, target[:, -1, ...] < 200)
            prompt_scale = torch.max(target[:,-1].flatten(1), dim=1)[0][
                    ..., None, None, None
                ]
            target_norm = target / prompt_scale
            return (
                name,
                total_iter,
                image,
                intrinsics,
                extrinsics,
                None,
                None,
                prompt_scale,
                None, # prompt_center,
                None,
                None, # prompt_diffmap,
                target,
                target_norm,
                valid_mask,
                None, # edge_mask,
                image_show,
            )
        else:
            return super().get_inputs(batch)

    @torch.no_grad()
    def infer(self, **batch):
        """
        Executes inference on a given input image and returns the predicted depth map and related outputs.

        Parameters:
        - image (Tensor): The input image tensor.
        - **kwargs: Additional keyword arguments containing depth information and other optional parameters.

        Returns:
        - DepthOutput: An object containing the predicted depth map and other related outputs.
        """

        # Set the model to evaluation mode.
        self.eval()
        self.sparse_nums = self.sparse_nums_test
        (
            name,
            total_iter,
            image,
            intrinsics,
            extrinsics,
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
        ) = self.get_inputs(batch, self.infer_with_teacher)

        align_gt = align_mask = None
        if self.match_input_res:
            align_gt = np.zeros_like(image_show[:,:,0])
        if self.match_input_res and self.post_align:
            align_mask = batch[self.align_mask_name]

        return self.postprocess(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image=image,
            target=target,
            prompt_depth=prompt_depth,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            image_show=image_show,
            pointmap_pred=target_norm,
            confidence_pred=None,
            invalid_mask_pred=None,
            gradient_pred=None,
            prompt_confidence_pred=None,
            align_gt=align_gt,
            align_mask=align_mask,
        )