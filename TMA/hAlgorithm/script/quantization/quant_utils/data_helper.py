import os, sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import cv2

import logging
from hAlgorithm.modules.utils.alignment import align_depth_least_square
from hAlgorithm.modules.pipelines.outputs import DepthOutput

def data_parser(data, model):
    image = data["image"].cuda()
    # image = ((image + 1) * 0.5 - model._mean) / model._std
    prompt_depth = data[model.prompt_name][:,-1:,:,:].cuda()
    prompt_scale = data[model.prompt_scale_name][
        :, None, None, None
    ].cuda()

    data_dict = {
        "image": image,
        "prompt_depth": prompt_depth,
        "prompt_scale": prompt_scale
    }

    return data_dict

def data_parser_fixresize_custom(data, model):
    image = data["image"].cuda()
    prompt_depth = data[model.prompt_name].cuda()
    prompt_scale = data[model.prompt_scale_name][
        :, None, None, None
    ].cuda()

    patch_size=14
    img_max_size = int(960/patch_size)*patch_size
    sparse_max_size = round(960/patch_size*8/patch_size)*patch_size
    assume_img_shape = (1080, 1920)
    img_ratio = img_max_size/float(assume_img_shape[1])
    depth_ratio = sparse_max_size/float(assume_img_shape[1])
    img_dsize = (int(assume_img_shape[0]*img_ratio/14)*14, int(assume_img_shape[1]*img_ratio/14)*14)
    depth_dsize = (int(assume_img_shape[0]*depth_ratio/14)*14, int(assume_img_shape[1]*depth_ratio/14)*14)

    image = F.interpolate(image, size=img_dsize, scale_factor=None, mode="bilinear", align_corners=False)
    prompt_depth = F.interpolate(prompt_depth, size=depth_dsize, scale_factor=None, mode="bilinear", align_corners=False)

    data_dict = {
        "image": image,
        "meta_data": data["meta_data"],
        "prompt_depth": prompt_depth,
        "prompt_scale": prompt_scale
    }

    return data_dict

class DataParser:
    def __init__(self, model):
        self.model = model
    
    def parse_data(self, batch):
        data_dict = {}

        data_dict['orig_image'] = batch["image"]
        image = batch["image"].clone()
        # image = ((image + 1) * 0.5 - self.model._mean.cpu()) / self.model._std.cpu()
        data_dict['image'] = image
        data_dict['prompt_depth'] = batch[self.model.prompt_name] ##[:,-1:,:,:]
        data_dict['prompt_scale'] = batch[self.model.prompt_scale_name][:, None, None, None]
        data_dict['intrinsic'] = batch.get('intrinsics', None)
        data_dict['image_show'] = batch['image_show']
        data_dict['depth_gt'] = batch[self.model.align_name]
        data_dict['depth_gt_valid_mask'] = batch[self.model.align_mask_name]
        data_dict['pointmap_gt'] = batch.get(self.model.target_name, None)
        
        return data_dict

def np_depth_to_point(depth, K):
    B, C, H, W = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5, indexing="xy")
    points = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=0).reshape(3, -1).astype(np.float32)
    rays_d = np.linalg.inv(K) @ points  # (3, HW)
    pts = depth.reshape(B, C, -1) * rays_d  # (B, 3, HW)
    pc = pts.reshape(B, 3, H, W)
    return pc

def depth_to_point(depth, K, device):
    B, C, H, W = depth.shape
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy"
    )
    points = (
        torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
        .reshape(3, -1)
        .float()
        .to(device)
    )
    rays_d = K.inverse().to(device) @ points  # (B, 3, HW)
    pts = depth.flatten(2) * rays_d
    depth = pts.reshape(B, 3, H, W)
    return depth

def filter_pointmap(pointmap_pred, confidence_pred, image, conf_thresh):
    pointmap_color = image.float().squeeze(0).numpy().transpose(1, 2, 0)
    pointmap_color = pointmap_color.reshape(-1, 3)
    filtered_pointmap = pointmap_pred[confidence_pred.reshape(-1) > conf_thresh]
    filtered_pointmap_color = pointmap_color[
        confidence_pred.reshape(-1) > conf_thresh
    ]
    return filtered_pointmap, filtered_pointmap_color

def normalize_depth(depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
    if center is not None:
        depth = depth - center
    if scale is not None:
        depth = depth / scale
    return depth

# todo: make them all np.array or tensor?
class NewDataParser:
    def __init__(
        self,
        prompt_name=None,
        prompt_mask_name=None,
        prompt_interpolate=None,
        prompt_diffmap_name=None,
        prompt_scale_name=None,
        prompt_center_name=None,
        target_name=None,
        target_mask_name=None,
        target_clip=None,
        edge_mask_name=None,
        prompt_set_none=False,
        debug=False,
        debug_size=False,
        debug_scale=False
    ):
        self.prompt_name = prompt_name
        self.prompt_mask_name = prompt_mask_name
        self.prompt_interpolate = prompt_interpolate
        self.prompt_diffmap_name = prompt_diffmap_name
        self.prompt_scale_name = prompt_scale_name
        self.prompt_center_name = prompt_center_name
        self.target_name = target_name
        self.target_mask_name = target_mask_name
        self.target_clip = target_clip
        self.edge_mask_name = edge_mask_name
        self.prompt_set_none = prompt_set_none
        self.debug = debug
        self.debug_size = debug_size
        self.debug_scale = debug_scale

    def get_inputs(self, batch):
        # Extract and move tensors to the appropriate device and dtype if necessary.
        name = batch["meta_data"]["name"][0]
        total_iter = batch.get("total_iter", None)
        image = batch["image"]

        intrinsics = batch.get("intrinsics", None)
        extrinsics = batch.get("extrinsics", None)  # [n, 4, 4]

        image_show = batch.get("image_show", None)
        if image_show is not None:
            if image_show.ndim == 4:
                image_show = image_show.float().squeeze(0).numpy().transpose(1, 2, 0)
            else:
                image_show = image_show.float().squeeze(0).numpy().transpose(0, 2, 3, 1)

        if self.prompt_name is not None:
            prompt_depth = batch[self.prompt_name]
            prompt_mask = (
                batch[self.prompt_mask_name] if self.prompt_mask_name else None
            )
            if self.prompt_interpolate:
                import knn_interpolate

                prompt_depth_interpolate = torch.zeros_like(prompt_depth)
                knn_interpolate.k1_interpolate_batch(
                    prompt_depth.float().contiguous(),
                    prompt_mask.int().contiguous(),
                    prompt_depth_interpolate,
                )
                if self.debug:
                    self.debug_prompt_interpolate(prompt_depth, prompt_depth_interpolate)

                prompt_depth = prompt_depth_interpolate
            else:
                prompt_diffmap = (
                    batch[self.prompt_diffmap_name]
                    if self.prompt_diffmap_name
                    else None
                )

            prompt_scale = batch[self.prompt_scale_name][
                ..., None, None, None
            ]

            if self.prompt_center_name is not None:
                prompt_center = batch[self.prompt_center_name][
                    ..., None, None
                ]
            else:
                # Otherwise, create a tensor of zeros with the same shape as prompt_scale.
                prompt_center = None

            # Normalize the prompt depth if it is provided, using the maximum range and center values.
            prompt_depth_norm = self.normalize(prompt_depth, prompt_scale, prompt_center)

        else:
            prompt_depth = prompt_depth_norm = prompt_scale = prompt_center = prompt_mask = (
                prompt_diffmap
            ) = None

        if self.target_name is not None and self.target_name in batch:
            target = batch[self.target_name]
            valid_mask = batch[self.target_mask_name]

            if self.debug:
                logging.info(
                    f'{valid_mask.reshape(valid_mask.shape[0], -1).sum(-1),} {batch["meta_data"]["data_info"]["rgb"]}'
                )

            if self.prompt_interpolate:
                prompt_diffmap = torch.abs(target - prompt_depth)

            target_norm = self.normalize(target, prompt_scale, prompt_center)

            # Optionally clip the target depth to a specified range.
            if self.target_clip is not None:
                target_norm = target_norm.clip(self.target_clip[0], self.target_clip[1])

        else:
            target = target_norm = valid_mask = None

        # Extract edge mask from the batch if it exists.
        edge_mask = batch.get(self.edge_mask_name, None)
        if edge_mask is not None:
            edge_mask = edge_mask

        # After normalize
        if self.prompt_set_none:
            prompt_depth = prompt_depth_norm = prompt_scale = prompt_center = prompt_mask = (
                prompt_diffmap
            ) = None
        
        if self.debug_size:
            logging.info(f"image: {image.shape}, target: {target_norm.shape}, prompt: {prompt_depth.shape}")
        if self.debug_scale:
            bs = image.shape[0]
            logging.info(f"prompt_scale: {prompt_scale.reshape(-1).cpu().numpy().tolist()}, target_norm:[{target_norm[:, 2].reshape(bs, -1).min(dim=1)[0].cpu().numpy().tolist()}, {target_norm[:, 2].reshape(bs, -1).max(dim=1)[0].cpu().numpy().tolist()}]")

        ret_data = {
            'name' : name,
            'total_iter' : total_iter,
            'image' : image,
            'intrinsics' : intrinsics,
            'extrinsics' : extrinsics,
            'prompt_depth' : prompt_depth,
            'prompt_depth_norm' : prompt_depth_norm,
            'prompt_scale' : prompt_scale,
            'prompt_center' : prompt_center,
            'prompt_mask' : prompt_mask,
            'prompt_diffmap' : prompt_diffmap,
            'target' : target,
            'target_norm' : target_norm,
            'valid_mask' : valid_mask,
            'edge_mask' : edge_mask,
            'image_show' : image_show,
        }

        return ret_data

    def debug_prompt_interpolate(self, prompt_depth, prompt_depth_interpolate):
        for bi in range(prompt_depth_interpolate.shape[0]):
            prompt_depth_interpolate_norm = (
                prompt_depth_interpolate[bi : bi + 1, 2].detach().cpu().numpy()
            )
            prompt_depth_interpolate_norm = (
                prompt_depth_interpolate_norm - prompt_depth_interpolate_norm.min()
            ) / (prompt_depth_interpolate_norm.max() - prompt_depth_interpolate_norm.min() + 1e-6)
            prompt_depth_interpolate_norm = colorize_depth_maps(
                prompt_depth_interpolate_norm,
                0,
                1,
                cmap="turbo",
                valid_mask=np.ones(prompt_depth_interpolate_norm.shape).astype(bool),
            )
            prompt_depth_interpolate_norm.save(f"{bi:04d}_dense_depth.jpg")

            prompt_depth_norm = prompt_depth[bi : bi + 1, 2].detach().cpu().numpy()
            prompt_depth_norm = (prompt_depth_norm - prompt_depth_norm.min()) / (
                prompt_depth_norm.max() - prompt_depth_norm.min() + 1e-6
            )
            prompt_depth_norm = colorize_depth_maps(
                prompt_depth_norm,
                0,
                1,
                cmap="turbo",
                valid_mask=np.ones(prompt_depth_norm.shape).astype(bool),
            )
            prompt_depth_norm.save(f"{bi:04d}_sparse_depth.jpg")

    def normalize(self, depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
        if center is not None:
            depth = depth - center
        if scale is not None:
            depth = depth / scale
        return depth

class PostProcessor:
    def __init__(
        self,
        match_input_res=False,
        pointmap_match_input_res=False,
        post_align=False,
        align_name = None,
        align_mask_name=None,
        output_depth2point=False,
        output_conf_thresh=0,
        output_global_pointmap=False,
    ):
        self.match_input_res = match_input_res
        self.pointmap_match_input_res = pointmap_match_input_res
        self.post_align = post_align
        self.align_name = align_name
        self.align_mask_name = align_mask_name
        self.output_depth2point = output_depth2point
        self.output_conf_thresh = output_conf_thresh
        self.output_global_pointmap = output_global_pointmap

        self.cache_dict = dict()

    def postprocess(self, raw_data, input_data, infer_data):
        image = input_data['image']
        target = input_data['target']
        intrinsics = input_data['intrinsics']
        extrinsics = input_data['extrinsics']
        prompt_depth = input_data['prompt_depth']
        prompt_scale = input_data['prompt_scale']
        prompt_center = input_data['prompt_center']
        image_show = input_data['image_show']
    
        pointmap_pred = infer_data['pointmap_pred']
        confidence_pred = infer_data['confidence_pred']
        gradient_pred = infer_data['gradient_pred']
        prompt_confidence_pred = infer_data['prompt_confidence_pred']

        align_gt = align_mask = None
        if self.match_input_res:
            align_gt = raw_data[self.align_name]
        if self.match_input_res and self.post_align:
            align_mask = raw_data[self.align_mask_name]

        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        if image_show is not None:
            pointmap_color = image_show
        else:
            pointmap_color = image.cpu().float().squeeze(0).numpy().transpose(1, 2, 0)
            pointmap_color = (pointmap_color + 1) * 0.5 * 255

        # Denormalize the predicted pointmap using the provided max range and center values.
        pointmap_pred = self.denormalize(pointmap_pred, scale=prompt_scale, center=prompt_center)

        if self.output_depth2point and pointmap_pred.shape[1] == 3:
            pointmap_pred = pointmap_pred[:, 2:3, :, :]

        # DepthMap trans to PointMap
        if pointmap_pred.shape[1] == 1:
            pointmap_pred = self.depth_to_point(pointmap_pred, K=intrinsics, device=pointmap_pred.device)

        # Adjust pointmap_color shape
        if (
            pointmap_pred.shape[2] != pointmap_color.shape[0]
            or pointmap_pred.shape[3] != pointmap_color.shape[1]
        ):
            pointmap_color = cv2.resize(
                pointmap_color,
                dsize=(pointmap_pred.shape[3], pointmap_pred.shape[2]),
                interpolation=cv2.INTER_LINEAR,
            )
        pointmap_color = pointmap_color.reshape(-1, 3)

        # Convert the predicted pointmap to a NumPy array and extract the depth channel.
        depth = pointmap_pred[0, 2].cpu().float().numpy().clip(1e-3)
        pointmap_h, pointmap_w = depth.shape[:2]
        pointmap_pred = (
            pointmap_pred.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
        )

        # Filtered noise pointmap base on confidence_pred
        if confidence_pred is not None:
            confidence_pred = confidence_pred.cpu()[0, 0].float().numpy()
            filtered_pointmap = pointmap_pred[confidence_pred.reshape(-1) > self.output_conf_thresh]
            filtered_pointmap_color = pointmap_color[
                confidence_pred.reshape(-1) > self.output_conf_thresh
            ]
        else:
            filtered_pointmap = filtered_pointmap_color = None

        # If matching input resolution is required, resize the predicted pointmap and color pointmap to match the ground truth depth.
        pointmap_pred_align = pointmap_color_align = None
        filtered_pointmap_pred_align = filtered_pointmap_color_align = None
        if self.match_input_res:
            align_gt = align_gt.squeeze().numpy()
            h, w = align_gt.shape
            depth = cv2.resize(
                depth,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            ratio_h, ratio_w = 1.0 * h / pointmap_h, 1.0 * w / pointmap_w
            intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio_w
            intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio_h
            intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio_w
            intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio_h

            if confidence_pred is not None:
                confidence_pred = cv2.resize(
                    confidence_pred,
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

            if self.pointmap_match_input_res:
                pointmap_pred_align = self.depth_to_point(
                    torch.from_numpy(depth)[None, None],
                    K=intrinsics.cpu(),
                    device="cpu",
                    cache=False,
                )
                pointmap_pred_align = (
                    pointmap_pred_align.squeeze(0).permute(1, 2, 0).numpy().reshape(-1, 3)
                )

                pointmap_color_align = cv2.resize(
                    pointmap_color.reshape(pointmap_h, pointmap_w, 3),
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                ).reshape(-1, 3)

                if confidence_pred is not None:
                    filtered_pointmap_pred_align = pointmap_pred_align[
                        confidence_pred.reshape(-1) > self.output_conf_thresh
                    ]
                    filtered_pointmap_color_align = pointmap_color_align[
                        confidence_pred.reshape(-1) > self.output_conf_thresh
                    ]

            if self.post_align:
                align_mask = align_mask.squeeze().numpy()
                depth = align_depth_least_square(
                    gt_arr=align_gt,
                    pred_arr=depth,
                    valid_mask_arr=align_mask,
                    return_scale_shift=False,
                    max_resolution=None,
                )

        # Optionally convert the ground truth pointmap to a NumPy array.
        if target is not None:
            pointmap_gt = target.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)

        if intrinsics is not None:
            intrinsics = intrinsics.cpu().squeeze(0).numpy()

        if extrinsics is not None:
            extrinsics = extrinsics.cpu().squeeze(0).numpy()

        # Global predicted point cloud
        pointmap_gt_global = pointmap_pred_global = None
        if self.output_global_pointmap and extrinsics is not None:
            extrinsics_inv = np.linalg.inv(extrinsics)
            R = extrinsics_inv[:3, :3]
            T = extrinsics_inv[:3, 3]
            if pointmap_gt is not None:
                pointmap_gt_global = np.dot(R, pointmap_gt.T).T + T
            pointmap_pred_global = np.dot(R, pointmap_pred.T).T + T

        if gradient_pred is not None:
            gradient_pred = gradient_pred.cpu().float().squeeze(0).numpy().transpose(1, 2, 0)

        if prompt_confidence_pred is not None:
            prompt_confidence_pred = prompt_confidence_pred.cpu().float()[0, 0].numpy()

        if prompt_depth is not None:
            _, _, prompt_h, prompt_w = prompt_depth.shape
            prompt_pointmap = (
                prompt_depth.squeeze(0)[:3].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
            )
        else:
            prompt_h = prompt_w = None

        return DepthOutput(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_align=depth,
            pointmap=pointmap_pred,
            pointmap_gt=pointmap_gt,
            pointmap_color=pointmap_color,
            pointmap_h=pointmap_h,
            pointmap_w=pointmap_w,
            confidence=confidence_pred,
            filtered_pointmap=filtered_pointmap,
            filtered_pointmap_color=filtered_pointmap_color,
            pointmap_gt_global=pointmap_gt_global,
            pointmap_global=pointmap_pred_global,
            depth_grad=gradient_pred,
            input_confidence=prompt_confidence_pred,
            prompt_pointmap=prompt_pointmap,
            prompt_h=prompt_h,
            prompt_w=prompt_w,
            pointmap_align=pointmap_pred_align,
            pointmap_color_align=pointmap_color_align,
            filtered_pointmap_align=filtered_pointmap_pred_align,
            filtered_pointmap_color_align=filtered_pointmap_color_align,
        )

    def depth_to_point(self, depth, K, device=None, cache=True):
        B, C, H, W = depth.shape
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
        pts = depth.flatten(2) * rays_d
        depth = pts.reshape(B, 3, H, W)
        return depth

    def denormalize(self, depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
        if scale is not None:
            depth = depth * scale
        if center is not None:
            depth = depth + center
        return depth

def summary_eval(frame_eval_results, eval_metrics):
    eval_results = {}
    if isinstance(eval_metrics.metrics[0], str):
        for metric_name in eval_metrics.metrics:
            eval_results[metric_name] = sum(
                result[metric_name].item() for result in frame_eval_results
            ) / len(frame_eval_results)
    else:
        for metric_obj in eval_metrics.metrics:
            for metric_name in metric_obj.metrics:
                eval_results[metric_name] = sum(
                    result[metric_name].item() for result in frame_eval_results
                ) / len(frame_eval_results)

    return eval_results