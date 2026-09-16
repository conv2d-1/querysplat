import logging

import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.mvfr_v3 import MVFRPipeline
from hAlgorithm.modules.pipelines2.utils.outputs import DenseMatchingOutput
from hAlgorithm.modules.models2.external.romav2.models.romav2.romav2 import _map_confidence
from hAlgorithm.utils import instantiate_from_config


class MVFRMatchQueryPipeline(MVFRPipeline):
    """Pipeline that replaces dense match head with query-based match prediction.

    Model output is organized by pair: (B, num_pair, Q, ...).
    GT warp is also per pair.

    Inherits from mvfr_v3.MVFRPipeline to reuse matching-pair data handling,
    GT warp computation, postprocess, and visualization. Overrides train_step
    and infer for query-based match training only (no depth/points/camera).
    """

    def __init__(
        self,
        query_match_loss=None,
        edge_mask_name=None,
        training_sub_pixel_scale=1,
        testing_sub_pixel_scale=1,
        pair_idx=[(0, 1)],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.query_match_loss = instantiate_from_config(query_match_loss)
        self.edge_mask_name = edge_mask_name
        self.training_sub_pixel_scale = training_sub_pixel_scale
        self.testing_sub_pixel_scale = testing_sub_pixel_scale

        self.pair_idx = pair_idx

    def _get_edge_mask(self, batch):
        edge_mask = None
        if self.edge_mask_name is not None and self.edge_mask_name in batch:
            edge_mask = batch[self.edge_mask_name].to(device=self.device)
        return edge_mask

    def _get_prediction_branches(self, results):
        branches = {
            "coarse": {
                "warp": results["warp"],
                "confidence": results["confidence"],
            }
        }

        refiner_scales = sorted(
            {
                int(key[len("refiner_"):-len("_warp")])
                for key in results
                if key.startswith("refiner_") and key.endswith("_warp")
            }
        )
        for scale in refiner_scales:
            warp_key = f"refiner_{scale}_warp"
            conf_key = f"refiner_{scale}_confidence"
            if conf_key in results:
                branches[f"refiner_{scale}"] = {
                    "warp": results[warp_key],
                    "confidence": results[conf_key],
                }
        return branches

    def _get_final_prediction_branch(self, branches):
        refiner_names = sorted(
            [name for name in branches if name.startswith("refiner_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        if refiner_names:
            return refiner_names[0], refiner_names[1:]
        return "coarse", []

    def _sample_gt_at_query(self, warp_gt, mask_gt, query, B, num_pair, H, W):
        uv = query.uv  # (Q, 2) in [0, 1]
        uv_grid = (uv * 2 - 1).unsqueeze(0).unsqueeze(2)  # (1, Q, 1, 2)
        uv_grid = uv_grid.expand(B * num_pair, -1, -1, -1)  # (B*P, Q, 1, 2)

        warp_gt_chw = warp_gt.view(B * num_pair, H, W, 2).permute(0, 3, 1, 2)
        mask_gt_chw = mask_gt.view(B * num_pair, H, W).unsqueeze(1).float()

        gt_warp_sampled = F.grid_sample(
            warp_gt_chw,
            uv_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        gt_mask_sampled = F.grid_sample(
            mask_gt_chw,
            uv_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        gt_warp_sampled = gt_warp_sampled.squeeze(-1).permute(0, 2, 1)
        gt_mask_sampled = gt_mask_sampled.squeeze(-1).squeeze(1)
        gt_mask_sampled = gt_mask_sampled >= 0.999

        return gt_warp_sampled, gt_mask_sampled

    def _compute_gt_warp_at_scale1(self, depths, intrinsics, extrinsics, camera_type="PINHOLE", meta_data=None):
        """Compute GT warp at full resolution for given pairs."""
        B, V, C, H, W = depths.shape
        warp_list, mask_list = [], []
        for pair in self.pair_idx:
            depth1 = depths[:, pair[0], -1]
            depth2 = depths[:, pair[1], -1]
            K1 = intrinsics[:, pair[0]]
            K2 = intrinsics[:, pair[1]]
            T1 = extrinsics[:, pair[0]]
            T2 = extrinsics[:, pair[1]]
            T_1to2 = T2 @ T1.inverse()

            if camera_type == "FISHEYE_BLENDER":
                from hAlgorithm.datasets_fisheye.utils.fisheye_warp import get_gt_warp_fisheye_blender as blender_depth_to_warp
                distort_k = meta_data["distort_k"]
                sensor_size = meta_data["sensor_size"]
                crop_offset = meta_data.get("crop_offset", None)
                assert distort_k is not None and sensor_size is not None

                distort_k = distort_k.to(depth1.device)
                sensor_size = sensor_size.to(depth1.device)
                if crop_offset is not None:
                    crop_offset = crop_offset.to(depth1.device)

                if distort_k.dim() == 3:
                    k_coeffs1 = distort_k[:, pair[0]]
                    k_coeffs2 = distort_k[:, pair[1]]
                else:
                    k_coeffs1 = k_coeffs2 = distort_k

                if sensor_size.dim() == 3:
                    ss1 = sensor_size[:, pair[0]]
                    ss2 = sensor_size[:, pair[1]]
                else:
                    ss1 = ss2 = sensor_size

                co1 = co2 = 0
                if crop_offset is not None:
                    if crop_offset.dim() == 3:
                        co1 = crop_offset[:, pair[0]]
                        co2 = crop_offset[:, pair[1]]
                    else:
                        co1 = co2 = crop_offset
                warp, warp_mask = blender_depth_to_warp(
                    depth1, depth2, T_1to2, K1, K2,
                    k_coeffs1, k_coeffs2, ss1, ss2,
                    crop_offset1=co1, crop_offset2=co2,
                    depth_interpolation_mode="bilinear",
                    H=H, W=W,
                )
            elif camera_type == "FISHEYE_EQUIDISTANT":
                from hAlgorithm.datasets_fisheye.utils.fisheye_warp import get_gt_warp_fisheye_equidistant as _warp_fn
                warp, warp_mask = _warp_fn(
                    depth1, depth2, T_1to2, K1, K2,
                    depth_interpolation_mode="bilinear",
                    H=H, W=W,
                )
            else:
                from hAlgorithm.modules.models2.external.romav2.utils.utils import get_gt_warp as _warp_fn
                warp, warp_mask = _warp_fn(
                    depth1, depth2, T_1to2, K1, K2,
                    depth_interpolation_mode="bilinear",
                    H=H, W=W,
                )

            warp_list.append(warp.to(dtype=depths.dtype))
            mask_list.append(warp_mask)

        warp_gt = torch.stack(warp_list, dim=1)  # (B, P, H, W, 2)
        mask_gt = torch.stack(mask_list, dim=1)  # (B, P, H, W)
        return warp_gt, mask_gt

    def train_step(self, batch):
        self.train()

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
            extrinsics_noise,
            prompt_extrinsics,
            rgb_mask,
        ) = self.get_inputs(batch)

        meta_data["sub_pixel_scale"] = self.training_sub_pixel_scale

        edge_mask = self._get_edge_mask(batch)
        target_match_gt_depth, target_match_gt_depth_intrinsics, target_match_gt_extrinsics = self.get_match_inputs(batch)

        camera_type = self.get_camera_type(meta_data)

        if prompt_extrinsics is not None:
            w2c = prompt_extrinsics
        elif extrinsics_noise is not None and extrinsics is not None:
            w2c = torch.matmul(extrinsics_noise, extrinsics)
        else:
            w2c = extrinsics

        results = self.model(
            image,
            edge_mask=edge_mask,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=ray_directions,
            w2c=w2c,
            ray_world=ray_world,
            meta_data=meta_data,
            pair_idx=self.pair_idx,
        )

        B = image.shape[0]
        H, W = image.shape[-2:]
        num_pair = len(self.pair_idx)

        query = results["query"]
        branches = self._get_prediction_branches(results)

        warp_gt, mask_gt = self._compute_gt_warp_at_scale1(
            target_match_gt_depth,
            target_match_gt_depth_intrinsics,
            target_match_gt_extrinsics,
            camera_type=camera_type,
            meta_data=meta_data,
        )
        gt_warp_sampled, gt_mask_sampled = self._sample_gt_at_query(
            warp_gt, mask_gt, query, B, num_pair, H, W
        )

        total_loss, total_loss_dict = 0, dict()

        if self.query_match_loss is not None:
            match_weight = self.task_weight.get("match", 1.0)
            total_loss_dict["match"] = 0
            for branch_name, branch in branches.items():
                pred_warp_flat = branch["warp"].view(B * num_pair, -1, 2)
                pred_conf_flat = branch["confidence"].view(
                    B * num_pair, -1, branch["confidence"].shape[-1]
                )
                loss, loss_dict = self.query_match_loss(
                    pred_warp=pred_warp_flat,
                    pred_conf=pred_conf_flat,
                    gt_warp=gt_warp_sampled,
                    gt_mask=gt_mask_sampled,
                    gt_h=H,
                    gt_w=W,
                    name=name,
                )

                branch_total = loss * match_weight
                total_loss += branch_total
                total_loss_dict[f"match_{branch_name}"] = branch_total
                total_loss_dict["match"] += branch_total
                for key, val in loss_dict.items():
                    total_loss_dict[f"match_{branch_name}_{key}"] = val * match_weight

        if scale is not None:
            total_loss_dict["scale"] = round(scale.max().cpu().item(), 2)
        if image is not None:
            aspect_ratio = image.shape[-1] / image.shape[-2]
            total_loss_dict["aspect_ratio"] = round(aspect_ratio, 2)
            total_loss_dict["bs"] = int(image.shape[0])
            total_loss_dict["view"] = int(image.shape[1])
            total_loss_dict["max_size"] = int(max(image.shape[-1], image.shape[-2]))

        if self.debug_rgb_path:
            if isinstance(meta_data["data_info"][0], (list, tuple)):
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0][0]['rgb']}")
            else:
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0]['rgb']}")

        return total_loss, total_loss_dict

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
            extrinsics_noise,
            prompt_extrinsics,
            rgb_mask,
        ) = self.get_inputs(batch)

        meta_data["sub_pixel_scale"] = self.testing_sub_pixel_scale

        target_match_gt_depth, target_match_gt_depth_intrinsics, target_match_gt_extrinsics = self.get_match_inputs(batch)
        camera_type = self.get_camera_type(meta_data)

        if prompt_extrinsics is not None:
            w2c = prompt_extrinsics
        elif extrinsics_noise is not None and extrinsics is not None:
            w2c = torch.matmul(extrinsics_noise, extrinsics)
        else:
            w2c = extrinsics

        if "use_amp" in batch and batch["use_amp"]:
            with torch.autocast("cuda", enabled=bool(batch["use_amp"]), dtype=batch["amp_dtype"]):
                results = self.model(
                    image,
                    scale=scale,
                    prompt_depth=prompt_depth,
                    intrinsics=intrinsics,
                    ray_directions=ray_directions,
                    w2c=w2c,
                    ray_world=ray_world,
                    meta_data=meta_data,
                    pair_idx=self.pair_idx,
                )
        else:
            results = self.model(
                image,
                scale=scale,
                prompt_depth=prompt_depth,
                intrinsics=intrinsics,
                ray_directions=ray_directions,
                w2c=w2c,
                ray_world=ray_world,
                meta_data=meta_data,
                pair_idx=self.pair_idx,
            )

        B, N, C, H, W = image_show.shape
        num_pair = len(self.pair_idx)
        query = results["query"]
        branches = self._get_prediction_branches(results)
        final_branch_name, refiner_branch_names = self._get_final_prediction_branch(branches)
        final_branch = branches[final_branch_name]
        coarse_branch = branches["coarse"]

        warp_AB = final_branch["warp"]  # (B, P, Q, 2)
        conf_AB = final_branch["confidence"]  # (B, P, Q, C)
        coarse_warp_AB = coarse_branch["warp"]
        coarse_conf_AB = coarse_branch["confidence"]

        # In eval mode, full_uv=True -> Q = H_q * W_q
        if query.full_uv:
            Q_h, Q_w = query.height, query.width
            warp_AB = warp_AB.view(B, num_pair, Q_h, Q_w, 2)
            conf_AB = conf_AB.view(B, num_pair, Q_h, Q_w, -1)
            coarse_warp_AB = coarse_warp_AB.view(B, num_pair, Q_h, Q_w, 2)
            coarse_conf_AB = coarse_conf_AB.view(B, num_pair, Q_h, Q_w, -1)

        if len(refiner_branch_names) > 0:
            warp_refine = {name.split("_")[1]:branches[name]["warp"] for name in refiner_branch_names}
            if query.full_uv:
                warp_refine = {s: warp.view(B, num_pair, Q_h, Q_w, 2) for s, warp in warp_refine.items()}
        else:
            warp_refine = None
        
        # Build overlap from confidence
        overlap_AB = conf_AB[..., :1]
        precision_AB = conf_AB[..., 1:4] # xyz confidence
        coarse_overlap_AB = coarse_conf_AB[..., :1]

        # Compute GT warp
        if target_match_gt_depth is not None:
            warp_gt, mask_gt = self._compute_gt_warp_at_scale1(
                target_match_gt_depth,
                target_match_gt_depth_intrinsics,
                target_match_gt_extrinsics,
                camera_type=camera_type,
                meta_data=meta_data,
            )
            # NOTE: 对齐训练逻辑
            depth_height, depth_width = target_match_gt_depth.shape[-2:]
            warp_gt, mask_gt = self._sample_gt_at_query(warp_gt, mask_gt, query, B, num_pair, depth_height, depth_width)
            warp_gt = warp_gt.reshape(B, num_pair, query.height, query.width, 2)
            mask_gt = mask_gt.reshape(B, num_pair, query.height, query.width)
        else:
            warp_gt = mask_gt = None

        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        mv_outputs = []
        for fi in range(frame_num):
            for vi in range(view_num):
                index = fi * view_num + vi

                single = self.postprocess(
                    pred_local_points=None,
                    pred_local_conf=None,
                    pred_global_points=None,
                    pred_global_conf=None,
                    pred_extrinsics=None,
                    pred_intrinsics=None,
                    image=image[:, index] if image is not None else None,
                    image_show=image_show[:, index] if image_show is not None else None,
                    scale=scale[:, index] if scale is not None else None,
                    prompt_depth=None,
                    target_local_depth=None,
                    target_global_points=None,
                    target_depth_mask=None,
                    intrinsics=intrinsics[:, index] if intrinsics is not None else None,
                    extrinsics=extrinsics[:, index] if extrinsics is not None else None,
                    align_data=align_data[:, 0] if align_data is not None else None,
                )

                single.rgb = (image[0, index].permute(1, 2, 0).cpu().numpy() + 1) * 0.5
                single.frame_index = fi
                single.view_index = vi
                single.total_index = index

                # Attach dense matching results for source views
                cur_pairs = [(pi, p) for pi, p in enumerate(self.pair_idx) if p[0] == index]
                matching_results = []
                for pair_i, cur_pair in cur_pairs:
                    image0 = np.ascontiguousarray(image_show[:, cur_pair[0]].reshape(-1, C, H, W).transpose(0, 2, 3, 1).astype(np.uint8))
                    image1 = np.ascontiguousarray(image_show[:, cur_pair[1]].reshape(-1, C, H, W).transpose(0, 2, 3, 1).astype(np.uint8))
                    matching_results.append(
                        DenseMatchingOutput(
                            image0=image0,
                            image1=image1,
                            warp=warp_AB[:, pair_i],
                            warp_coarse=coarse_warp_AB[:, pair_i],
                            warp_refine={s:warp[:, pair_i] for s, warp in warp_refine.items()} if warp_refine is not None else None,
                            overlap=overlap_AB[:, pair_i] if overlap_AB is not None else None,
                            overlap_coarse=coarse_overlap_AB[:, pair_i] if coarse_overlap_AB is not None else None,
                            warp_gt=warp_gt[:, pair_i] if warp_gt is not None else None,
                            overlap_gt=mask_gt[:, pair_i][..., None] if mask_gt is not None else None,
                            pred_covariance=precision_AB[:, pair_i] if precision_AB is not None else None,
                            extrinsics_image0=target_match_gt_extrinsics[:, cur_pair[0]].cpu()[0].numpy() if target_match_gt_extrinsics is not None else None,
                            extrinsics_image1=target_match_gt_extrinsics[:, cur_pair[1]].cpu()[0].numpy() if target_match_gt_extrinsics is not None else None,
                            intrinsics_image0=intrinsics[:, cur_pair[0]].cpu()[0].numpy() if intrinsics is not None else None,
                            intrinsics_image1=intrinsics[:, cur_pair[1]].cpu()[0].numpy() if intrinsics is not None else None,
                        )
                    )
                single.dense_matching = matching_results
                mv_outputs.append(single)

        return mv_outputs
