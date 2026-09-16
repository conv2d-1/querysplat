import logging
import os

import cv2
import numpy as np
import torch

from hAlgorithm.modules.pipelines2.base import Pipeline
from hAlgorithm.modules.pipelines2.utils.outputs import DenseMatchingOutput
from hAlgorithm.modules.models2.external.romav2.models.romav2.romav2 import _map_confidence
import torch.nn.functional as F
from hAlgorithm.modules.pipelines2.utils.visualize import save_warp
from hAlgorithm.modules.models2.external.romav2.models.romav2.geometry import bhwc_interpolate

class RomaPipeline(Pipeline):
    """Pipeline for roma."""

    def __init__(
        self, 
        intrinsics_name=None,
        extrinsics_name=None,
        save_output_cfg=None,
        flow_supervise=False,
        training_config=None,
        **kwargs
    ):
        super(RomaPipeline, self).__init__(**kwargs)

        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.flow_supervise = flow_supervise

        self.save_output_cfg = dict(
            save_everything=True,
            gt_out_dir="gt",
        )
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)
        
        self.training_config = dict(
            match_scales=[1, 2, 4],
            loss_weight=dict(
                coarse=1,
                refiner_1=1,
                refiner_2=1,
                refiner_4=1,
            ),
            alpha=0.5,
            scale_c=1e-4,
            conf_weight=0.01,
        )
        if training_config is not None:
            self.training_config.update(training_config)
        assert 1 in self.training_config["match_scales"], "Must Contain scale 1"

    def get_inputs(self, batch):
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter

        image = batch["image"].to(device=self.device, dtype=self.dtype)

        intrinsics = extrinsics = image_show = None
        warp = warp_mask = None
        
        if "warp" in batch:
            warp = batch["warp"].to(device=self.device)
            warp_mask = batch["warp_mask"].to(device=self.device)

        if self.intrinsics_name is not None and self.intrinsics_name in batch:
            intrinsics = batch[self.intrinsics_name].to(device=self.device)

        if self.extrinsics_name is not None and self.extrinsics_name in batch:
            extrinsics = batch[self.extrinsics_name].to(device=self.device)

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                image_show = image_show.float().numpy()

        return (name, total_iter, meta_data, image, image_show, warp, warp_mask, intrinsics, extrinsics)

    def get_warp_scales(self, batch):
        warp_scales = {}
        warp_mask_scales = {}
        if "warp" in batch:
            warp_scales[1] = batch["warp"].to(device=self.device)
            warp_mask_scales[1] = batch["warp_mask"].to(device=self.device)
        
        for scale in self.training_config["match_scales"]:
            if scale == 1:
                continue
            if f"warp_{scale}" in batch:
                warp_scales[scale] = batch[f"warp_{scale}"].to(device=self.device)
                warp_mask_scales[scale] = batch[f"warp_mask_{scale}"].to(device=self.device)
        return warp_scales, warp_mask_scales

    def calc_gt_warp(self, depths, intrinsics, extrinsics, pair_idx, fisheye=False):
        warp_scales = {scale: [] for scale in self.training_config["match_scales"]}
        warp_mask_scales = {scale: [] for scale in self.training_config["match_scales"]}
        B, V, C, H, W = depths.shape
        for pair in pair_idx:
            depth1 = depths[:, pair[0], -1] # B, H, W
            depth2 = depths[:, pair[1], -1]
            
            K1 = intrinsics[:, pair[0]] # B, 3, 3
            K2 = intrinsics[:, pair[1]]
            
            T1 = extrinsics[:, pair[0]] # B, 4, 4
            T2 = extrinsics[:, pair[1]]
            
            T_1to2 = (T2 @ T1.inverse())
            for scale in self.training_config["match_scales"]:
                h1, w1 = int(H / scale), int(W / scale)
                if fisheye:
                    from hAlgorithm.datasets_fisheye.utils.fisheye_warp import get_gt_warp_fisheye_equidistant as fisheye_depth_to_warp
                    warp, warp_mask = fisheye_depth_to_warp(
                        depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode='bilinear',
                        H=h1, W=w1
                    )
                else:
                    from hAlgorithm.modules.models2.external.romav2.utils.utils import get_gt_warp as romav1_depth_to_warp
                    warp, warp_mask = romav1_depth_to_warp(
                        depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode='bilinear',
                        H=h1, W=w1
                    )
                warp_scales[scale].append(warp.to(dtype=depths.dtype))
                warp_mask_scales[scale].append(warp_mask)
        for k, v in warp_scales.items():
            warp_scales[k] = torch.stack(v, dim=1)
        for k, v in warp_mask_scales.items():
            warp_mask_scales[k] = torch.stack(v, dim=1)
        return warp_scales, warp_mask_scales

    def train_step(self, batch):
        self.train()

        (
            name,
            total_iter,
            meta_data,
            image,
            image_show,
            warp,
            warp_mask,
            intrinsics,
            extrinsics,
        ) = self.get_inputs(batch)
        warp_scales, warp_mask_scales = self.get_warp_scales(batch)
        
        B, V, C, H, W = image.shape
        results = self.model(
            image=image,
        )
        matching = results["match"]
        
        # warp_AB, confidence_AB = matching["final"]["warp_AB"], matching["final"]["confidence_AB"]
        # coarse_warp_AB, coarse_confidence_AB = matching["coarse"]["warp_AB"], matching["coarse"]["confidence_AB"]
        
        # generate gt online
        
        pair_idx = matching.pop("pair_idx")
        num_pair = len(pair_idx)
        
        if len(warp_scales) == 0:
            depth = batch["depth"].to(self.device)
            camera_type = meta_data.get("camera_type", "PINHOLE")
            if isinstance(camera_type, (list, tuple)):
                camera_type = camera_type[0]
            fisheye = (camera_type == "FISHEYE_EQUIDISTANT")
            warp_scales, warp_mask_scales = self.calc_gt_warp(depth, intrinsics, extrinsics, pair_idx, fisheye=fisheye)

        total_loss_dict = dict(ce_loss={}, reg_loss={})
        loss_weight = self.training_config["loss_weight"]
        # Compute losses
        alpha = self.training_config["alpha"]
        scale_c = self.training_config["scale_c"]
        conf_weight = self.training_config["conf_weight"]
        
        for key, val in matching.items():
            if key in ["final", "pair_idx"]:
                # final is the same as refiner_1
                continue
            if key in ["coarse"]:
                scale = 4
            elif key.startswith("refiner"):
                scale = int(key[-1])
            else:
                continue
            
            loss_w = loss_weight.get(key, 0)
            
            # pred_warp_AB = val["warp_AB"][:,0]
            # pred_conf_AB = val["confidence_AB"][:,0]
            
            # warp_gt = warp_scales[scale][:, 0]
            # warp_mask_gt = warp_mask_scales[scale][:, 0]
            # breakpoint()
            
            def reshape_pair_to_batch(value):
                return value.reshape(B*num_pair, *value.shape[2:]) if value is not None else None

            pred_warp_AB = reshape_pair_to_batch(val["warp_AB"]).float()
            pred_conf_AB = reshape_pair_to_batch(val["confidence_AB"]).float()
            
            warp_gt = reshape_pair_to_batch(warp_scales[scale])
            warp_mask_gt = reshape_pair_to_batch(warp_mask_scales[scale])
            
            # overlap_AB, precision_AB = _map_confidence(
            #     confidence=confidence_AB, threshold=None
            # )
            
            epe = (pred_warp_AB - warp_gt).norm(dim=-1)
            epe = torch.clamp(epe, min=1e-6)
            ce_loss = F.binary_cross_entropy_with_logits(pred_conf_AB[..., 0], warp_mask_gt)
            
            a = alpha
            cs = scale_c * scale
            x = epe[warp_mask_gt > 0.99]
            reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
            
            if not torch.any(reg_loss):
                reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
            total_loss_dict["ce_loss"][key] = ce_loss.mean() * conf_weight * loss_w
            total_loss_dict["reg_loss"][key] = reg_loss.mean() * loss_w

        total_loss_dict = {key: sum(val.values()) / len(val) for key, val in total_loss_dict.items()}
        total_loss = sum(total_loss_dict.values())
        # total_loss_dict["loss"] = total_loss.item()

        return total_loss, total_loss_dict

    @torch.no_grad()
    def infer(self, **batch):
        self.eval()

        (
            name,
            total_iter,
            meta_data,
            image,
            image_show,
            warp,
            warp_mask,
            intrinsics,
            extrinsics,
        ) = self.get_inputs(batch)

        results = self.model(
            image=image,
        )

        B, V, C, H, W = image_show.shape
        
        assert V == 2
        image0 = np.ascontiguousarray(image_show[:, :1].reshape(-1, C, H, W).transpose(0, 2, 3, 1).astype(np.uint8))
        image1 = np.ascontiguousarray(image_show[:, 1:].reshape(-1, C, H, W).transpose(0, 2, 3, 1).astype(np.uint8))

        matching = results["match"]
        warp_AB, confidence_AB = matching["final"]["warp_AB"], matching["final"]["confidence_AB"]
        coarse_warp_AB, coarse_confidence_AB = matching["coarse"]["warp_AB"], matching["coarse"]["confidence_AB"]
        
        overlap_AB, precision_AB = _map_confidence(
            confidence=confidence_AB, threshold=None
        )
        overlap_AB[overlap_AB>0.5] = 1
        overlap_AB[overlap_AB<0.5] = 0
        
        coarse_overlap_AB = coarse_confidence_AB
        
        if warp is None:
            depth = batch["depth"].to(self.device)
            camera_type = meta_data.get("camera_type", "PINHOLE")
            if isinstance(camera_type, (list, tuple)):
                camera_type = camera_type[0]
            fisheye = (camera_type == "FISHEYE_EQUIDISTANT")
            warp_scales, warp_mask_scales = self.calc_gt_warp(depth, intrinsics, extrinsics, matching["pair_idx"], fisheye=fisheye)
            warp = warp_scales[1]
            warp_mask = warp_mask_scales[1]
        
        if extrinsics is not None:
            extrinsics_image0 = extrinsics[:, 0:1].expand(-1, V - 1, -1, -1).cpu().float().numpy().reshape(-1, 4, 4)
            extrinsics_image1 = extrinsics[:, 1:].cpu().float().numpy().reshape(-1, 4, 4)
        else:
            extrinsics_image0 = extrinsics_image1 = None

        if intrinsics is not None:
            intrinsics_image0 = intrinsics[:, 0:1].expand(-1, V - 1, -1, -1).cpu().float().numpy().reshape(-1, 3, 3)
            intrinsics_image1 = intrinsics[:, 1:].cpu().float().numpy().reshape(-1, 3, 3)
        else:
            intrinsics_image0 = intrinsics_image1 = None
        
        num_pair = len(matching["pair_idx"])
        outputs_list = []
        for i in range(num_pair):
            outputs = DenseMatchingOutput(
                image0=image0,
                image1=image1,
                warp=warp_AB[:, i],
                overlap=overlap_AB[:, i],
                warp_coarse=bhwc_interpolate(coarse_warp_AB[:, i], (H, W)),
                overlap_coarse=bhwc_interpolate(coarse_overlap_AB[:, i], (H, W)),
                warp_gt=warp[:, i],
                overlap_gt=warp_mask[:, i][..., None],
                pred_covariance=precision_AB[:, i],
                intrinsics_image0=intrinsics_image0[0] if intrinsics_image0 is not None else None,
                intrinsics_image1=intrinsics_image1[0] if intrinsics_image1 is not None else None,
                extrinsics_image0=extrinsics_image0[0] if extrinsics_image0 is not None else None,
                extrinsics_image1=extrinsics_image1[0] if extrinsics_image1 is not None else None,
            )
            outputs_list.append(outputs)
        return outputs_list

    def get_out_dir(self, out_dir, data_idx=None):
        if data_idx is not None:
            matching_out_dir = os.path.join(out_dir, f"matching/{data_idx:06d}")
        else:
            matching_out_dir = os.path.join(out_dir, "matching")
        return matching_out_dir

    def get_gt_out_dir(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None
        gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        return gt_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        data_idx = meta_data["data_idx"][0]

        gt_out_dir = self.get_gt_out_dir(out_dir)
        out_dir = self.get_out_dir(out_dir, data_idx=data_idx)
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(len(outputs_list)):
            outputs = outputs_list[idx]
            image0 = outputs.image0[0]
            image1 = outputs.image1[0]
            warp = outputs.warp
            overlap = outputs.overlap
            warp_coarse = outputs.warp_coarse
            overlap_coarse = outputs.overlap_coarse
            
            warp_gt = outputs.warp_gt
            overlap_gt = outputs.overlap_gt
            
            p1 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
            p2 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            
            # Images Visualization
            if gt_out_dir is not None and self.save_output_cfg["save_everything"]:
                image0_save_path = os.path.join(gt_out_dir, f"images/{data_idx:03d}/{idx:03d}/img0.jpg")
                image1_save_path = os.path.join(gt_out_dir, f"images/{data_idx:03d}/{idx:03d}/img1.jpg")

                if not os.path.exists(image0_save_path) or not os.path.exists(image1_save_path):
                    os.makedirs(os.path.dirname(image0_save_path), exist_ok=True)
                    cv2.imwrite(image0_save_path, p1)
                    cv2.imwrite(image1_save_path, p2)

                warp_gt_path = os.path.join(gt_out_dir, f"matching/{data_idx:03d}")
                if warp_gt is not None and not os.path.exists(warp_gt_path):
                    os.makedirs(warp_gt_path, exist_ok=True)
                    save_warp(p1, p2, warp_gt, overlap_gt,
                        warp_gt_path, "warp", idx
                    )

            # Prediction
            if warp is not None:
                save_warp(p1, p2, warp, overlap, out_dir, "warp", idx)

            if warp_coarse is not None:
                save_warp(p1, p2, warp_coarse, overlap, out_dir, "coarse_warp", idx)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        pass
