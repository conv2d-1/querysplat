import logging
import os

import numpy as np
import torch

from hAlgorithm.modules.pipelines.visualize import save_image
from hAlgorithm.modules.utils.gaussians.camera import Camera, CameraList

from .outputs import ReconstructOutput
from .prompt_pointmap_pipeline_v2 import PromptPointMapPipeline


class DiFixDataPipeline(PromptPointMapPipeline):

    def __init__(
        self,
        extrinsics_name="extrinsics_reff",
        **kwargs,
    ):
        super(DiFixDataPipeline, self).__init__(extrinsics_name=extrinsics_name, **kwargs)

        self.output_type = ReconstructOutput
        self.extrinsics_c2w = self.model.extrinsics_c2w

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

    def train_step(self, batch, iterations):
        pass

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
            intrinsics = intrinsics.clone()

            results = self.render_custom(extrinsics, intrinsics, image.shape[-2], image.shape[-1])

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

    def visualize(self, outputs_list, meta_data, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f"save: {out_dir}")

        frame_num = len(outputs_list)
        data_idx = [int(meta_data_i["data_idx"][0]) for meta_data_i in meta_data]

        render_rgb_paths = []
        render_depth_paths = []
        for index in range(frame_num):
            prefix = f"index{index:03d}_"
            outputs = outputs_list[index]

            save_path = save_image(
                outputs.render_rgb.copy(),
                out_dir,
                f"{prefix}render_rgb",
                data_idx[index],
                info=False,
            )
            # render_rgb_paths.append(save_path)

            save_path = save_image(
                outputs.rgb.copy(),
                out_dir,
                f"{prefix}rgb",
                data_idx[index],
                info=False,
            )
            # render_rgb_paths.append(save_path)

            # if outputs.render_depth is not None:
            #     save_path = save_depth_map(
            #         outputs.render_depth.copy(),
            #         out_dir,
            #         f"000000/{prefix}depth",
            #         data_idx[index],
            #         info=False,
            #     )
            #     render_depth_paths.append(save_path)

        # if len(render_rgb_paths) > 0:
        #     save_path = os.path.join(out_dir, f"merge_render_rgb.jpg")
        #     grid_images(
        #         save_path=save_path,
        #         paths=render_rgb_paths,
        #         col=min(4, len(render_rgb_paths)),
        #     )

        # if len(render_depth_paths) > 0:
        #     save_path = os.path.join(out_dir, f"merge_render_depth.jpg")
        #     grid_images(
        #         save_path=save_path,
        #         paths=render_depth_paths,
        #         col=min(4, len(render_depth_paths)),
        #     )
