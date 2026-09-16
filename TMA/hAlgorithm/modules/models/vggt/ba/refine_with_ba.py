import os
import cv2
import copy
import torch
import trimesh
import pycolmap
import numpy as np
import torch.nn.functional as F

from .track_predict import predict_tracks
from .vggsfm_utils import build_vggsfm_tracker, initialize_feature_extractors
from .np_to_pycolmap import batch_np_matrix_to_pycolmap, pycolmap_to_batch_np_matrix


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


def get_shape(image, max_size):
    height, width = image.shape[-2:]
    if width < max_size and height < max_size:
        new_width = width
        new_height = height
    elif width >= height:
        new_width = max_size
        new_height = round(height * (new_width / width))
    else:
        new_height = max_size
        new_width = round(width * (new_height / height))

    return new_width, new_height


class Ba(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.tracker = None

    def prepare_models(self, dtype, device):
        if self.tracker is None:
            self.tracker = build_vggsfm_tracker(model_path=self.cfg["model_path"]).to(device, dtype)
            self.keypoint_extractors = initialize_feature_extractors(
                self.cfg["max_query_pts"],
                extractor_method=self.cfg["extractor_method"],
                device=device,
            )

            self.dino_v2_model = torch.hub.load(
                "hAlgorithm/modules/models/facebookresearch_dinov2_main",
                model_name=self.cfg.get("dino_name", "dinov2_vitb14_reg"),
                source="local",
                pretrained=False,
            )
            self.dino_v2_model.load_state_dict(
                torch.load(self.cfg["dino_path"], map_location="cpu", weights_only=False),
                strict=True,
            )

            self.dino_v2_model.eval()
            self.dino_v2_model = self.dino_v2_model.to(device)

    def __call__(
        self,
        images,
        outputs_list,
        image_size_hw,
        dtype=torch.float16,
    ):
        # self.prepare_models(device=images.device, dtype=images.dtype)

        if self.cfg.get("local2glb_points", True):
            points_3d = np.stack([outputs.local2glb_pointmap for outputs in outputs_list], axis=0)
            depth_conf = np.stack(
                [outputs.local2glb_confidence for outputs in outputs_list], axis=0
            )
        else:
            points_3d = np.stack([outputs.glb_mv_pointmap for outputs in outputs_list])
            depth_conf = np.stack([outputs.glb_mv_confidence for outputs in outputs_list], axis=0)

        points_3d = points_3d.reshape(-1, outputs_list[0].pointmap_h, outputs_list[0].pointmap_w, 3)
        intrinsic = np.stack([outputs.intrinsics_pred for outputs in outputs_list], axis=0)
        extrinsic = np.stack([outputs.extrinsics_pred for outputs in outputs_list], axis=0)

        new_width, new_height = get_shape(images, max_size=self.cfg["track_max_size"])
        images = F.interpolate(
            images, size=(new_height, new_width), mode="bilinear", align_corners=False
        )
        images = images / 255.0

        new_image_size = np.array([new_height, new_width])
        scale = (new_image_size / image_size_hw)[[1, 0]]
        shared_camera = self.cfg["shared_camera"]

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,  # [N, 3, H, W]
                conf=depth_conf,  # [N, H, W]
                points_3d=points_3d,  # [N, H, W, 3]
                masks=None,
                max_query_pts=self.cfg["max_query_pts"],
                query_frame_num=self.cfg["query_frame_num"],
                keypoint_extractor="aliked+sp",
                fine_tracking=self.cfg["fine_tracking"],
                model_path=self.cfg.get("model_path", None),
                dino_name=self.cfg.get("dino_name", None),
                dino_path=self.cfg.get("dino_path", None),
            )
            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        # intrinsic[:, :2, :] *= scale
        intrinsic[:, 0, :] *= scale[0]
        intrinsic[:, 1, :] *= scale[1]
        track_mask = pred_vis_scores > self.cfg["vis_thresh"]

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            new_image_size,
            masks=track_mask,
            max_reproj_error=self.cfg["max_reproj_error"],
            shared_camera=shared_camera,
            camera_type=self.cfg["camera_type"],
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        # trimesh.PointCloud(points_3d, colors=points_rgb).export("points.ply")

        return reconstruction

    