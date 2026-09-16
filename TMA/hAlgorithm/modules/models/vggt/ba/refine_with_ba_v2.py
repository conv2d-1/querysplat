import os
import cv2
import copy
import torch
import trimesh
import pycolmap
import numpy as np
import torch.nn.functional as F

from .track_predict import *
from .vggsfm_utils import build_vggsfm_tracker, initialize_feature_extractors
from .np_to_pycolmap import batch_np_matrix_to_pycolmap, pycolmap_to_batch_np_matrix

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
        self.prepare_models(device=images.device, dtype=images.dtype)

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
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = self.predict_tracks(
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

    
    def predict_tracks(
        self,
        images,
        conf=None,
        points_3d=None,
        masks=None,
        max_query_pts=2048,
        query_frame_num=5,
        keypoint_extractor="aliked+sp",
        max_points_num=163840,
        fine_tracking=True,
        complete_non_vis=True,
        model_path=None,
        dino_name=None,
        dino_path=None,
    ):
        """
        Predict tracks for the given images and masks.

        TODO: support non-square images
        TODO: support masks


        This function predicts the tracks for the given images and masks using the specified query method
        and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

        Args:
            images: Tensor of shape [S, 3, H, W] containing the input images.
            conf: Tensor of shape [S, 1, H, W] containing the confidence scores. Default is None.
            points_3d: Tensor containing 3D points. Default is None.
            masks: Optional tensor of shape [S, 1, H, W] containing masks. Default is None.
            max_query_pts: Maximum number of query points. Default is 2048.
            query_frame_num: Number of query frames to use. Default is 5.
            keypoint_extractor: Method for keypoint extraction. Default is "aliked+sp".
            max_points_num: Maximum number of points to process at once. Default is 163840.
            fine_tracking: Whether to use fine tracking. Default is True.
            complete_non_vis: Whether to augment non-visible frames. Default is True.

        Returns:
            pred_tracks: Numpy array containing the predicted tracks.
            pred_vis_scores: Numpy array containing the visibility scores for the tracks.
            pred_confs: Numpy array containing the confidence scores for the tracks.
            pred_points_3d: Numpy array containing the 3D points for the tracks.
            pred_colors: Numpy array containing the point colors for the tracks. (0, 255)
        """

        device = images.device
        # dtype = images.dtype
        # tracker = build_vggsfm_tracker(model_path=model_path).to(device, dtype)
        tracker = self.tracker

        # Find query frames
        query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device, model_name=dino_name, pretrain=dino_path, dino_v2_model=self.dino_v2_model)

        # Add the first image to the front if not already present
        if 0 in query_frame_indexes:
            query_frame_indexes.remove(0)
        query_frame_indexes = [0, *query_frame_indexes]

        # TODO: add the functionality to handle the masks
        # keypoint_extractors = initialize_feature_extractors(
        #     max_query_pts, extractor_method=keypoint_extractor, device=device
        # )
        keypoint_extractors = self.keypoint_extractors

        pred_tracks = []
        pred_vis_scores = []
        pred_confs = []
        pred_points_3d = []
        pred_colors = []

        fmaps_for_tracker = tracker.process_images_to_fmaps(images)

        if fine_tracking:
            print("For faster inference, consider disabling fine_tracking")

        for query_index in query_frame_indexes:
            print(f"Predicting tracks for query frame {query_index}")
            pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = self._forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                keypoint_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device,
            )

            pred_tracks.append(pred_track)
            pred_vis_scores.append(pred_vis)
            pred_confs.append(pred_conf)
            pred_points_3d.append(pred_point_3d)
            pred_colors.append(pred_color)

        if complete_non_vis:
            pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = self._augment_non_visible_frames(
                pred_tracks,
                pred_vis_scores,
                pred_confs,
                pred_points_3d,
                pred_colors,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                keypoint_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                min_vis=500,
                non_vis_thresh=0.1,
                device=device,
            )

        pred_tracks = np.concatenate(pred_tracks, axis=1)
        pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
        pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
        pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
        pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

        # from vggt.utils.visual_track import visualize_tracks_on_images
        # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>0.2, out_dir="track_visuals")

        return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


    def _forward_on_query(
        self,
        query_index,
        images,
        conf,
        points_3d,
        fmaps_for_tracker,
        keypoint_extractors,
        tracker,
        max_points_num,
        fine_tracking,
        # conf_ratio=None,
        # conf_thresh=1.2,
        device,
    ):
        """
        Process a single query frame for track prediction.

        Args:
            query_index: Index of the query frame
            images: Tensor of shape [S, 3, H, W] containing the input images
            conf: Confidence tensor
            points_3d: 3D points tensor
            fmaps_for_tracker: Feature maps for the tracker
            keypoint_extractors: Initialized feature extractors
            tracker: VGG-SFM tracker
            max_points_num: Maximum number of points to process at once
            fine_tracking: Whether to use fine tracking
            device: Device to use for computation

        Returns:
            pred_track: Predicted tracks
            pred_vis: Visibility scores for the tracks
            pred_conf: Confidence scores for the tracks
            pred_point_3d: 3D points for the tracks
            pred_color: Point colors for the tracks (0, 255)
        """
        frame_num, _, height, width = images.shape

        query_image = images[query_index]
        query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
        query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

        # Extract the color at the keypoint locations
        query_points_long = query_points.squeeze(0).round().long()
        pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
        pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

        # Query the confidence and points_3d at the keypoint locations
        if (conf is not None) and (points_3d is not None):
            # assert height == width
            # assert conf.shape[-2] == conf.shape[-1]
            # assert conf.shape[:3] == points_3d.shape[:3]
            # scale = conf.shape[-1] / width

            # query_points_scaled = (query_points.squeeze(0) * scale).round().long()
            # query_points_scaled = query_points_scaled.cpu().numpy()
            
            scale_w = conf.shape[-1] / width
            scale_h = conf.shape[-2] / height
            query_points_scaled = query_points.squeeze(0).clone()
            query_points_scaled[:, 0] = query_points_scaled[:, 0] * scale_w
            query_points_scaled[:, 1] = query_points_scaled[:, 1] * scale_h
            query_points_scaled = query_points_scaled.round().long().cpu().numpy()

            pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
            pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

            # heuristic to remove low confidence points
            # should I export this as an input parameter?
            # valid_mask = pred_conf > 1.2

            conf_thresh = np.percentile(conf[query_index].reshape(-1), 30)
            valid_mask = pred_conf > conf_thresh

            if valid_mask.sum() > 512:
                query_points = query_points[:, valid_mask]  # Make sure shape is compatible
                pred_conf = pred_conf[valid_mask]
                pred_point_3d = pred_point_3d[valid_mask]
                pred_color = pred_color[valid_mask]
        else:
            pred_conf = None
            pred_point_3d = None

        reorder_index = calculate_index_mappings(query_index, frame_num, device=device)

        images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
        images_feed = images_feed[None]  # add batch dimension
        fmaps_feed = fmaps_feed[None]  # add batch dimension

        all_points_num = images_feed.shape[1] * query_points.shape[1]

        # Don't need to be scared, this is just chunking to make GPU happy
        if all_points_num > max_points_num:
            num_splits = (all_points_num + max_points_num - 1) // max_points_num
            query_points = torch.chunk(query_points, num_splits, dim=1)
        else:
            query_points = [query_points]

        pred_track, pred_vis, _ = predict_tracks_in_chunks(
            tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
        )

        pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)

        pred_track = pred_track.squeeze(0).float().cpu().numpy()
        pred_vis = pred_vis.squeeze(0).float().cpu().numpy()

        return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


    def _augment_non_visible_frames(
        self,
        pred_tracks: list,  # ← running list of np.ndarrays
        pred_vis_scores: list,  # ← running list of np.ndarrays
        pred_confs: list,  # ← running list of np.ndarrays for confidence scores
        pred_points_3d: list,  # ← running list of np.ndarrays for 3D points
        pred_colors: list,  # ← running list of np.ndarrays for colors
        images: torch.Tensor,
        conf,
        points_3d,
        fmaps_for_tracker,
        keypoint_extractors,
        tracker,
        max_points_num: int,
        fine_tracking: bool,
        *,
        min_vis: int = 500,
        non_vis_thresh: float = 0.1,
        device: torch.device = None,
    ):
        """
        Augment tracking for frames with insufficient visibility.

        Args:
            pred_tracks: List of numpy arrays containing predicted tracks.
            pred_vis_scores: List of numpy arrays containing visibility scores.
            pred_confs: List of numpy arrays containing confidence scores.
            pred_points_3d: List of numpy arrays containing 3D points.
            pred_colors: List of numpy arrays containing point colors.
            images: Tensor of shape [S, 3, H, W] containing the input images.
            conf: Tensor of shape [S, 1, H, W] containing confidence scores
            points_3d: Tensor containing 3D points
            fmaps_for_tracker: Feature maps for the tracker
            keypoint_extractors: Initialized feature extractors
            tracker: VGG-SFM tracker
            max_points_num: Maximum number of points to process at once
            fine_tracking: Whether to use fine tracking
            min_vis: Minimum visibility threshold
            non_vis_thresh: Non-visibility threshold
            device: Device to use for computation

        Returns:
            Updated pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, and pred_colors lists.
        """
        last_query = -1
        final_trial = False
        cur_extractors = keypoint_extractors  # may be replaced on the final trial

        while True:
            # Visibility per frame
            vis_array = np.concatenate(pred_vis_scores, axis=1)

            # Count frames with sufficient visibility using numpy
            sufficient_vis_count = (vis_array > non_vis_thresh).sum(axis=-1)
            non_vis_frames = np.where(sufficient_vis_count < min_vis)[0].tolist()

            if len(non_vis_frames) == 0:
                break

            print("Processing non visible frames:", non_vis_frames)

            # Decide the frames & extractor for this round
            if non_vis_frames[0] == last_query:
                # Same frame failed twice - final "all-in" attempt
                final_trial = True
                cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
                query_frame_list = non_vis_frames  # blast them all at once
            else:
                query_frame_list = [non_vis_frames[0]]  # Process one at a time

            last_query = non_vis_frames[0]

            # Run the tracker for every selected frame
            for query_index in query_frame_list:
                new_track, new_vis, new_conf, new_point_3d, new_color = self._forward_on_query(
                    query_index,
                    images,
                    conf,
                    points_3d,
                    fmaps_for_tracker,
                    cur_extractors,
                    tracker,
                    max_points_num,
                    fine_tracking,
                    device,
                )
                pred_tracks.append(new_track)
                pred_vis_scores.append(new_vis)
                pred_confs.append(new_conf)
                pred_points_3d.append(new_point_3d)
                pred_colors.append(new_color)

            if final_trial:
                break  # Stop after final attempt

        return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors

