from typing import Union

import numpy as np
from diffusers.utils import BaseOutput
from PIL import Image


class DepthOutput(BaseOutput):
    """
    Output class for monocular depth prediction pipeline.

    Args:
        intrinsics (np.ndarray, optional):
            Camera intrinsics parameters as a 3x3 matrix, used to project 2D image coordinates into 3D space.
            Shape: [3, 3]. Defaults to None.

        extrinsics (np.ndarray, optional):
            Camera extrinsics parameters as a 4x4 matrix, used to project camera 3D space into global 3D space.
            Shape: [4, 4]. Defaults to None.

        depth (np.ndarray):
            Predicted depth map as a NumPy array, with depth values normalized to the range [0, 1].
            Shape: [H, W], where H is the height and W is the width of the image.

        depth_norm (np.ndarray, optional):
            Normalized depth map as a NumPy array, scaled to a specific range (e.g., meters).
            Shape: [H, W]. Defaults to None.

        depth_align (np.ndarray, optional):
            Aligned depth map as a NumPy array, adjusted to match the ground truth or reference depth.
            Shape: [H, W]. Defaults to None.

        pointmap (np.ndarray, optional):
            Point cloud representation of the depth map, where each pixel corresponds to a 3D point in space.
            Shape: [H, W, 3], where the third dimension represents the x, y, z coordinates of the point. Defaults to None.

        mask (np.ndarray, optional):
            Binary mask indicating valid depth regions. Pixels with a value of True are considered valid, while False indicates invalid or occluded regions.
            Shape: [H, W]. Defaults to None.

        pointmap_gt (np.ndarray, optional):
            Ground truth point cloud, if available. This can be used for evaluating the accuracy of the predicted point cloud.
            Shape: [H, W, 3]. Defaults to None.

        pointmap_color (np.ndarray, optional):
            Color information for the point cloud, typically obtained from the input RGB image.
            Shape: [H, W, 3]. Defaults to None.

        filtered_pointmap (np.ndarray, optional):
            Filtered point cloud based on confidence or other criteria.
            Shape: [H, W, 3]. Defaults to None.

        filtered_pointmap_color (np.ndarray, optional):
            Color information for the filtered point cloud.
            Shape: [H, W, 3]. Defaults to None.

        inconf_filtered_pointmap (np.ndarray, optional):
            Filtered point cloud for low-confidence regions.
            Shape: [H, W, 3]. Defaults to None.

        inconf_filtered_pointmap_color (np.ndarray, optional):
            Color information for the low-confidence filtered point cloud.
            Shape: [H, W, 3]. Defaults to None.

        invalid_mask (np.ndarray, optional):
            Boolean mask indicating invalid depth regions. Pixels with a value of True are considered invalid, while False indicates valid regions.
            Shape: [H, W]. Defaults to None.

        input_confidence (np.ndarray, optional):
            Confidence estimate for the depth prediction, typically calculated as the Median Absolute Deviation (MAD) from ensembling multiple predictions.
            Shape: [H, W]. Defaults to None.

        depth_colored (Image.Image, optional):
            Colorized depth map as a PIL Image, with the shape [H, W] and values in the range [0, 1].
            The colorization is typically done using a colormap to visualize the depth values more intuitively. Defaults to None.

        confidence (np.ndarray, optional):
            Confidence map for the depth prediction. Shape: [H, W]. Defaults to None.

        depth_grad (np.ndarray, optional):
            Gradient map for the depth prediction. Shape: [2, H, W]. Defaults to None.

        pointmap_gt_global (np.ndarray, optional):
            Global ground truth point cloud. Shape: [H, W, 3]. Defaults to None.

        pointmap_global (np.ndarray, optional):
            Global predicted point cloud. Shape: [H, W, 3]. Defaults to None.

    """

    intrinsics: np.ndarray = None
    extrinsics: np.ndarray = None

    depth: np.ndarray = None
    depth_norm: np.ndarray = None
    depth_align: np.ndarray = None

    rel_depth: np.ndarray = None

    pointmap: np.ndarray = None
    pointmap_h: int = None
    pointmap_w: int = None
    mask: np.ndarray = None
    pointmap_gt: np.ndarray = None
    pointmap_gt_h: int = None
    pointmap_gt_w: int = None
    pointmap_color: np.ndarray = None

    filtered_pointmap: np.ndarray = None
    filtered_pointmap_color: np.ndarray = None
    inconf_filtered_pointmap: np.ndarray = None
    inconf_filtered_pointmap_color: np.ndarray = None
    filtered_pointmap_global: np.ndarray = None

    invalid_mask: np.ndarray = None
    input_confidence: np.ndarray = None
    depth_colored: Union[None, Image.Image] = None
    confidence: Union[None, np.ndarray] = None
    depth_grad: Union[None, np.ndarray] = None

    pointmap_gt_global: np.ndarray = None
    pointmap_global: np.ndarray = None

    prompt_pointmap: np.ndarray = None

    pointmap_align: np.ndarray = None
    pointmap_color_align: np.ndarray = None
    filtered_pointmap_align: np.ndarray = None
    filtered_pointmap_color_align: np.ndarray = None

    prompt_h: int = None
    prompt_w: int = None

    disparity: np.ndarray = None
    seq_disparity: list = None

    prompt_scale: float = None


class ReconstructOutput(DepthOutput):
    gaussians = None
    gs_xyz: np.ndarray = None

    rgb: np.ndarray = None
    object_mask: np.ndarray = None

    render_rgb: np.ndarray = None
    render_depth: np.ndarray = None
    dgs_render_rgb: np.ndarray = None
    dgs_render_depth: np.ndarray = None
    render_num: int = None

    intrinsics_pred: np.ndarray = None
    extrinsics_pred: np.ndarray = None

    sf_pointmap: np.ndarray = None
    sf_confidence: np.ndarray = None

    glb_mv_pointmap: np.ndarray = None
    glb_mv_confidence: np.ndarray = None
    glb2local_pointmap: np.ndarray = None

    local2glb_pointmap: np.ndarray = None
    local2glb_confidence: np.ndarray = None

    frame_index: int = None
    view_index: int = None
    total_index: int = None

    track_pred: np.ndarray = None
    track_vis_pred: np.ndarray = None
    track_confidence_pred: np.ndarray = None

    track_gt: np.ndarray = None
    track_vis_gt: np.ndarray = None

    # Sparse motion query metadata (for visualization)
    motion_queries_uv: np.ndarray = None        # [Q, 2] source pixel coords
    motion_queries_tgt_frame: np.ndarray = None  # [Q]    target frame index per query
    motion_queries_gt_3d_src: np.ndarray = None   # [Q, 3] GT source 3D in camera frame (abs scale)

    normal: np.ndarray = None
    motion_mask: np.ndarray = None

    normal_gt: np.ndarray = None
    normal_mask: np.ndarray = None
    invalid_mask_gt: np.ndarray = None
    motion_mask_gt: np.ndarray = None

    depth_mask: np.ndarray = None
    
    dense_matching: list = None

    ray_directions: np.ndarray = None
    ray_directions_gt: np.ndarray = None

    track_3d: dict = None


class MatchingOutput(BaseOutput):
    image0 = None
    image1 = None
    matches = None
    matches_gt = None
    cov0 = None
    cov1 = None

    extrinsics_image0 = None
    extrinsics_image1 = None

class DenseMatchingOutput(BaseOutput):
    image0 = None
    image1 = None
    
    warp = None
    warp_coarse = None
    warp_refine = None
    warp_gt = None
    
    overlap = None
    overlap_coarse = None
    overlap_gt = None

    pred_covariance = None
    cov0 = None
    cov1 = None
    
    extrinsics_image0 = None
    extrinsics_image1 = None
    
    intrinsics_image0 = None
    intrinsics_image1 = None
    matches = None
    matches_gt = None

class Track3DOutput(BaseOutput):
    query_uv = None
    query_width = None
    query_height = None

    warp3d = None
    warp3d_uv = None
    warp2d = None

    src_index = None
    tgt_index = None
    
    src_3d_gt = None
    tgt_3d_gt = None

    src_2d_gt = None
    tgt_2d_gt = None

    src_valids_gt = None
    tgt_valids_gt = None

    src_visibs_gt = None
    tgt_visibs_gt = None

    motion_mask = None

    warp3d_delta = None
    src_points = None
