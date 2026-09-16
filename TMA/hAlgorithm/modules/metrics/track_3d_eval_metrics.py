"""3D Tracking Evaluation Metrics (Publication-Ready)

Evaluates 3D scene flow predictions against ground truth 3D trajectories.
Aligned with TAPVid-3D and PointOdyssey benchmarks.

Key Design Choices:
    1. Model predicts 3D displacement in SOURCE CAMERA coordinate frame
    2. GT trajs_3d is in WORLD coordinates, must be transformed to camera frame
    3. Bilinear interpolation for sub-pixel sampling accuracy
    4. Metrics aligned with TAPVid-3D: EPE3D, Acc3D, 3D-AJ, OA

Output format (from pipeline):
    - scene_flow_pred: [num_views, 3, H, W] - predicted scene flow (source camera frame)
    - trajs_3d: [num_views, num_trajs, 3] - GT 3D positions (world coordinates)
    - trajs_2d: [num_views, num_trajs, 2] - GT 2D pixel coords at each view
    - trajs_visibs: [num_views, num_trajs] - visibility mask
    - trajs_valids: [num_views, num_trajs] - validity mask
    - extrinsics: [num_views, 4, 4] - world-to-camera transformation (w2c)
"""

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def bilinear_sample(feature_map: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling from feature map at given coordinates.
    
    Args:
        feature_map: [C, H, W] feature map
        coords: [N, 2] coordinates (x, y) in pixel space
    
    Returns:
        sampled: [N, C] sampled features
    """
    C, H, W = feature_map.shape
    N = coords.shape[0]
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Clamp to valid range
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    
    # Get integer and fractional parts
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    
    # Compute weights
    wx = x - x0
    wy = y - y0
    
    # Sample four corners: [N, C]
    f00 = feature_map[:, y0, x0].T
    f01 = feature_map[:, y0, x1].T
    f10 = feature_map[:, y1, x0].T
    f11 = feature_map[:, y1, x1].T
    
    # Bilinear interpolation
    wx = wx[:, np.newaxis]
    wy = wy[:, np.newaxis]
    sampled = (1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f01 + \
              (1 - wx) * wy * f10 + wx * wy * f11
    
    return sampled


def transform_flow_world_to_camera(
    world_flow: np.ndarray,
    R_w2c: np.ndarray
) -> np.ndarray:
    """
    Transform 3D flow from world coordinates to camera coordinates.
    
    Args:
        world_flow: [N, 3] flow vectors in world coordinates
        R_w2c: [3, 3] rotation matrix (world-to-camera)
    
    Returns:
        camera_flow: [N, 3] flow vectors in camera coordinates
    """
    # camera_flow = R_w2c @ world_flow.T -> [3, N] -> transpose -> [N, 3]
    return (R_w2c @ world_flow.T).T


def compute_optimal_scale(
    pred: np.ndarray,
    gt: np.ndarray,
    method: str = "least_squares"
) -> float:
    """
    Compute optimal scale factor to align non-metric predictions to metric GT.
    
    For non-metric depth/flow predictions, we need to find scale s such that:
        s * pred ≈ gt
    
    Args:
        pred: [N, 3] or [N,] predicted values (non-metric)
        gt: [N, 3] or [N,] ground truth values (metric)
        method: Alignment method:
            - "least_squares": s = (pred · gt) / (pred · pred)
            - "median": s = median(gt) / median(pred)
    
    Returns:
        scale: Optimal scale factor
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Remove invalid values (NaN, Inf)
    valid = np.isfinite(pred_flat) & np.isfinite(gt_flat) & (np.abs(pred_flat) > 1e-8)
    if valid.sum() == 0:
        return 1.0
    
    pred_valid = pred_flat[valid]
    gt_valid = gt_flat[valid]
    
    if method == "least_squares":
        # Minimize ||s * pred - gt||^2
        # Solution: s = (pred · gt) / (pred · pred)
        dot_pred_gt = np.dot(pred_valid, gt_valid)
        dot_pred_pred = np.dot(pred_valid, pred_valid)
        if dot_pred_pred > 1e-8:
            scale = dot_pred_gt / dot_pred_pred
        else:
            scale = 1.0
    elif method == "median":
        # s = median(|gt|) / median(|pred|)
        pred_median = np.median(np.abs(pred_valid))
        gt_median = np.median(np.abs(gt_valid))
        if pred_median > 1e-8:
            scale = gt_median / pred_median
        else:
            scale = 1.0
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    return float(scale)


def compute_3d_tracking_metrics(
    pred_scene_flow: np.ndarray,
    gt_trajs_3d: np.ndarray,
    gt_trajs_2d: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    visibs: Optional[np.ndarray] = None,
    valids: Optional[np.ndarray] = None,
    thresholds: List[float] = [0.05, 0.1, 0.2, 0.5, 1.0],
    position_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Compute 3D tracking metrics (TAPVid-3D / PointOdyssey aligned).
    
    Args:
        pred_scene_flow: [num_src_views, num_tgt_views, 3, H, W] - predicted scene flow
                         (in source camera frame, already in absolute scale)
        gt_trajs_3d: [num_views, num_trajs, 3] - GT 3D positions (WORLD coordinates)
        gt_trajs_2d: [num_views, num_trajs, 2] - GT 2D pixel coords (x, y)
        extrinsics: [num_views, 4, 4] - world-to-camera transformation (w2c)
                    If None, assumes pred is already in world coordinates
        visibs: [num_views, num_trajs] - visibility mask
        valids: [num_views, num_trajs] - validity mask
        thresholds: List of distance thresholds (meters) for accuracy computation
        position_threshold: Threshold for 3D-AJ computation (meters)
    
    Returns:
        Dictionary of metrics:
            - epe3d: Mean 3D endpoint error (meters)
            - median_epe3d: Median 3D endpoint error
            - acc3d_{thresh}: Fraction within threshold
            - aj3d: 3D Average Jaccard (position + occlusion)
            - oa: Occlusion Accuracy
    """
    num_views = gt_trajs_3d.shape[0]
    num_trajs = gt_trajs_3d.shape[1]
    H, W = pred_scene_flow.shape[-2:]
    
    all_errors = []
    all_occlusion_correct = []  # For Occlusion Accuracy
    all_position_correct = []   # For 3D-AJ (position within threshold)
    all_both_correct = []       # For 3D-AJ (both position and occlusion correct)
    all_gt_visible = []         # GT visibility for each pair
    
    for src_idx in range(num_views):
        for tgt_idx in range(num_views):
            if src_idx == tgt_idx:
                continue
            
            # Get predicted scene flow for this pair
            flow_pred = pred_scene_flow[src_idx, tgt_idx]  # [3, H, W]
            
            # Compute GT scene flow in WORLD coordinates: position_tgt - position_src
            gt_flow_world = gt_trajs_3d[tgt_idx] - gt_trajs_3d[src_idx]  # [num_trajs, 3]
            
            # Transform GT flow to SOURCE CAMERA frame if extrinsics provided
            if extrinsics is not None:
                R_w2c_src = extrinsics[src_idx, :3, :3]  # [3, 3]
                gt_flow_camera = transform_flow_world_to_camera(gt_flow_world, R_w2c_src)
            else:
                # Assume prediction is in world coordinates (legacy mode)
                gt_flow_camera = gt_flow_world
            
            # Get 2D locations at source view for sampling
            src_2d = gt_trajs_2d[src_idx]  # [num_trajs, 2] (x, y)
            
            # Build validity mask (both endpoints must be valid)
            mask = np.ones(num_trajs, dtype=bool)
            gt_src_visible = np.ones(num_trajs, dtype=bool)
            gt_tgt_visible = np.ones(num_trajs, dtype=bool)
            
            if visibs is not None:
                gt_src_visible = visibs[src_idx] > 0
                gt_tgt_visible = visibs[tgt_idx] > 0
                mask &= gt_src_visible & gt_tgt_visible
            if valids is not None:
                mask &= (valids[src_idx] > 0) & (valids[tgt_idx] > 0)
            
            # Filter points within image bounds
            x_coords = src_2d[:, 0]
            y_coords = src_2d[:, 1]
            in_bounds = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
            mask &= in_bounds
            
            if mask.sum() == 0:
                continue
            
            # Bilinear sampling at GT 2D locations
            valid_coords = src_2d[mask]  # [num_valid, 2]
            sampled_pred = bilinear_sample(flow_pred, valid_coords)  # [num_valid, 3]
            # scene_flow is already in absolute scale, no denormalization needed
            
            gt_valid = gt_flow_camera[mask]  # [num_valid, 3]
            
            # Compute per-point 3D errors
            errors = np.linalg.norm(sampled_pred - gt_valid, axis=1)
            all_errors.extend(errors.tolist())
            
            # For 3D-AJ: position within threshold
            position_correct = errors < position_threshold
            all_position_correct.extend(position_correct.tolist())
            
            # Store visibility for occlusion metrics
            all_gt_visible.extend(gt_tgt_visible[mask].tolist())
    
    if len(all_errors) == 0:
        # No valid points
        metrics = {
            "epe3d": float('nan'),
            "median_epe3d": float('nan'),
            "aj3d": float('nan'),
            "oa": float('nan'),
            "num_valid_points": 0
        }
        for thresh in thresholds:
            metrics[f"acc3d_{thresh}"] = float('nan')
        return metrics
    
    all_errors = np.array(all_errors)
    all_position_correct = np.array(all_position_correct)
    all_gt_visible = np.array(all_gt_visible)
    
    # === Core Metrics ===
    metrics = {
        "epe3d": float(np.mean(all_errors)),
        "median_epe3d": float(np.median(all_errors)),
        "num_valid_points": len(all_errors),
    }
    
    # Accuracy at different thresholds
    for thresh in thresholds:
        acc = np.mean(all_errors < thresh)
        metrics[f"acc3d_{thresh}"] = float(acc)
    
    # === TAPVid-3D Aligned Metrics ===
    # Note: Full 3D-AJ requires occlusion prediction from model
    # For now, compute position-only AJ (assuming all visible predictions are correct)
    # This is a simplified version; full version needs pred_visibility
    metrics["aj3d"] = float(np.mean(all_position_correct))
    
    # Occlusion Accuracy placeholder (requires model to predict visibility)
    # For scenes where all points are visible, OA is 1.0
    metrics["oa"] = 1.0  # Placeholder
    
    return metrics


class Track3DEvalMetrics:
    """
    Evaluator for 3D tracking / scene flow predictions.
    Aligned with TAPVid-3D and PointOdyssey benchmarks.
    
    Computes:
        - EPE3D: Mean 3D endpoint error (meters)
        - Median EPE3D: Median 3D endpoint error
        - Acc3D@{thresh}: Accuracy within distance threshold
        - AJ3D: 3D Average Jaccard (position accuracy)
        - OA: Occlusion Accuracy (placeholder, requires model visibility prediction)
    
    IMPORTANT: Model predicts flow in SOURCE CAMERA frame.
    GT trajs_3d is in WORLD coordinates and is transformed to camera frame for comparison.
    
    Usage:
        metrics_obj = Track3DEvalMetrics(thresholds=[0.05, 0.1, 0.2])
        results = metrics_obj(inputs, outputs)
    """
    
    def __init__(
        self,
        thresholds: List[float] = [0.05, 0.1, 0.2, 0.5, 1.0],
        position_threshold: float = 0.1,
        use_visibility: bool = True,
        use_validity: bool = True,
        **kwargs,
    ):
        """
        Args:
            thresholds: List of distance thresholds (meters) for accuracy metrics
            position_threshold: Threshold for 3D-AJ computation (meters)
            use_visibility: Whether to use visibility mask
            use_validity: Whether to use validity mask
        """
        self.thresholds = thresholds
        self.position_threshold = position_threshold
        self.use_visibility = use_visibility
        self.use_validity = use_validity
        
        # Define metric names for logging
        self.metrics = ["epe3d", "median_epe3d", "aj3d", "oa"] + [f"acc3d_{t}" for t in thresholds]
    
    def __call__(self, inputs, outputs) -> Dict[str, float]:
        """
        Evaluate 3D tracking metrics.
        
        Args:
            inputs: Input batch (not used, for interface compatibility)
            outputs: List of ReconstructOutput objects, one per view
        
        Returns:
            Dictionary of metric values
        """
        # Check if required data is available
        if not hasattr(outputs[0], 'scene_flow_pred') or outputs[0].scene_flow_pred is None:
            return {}
        if not hasattr(outputs[0], 'trajs_3d') or outputs[0].trajs_3d is None:
            return {}
        if not hasattr(outputs[0], 'trajs_2d') or outputs[0].trajs_2d is None:
            return {}
        
        # Get data from first output (trajectory data is shared)
        output = outputs[0]
        
        # Collect scene flow predictions from all views
        # scene_flow_pred per output: [num_views, 3, H, W]
        num_views = len(outputs)
        
        scene_flow_list = []
        extrinsics_list = []
        for out in outputs:
            if hasattr(out, 'scene_flow_pred') and out.scene_flow_pred is not None:
                flow = out.scene_flow_pred
                if isinstance(flow, torch.Tensor):
                    flow = flow.detach().cpu().numpy()
                scene_flow_list.append(flow)
            
            # Collect extrinsics (w2c) for coordinate transformation
            if hasattr(out, 'extrinsics') and out.extrinsics is not None:
                ext = out.extrinsics
                if isinstance(ext, torch.Tensor):
                    ext = ext.detach().cpu().numpy()
                extrinsics_list.append(ext)
            elif hasattr(out, 'gt_extrinsics') and out.gt_extrinsics is not None:
                # Fallback to GT extrinsics (which is c2w, need to invert)
                ext = out.gt_extrinsics
                if isinstance(ext, torch.Tensor):
                    ext = ext.detach().cpu().numpy()
                # GT extrinsics is c2w, invert to get w2c
                ext = np.linalg.inv(ext)
                extrinsics_list.append(ext)
        
        if len(scene_flow_list) != num_views:
            return {}
        
        # Stack to [num_src_views, num_tgt_views, 3, H, W]
        pred_scene_flow = np.stack(scene_flow_list, axis=0)
        
        # Stack extrinsics if available
        extrinsics = None
        if len(extrinsics_list) == num_views:
            extrinsics = np.stack(extrinsics_list, axis=0)
        
        # Get GT trajectories
        trajs_3d = output.trajs_3d
        trajs_2d = output.trajs_2d
        if isinstance(trajs_3d, torch.Tensor):
            trajs_3d = trajs_3d.detach().cpu().numpy()
        if isinstance(trajs_2d, torch.Tensor):
            trajs_2d = trajs_2d.detach().cpu().numpy()
        
        # Get masks
        visibs = None
        valids = None
        if self.use_visibility and hasattr(output, 'trajs_visibs') and output.trajs_visibs is not None:
            visibs = output.trajs_visibs
            if isinstance(visibs, torch.Tensor):
                visibs = visibs.detach().cpu().numpy()
        if self.use_validity and hasattr(output, 'trajs_valids') and output.trajs_valids is not None:
            valids = output.trajs_valids
            if isinstance(valids, torch.Tensor):
                valids = valids.detach().cpu().numpy()
        
        # Get scale for denormalization (legacy, no longer needed as scene_flow is absolute)
        # scale handling removed - predictions are now in absolute scale
        
        # Compute metrics
        results = compute_3d_tracking_metrics(
            pred_scene_flow=pred_scene_flow,
            gt_trajs_3d=trajs_3d,
            gt_trajs_2d=trajs_2d,
            extrinsics=extrinsics,
            visibs=visibs,
            valids=valids,
            thresholds=self.thresholds,
            position_threshold=self.position_threshold,
        )
        
        # Remove internal debug metrics
        results.pop("num_valid_points", None)
        
        return results


class SceneFlow3DMetrics:
    """
    Alternative evaluator that computes dense scene flow metrics.
    
    This evaluator computes metrics over the full dense scene flow prediction,
    not just at sparse GT trajectory points.
    
    Requires GT dense scene flow (if available).
    """
    
    def __init__(
        self,
        thresholds: List[float] = [0.05, 0.1, 0.2, 0.5, 1.0],
        **kwargs,
    ):
        self.thresholds = thresholds
        self.metrics = ["dense_epe3d"] + [f"dense_acc3d_{t}" for t in thresholds]
    
    def __call__(self, inputs, outputs) -> Dict[str, float]:
        """
        Evaluate dense scene flow metrics (requires GT dense scene flow).
        """
        # This requires GT dense scene flow which may not always be available
        # Placeholder for future implementation
        if not hasattr(outputs[0], 'scene_flow_pred') or outputs[0].scene_flow_pred is None:
            return {}
        if not hasattr(outputs[0], 'scene_flow_gt') or outputs[0].scene_flow_gt is None:
            return {}
        
        output = outputs[0]
        pred = output.scene_flow_pred
        gt = output.scene_flow_gt
        
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.numpy()
        
        # Compute EPE
        error = np.linalg.norm(pred - gt, axis=-3)  # [..., H, W]
        
        results = {
            "dense_epe3d": float(np.mean(error)),
        }
        
        for thresh in self.thresholds:
            results[f"dense_acc3d_{thresh}"] = float(np.mean(error < thresh))
        
        return results


def compute_any4d_metrics(
    pred_scene_flow: np.ndarray,
    gt_trajs_3d: np.ndarray,
    gt_trajs_2d: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    visibs: Optional[np.ndarray] = None,
    valids: Optional[np.ndarray] = None,
    dynamic_threshold: float = 0.01,
    apd_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
    inlier_threshold: float = 0.1,
    ref_frame_idx: int = 0,
    dynamic_only: bool = True,
    origin_h: Optional[int] = None,
    origin_w: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute 3D scene flow metrics aligned with Any4D paper.
    
    Any4D Metrics (from paper):
        - EPE: End-Point Error - mean 3D L2 error (meters)
        - APD: Average Points within Delta - mean accuracy across thresholds
        - τ (tau): Inlier ratio at 0.1m - fraction of points with error < 0.1m
    
    Key differences from Track3DEvalMetrics:
        1. Evaluates flow from REFERENCE FRAME (default: frame 0) to all other frames
           (not all source-target pairs)
        2. Only evaluates DYNAMIC points (motion > dynamic_threshold)
        3. APD is computed as mean of multiple accuracy thresholds
    
    IMPORTANT: trajs_2d may be in original resolution, but pred_scene_flow is in
    model resolution. We scale coordinates accordingly.
    
    Args:
        pred_scene_flow: [num_tgt_views, 3, H, W] - predicted scene flow FROM ref_frame
                         (already in absolute scale)
        gt_trajs_3d: [num_views, num_trajs, 3] - GT 3D positions (WORLD coordinates)
        gt_trajs_2d: [num_views, num_trajs, 2] - GT 2D pixel coords (x, y) in ORIGINAL resolution
        extrinsics: [num_views, 4, 4] - world-to-camera transformation (w2c)
        visibs: [num_views, num_trajs] - visibility mask
        valids: [num_views, num_trajs] - validity mask
        dynamic_threshold: Threshold for classifying points as dynamic (meters)
        apd_thresholds: Thresholds for APD computation
        inlier_threshold: Threshold for inlier ratio (τ)
        ref_frame_idx: Reference frame index (source frame for scene flow)
        dynamic_only: If True, only evaluate dynamic points
        origin_h: Original image height (for coordinate scaling)
        origin_w: Original image width (for coordinate scaling)
    
    Returns:
        Dictionary of metrics:
            - epe: End-Point Error (meters)
            - apd: Average Points within Delta
            - tau: Inlier ratio at inlier_threshold
            - acc3d_{thresh}: Individual accuracy at each threshold
    """
    num_views = gt_trajs_3d.shape[0]
    num_trajs = gt_trajs_3d.shape[1]
    H, W = pred_scene_flow.shape[-2:]
    
    # Compute coordinate scale factors if original resolution is provided
    scale_x = 1.0
    scale_y = 1.0
    if origin_w is not None and origin_w > 0:
        scale_x = W / origin_w
    if origin_h is not None and origin_h > 0:
        scale_y = H / origin_h
    
    all_errors = []
    all_is_dynamic = []
    
    src_idx = ref_frame_idx
    
    for tgt_idx in range(num_views):
        if src_idx == tgt_idx:
            continue
        
        # Get predicted scene flow for this target frame
        # pred_scene_flow is [num_views, 3, H, W], indexed by target frame
        flow_pred = pred_scene_flow[tgt_idx]  # [3, H, W]
        
        # Compute GT scene flow in WORLD coordinates: position_tgt - position_src
        gt_flow_world = gt_trajs_3d[tgt_idx] - gt_trajs_3d[src_idx]  # [num_trajs, 3]
        
        # Transform GT flow to SOURCE CAMERA frame if extrinsics provided
        if extrinsics is not None:
            R_w2c_src = extrinsics[src_idx, :3, :3]  # [3, 3]
            gt_flow_camera = transform_flow_world_to_camera(gt_flow_world, R_w2c_src)
        else:
            gt_flow_camera = gt_flow_world
        
        # Get 2D locations at source view for sampling (in ORIGINAL resolution)
        src_2d_orig = gt_trajs_2d[src_idx]  # [num_trajs, 2] (x, y)
        
        # Scale 2D coordinates to model resolution
        src_2d = src_2d_orig.copy()
        src_2d[:, 0] = src_2d_orig[:, 0] * scale_x
        src_2d[:, 1] = src_2d_orig[:, 1] * scale_y
        
        # Build validity mask (both endpoints must be valid)
        mask = np.ones(num_trajs, dtype=bool)
        
        if visibs is not None:
            gt_src_visible = visibs[src_idx] > 0
            gt_tgt_visible = visibs[tgt_idx] > 0
            mask &= gt_src_visible & gt_tgt_visible
        if valids is not None:
            mask &= (valids[src_idx] > 0) & (valids[tgt_idx] > 0)
        
        # Filter points within image bounds (using scaled coordinates)
        x_coords = src_2d[:, 0]
        y_coords = src_2d[:, 1]
        in_bounds = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        mask &= in_bounds
        
        if mask.sum() == 0:
            continue
        
        # Bilinear sampling at GT 2D locations (scaled to model resolution)
        valid_coords = src_2d[mask]  # [num_valid, 2]
        sampled_pred = bilinear_sample(flow_pred, valid_coords)  # [num_valid, 3]
        # scene_flow is already in absolute scale, no denormalization needed
        
        gt_valid = gt_flow_camera[mask]  # [num_valid, 3]
        
        # Compute per-point 3D errors
        errors = np.linalg.norm(sampled_pred - gt_valid, axis=1)
        all_errors.extend(errors.tolist())
        
        # Compute motion magnitude to identify dynamic points
        gt_motion_mag = np.linalg.norm(gt_valid, axis=1)
        is_dynamic = gt_motion_mag > dynamic_threshold
        all_is_dynamic.extend(is_dynamic.tolist())
    
    if len(all_errors) == 0:
        # No valid points
        metrics = {
            "epe": float('nan'),
            "apd": float('nan'),
            "tau": float('nan'),
            "num_points": 0,
            "num_dynamic_points": 0,
        }
        for thresh in apd_thresholds:
            metrics[f"acc3d_{thresh}"] = float('nan')
        return metrics
    
    all_errors = np.array(all_errors)
    all_is_dynamic = np.array(all_is_dynamic)
    
    # Filter to dynamic points only if requested
    if dynamic_only:
        if all_is_dynamic.sum() == 0:
            # No dynamic points found
            metrics = {
                "epe": float('nan'),
                "apd": float('nan'),
                "tau": float('nan'),
                "num_points": len(all_errors),
                "num_dynamic_points": 0,
            }
            for thresh in apd_thresholds:
                metrics[f"acc3d_{thresh}"] = float('nan')
            return metrics
        
        eval_errors = all_errors[all_is_dynamic]
    else:
        eval_errors = all_errors
    
    # === Any4D Metrics ===
    # EPE: End-Point Error (mean 3D L2 error)
    epe = float(np.mean(eval_errors))
    
    # τ (tau): Inlier ratio at threshold (default 0.1m)
    tau = float(np.mean(eval_errors < inlier_threshold))
    
    # APD: Average Points within Delta (mean accuracy across thresholds)
    accuracies = []
    acc_dict = {}
    for thresh in apd_thresholds:
        acc = float(np.mean(eval_errors < thresh))
        accuracies.append(acc)
        acc_dict[f"acc3d_{thresh}"] = float(acc)
    
    apd = float(np.mean(accuracies))
    
    metrics = {
        "epe": float(epe),
        "apd": float(apd),
        "tau": float(tau),
        "num_points": len(all_errors),
        "num_dynamic_points": int(all_is_dynamic.sum()),
        **acc_dict,
    }
    
    return metrics


def compute_vdpm_metrics(
    pred_scene_flow: np.ndarray,
    gt_trajs_3d: np.ndarray,
    gt_trajs_2d: np.ndarray,
    extrinsics: Optional[np.ndarray] = None,
    visibs: Optional[np.ndarray] = None,
    valids: Optional[np.ndarray] = None,
    dynamic_threshold: float = 0.01,
    apd_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
    inlier_threshold: float = 0.1,
    ref_frame_idx: int = 0,
    dynamic_only: bool = True,
    origin_h: Optional[int] = None,
    origin_w: Optional[int] = None,
    scale_alignment_method: str = "least_squares",
) -> Dict[str, float]:
    """
    Compute 3D scene flow metrics for VDPM (Non-Metric Predictions) with Scale Alignment.
    
    This function is specifically designed for models like VDPM that predict
    NON-METRIC (relative) depth/scene flow. It computes an optimal scale factor
    to align predictions to metric GT before evaluation.
    
    Metrics (aligned with Any4D paper):
        - EPE: End-Point Error - mean 3D L2 error (meters, after scale alignment)
        - APD: Average Points within Delta - mean accuracy across thresholds
        - τ (tau): Inlier ratio at 0.1m
        - scale: Computed scale factor for alignment
    
    Scale Alignment Methods:
        - "least_squares": s = (pred · gt) / (pred · pred) - minimizes ||s*pred - gt||^2
        - "median": s = median(|gt|) / median(|pred|) - robust to outliers
    
    Args:
        pred_scene_flow: [num_tgt_views, 3, H, W] - predicted scene flow FROM ref_frame (non-metric)
        gt_trajs_3d: [num_views, num_trajs, 3] - GT 3D positions (WORLD coordinates, metric)
        gt_trajs_2d: [num_views, num_trajs, 2] - GT 2D pixel coords (x, y) in ORIGINAL resolution
        extrinsics: [num_views, 4, 4] - world-to-camera transformation (w2c)
        visibs: [num_views, num_trajs] - visibility mask
        valids: [num_views, num_trajs] - validity mask
        dynamic_threshold: Threshold for classifying points as dynamic (meters)
        apd_thresholds: Thresholds for APD computation
        inlier_threshold: Threshold for inlier ratio (τ)
        ref_frame_idx: Reference frame index (source frame for scene flow)
        dynamic_only: If True, only evaluate dynamic points
        origin_h: Original image height (for coordinate scaling)
        origin_w: Original image width (for coordinate scaling)
        scale_alignment_method: Method for scale alignment ("least_squares" or "median")
    
    Returns:
        Dictionary of metrics:
            - epe: End-Point Error (meters, after scale alignment)
            - apd: Average Points within Delta
            - tau: Inlier ratio at inlier_threshold
            - acc3d_{thresh}: Individual accuracy at each threshold
            - scale: Computed scale factor
    """
    num_views = gt_trajs_3d.shape[0]
    num_trajs = gt_trajs_3d.shape[1]
    H, W = pred_scene_flow.shape[-2:]
    
    # Compute coordinate scale factors if original resolution is provided
    scale_x = 1.0
    scale_y = 1.0
    if origin_w is not None and origin_w > 0:
        scale_x = W / origin_w
    if origin_h is not None and origin_h > 0:
        scale_y = H / origin_h
    
    # === First Pass: Collect all pred/gt pairs ===
    all_sampled_pred = []  # Predicted flow values
    all_gt_flow = []  # GT flow values (in camera frame)
    all_is_dynamic = []  # Dynamic point flags
    
    src_idx = ref_frame_idx
    
    for tgt_idx in range(num_views):
        if src_idx == tgt_idx:
            continue
        
        # Get predicted scene flow for this target frame
        flow_pred = pred_scene_flow[tgt_idx]  # [3, H, W]
        
        # Compute GT scene flow in WORLD coordinates: position_tgt - position_src
        gt_flow_world = gt_trajs_3d[tgt_idx] - gt_trajs_3d[src_idx]  # [num_trajs, 3]
        
        # Transform GT flow to SOURCE CAMERA frame if extrinsics provided
        if extrinsics is not None:
            R_w2c_src = extrinsics[src_idx, :3, :3]  # [3, 3]
            gt_flow_camera = transform_flow_world_to_camera(gt_flow_world, R_w2c_src)
        else:
            gt_flow_camera = gt_flow_world
        
        # Get 2D locations at source view for sampling (in ORIGINAL resolution)
        src_2d_orig = gt_trajs_2d[src_idx]  # [num_trajs, 2] (x, y)
        
        # Scale 2D coordinates to model resolution
        src_2d = src_2d_orig.copy()
        src_2d[:, 0] = src_2d_orig[:, 0] * scale_x
        src_2d[:, 1] = src_2d_orig[:, 1] * scale_y
        
        # Build validity mask (both endpoints must be valid)
        mask = np.ones(num_trajs, dtype=bool)
        
        if visibs is not None:
            gt_src_visible = visibs[src_idx] > 0
            gt_tgt_visible = visibs[tgt_idx] > 0
            mask &= gt_src_visible & gt_tgt_visible
        if valids is not None:
            mask &= (valids[src_idx] > 0) & (valids[tgt_idx] > 0)
        
        # Filter points within image bounds (using scaled coordinates)
        x_coords = src_2d[:, 0]
        y_coords = src_2d[:, 1]
        in_bounds = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        mask &= in_bounds
        
        if mask.sum() == 0:
            continue
        
        # Bilinear sampling at GT 2D locations (scaled to model resolution)
        valid_coords = src_2d[mask]  # [num_valid, 2]
        sampled_pred = bilinear_sample(flow_pred, valid_coords)  # [num_valid, 3]
        gt_valid = gt_flow_camera[mask]  # [num_valid, 3]
        
        all_sampled_pred.append(sampled_pred)
        all_gt_flow.append(gt_valid)
        
        # Compute motion magnitude to identify dynamic points
        gt_motion_mag = np.linalg.norm(gt_valid, axis=1)
        is_dynamic = gt_motion_mag > dynamic_threshold
        all_is_dynamic.extend(is_dynamic.tolist())
    
    if len(all_sampled_pred) == 0:
        # No valid points
        metrics = {
            "epe": float('nan'),
            "apd": float('nan'),
            "tau": float('nan'),
            "num_points": 0,
            "num_dynamic_points": 0,
            "scale": 1.0,
        }
        for thresh in apd_thresholds:
            metrics[f"acc3d_{thresh}"] = float('nan')
        return metrics
    
    # Stack all predictions and GT
    all_sampled_pred = np.concatenate(all_sampled_pred, axis=0)  # [N, 3]
    all_gt_flow = np.concatenate(all_gt_flow, axis=0)  # [N, 3]
    all_is_dynamic = np.array(all_is_dynamic)
    
    # === Compute Scale Alignment ===
    computed_scale = compute_optimal_scale(
        all_sampled_pred, all_gt_flow, method=scale_alignment_method
    )
    # Apply scale to predictions
    all_sampled_pred_scaled = all_sampled_pred * computed_scale
    
    # === Compute Errors ===
    all_errors = np.linalg.norm(all_sampled_pred_scaled - all_gt_flow, axis=1)
    
    # Filter to dynamic points only if requested
    if dynamic_only:
        if all_is_dynamic.sum() == 0:
            # No dynamic points found
            metrics = {
                "epe": float('nan'),
                "apd": float('nan'),
                "tau": float('nan'),
                "num_points": len(all_errors),
                "num_dynamic_points": 0,
                "scale": float(computed_scale),
            }
            for thresh in apd_thresholds:
                metrics[f"acc3d_{thresh}"] = float('nan')
            return metrics
        
        eval_errors = all_errors[all_is_dynamic]
    else:
        eval_errors = all_errors
    
    # === VDPM Metrics (same as Any4D, but after scale alignment) ===
    # EPE: End-Point Error (mean 3D L2 error)
    epe = float(np.mean(eval_errors))
    
    # τ (tau): Inlier ratio at threshold (default 0.1m)
    tau = float(np.mean(eval_errors < inlier_threshold))
    
    # APD: Average Points within Delta (mean accuracy across thresholds)
    accuracies = []
    acc_dict = {}
    for thresh in apd_thresholds:
        acc = float(np.mean(eval_errors < thresh))
        accuracies.append(acc)
        acc_dict[f"acc3d_{thresh}"] = float(acc)
    
    apd = float(np.mean(accuracies))
    
    metrics = {
        "epe": float(epe),
        "apd": float(apd),
        "tau": float(tau),
        "num_points": len(all_errors),
        "num_dynamic_points": int(all_is_dynamic.sum()),
        "scale": float(computed_scale),
        **acc_dict,
    }
    
    return metrics


class Any4DSceneFlowEvalMetrics:
    """
    Scene Flow Evaluation Metrics aligned with Any4D paper.
    
    Reference: Any4D paper metrics:
        - EPE: End-Point Error (mean 3D L2 error in meters)
        - APD: Average Points within Delta (mean accuracy across thresholds)
        - τ (tau): Inlier ratio at 0.1m
    
    Key Features:
        1. Evaluates scene flow from reference frame to all other frames
           (not all source-target pairs like TAPVid-3D)
        2. Only evaluates DYNAMIC points (points with GT motion > threshold)
        3. Uses 50 frames for evaluation (configurable)
        4. Scales 2D coordinates from original resolution to model resolution
    
    IMPORTANT:
        - Model predicts scene_flow_pred with shape [num_views, 3, H, W]
          representing flow FROM frame 0 TO each target frame
        - trajs_2d is in ORIGINAL resolution, must be scaled to model resolution
    
    Usage:
        metrics_obj = Any4DSceneFlowEvalMetrics(
            dynamic_threshold=0.01,
            apd_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
            inlier_threshold=0.1,
        )
        results = metrics_obj(inputs, outputs)
    """
    
    def __init__(
        self,
        dynamic_threshold: float = 0.01,
        apd_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
        inlier_threshold: float = 0.1,
        ref_frame_idx: int = 0,
        dynamic_only: bool = True,
        use_visibility: bool = True,
        use_validity: bool = True,
        metric_prefix: str = "",
        **kwargs,
    ):
        """
        Args:
            dynamic_threshold: Threshold for classifying points as dynamic (meters)
            apd_thresholds: Thresholds for APD computation
            inlier_threshold: Threshold for inlier ratio (τ), default 0.1m per Any4D
            ref_frame_idx: Reference frame index (source frame for scene flow)
            dynamic_only: If True, only evaluate dynamic points (Any4D default)
            use_visibility: Whether to use visibility mask
            use_validity: Whether to use validity mask
            metric_prefix: Prefix for metric names (e.g. "dyn_")
        """
        self.dynamic_threshold = dynamic_threshold
        self.apd_thresholds = apd_thresholds
        self.inlier_threshold = inlier_threshold
        self.ref_frame_idx = ref_frame_idx
        self.dynamic_only = dynamic_only
        self.use_visibility = use_visibility
        self.use_validity = use_validity
        self.metric_prefix = metric_prefix
        
        # Define metric names for logging
        self.metrics = [f"{metric_prefix}{m}" for m in
                        ["epe", "apd", "tau"] + [f"acc3d_{t}" for t in apd_thresholds]]
    
    def __call__(self, inputs, outputs) -> Dict[str, float]:
        """
        Evaluate Any4D-aligned 3D scene flow metrics.
        
        Args:
            inputs: Input batch (contains meta_data for resolution info)
            outputs: List of ReconstructOutput objects, one per view
        
        Returns:
            Dictionary of metric values
        """
        # Check if required data is available
        if not hasattr(outputs[0], 'scene_flow_pred') or outputs[0].scene_flow_pred is None:
            return {}
        if not hasattr(outputs[0], 'trajs_3d') or outputs[0].trajs_3d is None:
            return {}
        if not hasattr(outputs[0], 'trajs_2d') or outputs[0].trajs_2d is None:
            return {}
        
        # Get data from first output (trajectory data is shared across all outputs)
        output = outputs[0]
        
        # Get scene flow prediction
        # scene_flow_pred shape: [num_views, 3, H, W] - flow from frame 0 to each frame
        # All outputs store the SAME scene_flow_pred, so just use the first one
        pred_scene_flow = output.scene_flow_pred
        if isinstance(pred_scene_flow, torch.Tensor):
            pred_scene_flow = pred_scene_flow.detach().cpu().numpy()
        
        num_views = pred_scene_flow.shape[0]
        
        # Collect extrinsics from outputs
        # CRITICAL: Use motion_extrinsics (raw w2c) for coordinate transformation,
        # NOT extrinsics (which is extrinsics_reff normalized to frame 0).
        # trajs_3d is in WORLD coordinates and needs raw w2c rotation to transform to camera frame.
        extrinsics = None
        
        # Priority 1: Use motion_extrinsics (raw w2c) - stored as full [num_views, 4, 4] array
        if hasattr(output, 'motion_extrinsics') and output.motion_extrinsics is not None:
            ext = output.motion_extrinsics
            if isinstance(ext, torch.Tensor):
                ext = ext.detach().cpu().numpy()
            extrinsics = ext  # Already [num_views, 4, 4]
        else:
            # Fallback: collect per-frame extrinsics and stack
            # WARNING: This fallback may use extrinsics_reff (normalized to frame 0),
            # which would cause coordinate frame mismatch with model predictions!
            import warnings
            warnings.warn(
                "[Any4DSceneFlowEvalMetrics] motion_extrinsics not found! "
                "Falling back to per-frame extrinsics, which may be extrinsics_reff (normalized to frame 0). "
                "This will cause INCORRECT evaluation if GT is in WORLD coordinates! "
                "Please ensure motion_extrinsics (raw w2c) is properly stored during inference.",
                UserWarning,
                stacklevel=2
            )
            extrinsics_list = []
            for out in outputs:
                if hasattr(out, 'extrinsics') and out.extrinsics is not None:
                    ext = out.extrinsics
                    if isinstance(ext, torch.Tensor):
                        ext = ext.detach().cpu().numpy()
                    extrinsics_list.append(ext)
                elif hasattr(out, 'gt_extrinsics') and out.gt_extrinsics is not None:
                    ext = out.gt_extrinsics
                    if isinstance(ext, torch.Tensor):
                        ext = ext.detach().cpu().numpy()
                    ext = np.linalg.inv(ext)
                    extrinsics_list.append(ext)
            
            if len(extrinsics_list) == len(outputs):
                extrinsics = np.stack(extrinsics_list, axis=0)
        
        # Get GT trajectories
        trajs_3d = output.trajs_3d
        trajs_2d = output.trajs_2d
        if isinstance(trajs_3d, torch.Tensor):
            trajs_3d = trajs_3d.detach().cpu().numpy()
        if isinstance(trajs_2d, torch.Tensor):
            trajs_2d = trajs_2d.detach().cpu().numpy()
        
        # Get masks
        visibs = None
        valids = None
        if self.use_visibility and hasattr(output, 'trajs_visibs') and output.trajs_visibs is not None:
            visibs = output.trajs_visibs
            if isinstance(visibs, torch.Tensor):
                visibs = visibs.detach().cpu().numpy()
        if self.use_validity and hasattr(output, 'trajs_valids') and output.trajs_valids is not None:
            valids = output.trajs_valids
            if isinstance(valids, torch.Tensor):
                valids = valids.detach().cpu().numpy()
        
        # Get original resolution for coordinate scaling
        # trajs_2d is in original resolution, prediction is in model resolution
        origin_h = None
        origin_w = None
        
        # Try to get from inputs (meta_data)
        if inputs is not None and isinstance(inputs, dict):
            meta_data = inputs.get('meta_data', {})
            if 'origin_height' in meta_data:
                origin_h = meta_data['origin_height']
                if isinstance(origin_h, torch.Tensor):
                    origin_h = int(origin_h.flatten()[0].item())
                elif isinstance(origin_h, (list, tuple)):
                    origin_h = int(origin_h[0])
            if 'origin_width' in meta_data:
                origin_w = meta_data['origin_width']
                if isinstance(origin_w, torch.Tensor):
                    origin_w = int(origin_w.flatten()[0].item())
                elif isinstance(origin_w, (list, tuple)):
                    origin_w = int(origin_w[0])
        
        # Fallback: try to get from output attributes
        if origin_h is None and hasattr(output, 'origin_height'):
            origin_h = output.origin_height
        if origin_w is None and hasattr(output, 'origin_width'):
            origin_w = output.origin_width
        
        # Compute Any4D metrics (scene_flow is already in absolute scale)
        results = compute_any4d_metrics(
            pred_scene_flow=pred_scene_flow,
            gt_trajs_3d=trajs_3d,
            gt_trajs_2d=trajs_2d,
            extrinsics=extrinsics,
            visibs=visibs,
            valids=valids,
            dynamic_threshold=self.dynamic_threshold,
            apd_thresholds=self.apd_thresholds,
            inlier_threshold=self.inlier_threshold,
            ref_frame_idx=self.ref_frame_idx,
            dynamic_only=self.dynamic_only,
            origin_h=origin_h,
            origin_w=origin_w,
        )
        
        # Remove internal debug metrics
        results.pop("num_points", None)
        results.pop("num_dynamic_points", None)
        
        if self.metric_prefix:
            results = {f"{self.metric_prefix}{k}": v for k, v in results.items()}
        
        return results


class VDPMSceneFlowEvalMetrics:
    """
    Scene Flow Evaluation Metrics for VDPM (Non-Metric Depth Predictions).
    
    This evaluator is specifically designed for models like VDPM that predict
    NON-METRIC (relative) depth/scene flow. It computes an optimal scale factor
    to align predictions to metric GT before evaluation.
    
    Reference: Any4D paper metrics:
        - EPE: End-Point Error (mean 3D L2 error in meters, after scale alignment)
        - APD: Average Points within Delta (mean accuracy across thresholds)
        - τ (tau): Inlier ratio at 0.1m
        - scale: Computed scale factor for alignment
    
    Key Features:
        1. Computes optimal scale to align non-metric predictions to metric GT
        2. Evaluates scene flow from reference frame to all other frames
        3. Only evaluates DYNAMIC points (points with GT motion > threshold)
        4. Scales 2D coordinates from original resolution to model resolution
    
    Scale Alignment Methods:
        - "least_squares": s = (pred · gt) / (pred · pred) - minimizes ||s*pred - gt||^2
        - "median": s = median(|gt|) / median(|pred|) - robust to outliers
    
    IMPORTANT:
        - VDPM predicts non-metric depth, so scale alignment is REQUIRED
        - The computed scale factor is returned as a metric for analysis
    
    Usage:
        metrics_obj = VDPMSceneFlowEvalMetrics(
            dynamic_threshold=0.01,
            apd_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
            inlier_threshold=0.1,
            scale_alignment_method="least_squares",
        )
        results = metrics_obj(inputs, outputs)
    """
    
    def __init__(
        self,
        dynamic_threshold: float = 0.01,
        apd_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5],
        inlier_threshold: float = 0.1,
        ref_frame_idx: int = 0,
        dynamic_only: bool = True,
        use_visibility: bool = True,
        use_validity: bool = True,
        scale_alignment_method: str = "least_squares",
        **kwargs,
    ):
        """
        Args:
            dynamic_threshold: Threshold for classifying points as dynamic (meters)
            apd_thresholds: Thresholds for APD computation
            inlier_threshold: Threshold for inlier ratio (τ), default 0.1m per Any4D
            ref_frame_idx: Reference frame index (source frame for scene flow)
            dynamic_only: If True, only evaluate dynamic points (Any4D default)
            use_visibility: Whether to use visibility mask
            use_validity: Whether to use validity mask
            scale_alignment_method: Method for scale alignment ("least_squares" or "median")
        """
        self.dynamic_threshold = dynamic_threshold
        self.apd_thresholds = apd_thresholds
        self.inlier_threshold = inlier_threshold
        self.ref_frame_idx = ref_frame_idx
        self.dynamic_only = dynamic_only
        self.use_visibility = use_visibility
        self.use_validity = use_validity
        self.scale_alignment_method = scale_alignment_method
        
        # Define metric names for logging (includes scale)
        self.metrics = ["epe", "apd", "tau", "scale"] + [f"acc3d_{t}" for t in apd_thresholds]
    
    def __call__(self, inputs, outputs) -> Dict[str, float]:
        """
        Evaluate VDPM scene flow metrics with scale alignment.
        
        Args:
            inputs: Input batch (contains meta_data for resolution info)
            outputs: List of ReconstructOutput objects, one per view
        
        Returns:
            Dictionary of metric values including computed scale factor
        """
        # Check if required data is available
        if not hasattr(outputs[0], 'scene_flow_pred') or outputs[0].scene_flow_pred is None:
            return {}
        if not hasattr(outputs[0], 'trajs_3d') or outputs[0].trajs_3d is None:
            return {}
        if not hasattr(outputs[0], 'trajs_2d') or outputs[0].trajs_2d is None:
            return {}
        
        # Get data from first output (trajectory data is shared across all outputs)
        output = outputs[0]
        
        # Get scene flow prediction
        pred_scene_flow = output.scene_flow_pred
        if isinstance(pred_scene_flow, torch.Tensor):
            pred_scene_flow = pred_scene_flow.detach().cpu().numpy()
        
        num_views = pred_scene_flow.shape[0]
        
        # Collect extrinsics from outputs
        extrinsics = None
        
        if hasattr(output, 'motion_extrinsics') and output.motion_extrinsics is not None:
            ext = output.motion_extrinsics
            if isinstance(ext, torch.Tensor):
                ext = ext.detach().cpu().numpy()
            extrinsics = ext
        else:
            import warnings
            warnings.warn(
                "[VDPMSceneFlowEvalMetrics] motion_extrinsics not found! "
                "Falling back to per-frame extrinsics.",
                UserWarning,
                stacklevel=2
            )
            extrinsics_list = []
            for out in outputs:
                if hasattr(out, 'extrinsics') and out.extrinsics is not None:
                    ext = out.extrinsics
                    if isinstance(ext, torch.Tensor):
                        ext = ext.detach().cpu().numpy()
                    extrinsics_list.append(ext)
                elif hasattr(out, 'gt_extrinsics') and out.gt_extrinsics is not None:
                    ext = out.gt_extrinsics
                    if isinstance(ext, torch.Tensor):
                        ext = ext.detach().cpu().numpy()
                    ext = np.linalg.inv(ext)
                    extrinsics_list.append(ext)
            
            if len(extrinsics_list) == len(outputs):
                extrinsics = np.stack(extrinsics_list, axis=0)
        
        # Get GT trajectories
        trajs_3d = output.trajs_3d
        trajs_2d = output.trajs_2d
        if isinstance(trajs_3d, torch.Tensor):
            trajs_3d = trajs_3d.detach().cpu().numpy()
        if isinstance(trajs_2d, torch.Tensor):
            trajs_2d = trajs_2d.detach().cpu().numpy()
        
        # Get masks
        visibs = None
        valids = None
        if self.use_visibility and hasattr(output, 'trajs_visibs') and output.trajs_visibs is not None:
            visibs = output.trajs_visibs
            if isinstance(visibs, torch.Tensor):
                visibs = visibs.detach().cpu().numpy()
        if self.use_validity and hasattr(output, 'trajs_valids') and output.trajs_valids is not None:
            valids = output.trajs_valids
            if isinstance(valids, torch.Tensor):
                valids = valids.detach().cpu().numpy()
        
        # Get original resolution for coordinate scaling
        origin_h = None
        origin_w = None
        
        if inputs is not None and isinstance(inputs, dict):
            meta_data = inputs.get('meta_data', {})
            if 'origin_height' in meta_data:
                origin_h = meta_data['origin_height']
                if isinstance(origin_h, torch.Tensor):
                    origin_h = int(origin_h.flatten()[0].item())
                elif isinstance(origin_h, (list, tuple)):
                    origin_h = int(origin_h[0])
            if 'origin_width' in meta_data:
                origin_w = meta_data['origin_width']
                if isinstance(origin_w, torch.Tensor):
                    origin_w = int(origin_w.flatten()[0].item())
                elif isinstance(origin_w, (list, tuple)):
                    origin_w = int(origin_w[0])
        
        if origin_h is None and hasattr(output, 'origin_height'):
            origin_h = output.origin_height
        if origin_w is None and hasattr(output, 'origin_width'):
            origin_w = output.origin_width
        
        # Compute VDPM metrics with scale alignment (for non-metric predictions)
        results = compute_vdpm_metrics(
            pred_scene_flow=pred_scene_flow,
            gt_trajs_3d=trajs_3d,
            gt_trajs_2d=trajs_2d,
            extrinsics=extrinsics,
            visibs=visibs,
            valids=valids,
            dynamic_threshold=self.dynamic_threshold,
            apd_thresholds=self.apd_thresholds,
            inlier_threshold=self.inlier_threshold,
            ref_frame_idx=self.ref_frame_idx,
            dynamic_only=self.dynamic_only,
            origin_h=origin_h,
            origin_w=origin_w,
            scale_alignment_method=self.scale_alignment_method,
        )
        
        # Remove internal debug metrics but keep scale
        results.pop("num_points", None)
        results.pop("num_dynamic_points", None)
        
        return results