"""Sparse Motion Pipeline — D4RT Hybrid Training

Dense depth/camera + sparse 3D motion prediction.
Encoder → Scene Representation F → Dense heads → Sparse Motion Decoder.
"""
from __future__ import annotations

import logging
import os
import traceback
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.base import Pipeline
from hAlgorithm.modules.pipelines2.utils.outputs import ReconstructOutput
from hAlgorithm.modules.pipelines2.utils.visualize_motion import vis_sparse_motion_3d_rerun, MotionVisualizer
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    maybe_instantiate, scalar, world_to_camera, sanitize,
    cat_query_dicts, FieldNames, LossAccumulator,
    get_ref_scale, get_ref_scale_flat,
)
from hAlgorithm.modules.models2.query_bank.motion_query import QueryConfig, QueryBuilder, MotionQueryBank
from hAlgorithm.utils import instantiate_from_config

logger = logging.getLogger(__name__)


class SparseMotionPipeline(Pipeline):
    """D4RT hybrid pipeline: dense depth/camera + sparse motion prediction."""

    def __init__(
        self,
        scale_name=None, intrinsics_name=None, extrinsics_name=None,
        motion_extrinsics_name="extrinsics", prompt_depth_name=None,
        target_local_depth_name=None, target_global_points_name=None,
        target_depth_mask_name=None, target_normal_name=None,
        target_normal_mask_name=None,
        local_depth_l1_loss=None, global_depth_l1_loss=None, camera_loss=None,
        pose_encoding_type="absT_quaR_FoV",
        sparse_motion_loss=None, sparse_displacement_loss=None,
        depth_consistency_loss=None, reprojection_loss=None, cycle_consistency_loss=None,
        aux_loss_gamma: float = 0.8,
        prediction_type: str = "world_position",
        task_weight=None,
        clip_level_time_norm=False,
        # ── sv_query-style query_bank config dict (preferred) ─────────────────
        query_bank: dict = None,
        # ── legacy flat query params (kept for backward compatibility) ─────────
        num_depth_consistency_samples: int = 64, cycle_consistency_prob: float = 0.3,
        num_queries_per_frame: int = 256, src_frame_idx: int = 0,
        default_max_frames: int = 1000, deterministic_queries: bool = False,
        min_dynamic_ratio: float = 0.0, sampler_dynamic_threshold: float = 0.002,
        train_use_all_points: bool = False, random_src_frame: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_level_time_norm = clip_level_time_norm

        self.fields = FieldNames(
            scale=scale_name, intrinsics=intrinsics_name, extrinsics=extrinsics_name,
            motion_extrinsics=motion_extrinsics_name, prompt_depth=prompt_depth_name,
            target_local_depth=target_local_depth_name,
            target_global_points=target_global_points_name,
            target_depth_mask=target_depth_mask_name,
        )

        # Build the MotionQueryBank: prefer the config-dict path so all
        # hyper-parameters live in one place in the config file.
        if query_bank is not None:
            self.query_bank: MotionQueryBank = instantiate_from_config(query_bank)
        else:
            # Legacy flat-param path — construct MotionQueryBank directly.
            self.query_bank = MotionQueryBank(
                num_queries_per_frame=num_queries_per_frame,
                src_frame_idx=src_frame_idx,
                default_max_frames=default_max_frames,
                deterministic=deterministic_queries,
                num_depth_consistency_samples=num_depth_consistency_samples,
                cycle_consistency_prob=cycle_consistency_prob,
                min_dynamic_ratio=min_dynamic_ratio,
                sampler_dynamic_threshold=sampler_dynamic_threshold,
                use_all_points=train_use_all_points,
                random_src_frame=random_src_frame,
            )

        # Expose QueryConfig for internal helpers that read qcfg directly.
        self.qcfg = self.query_bank.cfg
        # Alias kept for any external code that still calls self.query_builder.
        self.query_builder = self.query_bank

        self.visualizer     = MotionVisualizer()
        self.aux_loss_gamma = aux_loss_gamma
        self.visualize_attention = kwargs.get("visualize_attention", False)
        self._attn_viz_rgb = None

        assert prediction_type in ("world_position", "displacement")
        self.prediction_type  = prediction_type
        self.pose_encoding_type = pose_encoding_type
        self.task_weight      = task_weight or {}

        self._losses = {
            'local_depth':       maybe_instantiate(local_depth_l1_loss),
            'global_depth':      maybe_instantiate(global_depth_l1_loss),
            'camera':            maybe_instantiate(camera_loss),
            'sparse_motion':     maybe_instantiate(sparse_motion_loss),
            'sparse_displacement': maybe_instantiate(sparse_displacement_loss),
            'depth_consistency': maybe_instantiate(depth_consistency_loss),
            'reprojection':      maybe_instantiate(reprojection_loss),
            'cycle':             maybe_instantiate(cycle_consistency_loss),
        }
        logger.info(
            "[SparseMotionPipeline] losses=%s | pred_type=%s | weights=%s",
            {k for k, v in self._losses.items() if v is not None},
            prediction_type, self.task_weight,
        )

    def _loss(self, name: str):
        return self._losses.get(name)

    @property
    def src(self) -> int:
        return self.qcfg.src_frame_idx

    @property
    def is_displacement(self) -> bool:
        return self.prediction_type == "displacement"

    @staticmethod
    def _split_sparse_motion_output(sparse_out):
        """Normalise sparse head output → (pred_3d, pred_log_variance, pred_flow)."""
        if isinstance(sparse_out, dict):
            return (sparse_out.get('pred_3d'),
                    sparse_out.get('pred_log_variance'),
                    sparse_out.get('pred_flow'))
        return sparse_out, None, None

    @staticmethod
    def _final_iter_pred(pred):
        """[B, num_iters, Q, 3] → [B, Q, 3]; no-op for 3-D tensors."""
        if pred is not None and pred.ndim == 4:
            return pred[:, -1]
        return pred

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _to_camera_safe(self, pts: torch.Tensor, ext) -> torch.Tensor:
        if ext is None:
            return pts
        return world_to_camera(pts, ext, self.src)

    @staticmethod
    def _resize_if_needed(arr: np.ndarray, h: int, w: int,
                          interp=cv2.INTER_LINEAR) -> np.ndarray:
        return cv2.resize(arr, (w, h), interpolation=interp) if (arr.shape[0] != h or arr.shape[1] != w) else arr

    def _ref_scale_3d(self, scale):
        """Extract ref-view scale as [B, 1, 1] for broadcasting to [B, Q, 3]."""
        return scale[:, 0, 0, 0, 0].view(-1, 1, 1) if scale is not None else None

    # ── Input parsing ─────────────────────────────────────────────────────────

    def _load_field(self, batch, name_key, normalize_with_scale=None):
        name = self.fields.get(name_key)
        if not name or name not in batch:
            return None
        t = batch[name].to(self.device)
        return self.normalize(t, normalize_with_scale) if normalize_with_scale is not None else t

    def _load_trajectories(self, batch, scale):
        trajs_2d = batch.get("trajs_2d")
        trajs_3d = batch.get("trajs_3d")
        valids   = batch.get("valids")
        visibs   = batch.get("visibs")
        if isinstance(trajs_2d, list):
            trajs_2d, trajs_3d, valids, visibs = self._pad_traj_batch(trajs_2d, trajs_3d, valids, visibs)
        if trajs_3d is not None and scale is not None:
            trajs_3d = trajs_3d / get_ref_scale(scale)
        return trajs_2d, trajs_3d, valids, visibs

    def _pad_traj_batch(self, trajs_2d_list, trajs_3d_list, valids_list, visibs_list):
        B     = len(trajs_2d_list)
        T     = trajs_2d_list[0].shape[0]
        max_N = max(t.shape[1] for t in trajs_2d_list)

        def _pad(data_list, channels=None):
            if data_list is None or data_list[0] is None:
                return None
            shape = (B, T, max_N, channels) if channels else (B, T, max_N)
            out = torch.zeros(shape, dtype=data_list[0].dtype, pin_memory=True)
            for i, t in enumerate(data_list):
                out[i, :, :t.shape[1]] = t
            return out.to(self.device, non_blocking=True)

        return _pad(trajs_2d_list, 2), _pad(trajs_3d_list, 3), _pad(valids_list), _pad(visibs_list)

    def _parse_time_info(self, meta_data):
        if "data_info" not in meta_data:
            return None, None
        data_info = meta_data["data_info"]
        B, T = len(data_info), len(data_info[0]) if data_info else 0
        if T == 0:
            return None, None

        if self.clip_level_time_norm:
            clip_starts = meta_data.get("clip_start_frame_id")
            clip_ends = meta_data.get("clip_end_frame_id")
            assert clip_starts is not None and clip_ends is not None, (
                "clip_level_time_norm=True requires clip_start_frame_id and "
                "clip_end_frame_id in meta_data (provided by BaseTrackDataset)"
            )

            time_idx_list = []
            for b in range(B):
                raw_ids = [float(fd.get("frame_id", 0)) for fd in data_info[b]]
                cs = float(clip_starts[b]) if hasattr(clip_starts, '__getitem__') else float(clip_starts)
                ce = float(clip_ends[b]) if hasattr(clip_ends, '__getitem__') else float(clip_ends)
                span = ce - cs
                if span < 1e-6:
                    normalized = [0.5] * len(raw_ids)
                else:
                    normalized = [(fid - cs) / span for fid in raw_ids]
                time_idx_list.append(normalized)
            time_idx = torch.tensor(time_idx_list, dtype=torch.float32, device=self.device)
        else:
            max_frames = scalar(meta_data.get('max_frames'), self.qcfg.default_max_frames)
            if max_frames == self.qcfg.default_max_frames:
                logger.warning("[SparseMotionPipeline] max_frames not in metadata, using default=%d", max_frames)
            time_idx = torch.tensor([
                [float(fd.get("frame_id", 0)) / max_frames for fd in data_info[b]]
                for b in range(B)
            ], dtype=torch.float32, device=self.device)

        meta_data["time_idx"] = time_idx
        return time_idx, time_idx[:, :1].clone()

    def get_inputs(self, batch):
        meta = batch["meta_data"]
        if (ti := batch.get("total_iter")) is not None:
            meta["total_iter"] = ti

        image = batch["image"].to(device=self.device, dtype=self.dtype)

        scale = None
        if (sn := self.fields.scale) and sn in batch:
            scale = batch[sn].to(self.device, self.dtype)[..., None, None, None]
        scale_flat = scale[..., 0, 0] if scale is not None else None

        intrinsics = self._load_field(batch, 'intrinsics')

        extrinsics = self._load_field(batch, 'extrinsics')
        if extrinsics is not None and scale_flat is not None:
            extrinsics[..., :3, 3] = self.normalize(extrinsics[..., :3, 3], scale_flat)

        motion_extrinsics = self._load_field(batch, 'motion_extrinsics')
        if motion_extrinsics is not None and scale is not None:
            motion_extrinsics[..., :3, 3] = self.normalize(
                motion_extrinsics[..., :3, 3], get_ref_scale_flat(scale)
            )

        time_idx, query_times = self._parse_time_info(meta)
        trajs_2d, trajs_3d, valids, visibs = self._load_trajectories(batch, scale)
        trajs_3d_original_world = trajs_3d

        if trajs_3d is not None and motion_extrinsics is not None and extrinsics is not None:
            raw_to_aligned = torch.linalg.inv(extrinsics[:, self.src]) @ motion_extrinsics[:, self.src]
            B, T, N, _ = trajs_3d.shape
            pts = trajs_3d.reshape(B, -1, 3)
            pts_h = torch.cat([pts, torch.ones(*pts.shape[:2], 1, device=pts.device, dtype=pts.dtype)], -1)
            trajs_3d = torch.bmm(raw_to_aligned, pts_h.transpose(1, 2)).transpose(1, 2)[..., :3].reshape(B, T, N, 3)

        return dict(
            name=meta["name"][0], total_iter=ti, meta_data=meta, image=image,
            intrinsics=intrinsics, extrinsics=extrinsics, scale=scale,
            prompt_depth=self._load_field(batch, 'prompt_depth', scale),
            target_local_depth=self._load_field(batch, 'target_local_depth', scale),
            target_global_points=self._load_field(batch, 'target_global_points', scale),
            target_depth_mask=self._load_field(batch, 'target_depth_mask'),
            time_idx=time_idx, query_times=query_times,
            trajs_2d=trajs_2d, trajs_3d=trajs_3d, valids=valids, visibs=visibs,
            motion_extrinsics=motion_extrinsics,
            trajs_3d_original_world=trajs_3d_original_world,
        )

    # ── GT preparation ────────────────────────────────────────────────────────

    def _prepare_gt(self, gt_tgt_world, motion_queries, ext):
        """GT displacement expressed in frame-0 camera coordinate system.

        The output coordinate frame is **always frame 0**, regardless of which
        frame was randomly chosen as the source during query construction.
        This mirrors ``compute_motion_head_loss(coord_frame_idx=0)`` used by
        MVFRMotionPipeline / ``prepare_scene_flow_gt``:

          cam_disp = R_frame0 @ (tgt_3d_world - src_3d_world)

        Note: ``motion_queries['src_frame_idx']`` (potentially random) still
        controls which frame's 2D UV and 3D anchor (``gt_3d_at_src``) are used
        for the query — only the *output coordinate system* is pinned to frame 0.
        """
        # Coordinate frame is always frame 0; consistent with MotionHead4RC /
        # compute_motion_head_loss(coord_frame_idx=0).
        COORD_FRAME = 0
        if self.is_displacement:
            assert ext is not None, "Displacement training requires extrinsics."
            R = ext[:, COORD_FRAME, :3, :3]
            disp = gt_tgt_world - motion_queries['gt_3d_at_src']
            return torch.bmm(R, disp.transpose(1, 2)).transpose(1, 2)
        return world_to_camera(gt_tgt_world, ext, COORD_FRAME)

    # ── Checkpoint loading (with legacy key remapping) ────────────────────────

    def _remap_legacy_heads(self, state_dict: dict) -> dict:
        """Remap pre-refactor flat key layouts to the current nested structure.

        Walks every submodule of ``self.model`` that supports legacy remapping
        (currently :class:`CrossAttnAdaLNFlowHead`) and rewrites state_dict keys
        in-place (on a new dict copy) *before* ``load_state_dict`` is called.

        Doing the remap here — rather than inside ``_load_from_state_dict`` —
        ensures PyTorch's top-level key-tracking sees the already-remapped keys,
        which prevents spurious entries in both ``unexpected_keys`` and
        ``missing_keys``.
        """
        from hAlgorithm.modules.models2.head.crossattn_adaln_flow_head import (
            CrossAttnAdaLNFlowHead,
        )
        for name, module in self.model.named_modules():
            if not isinstance(module, CrossAttnAdaLNFlowHead):
                continue
            prefix = f"{name}." if name else ""
            has_legacy = any(
                k.startswith(prefix + old_pfx)
                for k in state_dict
                for old_pfx in CrossAttnAdaLNFlowHead._LEGACY_KEY_MAP
            )
            if has_legacy:
                msg = (
                    f"[SparseMotionPipeline] DEPRECATED checkpoint layout detected "
                    f"for module '{name}': flat keys without 'query_feats_aggregator' / "
                    f"'query_decoder' nesting (introduced in the v48 refactor). "
                    f"Keys are remapped automatically. Re-save the checkpoint with the "
                    f"current model architecture to suppress this warning."
                )
                logger.warning(msg)
                warnings.warn(msg, DeprecationWarning, stacklevel=3)
                state_dict = CrossAttnAdaLNFlowHead._remap_legacy_state_dict(
                    state_dict, prefix
                )
        return state_dict

    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        """Load a checkpoint, transparently remapping any legacy key layouts."""
        if ckpt_path is not None:
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
            else:
                state_dict = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
        if state_dict is not None:
            state_dict = self._remap_legacy_heads(state_dict)
            res = self.model.load_state_dict(state_dict, strict=False)
            logger.info("Model parameters are loaded from %s", ckpt_path)
            logger.info("unexpected_keys: %s", res.unexpected_keys)
            logger.info("missing_keys: %s", res.missing_keys)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(self, batch):
        self.train()
        inp = self.get_inputs(batch)

        image = inp['image']
        B, V, _, H, W = image.shape
        meta   = inp['meta_data']
        H_orig = scalar(meta.get("origin_height"), H)
        W_orig = scalar(meta.get("origin_width"),  W)

        queries, splits, motion_queries, gt_positions, valid_mask, dc_queries = \
            self._build_train_queries(inp, B, V, H_orig, W_orig, image.device)

        combined = cat_query_dicts(queries) if queries else None
        results  = self.model(
            rgb=image, scale=inp['scale'], prompt_depth=inp['prompt_depth'],
            intrinsics=inp['intrinsics'], w2c=inp['extrinsics'],
            time_idx=inp['time_idx'], meta_data=meta,
            motion_queries=combined, H_orig=H_orig, W_orig=W_orig,
        )

        all_pred, all_pred_logvar, all_pred_flow = self._split_sparse_motion_output(results.get('sparse_motion_pred'))
        if all_pred is not None:
            n_bad = (~torch.isfinite(all_pred)).sum().item()
            if n_bad / max(all_pred.numel(), 1) > 0.5:
                raise RuntimeError(
                    f"[SparseMotion] Batch severely corrupted: "
                    f"{n_bad}/{all_pred.numel()} NaN/Inf in predictions. Skipping."
                )
            all_pred = sanitize(all_pred)
        if all_pred_logvar is not None:
            all_pred_logvar = torch.nan_to_num(all_pred_logvar, nan=0.0, posinf=6.0, neginf=-6.0)

        def _slice(name):
            if all_pred is None or name not in splits:
                return None
            s, e = splits[name]
            return all_pred[:, :, s:e] if all_pred.ndim == 4 else all_pred[:, s:e]

        def _slice_logvar(name):
            if all_pred_logvar is None or name not in splits:
                return None
            s, e = splits[name]
            return all_pred_logvar[:, s:e]

        def _slice_flow():
            if all_pred_flow is None or 'main' not in splits:
                return None
            s, e = splits['main']
            return all_pred_flow[:, s:e]

        acc = LossAccumulator(self.device, self.task_weight)
        self._dense_losses(inp, results, image, acc)
        self._sparse_losses(inp, results, _slice, _slice_logvar,
                            motion_queries, gt_positions, valid_mask, acc,
                            pred_flow=_slice_flow())
        self._auxiliary_losses(inp, results, _slice, motion_queries,
                               gt_positions, valid_mask, dc_queries, acc)

        if inp['scale'] is not None:
            acc.add_meta("scale", round(inp['scale'].max().cpu().item(), 2))
        acc.add_meta("bs", int(B))

        total, breakdown = acc.result()
        if total.grad_fn is None:
            logger.error(
                "[train_step] No differentiable loss accumulated (data=%s). "
                "This should not happen — all query/loss paths must maintain "
                "the computation graph. Check loss functions and query builder.",
                inp['name'],
            )
            raise RuntimeError(
                f"[SparseMotion] Loss has no grad_fn (data={inp['name']}). "
                "All code paths must produce a gradient-connected loss for DDP."
            )
        return total, breakdown

    def _build_train_queries(self, inp, B, V, H_orig, W_orig, device):
        batches, splits, offset = [], {}, 0
        motion_queries = gt_positions = valid_mask = dc_queries = None

        has_motion = (
            (self._loss('sparse_motion') or self._loss('sparse_displacement'))
            and inp['trajs_2d'] is not None and inp['trajs_3d'] is not None
        )
        if has_motion:
            motion_queries, gt_positions, valid_mask = self.query_builder.motion(
                inp['trajs_2d'], inp['trajs_3d'], inp['time_idx'],
                inp['visibs'], inp['valids'], (H_orig, W_orig),
            )
            n = motion_queries['uv'].shape[1]
            batches.append(motion_queries)
            splits['main'] = (offset, offset + n); offset += n

        if self._loss('depth_consistency') is not None:
            dc_queries = self.query_builder.depth_consistency(B, V, inp['time_idx'], device)
            n = dc_queries['uv'].shape[1]
            batches.append(dc_queries)
            splits['dc'] = (offset, offset + n); offset += n

        if (self._loss('cycle') is not None and motion_queries is not None
                and torch.rand(1).item() < self.qcfg.cycle_consistency_prob):
            rev = self.query_builder.reverse(motion_queries)
            if rev is not None:
                n = rev['uv'].shape[1]
                batches.append(rev)
                splits['cycle'] = (offset, offset + n); offset += n

        return batches, splits, motion_queries, gt_positions, valid_mask, dc_queries

    # ── Loss computation ──────────────────────────────────────────────────────

    def _dense_losses(self, inp, results, image, acc: LossAccumulator):
        for loss_name, pred_key, target_key, conf_key in (
            ('local_depth',  'depth',        'target_local_depth',   'confidence'),
            ('global_depth', 'global_points', 'target_global_points', 'global_confidence'),
        ):
            loss_fn = self._loss(loss_name)
            target, pred = inp.get(target_key), results.get(pred_key)
            if loss_fn is not None and target is not None and pred is not None:
                acc.add(loss_name, self._eval_depth_loss(
                    loss_fn, inp['name'], pred, target,
                    inp['target_depth_mask'], results.get(conf_key),
                ))

        cam_fn, pred_pose = self._loss('camera'), results.get('pose_enc')
        if cam_fn is not None and pred_pose is not None and inp['extrinsics'] is not None:
            loss, _ = cam_fn(
                name=inp['name'], pose_enc=pred_pose,
                target_intrinsic=inp['intrinsics'], target_extrinsics=inp['extrinsics'],
                scale=inp['scale'], image_size_hw=image.shape[-2:],
                valid_mask=inp['target_depth_mask'],
            )
            acc.add("camera", loss)

    def _sparse_losses(self, inp, results, _slice, _slice_logvar,
                       motion_queries, gt_positions, valid_mask, acc,
                       pred_flow=None):
        """Sparse displacement/position loss; handles single-pass and iterative outputs."""
        pred_pos    = _slice('main')
        pred_logvar = _slice_logvar('main')
        if motion_queries is None or gt_positions is None or pred_pos is None:
            return
        if not torch.isfinite(pred_pos).all():
            logger.warning("[_sparse_losses] NaN/Inf in pred_pos %s — skipping.", tuple(pred_pos.shape))
            return

        with torch.no_grad():
            gt_cam = self._prepare_gt(gt_positions, motion_queries, inp['extrinsics'])

        if self.is_displacement:
            loss_fn = self._loss('sparse_displacement') or self._loss('sparse_motion')
        else:
            loss_fn = self._loss('sparse_motion')
        if loss_fn is None:
            return

        def _call(pred_single):
            if self.is_displacement:
                return loss_fn(pred_displacement=pred_single, gt_displacement=gt_cam,
                               valid_mask=valid_mask, scale=inp['scale'],
                               pred_log_variance=pred_logvar)
            return loss_fn(pred_positions=pred_single, gt_positions=gt_cam,
                           valid_mask=valid_mask, pred_log_variance=pred_logvar)

        if pred_pos.ndim == 3:
            result = _call(pred_pos)
            if isinstance(result, tuple):
                acc.add("sparse_motion", result[0]); acc.breakdown.update(result[1])
            else:
                acc.add("sparse_motion", result)
        else:
            num_iters, gamma = pred_pos.shape[1], self.aux_loss_gamma
            for k in range(num_iters):
                w      = gamma ** (num_iters - 1 - k)
                result = _call(pred_pos[:, k])
                if isinstance(result, tuple):
                    acc.add("sparse_motion", result[0] * w)
                    if k == num_iters - 1:
                        acc.breakdown.update(result[1])
                else:
                    acc.add("sparse_motion", result * w)

        # ── Auxiliary 2D flow loss (pred_flow supervised by GT trajs_2d) ──
        if pred_flow is not None and motion_queries is not None:
            gt_2d_tgt = motion_queries.get('gt_2d_at_tgt')
            if gt_2d_tgt is not None:
                gt_flow = gt_2d_tgt - motion_queries['uv']           # [B, Q, 2]
                flow_err = (pred_flow - gt_flow).abs()               # L1
                if valid_mask is not None:
                    vm = valid_mask.unsqueeze(-1)                     # [B, Q, 1]
                    flow_loss = (flow_err * vm).sum() / vm.sum().clamp(min=1.0)
                else:
                    flow_loss = flow_err.mean()
                acc.add("sparse_flow", flow_loss)

    def _auxiliary_losses(self, inp, results, _slice, motion_queries,
                          gt_positions, valid_mask, dc_queries, acc):
        """D4RT auxiliary: depth consistency, reprojection, cycle."""
        B, V, _, H, W = inp['image'].shape

        dc_pred = _slice('dc')
        if (dc_pred is not None and dc_queries is not None
                and not self.is_displacement and results.get('depth') is not None):
            gt_z, dc_valid = self._sample_depth_gt(results['depth'], dc_queries)
            acc.add("d4rt_dc", self._loss('depth_consistency')(dc_pred[..., 2], gt_z, dc_valid))

        pred_pos = self._final_iter_pred(_slice('main'))
        if (self._loss('reprojection') is not None and not self.is_displacement
                and pred_pos is not None and motion_queries is not None):
            gt_2d = motion_queries.get('gt_2d_at_tgt')
            if gt_2d is not None and inp['intrinsics'] is not None:
                gt_2d_px = gt_2d.clone()
                gt_2d_px[..., 0] *= (W - 1); gt_2d_px[..., 1] *= (H - 1)
                scale_s  = self._ref_scale_3d(inp['scale'])
                pred_abs = pred_pos * scale_s if scale_s is not None else pred_pos
                acc.add("d4rt_reproj", self._loss('reprojection')(
                    pred_3d=pred_abs, intrinsics=inp['intrinsics'][:, self.src],
                    gt_2d=gt_2d_px, valid_mask=valid_mask, image_size=(H, W),
                ))

        cycle_pred = self._final_iter_pred(_slice('cycle'))
        if cycle_pred is not None and motion_queries is not None and self._loss('cycle') is not None:
            if self.is_displacement:
                fwd = self._final_iter_pred(_slice('main'))
                if fwd is not None:
                    acc.add("d4rt_cycle", self._loss('cycle')(
                        original_3d=-fwd, cycled_3d=cycle_pred, valid_mask=valid_mask,
                    ))
            else:
                gt_src = motion_queries.get('gt_3d_at_src')
                if gt_src is not None:
                    acc.add("d4rt_cycle", self._loss('cycle')(
                        original_3d=self._to_camera_safe(gt_src, inp['extrinsics']),
                        cycled_3d=cycle_pred, valid_mask=valid_mask,
                    ))

    @staticmethod
    def _eval_depth_loss(loss_fn, name, pred, target, mask, conf):
        """Flatten [B,V,...] → [B*V,...] for dense depth losses."""
        def _flat(t, squeeze=False):
            t = t.reshape(-1, *t.shape[2:]).permute(0, 2, 3, 1).contiguous()
            return t.squeeze(1) if squeeze else t

        mask_f = None
        if mask is not None:
            mask_f = mask.reshape(-1, *mask.shape[2:])
            if mask_f.ndim == 4:
                mask_f = mask_f.squeeze(1)
            mask_f = mask_f.bool()

        # _flat already produces [B*V, H, W, 1]; no further squeezing needed.
        conf_f = _flat(conf, squeeze=False) if conf is not None else None
        return loss_fn(name=name, pred_depth=_flat(pred), target_depth=_flat(target),
                       pred_conf=conf_f, valid_mask=mask_f)

    def _sample_depth_gt(self, dense_depth, dc_queries):
        src_depth = dense_depth[:, self.src, :1]
        uv = dc_queries['uv']
        grid = torch.stack([2 * uv[..., 0] - 1, 2 * uv[..., 1] - 1], -1).unsqueeze(2)
        gt_z = F.grid_sample(src_depth, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)[:, 0, :, 0]
        return gt_z, (gt_z > 1e-4).float()

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, **batch):
        self.eval()
        inp = self.get_inputs(batch)

        image  = inp['image']
        B, V, _, H, W = image.shape
        meta   = inp['meta_data']
        H_orig = scalar(meta.get("origin_height"), H)
        W_orig = scalar(meta.get("origin_width"),  W)
        scale  = inp['scale']

        has_trajs = inp['trajs_2d'] is not None and inp['trajs_3d'] is not None

        if has_trajs:
            results, pred_pos, gt_cam, valid_mask, motion_queries, pred_flow_2d = \
                self._infer_per_target_frame(inp, B, V, H, W, H_orig, W_orig)
        else:
            pred_flow_2d = None
            motion_queries = self.query_builder.grid(B, V, inp['time_idx'], image.device)
            valid_mask = None
            if motion_queries:
                Q = motion_queries['uv'].shape[1]
                valid_mask = torch.ones((B, Q), device=image.device)
                logger.info("[SparseMotionPipeline] Grid queries: %d points", Q)
            results = self.model(
                rgb=image, scale=scale, prompt_depth=inp['prompt_depth'],
                intrinsics=inp['intrinsics'], w2c=inp['extrinsics'],
                time_idx=inp['time_idx'], meta_data=meta,
                motion_queries=motion_queries, H_orig=H_orig, W_orig=W_orig,
            )
            pred_pos, gt_cam = self._postprocess_predictions(
                results, inp, motion_queries, None, B, scale,
            )

        return self._build_outputs(
            inp, results, pred_pos, gt_cam, valid_mask,
            has_trajs, H_orig, W_orig, B, V, H, W,
            motion_queries=motion_queries, pred_flow_2d=pred_flow_2d,
        )

    @torch.no_grad()
    def _infer_per_target_frame(self, inp, B, V, H, W, H_orig, W_orig):
        """Run encoder once, decode one target frame at a time (O(N×hidden) peak memory)."""
        image, scale, meta = inp['image'], inp['scale'], inp['meta_data']
        ext = inp['extrinsics']

        results = self.model(
            rgb=image, scale=scale, prompt_depth=inp['prompt_depth'],
            intrinsics=inp['intrinsics'], w2c=ext,
            time_idx=inp['time_idx'], meta_data=meta,
            motion_queries=None, return_cached=True, H_orig=H_orig, W_orig=W_orig,
        )
        features        = results.pop('_cached_features')
        patch_start_idx = results.pop('_cached_patch_start_idx')

        unwrapped   = self.model.module if hasattr(self.model, 'module') else self.model
        sparse_head = unwrapped.sparse_motion_head
        use_fs      = getattr(sparse_head, 'use_feature_sampling', False)
        mem         = None if use_fs else sparse_head._build_memory(
            features, patch_start_idx, B, img_size=(H_orig, W_orig),
        )

        want_attn_viz = (self.visualize_attention
                         and hasattr(sparse_head, 'viz_mode'))

        scale_s = self._ref_scale_3d(scale)
        all_preds, all_gt, all_valid, all_queries, all_flows = [], [], [], [], []

        if want_attn_viz:
            sparse_head._viz_cache = []
            self._attn_viz_rgb = image.detach().cpu()

        for tgt_frame in (t for t in range(V) if t != self.src):
            qdata = self.query_builder.motion_single_target(
                inp['trajs_2d'], inp['trajs_3d'], inp['time_idx'],
                inp['visibs'], inp['valids'], (H_orig, W_orig), tgt_frame=tgt_frame,
            )
            if qdata is None:
                continue
            frame_queries, frame_gt_world, frame_valid = qdata

            if use_fs:
                frame_out = sparse_head(features=features, rgb_images=image,
                                        queries=frame_queries, img_size=(H_orig, W_orig),
                                        patch_start_idx=patch_start_idx)
            else:
                frame_out = sparse_head.decode_with_memory(mem, image, frame_queries)

            frame_pred, _, frame_flow = self._split_sparse_motion_output(frame_out)
            frame_pred    = sanitize(self._final_iter_pred(frame_pred))
            frame_gt_cam  = self._prepare_gt(frame_gt_world, frame_queries, ext)
            frame_pred, frame_gt_cam = self._denormalize_pred_gt(
                frame_pred, frame_gt_cam, frame_queries, ext, scale_s,
            )
            all_preds.append(frame_pred); all_gt.append(frame_gt_cam)
            all_valid.append(frame_valid); all_queries.append(frame_queries)
            all_flows.append(frame_flow)

        if all_preds:
            pred_pos       = torch.cat(all_preds, dim=1)
            gt_cam         = torch.cat(all_gt,    dim=1)
            valid_mask     = torch.cat(all_valid, dim=1)
            motion_queries = cat_query_dicts(all_queries)
            # Preserve 2D flow predictions if the head produced them.
            valid_flows = [f for f in all_flows if f is not None]
            pred_flow_2d = torch.cat(valid_flows, dim=1) if valid_flows else None
            logger.info("[infer] %d targets × %d tracks/target = %d total (%d valid)",
                        len(all_preds), all_preds[0].shape[1],
                        pred_pos.shape[1], int(valid_mask.sum()))
        else:
            pred_pos = gt_cam = valid_mask = motion_queries = pred_flow_2d = None

        return results, pred_pos, gt_cam, valid_mask, motion_queries, pred_flow_2d

    def _denormalize_pred_gt(self, pred, gt_cam, queries, ext, scale_s):
        """Denormalize to absolute camera-frame coords."""
        if self.is_displacement:
            gt_src_cam = self._to_camera_safe(queries['gt_3d_at_src'], ext)
            if scale_s is not None:
                gt_cam, gt_src_cam, pred = gt_cam * scale_s, gt_src_cam * scale_s, pred * scale_s
            return gt_src_cam + pred, gt_src_cam + gt_cam
        if scale_s is not None:
            pred, gt_cam = pred * scale_s, gt_cam * scale_s
        return pred, gt_cam

    def _postprocess_predictions(self, results, inp, motion_queries, gt_positions, B, scale):
        """Denormalize grid-fallback predictions to absolute camera coords."""
        pred_pos, _, _ = self._split_sparse_motion_output(results.get('sparse_motion_pred'))
        pred_pos     = self._final_iter_pred(pred_pos)
        scale_s, ext = self._ref_scale_3d(scale), inp['extrinsics']
        gt_cam = None

        if gt_positions is not None and motion_queries is not None:
            gt_cam = self._prepare_gt(gt_positions, motion_queries, ext)
            if pred_pos is not None:
                pred_pos, gt_cam = self._denormalize_pred_gt(pred_pos, gt_cam, motion_queries, ext, scale_s)
            elif scale_s is not None:
                gt_cam = gt_cam * scale_s
        else:
            if gt_positions is not None:
                gt_cam = self._to_camera_safe(gt_positions, ext)
                if scale_s is not None:
                    gt_cam = gt_cam * scale_s
            if pred_pos is not None and scale_s is not None:
                pred_pos = pred_pos * scale_s

        return pred_pos, gt_cam

    # ── Output construction ───────────────────────────────────────────────────

    def _compute_gt_src_cam(self, inp, motion_queries):
        if motion_queries is None or 'gt_3d_at_src' not in motion_queries:
            return None
        ext     = inp['extrinsics'] if inp['extrinsics'] is not None else inp.get('motion_extrinsics')
        scale_s = self._ref_scale_3d(inp['scale'])
        gt      = self._to_camera_safe(motion_queries['gt_3d_at_src'], ext)
        return gt * scale_s if scale_s is not None else gt

    def _build_outputs(self, inp, results, pred_pos, gt_cam, valid_mask,
                       has_trajs, H_orig, W_orig, B, V, H, W,
                       motion_queries=None, pred_flow_2d=None):
        extrinsics = inp['extrinsics'] if inp['extrinsics'] is not None else inp.get('motion_extrinsics')
        if extrinsics is not None and inp['scale'] is not None:
            extrinsics = extrinsics.clone()
            extrinsics[..., :3, 3] = self.denormalize(extrinsics[..., :3, 3], inp['scale'][..., 0, 0])

        def _depth_to_pointmap(d):
            if d is None:
                return None
            if d.shape[-3] == 1:
                d = self.depth_to_points(d, K=inp['intrinsics'])
            if inp['scale'] is not None:
                d = self.denormalize(d, scale=inp['scale'])
            return d

        pred_depth  = _depth_to_pointmap(results.get('depth'))
        gt_pointmap = _depth_to_pointmap(inp.get('target_local_depth'))
        if gt_pointmap is None:
            logger.warning("[_build_outputs] gt_pointmap is None, falling back to pred_depth.")

        frame_num      = scalar(inp['meta_data'].get("frames"), 1)
        view_num       = scalar(inp['meta_data'].get("views"),  V)
        gt_3d_src_cam  = self._compute_gt_src_cam(inp, motion_queries)

        outputs = []
        for fi in range(frame_num):
            for vi in range(view_num):
                idx = fi * view_num + vi
                try:
                    out = self._build_single_output(
                        pred_depth, results.get('confidence'),
                        inp['intrinsics'], extrinsics, idx, H_orig, W_orig, src_h=H, src_w=W,
                        gt_pointmap=gt_pointmap,
                    )
                    self._attach_sparse_data(out, pred_pos, gt_cam, valid_mask, motion_queries,
                                             gt_3d_src_cam, pred_flow_2d=pred_flow_2d)
                    self._attach_trajectory_data(out, inp, has_trajs)
                    rgb = (inp['image'][0, idx].permute(1, 2, 0).cpu().float().numpy() + 1) * 0.5
                    out.rgb         = self._resize_if_needed(rgb, H_orig, W_orig)
                    out.frame_index = fi
                    out.view_index  = vi
                    out.total_index = idx
                    outputs.append(out)
                except Exception as e:
                    logger.error("[_build_outputs] frame=%d view=%d: %s\n%s",
                                 fi, vi, e, traceback.format_exc())

        # When running without GT trajectories on a displacement model,
        # `track_pred` holds raw displacement vectors (not absolute camera-space
        # positions).  Tag each output so the 3-D visualiser can recover absolute
        # target positions by adding the source 3-D position sampled from the
        # predicted pointmap.
        if self.is_displacement and not has_trajs:
            for out in outputs:
                out.track_pred_is_displacement = True

        logger.info("[_build_outputs] %d outputs (frames=%d, views=%d), pointmap=%d",
                    len(outputs), frame_num, view_num,
                    sum(1 for o in outputs if getattr(o, 'pointmap', None) is not None))
        return outputs

    @staticmethod
    def _attach_sparse_data(out, pred_pos, gt_cam, valid_mask, motion_queries,
                            gt_3d_src_cam=None, pred_flow_2d=None):
        if pred_pos    is not None: out.track_pred       = pred_pos[0].cpu().float().numpy()
        if gt_cam      is not None: out.track_gt         = gt_cam[0].cpu().float().numpy()
        if valid_mask  is not None:
            vm = valid_mask[0].cpu().float().numpy()
            if pred_pos is not None and vm.shape[0] != pred_pos[0].shape[0]:
                logger.warning(
                    "[_attach_sparse_data] valid_mask Q=%d != pred_pos Q=%d; "
                    "truncating valid_mask to match",
                    vm.shape[0], pred_pos[0].shape[0],
                )
                vm = vm[:pred_pos[0].shape[0]]
            out.track_vis_pred = vm
        if motion_queries is not None:
            if 'uv'            in motion_queries: out.motion_queries_uv        = motion_queries['uv'][0].cpu().float().numpy()
            if 'tgt_frame_idx' in motion_queries: out.motion_queries_tgt_frame = motion_queries['tgt_frame_idx'][0].cpu().numpy()
            # GT 2D position in target frame (normalized UV, [0,1]); used for 2D flow visualization.
            if 'gt_2d_at_tgt'  in motion_queries: out.motion_queries_gt_2d_tgt = motion_queries['gt_2d_at_tgt'][0].cpu().float().numpy()
        if gt_3d_src_cam is not None: out.motion_queries_gt_3d_src = gt_3d_src_cam[0].cpu().float().numpy()
        # 2D flow prediction in normalized UV space: pred_target_uv = src_uv + pred_flow_2d
        if pred_flow_2d is not None: out.pred_flow_2d = pred_flow_2d[0].cpu().float().numpy()

    @staticmethod
    def _attach_trajectory_data(out, inp, has_trajs):
        """Attach GT trajectory data in absolute-scale original-world coords."""
        trajs_3d_raw = inp.get('trajs_3d_original_world')
        if not (has_trajs and trajs_3d_raw is not None):
            return
        scale = inp['scale']
        trajs_abs = trajs_3d_raw * get_ref_scale(scale) if scale is not None else trajs_3d_raw
        out.trajs_3d = trajs_abs[0].cpu()
        out.trajs_2d = inp['trajs_2d'][0].cpu()
        if inp['visibs'] is not None: out.trajs_visibs = inp['visibs'][0].cpu()
        if inp['valids'] is not None: out.trajs_valids = inp['valids'][0].cpu()

        me = inp.get('motion_extrinsics')
        if me is not None:
            me_out = me.clone()
            if scale is not None:
                me_out[..., :3, 3] = me_out[..., :3, 3] * get_ref_scale_flat(scale)
            out.motion_extrinsics = me_out[0].cpu().numpy()

    @classmethod
    def _build_single_output(cls, pred_depth, pred_conf, intrinsics, extrinsics,
                             idx, H_orig, W_orig, src_h=None, src_w=None, gt_pointmap=None):
        depth_align = cls._extract_depth_align(pred_depth, idx, H_orig, W_orig)
        pmap        = cls._extract_pointmap(gt_pointmap, pred_depth, idx, H_orig, W_orig)

        conf = None
        if pred_conf is not None and idx < pred_conf.shape[1]:
            conf = cls._resize_if_needed(pred_conf[0, idx, 0].cpu().float().numpy(), H_orig, W_orig)

        intri = (intrinsics[0, idx].cpu().float().numpy().copy()
                 if intrinsics is not None and idx < intrinsics.shape[1] else None)
        sx, sy = 1.0, 1.0
        if intri is not None and src_h and src_w:
            sx = float(W_orig) / float(src_w); sy = float(H_orig) / float(src_h)
            intri[0, 0] *= sx; intri[1, 1] *= sy; intri[0, 2] *= sx; intri[1, 2] *= sy
        extri = (extrinsics[0, idx].cpu().float().numpy()
                 if extrinsics is not None and idx < extrinsics.shape[1] else None)

        out = ReconstructOutput(
            depth_align=depth_align, pointmap=pmap,
            pointmap_h=H_orig if (depth_align is not None or pmap is not None) else 0,
            pointmap_w=W_orig if (depth_align is not None or pmap is not None) else 0,
            confidence=conf, intrinsics=intri, extrinsics=extri,
        )
        out.intrinsics_src_h     = src_h
        out.intrinsics_src_w     = src_w
        out.intrinsics_scale_xy  = (sx, sy)
        return out

    @classmethod
    def _extract_depth_align(cls, pred_depth, idx, H_orig, W_orig):
        if pred_depth is None or idx >= pred_depth.shape[1]:
            return None
        d = pred_depth[0, idx]
        return cls._resize_if_needed(d[-1].cpu().float().numpy(), H_orig, W_orig) if d.shape[0] >= 1 else None

    @classmethod
    def _extract_pointmap(cls, gt_pointmap, pred_depth, idx, H_orig, W_orig):
        src = gt_pointmap if gt_pointmap is not None else pred_depth
        if src is None or idx >= src.shape[1]:
            return None
        d = src[0, idx]
        if d.shape[0] == 3:
            pmap = d.permute(1, 2, 0).cpu().float().numpy()
            return cls._resize_if_needed(pmap, H_orig, W_orig, cv2.INTER_NEAREST).reshape(-1, 3)
        if d.shape[0] != 1:
            logger.warning("[_extract_pointmap] idx=%d: unexpected channels=%d", idx, d.shape[0])
        return None

    # ── Visualisation & saving ────────────────────────────────────────────────

    def visualize(self, outputs_list, meta_data, out_dir):
        if not outputs_list:
            return
        data_idx   = meta_data["data_idx"][0]
        motion_dir = os.path.join(out_dir, f"sparse_motion/{data_idx:06d}")
        os.makedirs(motion_dir, exist_ok=True)
        self.visualizer.render(outputs_list[0], motion_dir, outputs_list=outputs_list)
        try:
            vis_sparse_motion_3d_rerun(cfg={}, mv_outputs=outputs_list,
                                       out_dir=out_dir, data_idx=data_idx, meta_data=meta_data)
        except Exception as e:
            logger.error("[SparseMotionPipeline] 3D vis failed: %s\n%s", e, traceback.format_exc())

        self._render_attention_viz_if_cached(motion_dir)

    def _render_attention_viz_if_cached(self, save_dir):
        """Render and clear any cached attention data from the sparse head."""
        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        sparse_head = getattr(unwrapped, 'sparse_motion_head', None)
        if sparse_head is None:
            return
        cache = getattr(sparse_head, '_viz_cache', None)
        if not cache:
            return
        try:
            attn_dir = os.path.join(save_dir, "attention")
            sparse_head.render_attention_viz(
                attn_dir,
                rgb_images=self._attn_viz_rgb,
                n_queries=8,
                dpi=150,
            )
        except Exception as e:
            logger.error("[attention_viz] %s\n%s", e, traceback.format_exc())
        finally:
            sparse_head._viz_cache = None
            self._attn_viz_rgb = None

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        if not outputs:
            return
        data_idx = meta_data["data_idx"][0]
        save_dir = os.path.join(out_dir, f"sparse_motion/{data_idx:06d}")
        os.makedirs(save_dir, exist_ok=True)
        fields = ('track_pred', 'track_gt', 'track_vis_pred',
                  'motion_queries_uv', 'motion_queries_tgt_frame', 'motion_queries_gt_3d_src')
        d = {k: getattr(outputs[0], k) for k in fields if getattr(outputs[0], k, None) is not None}
        if d:
            np.savez(os.path.join(save_dir, "sparse_motion_pred.npz"), **d)
