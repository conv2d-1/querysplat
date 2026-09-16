"""MVFRQueryMotionPipeline — unified dense-3D + sparse-motion training pipeline.

Extends ``MVFRQueryPipeline`` (v2) with sparse-motion supervision.

Query group design
------------------
The SDK model (``MVQueryUnified``) owns three named registries:
  - ``query_banks``    : QueryBank5 (model-managed) + MotionQueryBank (pipeline-managed)
  - ``aggregators``    : Aggregator2 (dense) + MotionAggregatorMLP (motion)
  - ``decoder_groups`` : shared MLP Head (both groups share the same weights)

``MotionQueryBank`` instances that are *pipeline-managed* are exposed by the
model via ``model.pipeline_managed_query_banks``.  This pipeline discovers
them on construction and uses them to build motion queries from GT trajectory
data during each ``train_step``.

The model is called with::

    results = self.model(rgb, ..., external_queries={"motion": motion_queries})

Loss terms
----------
  Part 1  lcl   — local depth          (from dense task_group)
  Part 2  glb   — global points        (from dense task_group)
  Part 3  cm    — camera               (from camera_head)
  Part 4  motion— 3D displacement      (from motion task_group)

No existing files are modified.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Optional

import torch
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.mvfr_query_v2 import MVFRQueryPipeline as _BaseQueryPipeline
from hAlgorithm.modules.pipelines2.utils.scale_alignment import align_sparse_motion
from hAlgorithm.modules.pipelines2.utils.motion_utils import (
    get_ref_scale, maybe_instantiate, sanitize, scalar, world_to_camera,
    cat_query_dicts,
)
from hAlgorithm.modules.pipelines2.utils.visualize_motion import vis_sparse_motion_3d_rerun, MotionVisualizer

logger = logging.getLogger(__name__)


class MVFRQueryMotionPipeline(_BaseQueryPipeline):
    """Dense-3D + sparse-motion unified pipeline.

    Inherits all dense-query logic from ``MVFRQueryPipeline`` (v2).  The
    motion query bank is discovered automatically from the SDK model's
    ``pipeline_managed_query_banks`` attribute rather than being constructed
    here, keeping the config unified inside ``query_banks``.

    Additional constructor args (beyond those of ``MVFRQueryPipeline``):
        motion_extrinsics_name: Batch key for the motion-space extrinsics
            (may differ from the dense ``extrinsics_name``).
        sparse_displacement_loss: Config for ``SparseDisplacementLoss``.
        sparse_motion_loss: Config for ``SparseMotionLoss`` (position-based
            alternative; used when ``prediction_type='world_position'``).
        prediction_type: ``'displacement'`` or ``'world_position'``.
        aux_loss_gamma: Exponential decay weight for auxiliary losses.
        motion_task_weight: Scalar multiplier for the total motion loss.
    """

    def __init__(
        self,
        motion_extrinsics_name: str = "extrinsics",
        sparse_displacement_loss: Optional[dict] = None,
        sparse_motion_loss: Optional[dict] = None,
        prediction_type: str = "displacement",
        aux_loss_gamma: float = 0.8,
        motion_task_weight: float = 1.0,
        flow_2d_loss: Optional[dict] = None,
        flow_2d_task_weight: float = 1.0,
        use_cached_motion_infer: bool = True,
        supervise_iter_predictions: bool = False,
        coarse_flow_2d_loss: Optional[dict] = None,
        coarse_flow_2d_task_weight: float = 0.05,
        coarse_displacement_loss: Optional[dict] = None,
        coarse_displacement_task_weight: float = 0.5,
        displacement_direction_loss: Optional[dict] = None,
        displacement_direction_weight: float = 1.0,
        infer_scale_align: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_cached_motion_infer = use_cached_motion_infer
        self._last_motion_caches: dict = {}

        self.motion_extrinsics_name = motion_extrinsics_name
        self.prediction_type = prediction_type
        self.aux_loss_gamma = aux_loss_gamma
        self.motion_task_weight = motion_task_weight
        self.flow_2d_task_weight = flow_2d_task_weight
        self.supervise_iter_predictions = supervise_iter_predictions
        self.coarse_flow_2d_task_weight = coarse_flow_2d_task_weight
        self.coarse_displacement_task_weight = coarse_displacement_task_weight
        self.displacement_direction_weight = displacement_direction_weight
        self.infer_scale_align = infer_scale_align

        assert prediction_type in ("displacement", "world_position")

        self._motion_losses = {
            "sparse_displacement": maybe_instantiate(sparse_displacement_loss),
            "sparse_motion":       maybe_instantiate(sparse_motion_loss),
        }

        # Optional 2-D optical flow loss, supervised by trajs_2d.
        # GT flow = gt_2d_at_tgt - uv (both normalized [0, 1]).
        self._flow2d_loss = maybe_instantiate(flow_2d_loss)

        # Auxiliary losses for coarse predictions from refine/dual-layer aggregators.
        self._coarse_flow2d_loss = maybe_instantiate(coarse_flow_2d_loss)
        self._coarse_disp_loss = maybe_instantiate(coarse_displacement_loss)

        # Direction loss for SphericalHead's direction branch.
        self._direction_loss = maybe_instantiate(displacement_direction_loss)

        # ── Discover pipeline-managed query banks from the SDK model ──────
        # MVQueryUnified exposes MotionQueryBank instances that need GT data
        # via model.pipeline_managed_query_banks.  We find the one that
        # belongs to a task_group containing sparse motion tasks.
        self.motion_query_bank = None
        self.motion_group_name: Optional[str] = None
        self.qcfg = None

        if hasattr(self.model, "pipeline_managed_query_banks") and \
                hasattr(self.model, "task_groups"):
            pmqb = self.model.pipeline_managed_query_banks
            for tg in self.model.task_groups:
                qb_name = tg.get("query_bank", "")
                tasks = set(tg.get("tasks", []))
                if qb_name in pmqb and tasks.intersection({"displacement", "flow_2d"}):
                    self.motion_group_name = tg["name"]
                    self.motion_query_bank = pmqb[qb_name]
                    self.qcfg = self.motion_query_bank.cfg
                    break

        logger.info(
            "[MVFRQueryMotionPipeline] motion_group='%s' | pred_type=%s | weight=%.3f"
            " | motion_losses=%s",
            self.motion_group_name, prediction_type, motion_task_weight,
            {k for k, v in self._motion_losses.items() if v is not None},
        )

    # ─────────────────────────────────────────────────────────────────────
    # Infer hooks — let super().infer() build the pyramid in its own call
    # ─────────────────────────────────────────────────────────────────────

    def _infer_extra_model_kwargs(self) -> dict:
        if self.use_cached_motion_infer and self.motion_group_name is not None:
            return {"return_cached": True}
        return {}

    def _on_infer_model_results(self, results: dict) -> None:
        self._last_motion_caches = results.pop("_motion_caches", {})

    # ─────────────────────────────────────────────────────────────────────
    # Motion-specific input helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_trajectories(self, batch, scale):
        """Load and optionally scale 3-D trajectories from the batch."""
        trajs_2d = batch.get("trajs_2d")
        trajs_3d = batch.get("trajs_3d")
        valids   = batch.get("valids")
        visibs   = batch.get("visibs")

        if isinstance(trajs_2d, list):
            B = len(trajs_2d)
            T = trajs_2d[0].shape[0]
            max_N = max(t.shape[1] for t in trajs_2d)

            def _pad(lst, ch=None):
                if lst is None or lst[0] is None:
                    return None
                shape = (B, T, max_N, ch) if ch else (B, T, max_N)
                out = torch.zeros(shape, dtype=lst[0].dtype)
                for i, t in enumerate(lst):
                    out[i, :, :t.shape[1]] = t
                return out.to(self.device, non_blocking=True)

            trajs_2d = _pad(trajs_2d, 2)
            trajs_3d = _pad(trajs_3d, 3) if trajs_3d is not None else None
            valids   = _pad(valids)
            visibs   = _pad(visibs)
        else:
            for attr in ("trajs_2d", "trajs_3d", "valids", "visibs"):
                v = locals()[attr]
                if v is not None:
                    locals()[attr]  # just reference to keep linter quiet
            if trajs_2d is not None:
                trajs_2d = trajs_2d.to(self.device)
            if trajs_3d is not None:
                trajs_3d = trajs_3d.to(self.device)
            if valids is not None:
                valids = valids.to(self.device)
            if visibs is not None:
                visibs = visibs.to(self.device)

        if trajs_3d is not None and scale is not None:
            trajs_3d = trajs_3d / get_ref_scale(scale)

        return trajs_2d, trajs_3d, valids, visibs

    def _load_motion_extrinsics(self, batch, scale):
        if self.motion_extrinsics_name not in batch:
            return None
        ext = batch[self.motion_extrinsics_name].to(self.device)
        if scale is not None:
            ext = ext.clone()
            ext[..., :3, 3] = self.normalize(
                ext[..., :3, 3],
                scale[:, 0, 0, 0, 0].view(-1, 1, 1).expand(-1, ext.shape[1], 1),
            )
        return ext

    def _parse_time_info(self, batch, meta_data):
        """Build per-frame normalised time index from meta_data."""
        if "data_info" not in meta_data:
            return None
        data_info = meta_data["data_info"]
        B = len(data_info)
        T = len(data_info[0]) if data_info else 0
        if T == 0:
            return None
        default_max = getattr(self.qcfg, "default_max_frames", 24) if self.qcfg else 24
        max_frames = scalar(meta_data.get("max_frames"), default_max)

        # Warn if max_frames is suspiciously small (likely using default)
        max_frame_id = max(
            fd.get("frame_id", 0) for b in range(B) for fd in data_info[b]
        )
        if max_frame_id > max_frames:
            logger.warning(
                "[_parse_time_info] max_frame_id=%d > max_frames=%d; "
                "time_idx will exceed [0,1]. Check dataset max_frames setting.",
                max_frame_id, max_frames,
            )

        time_idx = torch.tensor(
            [[float(fd.get("frame_id", 0)) / max_frames for fd in data_info[b]]
             for b in range(B)],
            dtype=torch.float32, device=self.device,
        )

        # Clamp to [0, 1] for safety (B-spline domain)
        if time_idx.max() > 1.0 or time_idx.min() < 0.0:
            logger.warning(
                "[_parse_time_info] time_idx range [%.3f, %.3f] out of [0,1]; clamping.",
                time_idx.min().item(), time_idx.max().item(),
            )
            time_idx = time_idx.clamp(0.0, 1.0)

        meta_data["time_idx"] = time_idx
        return time_idx

    # ─────────────────────────────────────────────────────────────────────
    # GT preparation for motion loss
    # ─────────────────────────────────────────────────────────────────────

    def _reduce_iterative_loss(
        self,
        pred: torch.Tensor,
        single_loss_fn,
        loss_name: str,
    ) -> torch.Tensor:
        """Optionally apply deep supervision over iterative predictions."""
        pred = sanitize(pred)
        if not torch.isfinite(pred).all():
            logger.warning("[%s] NaN/Inf in pred — skipping.", loss_name)
            return torch.tensor(0.0, device=pred.device)

        if pred.ndim == 3:
            return single_loss_fn(pred)

        if pred.ndim != 4:
            logger.warning(
                "[%s] Unsupported prediction rank %d; expected 3 or 4 dims.",
                loss_name, pred.ndim,
            )
            return torch.tensor(0.0, device=pred.device)

        if not self.supervise_iter_predictions:
            return single_loss_fn(pred[:, -1])

        num_iters = pred.shape[1]
        total = pred.new_tensor(0.0)
        for k in range(num_iters):
            weight = self.aux_loss_gamma ** (num_iters - 1 - k)
            loss_k = single_loss_fn(pred[:, k])
            if torch.is_tensor(loss_k) and torch.isfinite(loss_k):
                total = total + loss_k * weight
        return total

    def _prepare_gt_displacement(self, gt_tgt_world, motion_queries, extrinsics):
        """Rotate GT displacement into frame-0 camera space.

        ``disp_cam = R_frame0 @ (tgt_3d_world − src_3d_world)``
        """
        COORD = 0
        R    = extrinsics[:, COORD, :3, :3]
        disp = gt_tgt_world - motion_queries["gt_3d_at_src"]
        return torch.bmm(R, disp.transpose(1, 2)).transpose(1, 2)

    # ─────────────────────────────────────────────────────────────────────
    # Motion loss
    # ─────────────────────────────────────────────────────────────────────

    def _compute_motion_loss(
        self,
        motion_out: dict,
        motion_queries: dict,
        gt_positions: torch.Tensor,
        valid_mask: torch.Tensor,
        extrinsics: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pred = motion_out.get("pred_3d_iters")
        if pred is None:
            pred = motion_out.get("displacement_iters")
        if pred is None:
            pred = motion_out.get("pred_3d")
        if pred is None:
            pred = motion_out.get("displacement")
        if pred is None:
            logger.warning("[motion_loss] No 'pred_3d' or 'displacement' in motion_out.")
            return torch.tensor(0.0, device=valid_mask.device)

        with torch.no_grad():
            gt_cam = self._prepare_gt_displacement(gt_positions, motion_queries, extrinsics)

        def _single_loss(pred_single: torch.Tensor) -> torch.Tensor:
            if self.prediction_type == "displacement":
                loss_fn = (self._motion_losses.get("sparse_displacement")
                           or self._motion_losses.get("sparse_motion"))
                if loss_fn is None:
                    return torch.tensor(0.0, device=pred_single.device)
                result = loss_fn(
                    pred_displacement=pred_single, gt_displacement=gt_cam,
                    valid_mask=valid_mask, scale=scale,
                )
            else:
                loss_fn = self._motion_losses.get("sparse_motion")
                if loss_fn is None:
                    return torch.tensor(0.0, device=pred_single.device)
                result = loss_fn(
                    pred_positions=pred_single, gt_positions=gt_cam, valid_mask=valid_mask,
                )
            return result[0] if isinstance(result, (tuple, list)) else result

        return self._reduce_iterative_loss(pred, _single_loss, loss_name="motion_loss")

    def _compute_flow2d_loss(
        self,
        motion_out: dict,
        motion_queries: dict,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 2-D optical flow loss supervised by ``trajs_2d``.

        GT flow = ``gt_2d_at_tgt - uv``, both in normalized [0, 1] image
        coordinates, giving a dimensionless displacement.  The loss function
        is identical in interface to ``SparseDisplacementLoss`` (accepts
        ``pred_displacement``, ``gt_displacement``, ``valid_mask``), so any
        existing 3-D displacement loss class works here with 2-D inputs.
        """
        pred = motion_out.get("flow_2d_iters")
        if pred is None:
            pred = motion_out.get("flow_2d")
        if pred is None:
            return torch.tensor(0.0, device=valid_mask.device)

        gt_2d_tgt = motion_queries.get("gt_2d_at_tgt")
        uv_src    = motion_queries.get("uv")
        if gt_2d_tgt is None or uv_src is None:
            logger.warning("[flow2d_loss] gt_2d_at_tgt or uv missing from queries — skipping.")
            return torch.tensor(0.0, device=pred.device)

        gt_flow = (gt_2d_tgt - uv_src).detach()   # [B, Q, 2], no grad through GT

        if self._flow2d_loss is None:
            return torch.tensor(0.0, device=pred.device)

        def _single_loss(pred_single: torch.Tensor) -> torch.Tensor:
            result = self._flow2d_loss(
                pred_displacement=pred_single,
                gt_displacement=gt_flow,
                valid_mask=valid_mask,
            )
            return result[0] if isinstance(result, (tuple, list)) else result

        return self._reduce_iterative_loss(pred, _single_loss, loss_name="flow2d_loss")

    def _compute_coarse_flow2d_loss(
        self,
        motion_out: dict,
        motion_queries: dict,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary GT supervision for the coarse 2-D flow from refine aggregators."""
        pred = motion_out.get("coarse_flow_2d")
        if pred is None or self._coarse_flow2d_loss is None:
            return torch.tensor(0.0, device=valid_mask.device)

        gt_2d_tgt = motion_queries.get("gt_2d_at_tgt")
        uv_src = motion_queries.get("uv")
        if gt_2d_tgt is None or uv_src is None:
            return torch.tensor(0.0, device=pred.device)

        gt_flow = (gt_2d_tgt - uv_src).detach()
        result = self._coarse_flow2d_loss(
            pred_displacement=pred, gt_displacement=gt_flow, valid_mask=valid_mask,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    def _compute_per_pass_flow2d_loss(
        self,
        motion_out: dict,
        motion_queries: dict,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-pass GT supervision for intermediate flow predictions.

        Each refinement pass produces a flow delta stored as
        ``pass_0_flow_2d``, ``pass_1_flow_2d``, etc.  We accumulate them
        and compare the cumulative flow at each pass against the full GT
        flow, applying ``aux_loss_gamma`` decay so earlier (less accurate)
        passes receive lower weight.

        Weight schedule (gamma=0.8, 2 passes)::

            pass_0  cumulative  weight = gamma^2 = 0.64
            pass_1  cumulative  weight = gamma^1 = 0.80
        """
        if self._flow2d_loss is None:
            return torch.tensor(0.0, device=valid_mask.device)

        gt_2d_tgt = motion_queries.get("gt_2d_at_tgt")
        uv_src = motion_queries.get("uv")
        if gt_2d_tgt is None or uv_src is None:
            return torch.tensor(0.0, device=valid_mask.device)

        gt_flow = (gt_2d_tgt - uv_src).detach()

        pass_flows = []
        for i in range(100):
            key = f"pass_{i}_flow_2d"
            if key not in motion_out:
                break
            pass_flows.append(motion_out[key])

        if not pass_flows:
            return torch.tensor(0.0, device=valid_mask.device)

        num_passes = len(pass_flows)
        total_loss = gt_flow.new_tensor(0.0)
        cumulative_flow = torch.zeros_like(gt_flow)

        for i, delta_flow in enumerate(pass_flows):
            cumulative_flow = cumulative_flow + delta_flow
            weight = self.aux_loss_gamma ** (num_passes - i)
            result = self._flow2d_loss(
                pred_displacement=cumulative_flow,
                gt_displacement=gt_flow,
                valid_mask=valid_mask,
            )
            loss_i = result[0] if isinstance(result, (tuple, list)) else result
            if torch.is_tensor(loss_i) and torch.isfinite(loss_i):
                total_loss = total_loss + loss_i * weight

        return total_loss

    def _compute_direction_loss(
        self,
        motion_out: dict,
        motion_queries: dict,
        gt_positions: torch.Tensor,
        valid_mask: torch.Tensor,
        extrinsics: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Direction + static-magnitude loss for SphericalHead.

        Dynamic points: cosine similarity on predicted direction.
        Static points: magnitude → 0 regularisation (when available).
        """
        pred_dir = motion_out.get("displacement_direction")
        if pred_dir is None or self._direction_loss is None:
            return torch.tensor(0.0, device=valid_mask.device)

        with torch.no_grad():
            gt_cam = self._prepare_gt_displacement(
                gt_positions, motion_queries, extrinsics,
            )

        pred_mag = motion_out.get("displacement_magnitude")

        result = self._direction_loss(
            pred_direction=pred_dir,
            gt_displacement=gt_cam,
            valid_mask=valid_mask,
            scale=scale,
            pred_magnitude=pred_mag,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    def _compute_coarse_displacement_loss(
        self,
        motion_out: dict,
        motion_queries: dict,
        gt_positions: torch.Tensor,
        valid_mask: torch.Tensor,
        extrinsics: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Auxiliary GT supervision for the coarse 3-D displacement from dual-layer aggregators."""
        pred = motion_out.get("coarse_displacement")
        if pred is None or self._coarse_disp_loss is None:
            return torch.tensor(0.0, device=valid_mask.device)

        with torch.no_grad():
            gt_cam = self._prepare_gt_displacement(gt_positions, motion_queries, extrinsics)

        result = self._coarse_disp_loss(
            pred_displacement=pred, gt_displacement=gt_cam,
            valid_mask=valid_mask, scale=scale,
        )
        return result[0] if isinstance(result, (tuple, list)) else result

    # ─────────────────────────────────────────────────────────────────────
    # Evaluation / inference
    # ─────────────────────────────────────────────────────────────────────

    def _ref_scale_3d(self, scale):
        """[B,1,1] scale tensor for broadcasting to [B, Q, 3]."""
        return scale[:, 0, 0, 0, 0].view(-1, 1, 1) if scale is not None else None

    @staticmethod
    def _attach_empty_motion_fields(out0) -> None:
        """Initialize optional sparse-motion fields as absent."""
        out0.track_pred = None
        out0.track_gt = None
        out0.track_vis_pred = None
        out0.motion_queries_gt_3d_src = None
        out0.motion_queries_uv = None
        out0.motion_queries_tgt_frame = None
        out0.motion_queries_gt_2d_tgt = None
        out0.pred_flow_2d = None

    @torch.no_grad()
    def infer(self, **batch):
        """Override parent infer to add motion inference aligned with SparseMotionPipeline.

        Mirrors ``SparseMotionPipeline._infer_per_target_frame``:
          - Iterates over every non-src target frame.
          - Uses ``motion_single_target()`` which queries ALL visible tracks
            (not a subsampled set), matching the reference evaluation protocol.
          - Denormalizes pred/gt to metric scale before storing in the output,
            so APD thresholds (0.05, 0.1 m …) are comparable to the reference.
          - Populates ``track_pred``, ``track_gt``, ``track_vis_pred``,
            ``motion_queries_gt_3d_src``, ``motion_queries_uv``,
            ``motion_queries_tgt_frame`` in ``mv_outputs[0]`` so that
            ``SparseMotionEvalMetrics`` and ``MotionVisualizer`` work correctly.
        """
        if "meta_data" in batch:
            self._parse_time_info(batch, batch["meta_data"])

        mv_outputs = super().infer(**batch)

        if self.motion_query_bank is None or not mv_outputs:
            return mv_outputs

        (
            name, total_iter, meta_data, image, edge_mask, query_image,
            intrinsics, extrinsics, scale, prompt_depth,
            target_local_depth, target_global_points, target_depth_mask,
            target_normal, target_normal_mask, target_motion_mask,
            target_invalid_mask, image_show, align_data,
            ray_directions, ray_world, extrinsics_noise, prompt_extrinsics,
        ) = self.get_inputs(batch)

        time_idx = meta_data.get("time_idx")
        trajs_2d, trajs_3d, valids, visibs = self._load_trajectories(batch, scale)
        motion_ext = self._load_motion_extrinsics(batch, scale)
        out0 = mv_outputs[0]
        self._attach_empty_motion_fields(out0)

        if trajs_2d is None or trajs_3d is None:
            return mv_outputs

        # Save pre-alignment, normalized trajs_3d for GT visualization.
        # Mirrors SparseMotionPipeline: out.trajs_3d = trajs_3d_raw * scale.
        trajs_3d_raw_norm = trajs_3d  # [B, T, N, 3] normalized, in motion coords

        # Align GT trajectory coords to extrinsics reference frame.
        src_idx = getattr(self.qcfg, "src_frame_idx", 0) if self.qcfg else 0
        if motion_ext is not None and extrinsics is not None:
            raw_to_aligned = (
                torch.linalg.inv(extrinsics[:, src_idx]) @ motion_ext[:, src_idx]
            )
            bsz, T_t, Nq, _ = trajs_3d.shape
            pts = trajs_3d.reshape(bsz, -1, 3)
            pts_h = torch.cat(
                [pts, torch.ones(*pts.shape[:2], 1, device=pts.device, dtype=pts.dtype)], -1,
            )
            trajs_3d = (
                torch.bmm(raw_to_aligned, pts_h.transpose(1, 2))
                .transpose(1, 2)[..., :3]
                .reshape(bsz, T_t, Nq, 3)
            )

        B, N = image.shape[:2]
        H_orig = scalar(meta_data.get("origin_height"), image.shape[-2])
        W_orig = scalar(meta_data.get("origin_width"),  image.shape[-1])

        if prompt_extrinsics is not None:
            w2c = prompt_extrinsics
        elif extrinsics_noise is not None and extrinsics is not None:
            w2c = torch.matmul(extrinsics_noise, extrinsics)
        else:
            w2c = extrinsics

        scale_s = self._ref_scale_3d(scale)  # [B, 1, 1] metric scale factor

        # ── Resolve cached pyramid (built inside super().infer() via hooks) ─
        # When use_cached_motion_infer=True, _infer_extra_model_kwargs() added
        # return_cached=True to the parent's model call, and
        # _on_infer_model_results() stashed the result in self._last_motion_caches.
        # When use_cached_motion_infer=False, we fall back to a full model call
        # per target frame (old behaviour, kept for ablation / debugging).
        use_cache = self.use_cached_motion_infer and bool(self._last_motion_caches)
        cached = self._last_motion_caches.get(self.motion_group_name) if use_cache else None

        if cached is None and self.use_cached_motion_infer:
            logger.warning(
                "[infer] Cached pyramid not available for group '%s'; "
                "falling back to full per-frame inference.",
                self.motion_group_name,
            )

        # Unwrap DDP once for decode_motion_with_cache.
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model

        all_preds, all_gt, all_valid, all_queries, all_flows = [], [], [], [], []

        # ── Per-target-frame inference / decoding ─────────────────────────
        for tgt_frame in (t for t in range(N) if t != src_idx):
            qdata = self.motion_query_bank.motion_single_target(
                trajs_2d, trajs_3d, time_idx, visibs, valids,
                (H_orig, W_orig), tgt_frame=tgt_frame,
            )
            if qdata is None:
                continue
            frame_queries, frame_gt_world, frame_valid = qdata

            if cached is not None:
                # Fast path: aggregator + MLP only, encoder already ran once.
                results = unwrapped.decode_motion_with_cache(cached, frame_queries, rgb=image)
            else:
                # Fallback: full forward pass (encoder + decoder) per frame.
                results = self.model(
                    image,
                    edge_mask=edge_mask,
                    query_rgb=query_image,
                    scale=scale,
                    prompt_depth=prompt_depth,
                    intrinsics=intrinsics,
                    ray_directions=ray_directions,
                    w2c=w2c,
                    ray_world=ray_world,
                    time_idx=time_idx,
                    external_queries={self.motion_group_name: frame_queries},
                    meta_data=meta_data,
                )

            motion_pred = results.get("sparse_motion_pred")
            if motion_pred is None:
                continue

            pred = motion_pred.get("pred_3d")
            if pred is None:
                pred = motion_pred.get("displacement")
            pred_flow_2d = motion_pred.get("flow_2d")
            if pred is None and pred_flow_2d is None:
                continue

            if pred is not None:
                pred = sanitize(pred)  # [B, Q, 3] — normalized displacement
            if pred_flow_2d is not None:
                pred_flow_2d = sanitize(pred_flow_2d)  # [B, Q, 2] — normalized UV delta

            # GT displacement in frame-0 camera space (normalized).
            gt_cam_norm = self._prepare_gt_displacement(
                frame_gt_world, frame_queries, extrinsics,
            )  # [B, Q, 3]

            # Denormalize to metric scale (mirrors SparseMotionPipeline._denormalize_pred_gt).
            gt_src_cam = world_to_camera(frame_queries["gt_3d_at_src"], extrinsics, src_idx)
            if scale_s is not None:
                if pred is not None:
                    pred = pred * scale_s  # [B, Q, 3]
                gt_cam_norm = gt_cam_norm * scale_s
                gt_src_cam  = gt_src_cam  * scale_s

            # Convert displacement → absolute target positions.
            gt_abs    = gt_src_cam + gt_cam_norm
            pred_abs = gt_src_cam + pred if pred is not None else None

            all_gt.append(gt_abs)
            all_valid.append(frame_valid)
            all_queries.append({
                **frame_queries,
                "gt_3d_at_src_metric": gt_src_cam,
            })
            if pred_abs is not None:
                all_preds.append(pred_abs)
            all_flows.append(pred_flow_2d)

        # Populate trajectory-side metadata even when a sample yields 0 valid
        # sparse motion queries, so callback-time eval/vis never touches None.
        if trajs_2d is not None:
            out0.trajs_2d = trajs_2d[0].cpu()       # [T, N, 2] — pixel coords

        if trajs_3d_raw_norm is not None and scale is not None:
            # Denormalize to metric: multiply back by the reference view scale.
            trajs_3d_metric = trajs_3d_raw_norm * get_ref_scale(scale)
            out0.trajs_3d = trajs_3d_metric[0].cpu()  # [T, N, 3] metric world

        if motion_ext is not None:
            # Store METRIC motion extrinsics (denormalize translation), matching
            # SparseMotionPipeline convention.  The visualizer applies:
            #   world_vis = c2w_vis_metric @ w2c_motion_metric @ pt_motion_world
            me_vis = motion_ext[0, 0].clone()
            if scale is not None:
                me_vis[:3, 3] = me_vis[:3, 3] * get_ref_scale(scale)[0, 0, 0, 0]
            out0.motion_extrinsics = me_vis.cpu()  # [4, 4] metric

        if visibs is not None:
            out0.trajs_visibs = visibs[0].cpu()     # [T, N]
        if valids is not None:
            out0.trajs_valids = valids[0].cpu()     # [T, N]

        if not all_queries:
            logger.info("[infer] 0 valid motion-query groups; exporting empty sparse outputs.")
            return mv_outputs

        gt_all     = torch.cat(all_gt,    dim=1)
        valid_all  = torch.cat(all_valid, dim=1)   # [B, Q_total]

        # Merge query dicts for visualizer fields.
        merged_queries = cat_query_dicts(all_queries)

        gt_src_all = torch.cat(
            [q["gt_3d_at_src_metric"] for q in all_queries], dim=1
        )  # [B, Q_total, 3] — metric source positions for dynamic classification

        pred_all = None
        if all_preds:
            if len(all_preds) == len(all_queries):
                pred_all = torch.cat(all_preds, dim=1)  # [B, Q_total, 3]
            else:
                logger.warning(
                    "[infer] displacement missing for %d/%d target frames; skipping 3D motion export.",
                    len(all_queries) - len(all_preds), len(all_queries),
                )

        pred_flow_all = None
        valid_flows = [f for f in all_flows if f is not None]
        if valid_flows:
            if len(valid_flows) == len(all_queries):
                pred_flow_all = torch.cat(valid_flows, dim=1)  # [B, Q_total, 2]
            else:
                logger.warning(
                    "[infer] flow_2d missing for %d/%d target frames; skipping flow export.",
                    len(all_queries) - len(valid_flows), len(all_queries),
                )

        total_queries = int(gt_all.shape[1])
        logger.info(
            "[infer] %d target frames × Q tracks = %d total (%d valid)",
            len(all_queries), total_queries, int(valid_all.sum()),
        )

        # ── Attach to outputs[0] ─────────────────────────────────────────
        out0.track_gt                = gt_all[0].cpu().float().numpy()     # [Q, 3]
        out0.track_vis_pred          = valid_all[0].cpu().float().numpy()  # [Q]
        out0.motion_queries_gt_3d_src = gt_src_all[0].cpu().float().numpy()  # [Q, 3]
        if pred_all is not None:
            out0.track_pred = pred_all[0].cpu().float().numpy()  # [Q, 3]
        if pred_flow_all is not None:
            out0.pred_flow_2d = pred_flow_all[0].cpu().float().numpy()  # [Q, 2]
        if "uv" in merged_queries:
            out0.motion_queries_uv = merged_queries["uv"][0].cpu().float().numpy()
        if "tgt_frame_idx" in merged_queries:
            out0.motion_queries_tgt_frame = merged_queries["tgt_frame_idx"][0].cpu().numpy()
        if "gt_2d_at_tgt" in merged_queries:
            out0.motion_queries_gt_2d_tgt = merged_queries["gt_2d_at_tgt"][0].cpu().float().numpy()

        if not self.training and self.infer_scale_align:
            align_sparse_motion(mv_outputs)

        return mv_outputs

    # ─────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────

    def visualize(self, outputs_list, meta_data, out_dir):
        """Visualize motion predictions (mirrors SparseMotionPipeline.visualize)."""
        if not outputs_list:
            return

        # Dense depth / camera vis from parent (writes to mvdepth/ sub-dir).
        try:
            super().visualize(outputs_list, meta_data, out_dir)
        except Exception:
            pass

        # Sparse motion / flow vis.
        track_pred = getattr(outputs_list[0], "track_pred", None)
        pred_flow_2d = getattr(outputs_list[0], "pred_flow_2d", None)
        has_track_pred = track_pred is not None and len(track_pred) > 0
        has_flow_2d = pred_flow_2d is not None and len(pred_flow_2d) > 0
        if not has_track_pred and not has_flow_2d:
            return

        data_idx   = meta_data["data_idx"][0]
        motion_dir = os.path.join(out_dir, f"sparse_motion/{data_idx:06d}")
        os.makedirs(motion_dir, exist_ok=True)

        if not hasattr(self, "_motion_visualizer"):
            self._motion_visualizer = MotionVisualizer()

        try:
            self._motion_visualizer.render(outputs_list[0], motion_dir, outputs_list=outputs_list)
        except Exception:
            logger.error("[visualize] MotionVisualizer failed:\n%s", traceback.format_exc())

        try:
            vis_sparse_motion_3d_rerun(
                cfg=self.save_output_cfg, mv_outputs=outputs_list,
                out_dir=out_dir, data_idx=data_idx, meta_data=meta_data,
            )
        except Exception:
            logger.error("[visualize] 3D vis failed:\n%s", traceback.format_exc())

    # ─────────────────────────────────────────────────────────────────────
    # Unified train_step
    # ─────────────────────────────────────────────────────────────────────

    def train_step(self, batch):  # noqa: C901
        self.train()

        (
            name, total_iter, meta_data, image, edge_mask, query_image,
            intrinsics, extrinsics, scale, prompt_depth,
            target_local_depth, target_global_points, target_depth_mask,
            target_normal, target_normal_mask, target_motion_mask,
            target_invalid_mask, image_show, align_data,
            ray_directions, ray_world, extrinsics_noise, prompt_extrinsics,
        ) = self.get_inputs(batch)

        meta_data["sub_pixel_scale"] = self.training_sub_pixel_scale

        B, N = image.shape[:2]
        H_orig = scalar(meta_data.get("origin_height"), image.shape[-2])
        W_orig = scalar(meta_data.get("origin_width"),  image.shape[-1])

        if prompt_extrinsics is not None:
            w2c = prompt_extrinsics
        elif extrinsics_noise is not None and extrinsics is not None:
            w2c = torch.matmul(extrinsics_noise, extrinsics)
        else:
            w2c = extrinsics

        # ── Motion-specific inputs ────────────────────────────────────────
        time_idx   = self._parse_time_info(batch, meta_data)
        trajs_2d, trajs_3d, valids, visibs = self._load_trajectories(batch, scale)
        motion_ext = self._load_motion_extrinsics(batch, scale)

        # Align GT trajectory coords to the extrinsics reference frame.
        src_idx = getattr(self.qcfg, "src_frame_idx", 0) if self.qcfg else 0
        if trajs_3d is not None and motion_ext is not None and extrinsics is not None:
            raw_to_aligned = (
                torch.linalg.inv(extrinsics[:, src_idx])
                @ motion_ext[:, src_idx]
            )
            bsz, T_t, Nq, _ = trajs_3d.shape
            pts = trajs_3d.reshape(bsz, -1, 3)
            pts_h = torch.cat(
                [pts, torch.ones(*pts.shape[:2], 1, device=pts.device, dtype=pts.dtype)], -1,
            )
            trajs_3d = (
                torch.bmm(raw_to_aligned, pts_h.transpose(1, 2))
                .transpose(1, 2)[..., :3]
                .reshape(bsz, T_t, Nq, 3)
            )

        # Build motion queries from GT trajectories.
        motion_queries = gt_positions = valid_mask = None
        has_motion = (
            self.motion_query_bank is not None
            and trajs_2d is not None
            and trajs_3d is not None
            and (
                any(v is not None for v in self._motion_losses.values())
                or self._flow2d_loss is not None
            )
        )
        if has_motion:
            motion_queries, gt_positions, valid_mask = self.motion_query_bank.motion(
                trajs_2d, trajs_3d, time_idx, visibs, valids, (H_orig, W_orig),
            )

        # Build external_queries dict keyed by the motion group name.
        external_queries = None
        if has_motion and motion_queries is not None and self.motion_group_name:
            external_queries = {self.motion_group_name: motion_queries}

        # ── Single forward pass ───────────────────────────────────────────
        results = self.model(
            image,
            edge_mask=edge_mask,
            query_rgb=query_image,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=ray_directions,
            w2c=w2c,
            ray_world=ray_world,
            time_idx=time_idx,
            external_queries=external_queries,
            meta_data=meta_data,
        )

        query_depth     = results.get("depth")
        query_conf      = results.get("confidence")
        query_glob_pts  = results.get("global_points")
        query_glob_conf = results.get("global_confidence")
        pose_enc        = results.get("pose_enc")
        query           = results.get("query")

        # Dense DPT heads produce 5-D [B, N, C, H, W] full-resolution maps;
        # query-based task groups produce 4-D [B, N, Q, C] per-query tensors.
        # Reshape only the 4-D query-level outputs (mirrors v2 train_step).
        dense_fullres = (
            (query_depth is not None and query_depth.ndim == 5)
            or (query_glob_pts is not None and query_glob_pts.ndim == 5)
        )
        if not dense_fullres:
            if query_depth is not None:
                query_depth = query_depth.unsqueeze(2)
            if query_conf is not None:
                query_conf = query_conf.unsqueeze(2)
            if query_glob_pts is not None:
                if query_glob_pts.ndim == 4:
                    query_glob_pts = query_glob_pts.permute(0, 1, 3, 2).contiguous().unsqueeze(-1)
                else:
                    logger.warning(
                        "[train_step] query_glob_pts.ndim=%d (expected 4 in "
                        "non-dense path); skipping permute to avoid crash. "
                        "Check the head that produced this tensor.",
                        query_glob_pts.ndim,
                    )
            if query_glob_conf is not None:
                query_glob_conf = query_glob_conf.unsqueeze(2)

        # Build UV grid for GT sampling at query positions (mirrors v2).
        query_uv_grid = None
        if query is not None:
            if query.uv is None:
                query_uv_grid = query.batch_uv * 2 - 1
                query_uv_grid = query_uv_grid.unsqueeze(2)
            else:
                query_uv_grid = query.uv * 2 - 1
                query_uv_grid = query_uv_grid.unsqueeze(0).unsqueeze(2)
                query_uv_grid = query_uv_grid.expand(B * N, -1, -1, -1)

        if self.pose_encoding_type == "absT_quaR_FoV":
            pose_enc = results.get("pose_enc")
        else:
            raise NotImplementedError

        # Keep the loss as a tensor even when all task losses are skipped.
        total_loss = image.sum() * 0.0
        total_loss_dict = dict()

        # Normalise target_depth_mask to [B*N, C, H, W] once (mirrors v2).
        if target_depth_mask is not None:
            if target_depth_mask.ndim == 4:
                target_depth_mask = target_depth_mask.view(
                    B * N, 1, *target_depth_mask.shape[-2:]
                )
            elif target_depth_mask.ndim == 5:
                target_depth_mask = target_depth_mask.view(
                    B * N, *target_depth_mask.shape[-3:]
                )
            else:
                raise NotImplementedError

        # ── Part 1: Local depth loss ──────────────────────────────────────
        if query_depth is not None or query_conf is not None:
            if dense_fullres and target_local_depth is not None:
                # Full-resolution DPT path: compare pixel-level predictions
                # directly against GT without grid-sampling.
                pred_d = query_depth.view(B, N, *query_depth.shape[-3:])
                pred_c = query_conf.view(B, N, *query_conf.shape[-3:]) if query_conf is not None else None
                tgt_d  = target_local_depth.view(B, N, *target_local_depth.shape[-3:])
                tgt_m  = target_depth_mask.view(B, N, *target_depth_mask.shape[-3:])
                if pred_d.shape[-2:] != tgt_d.shape[-2:]:
                    _pd = F.interpolate(
                        pred_d.view(B * N, *pred_d.shape[-3:]),
                        tgt_d.shape[-2:], mode="bilinear", align_corners=False,
                    )
                    pred_d = _pd.view(B, N, *_pd.shape[1:])
                    if pred_c is not None:
                        _pc = F.interpolate(
                            pred_c.view(B * N, *pred_c.shape[-3:]),
                            tgt_d.shape[-2:], mode="bilinear", align_corners=False,
                        )
                        pred_c = _pc.view(B, N, *_pc.shape[1:])
                loss, loss_dict = self.get_base_loss(
                    name=name, image=image, coord="local",
                    pred_depth=pred_d, pred_conf=pred_c,
                    target_depth=tgt_d, valid_mask=tgt_m == 1, scale=scale,
                )
                total_loss, total_loss_dict = self.add_loss(
                    total_loss, total_loss_dict, loss, loss_dict, task_name="lcl",
                )
            elif query_uv_grid is not None:
                tld = target_local_depth.view(B * N, *target_local_depth.shape[-3:])

                tgt_q_depth = F.grid_sample(
                    tld, query_uv_grid, mode="bilinear", padding_mode="border", align_corners=False,
                )
                tgt_q_mask = F.grid_sample(
                    target_depth_mask.float(), query_uv_grid,
                    mode="bilinear", padding_mode="border", align_corners=False,
                )

                if query.full_uv:
                    query_depth = query_depth.squeeze(-1).view(
                        *query_depth.shape[:3], query.height, query.width
                    )
                    query_conf = query_conf.squeeze(-1).view(
                        *query_conf.shape[:3], query.height, query.width
                    )
                    if query_depth.shape[-3] == 1:
                        query_depth = self.depth_to_points(query_depth, K=intrinsics)
                    tgt_q_depth = tgt_q_depth.squeeze(-1).view(B, N, *query_depth.shape[-3:])
                    tgt_q_mask  = tgt_q_mask.squeeze(-1).view(B, N, *query_conf.shape[-3:]) == 1
                else:
                    tgt_q_depth = tgt_q_depth.squeeze(-1).view(B, N, *tgt_q_depth.shape[-3:])
                    tgt_q_mask  = tgt_q_mask.squeeze(-1).view(B, N, *tgt_q_mask.shape[-3:]) == 1

                loss, loss_dict = self.get_base_loss(
                    name=name, image=image, coord="local",
                    pred_depth=query_depth, pred_conf=query_conf,
                    target_depth=tgt_q_depth, valid_mask=tgt_q_mask, scale=scale,
                )
                total_loss, total_loss_dict = self.add_loss(
                    total_loss, total_loss_dict, loss, loss_dict, task_name="lcl",
                )

        # ── Part 2: Global points loss ────────────────────────────────────
        if query_glob_pts is not None or query_glob_conf is not None:
            if dense_fullres and target_global_points is not None:
                pred_g = query_glob_pts.view(B, N, *query_glob_pts.shape[-3:])
                pred_gc = query_glob_conf.view(B, N, *query_glob_conf.shape[-3:]) if query_glob_conf is not None else None
                tgt_g  = target_global_points.view(B, N, *target_global_points.shape[-3:])
                tgt_m  = target_depth_mask.view(B, N, *target_depth_mask.shape[-3:])
                if pred_g.shape[-2:] != tgt_g.shape[-2:]:
                    _pg = F.interpolate(
                        pred_g.view(B * N, *pred_g.shape[-3:]),
                        tgt_g.shape[-2:], mode="bilinear", align_corners=False,
                    )
                    pred_g = _pg.view(B, N, *_pg.shape[1:])
                    if pred_gc is not None:
                        _pgc = F.interpolate(
                            pred_gc.view(B * N, *pred_gc.shape[-3:]),
                            tgt_g.shape[-2:], mode="bilinear", align_corners=False,
                        )
                        pred_gc = _pgc.view(B, N, *_pgc.shape[1:])
                loss, loss_dict = self.get_base_loss(
                    name=name, image=image, coord="global",
                    pred_depth=pred_g, pred_conf=pred_gc,
                    target_depth=tgt_g, valid_mask=tgt_m == 1, scale=scale,
                )
                total_loss, total_loss_dict = self.add_loss(
                    total_loss, total_loss_dict, loss, loss_dict, task_name="glb",
                )
            elif query_uv_grid is not None:
                tgp = target_global_points.view(B * N, *target_global_points.shape[-3:])

                tgt_q_glb = F.grid_sample(
                    tgp, query_uv_grid, mode="bilinear", padding_mode="border", align_corners=False,
                )
                tgt_q_mask_glb = F.grid_sample(
                    target_depth_mask.float(), query_uv_grid,
                    mode="bilinear", padding_mode="border", align_corners=False,
                )

                if query.full_uv:
                    query_glob_pts  = query_glob_pts.view(
                        *query_glob_pts.shape[:3], query.height, query.width
                    )
                    query_glob_conf = query_glob_conf.squeeze(-1).view(
                        *query_glob_conf.shape[:3], query.height, query.width
                    )
                    tgt_q_glb      = tgt_q_glb.view(B, N, *query_glob_pts.shape[-3:])
                    tgt_q_mask_glb = tgt_q_mask_glb.squeeze(-1).view(
                        B, N, *query_glob_conf.shape[-3:]
                    ) == 1
                else:
                    tgt_q_glb      = tgt_q_glb.view(B, N, *tgt_q_glb.shape[-3:])
                    tgt_q_mask_glb = tgt_q_mask_glb.squeeze(-1).view(
                        B, N, *tgt_q_mask_glb.shape[-3:]
                    ) == 1

                loss, loss_dict = self.get_base_loss(
                    name=name, image=image, coord="global",
                    pred_depth=query_glob_pts, pred_conf=query_glob_conf,
                    target_depth=tgt_q_glb, valid_mask=tgt_q_mask_glb, scale=scale,
                )
                total_loss, total_loss_dict = self.add_loss(
                    total_loss, total_loss_dict, loss, loss_dict, task_name="glb",
                )

        # ── Part 3: Camera loss ───────────────────────────────────────────
        if self.camera_loss is not None and pose_enc is not None and pose_enc[0].requires_grad:
            loss, loss_dict = self.camera_loss(
                name=name, pose_enc=pose_enc,
                target_intrinsic=intrinsics, target_extrinsics=extrinsics,
                scale=scale, image_size_hw=image.shape[-2:],
                valid_mask=target_depth_mask,
            )
            total_loss, total_loss_dict = self.add_loss(
                total_loss, total_loss_dict, loss, loss_dict, task_name="cm",
            )

        # ── Part 4: Motion displacement loss ─────────────────────────────
        motion_pred = results.get("sparse_motion_pred")
        if (
            motion_pred is not None
            and motion_queries is not None
            and gt_positions is not None
            and (motion_pred.get("pred_3d") is not None or motion_pred.get("displacement") is not None)
        ):
            motion_loss = self._compute_motion_loss(
                motion_out=motion_pred,
                motion_queries=motion_queries,
                gt_positions=gt_positions,
                valid_mask=valid_mask,
                extrinsics=extrinsics,
                scale=scale,
            )
            if torch.is_tensor(motion_loss) and torch.isfinite(motion_loss):
                total_loss = total_loss + motion_loss * self.motion_task_weight
                total_loss_dict["motion"] = round(float(motion_loss.detach().cpu()), 6)

        # ── Part 4b: Direction loss for SphericalHead ─────────────────────
        if (motion_pred is not None and motion_queries is not None
                and gt_positions is not None and valid_mask is not None
                and self._direction_loss is not None):
            dir_loss = self._compute_direction_loss(
                motion_out=motion_pred,
                motion_queries=motion_queries,
                gt_positions=gt_positions,
                valid_mask=valid_mask,
                extrinsics=extrinsics,
                scale=scale,
            )
            if torch.is_tensor(dir_loss) and torch.isfinite(dir_loss):
                total_loss = total_loss + dir_loss * self.displacement_direction_weight
                total_loss_dict["dir"] = round(float(dir_loss.detach().cpu()), 6)

        # ── Part 5: 2-D optical flow loss ─────────────────────────────────
        if (motion_pred is not None and motion_queries is not None
                and valid_mask is not None and self._flow2d_loss is not None):
            flow2d_loss = self._compute_flow2d_loss(
                motion_out=motion_pred,
                motion_queries=motion_queries,
                valid_mask=valid_mask,
            )
            if torch.is_tensor(flow2d_loss) and torch.isfinite(flow2d_loss):
                total_loss = total_loss + flow2d_loss * self.flow_2d_task_weight
                total_loss_dict["flow_2d"] = round(float(flow2d_loss.detach().cpu()), 6)

        # ── Part 5b: Per-pass intermediate flow supervision ─────────────────
        if (motion_pred is not None and motion_queries is not None
                and valid_mask is not None and self._flow2d_loss is not None):
            pass_flow_loss = self._compute_per_pass_flow2d_loss(
                motion_out=motion_pred,
                motion_queries=motion_queries,
                valid_mask=valid_mask,
            )
            if torch.is_tensor(pass_flow_loss) and torch.isfinite(pass_flow_loss):
                total_loss = total_loss + pass_flow_loss * self.flow_2d_task_weight
                total_loss_dict["pass_flow"] = round(float(pass_flow_loss.detach().cpu()), 6)

        # ── Part 6: Coarse flow 2-D auxiliary loss ─────────────────────────
        if (motion_pred is not None and motion_queries is not None
                and valid_mask is not None and self._coarse_flow2d_loss is not None):
            coarse_flow_loss = self._compute_coarse_flow2d_loss(
                motion_out=motion_pred,
                motion_queries=motion_queries,
                valid_mask=valid_mask,
            )
            if torch.is_tensor(coarse_flow_loss) and torch.isfinite(coarse_flow_loss):
                total_loss = total_loss + coarse_flow_loss * self.coarse_flow_2d_task_weight
                total_loss_dict["coarse_flow"] = round(float(coarse_flow_loss.detach().cpu()), 6)

        # ── Part 7: Coarse displacement auxiliary loss ─────────────────────
        if (motion_pred is not None and motion_queries is not None
                and gt_positions is not None and valid_mask is not None
                and self._coarse_disp_loss is not None):
            coarse_disp_loss = self._compute_coarse_displacement_loss(
                motion_out=motion_pred,
                motion_queries=motion_queries,
                gt_positions=gt_positions,
                valid_mask=valid_mask,
                extrinsics=extrinsics,
                scale=scale,
            )
            if torch.is_tensor(coarse_disp_loss) and torch.isfinite(coarse_disp_loss):
                total_loss = total_loss + coarse_disp_loss * self.coarse_displacement_task_weight
                total_loss_dict["coarse_disp"] = round(float(coarse_disp_loss.detach().cpu()), 6)

        # ── Bookkeeping ───────────────────────────────────────────────────
        if scale is not None:
            total_loss_dict["scale"] = round(scale.max().cpu().item(), 2)
        if image is not None:
            total_loss_dict["aspect_ratio"] = round(image.shape[-1] / image.shape[-2], 2)
            total_loss_dict["bs"]            = int(B)
            total_loss_dict["view"]          = int(N)
            total_loss_dict["max_size"]      = int(max(image.shape[-1], image.shape[-2]))

        if self.debug_rgb_path:
            if isinstance(meta_data["data_info"][0], (list, tuple)):
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0][0]['rgb']}")
            else:
                logging.debug(f"rgb: {name}, {meta_data['data_info'][0]['rgb']}")

        if not torch.is_tensor(total_loss):
            total_loss = image.sum() * 0.0 + float(total_loss)

        return total_loss, total_loss_dict
