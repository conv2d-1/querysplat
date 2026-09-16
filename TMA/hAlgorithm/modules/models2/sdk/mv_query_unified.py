"""MVQueryUnified — shared ViT-G encoder with registry-based multi-task query groups.

Design
------
Three named registries are built at construction time from config lists:

  ``query_banks``     — { name → QueryBank instance }
                        Query banks whose type is ``MotionQueryBank`` are
                        *not* called by the model; they are instantiated here
                        but handed to the pipeline via
                        ``self.pipeline_managed_query_banks``.

  ``aggregators``     — { name → Aggregator instance }
                        Dispatched via the ``AGGREGATOR_TYPE`` class attribute:
                        ``"motion"``   → ``(x, motion_queries, B, N, meta_data)``
                        ``"standard"`` → ``(x, query, query_rgb, meta_data)``

  ``decoder_groups``  — { name → Decoder instance }
                        Dispatch is selected by the optional ``DECODER_TYPE``
                        class attribute:
                        ``"standard"``          → ``decoder(x=..., meta_data=...)``
                        ``"motion_structured"`` → ``decoder(inputs=..., meta_data=...)``
                        Same name = same weight instance.

``task_groups`` is a list of routing dicts that wire the three registries
together and specify which output keys each group claims from the decoder.

Time encoding
-------------
An optional ``time_encoder`` (``TimeTokenEncoder``) injects per-frame
sinusoidal time embeddings into the DA3 backbone **before** ``fuse_encoder``.
The embedding is added onto the camera token exactly as in
``MVBaseMotionWithTime``; when no ``camera_encoder`` is configured the time
embedding itself becomes the camera token.  Set ``embed_dim`` to the
backbone's pre-cat hidden dimension (1536 for ViT-G with ``cat_token=True``).

Forward
-------
::

    RGB [B, N, C, H, W]
        │
        ▼  DinoV2 ViT-G  (shared, one forward)
    patch_features
        │
        ├─ dense task_group ────────────────────────────────┐
        │   QueryBank5 (model-managed)                      │
        │   → query  [B*N, Q_d, 2]                          │
        │   → Aggregator2  [B*N, Q_d, 256]                  │
        │   → shared MLP Head  [B*N, Q_d, 9]                │
        │   → take depth/conf/global_pts/glb_conf           │
        │   → reshape [B, N, Q_d, D]                        │
        │                                                   │
        └─ motion task_group ───────────────────────────────┘
            external_queries["motion"]  (built by pipeline)
            → MotionAggregatorMLP  [B, Q_m, 256]
            → shared MLP Head  [B, Q_m, 9]   ← same instance
            → take displacement  [B, Q_m, 3]

The shared MLP Head always outputs all 9 dimensions; each task_group
silently discards outputs it does not claim.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from hAlgorithm.modules.models2.sdk.base import MVBase2
from hAlgorithm.utils import instantiate_from_config

logger = logging.getLogger(__name__)


class MVQueryUnified(MVBase2):
    """Registry-based unified query model: dense 3D + sparse motion in one framework.

    Args:
        query_banks: List of query-bank configs.  Each dict must have a
            ``"name"`` key plus the standard ``"type"`` + constructor kwargs.
            Banks of type ``MotionQueryBank`` are pipeline-managed and exposed
            via :attr:`pipeline_managed_query_banks`.
        aggregators: List of aggregator configs.  Same ``name``/``type`` format.
            Aggregators with ``AGGREGATOR_TYPE == "motion"`` receive
            ``(x, motion_queries, B, N, meta_data)``; all others receive
            ``(x, query, query_rgb, meta_data)``.
        decoder_groups: List of decoder-group configs, each with a ``"name"``
            key and a ``"decoder"`` sub-config.  **Same name = shared instance.**
            Structured motion decoders can expose
            ``DECODER_TYPE="motion_structured"``.
        task_groups: List of routing dicts.  Each entry specifies:
            - ``name``          – unique group identifier
            - ``query_bank``    – name in the query_banks registry
            - ``aggregator``    – name in the aggregators registry
            - ``decoder_group`` – name in the decoder_groups registry
            - ``tasks``         – list of output keys this group claims
        freeze_encoders_for_motion: Freeze encoder weights during training.
        **kwargs: Forwarded to ``MVBase2`` (``fuse_encoder``, ``camera_head``, …).
    """

    _ENCODER_MODULE_NAMES: Tuple[str, ...] = (
        "fuse_encoder",
        "rgb_encoder",
        "camera_encoder",
        "ray_encoder",
        "ray_in_world_encoder",
        "depth_encoder",
        "extra_encoder",
    )

    def __init__(
        self,
        query_banks: Optional[List[dict]] = None,
        aggregators: Optional[List[dict]] = None,
        decoder_groups: Optional[List[dict]] = None,
        task_groups: Optional[List[dict]] = None,
        freeze_encoders_for_motion: bool = False,
        time_encoder: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Optional time encoder: sinusoidal embeddings added to the camera
        # token before fuse_encoder so the backbone sees temporal ordering.
        self.time_encoder = self._instantiate_and_register(time_encoder, "time_encoder")

        # ── Query bank registry ───────────────────────────────────────────
        # Pipeline-managed banks (e.g. MotionQueryBank) are stored separately
        # and not called inside forward(); the pipeline calls them externally.
        self._qb_reg: Dict[str, Any] = {}
        self.pipeline_managed_query_banks: Dict[str, Any] = {}

        for cfg in (query_banks or []):
            name, inst = self._build_named(cfg)
            if self._is_pipeline_managed(inst):
                self.pipeline_managed_query_banks[name] = inst
                # Register as submodule only if it carries learnable params.
                if sum(p.numel() for p in inst.parameters()) > 0:
                    self.add_module(f"qb_{name}", inst)
            else:
                self._qb_reg[name] = inst
                self.add_module(f"qb_{name}", inst)

        # ── Aggregator registry ───────────────────────────────────────────
        self._agg_reg: Dict[str, Any] = {}
        for cfg in (aggregators or []):
            name, inst = self._build_named(cfg)
            self._agg_reg[name] = inst
            self.add_module(f"agg_{name}", inst)

        # ── Decoder registry (same name → same nn.Module instance) ───────
        self._dec_reg: Dict[str, Any] = {}
        for cfg in (decoder_groups or []):
            name = cfg["name"]
            inst = instantiate_from_config(cfg["decoder"])
            self._dec_reg[name] = inst
            self.add_module(f"dec_{name}", inst)

        # Task groups are pure routing configs — no instantiation needed.
        self.task_groups: List[dict] = task_groups or []
        self.freeze_encoders_for_motion = freeze_encoders_for_motion

    # ─────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_named(cfg: dict):
        """Pop ``"name"`` from a config dict and instantiate the remainder."""
        cfg = dict(cfg)
        name = cfg.pop("name")
        return name, instantiate_from_config(cfg)

    @staticmethod
    def _is_pipeline_managed(inst) -> bool:
        """True for query banks that require GT batch data.

        Pipeline-managed banks (e.g. MotionQueryBank, TrajectoryQueryBank) are
        NOT called inside forward(); instead the pipeline calls them externally
        with GT trajectory data and passes the result via ``external_queries``.

        Query banks opt-in by setting class attribute ``PIPELINE_MANAGED = True``.
        """
        return getattr(inst, "PIPELINE_MANAGED", False)

    # ─────────────────────────────────────────────────────────────────────
    # Encoder freeze helper
    # ─────────────────────────────────────────────────────────────────────

    def _freeze_encoders(self) -> None:
        for name in self._ENCODER_MODULE_NAMES:
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    # ─────────────────────────────────────────────────────────────────────
    # Time encoding hook
    # ─────────────────────────────────────────────────────────────────────

    def _modify_cam_token(self, cam_token, b, n, device, meta_data):
        """Inject per-frame time embeddings into the backbone camera token.

        Mirrors ``MVBaseMotionWithTime._modify_cam_token``: the sinusoidal
        time embedding (shape ``[B, N, embed_dim]``) is *added* onto the
        existing camera token.  When no ``camera_encoder`` is configured
        (``cam_token is None``), the time embedding itself becomes the
        camera token that is forwarded to ``fuse_encoder``.

        Normalized timestamps are read from ``meta_data["time_idx"]``
        (shape ``[B, N]``, range [0, 1]) when present; otherwise they are
        computed as a uniform grid from 0 to 1 over ``n`` frames.
        """
        if self.time_encoder is None:
            return cam_token

        if meta_data is not None and "time_idx" in meta_data:
            frame_timestamps = meta_data["time_idx"].float().view(b, n)
        else:
            if n <= 1:
                frame_timestamps = torch.zeros(b, 1, device=device, dtype=torch.float32)
            else:
                frame_timestamps = (
                    torch.linspace(0, 1, n, device=device, dtype=torch.float32)
                    .unsqueeze(0).expand(b, -1)
                )

        # time_encoder: [B, N] -> [B, N, 1, embed_dim] -> [B, N, embed_dim]
        time_emb = self.time_encoder(frame_timestamps).squeeze(-2)

        if cam_token is not None:
            return cam_token + time_emb
        return time_emb

    # ─────────────────────────────────────────────────────────────────────
    # Aggregator dispatch
    # ─────────────────────────────────────────────────────────────────────

    def _call_aggregator(
        self,
        aggregator,
        patch_tokens: List[torch.Tensor],
        queries,
        B: int,
        N: int,
        rgb: Optional[torch.Tensor],
        query_rgb: Optional[torch.Tensor],
        meta_data: dict,
        cam_tokens: Optional[List[torch.Tensor]] = None,
    ) -> Any:
        """Dispatch aggregator call based on its ``AGGREGATOR_TYPE`` attribute.

        ``"motion"``   → ``forward(..., cam_tokens=...)`` when provided.
        ``"standard"`` → ``forward(x, query, query_rgb, meta_data)``
        """
        agg_type = getattr(aggregator, "AGGREGATOR_TYPE", "standard")
        if agg_type == "motion":
            motion_kw: Dict[str, Any] = {}
            if cam_tokens is not None and self._motion_forward_accepts_cam_tokens(aggregator):
                motion_kw["cam_tokens"] = cam_tokens
            return aggregator(
                x=patch_tokens, motion_queries=queries,
                B=B, N=N, meta_data=meta_data, rgb=rgb,
                **motion_kw,
            )
        else:
            return aggregator(
                x=patch_tokens, query=queries,
                query_rgb=query_rgb, meta_data=meta_data,
            )

    @staticmethod
    def _cast_decoder_inputs_fp32(decoder_inputs):
        """Cast only floating-point tensors for fp32 decoder execution."""
        if torch.is_tensor(decoder_inputs):
            return decoder_inputs.float() if decoder_inputs.is_floating_point() else decoder_inputs

        to_floating_dtype = getattr(decoder_inputs, "to_floating_dtype", None)
        if callable(to_floating_dtype):
            return to_floating_dtype(dtype=torch.float32)

        if isinstance(decoder_inputs, dict):
            return {
                k: MVQueryUnified._cast_decoder_inputs_fp32(v)
                for k, v in decoder_inputs.items()
            }
        if isinstance(decoder_inputs, list):
            return [MVQueryUnified._cast_decoder_inputs_fp32(v) for v in decoder_inputs]
        if isinstance(decoder_inputs, tuple):
            return tuple(MVQueryUnified._cast_decoder_inputs_fp32(v) for v in decoder_inputs)
        return decoder_inputs

    @staticmethod
    def _extract_aggregator_aux(decoder_inputs):
        """Split aggregator output into decoder features and auxiliary dict.

        When an aggregator returns a plain tensor, ``aux`` is empty.
        When it returns ``{"features": Tensor, "aux": dict}``, the dict is
        extracted and forwarded to the pipeline for residual corrections
        and/or auxiliary supervision.
        """
        if isinstance(decoder_inputs, dict) and "features" in decoder_inputs:
            return decoder_inputs["features"], decoder_inputs.get("aux", {})
        return decoder_inputs, {}

    @staticmethod
    def _extract_encoder_cam_tokens(
        patch_features: Any,
        B: int,
        N: int,
    ) -> Optional[List[torch.Tensor]]:
        """Collect camera-slot (index 0) embeddings per encoder hook.

        The backbone places the learnable / conditioned camera embedding at
        sequence position 0; patch tokens start at ``patch_start_idx``.  Motion
        aggregators receive patch-only tensors, so camera slots are extracted
        here once and passed alongside.

        Args:
            patch_features: List/tuple of tensors ``[B, N, L, C]`` or
                ``[B * N, L, C]`` from the fuse encoder.
            B, N: Batch size and view count.

        Returns:
            One ``[B, N, C]`` tensor per hook, or ``None`` if unavailable.
        """
        if patch_features is None:
            return None
        feats = patch_features
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        out: List[torch.Tensor] = []
        for f in feats:
            if not torch.is_tensor(f):
                continue
            if f.ndim == 4:
                out.append(f[:, :, 0, :].contiguous())
            elif f.ndim == 3:
                bn, _, c = f.shape
                if bn != B * N:
                    logger.warning(
                        "[MVQueryUnified] cam token extract: expected B*N=%d, got %d — skip hook.",
                        B * N,
                        bn,
                    )
                    continue
                out.append(f.view(B, N, -1, c)[:, :, 0, :].contiguous())
        return out if out else None

    @staticmethod
    def _callable_accepts_kw(fn, name: str) -> bool:
        """True if ``fn`` exposes a parameter called ``name`` or ``**kwargs``."""
        sig = inspect.signature(fn)
        if name in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    @staticmethod
    def _motion_forward_accepts_cam_tokens(aggregator) -> bool:
        """Whether ``aggregator.forward`` can take a ``cam_tokens`` keyword."""
        return MVQueryUnified._callable_accepts_kw(aggregator.forward, "cam_tokens")

    @staticmethod
    def _forward_with_pyramid_accepts_cam_tokens(aggregator) -> bool:
        """Whether ``forward_with_pyramid`` accepts ``cam_tokens``."""
        fn = getattr(aggregator, "forward_with_pyramid", None)
        if fn is None:
            return False
        return MVQueryUnified._callable_accepts_kw(fn, "cam_tokens")

    @staticmethod
    def _call_decoder(decoder, decoder_inputs, meta_data: Optional[dict]):
        """Dispatch decoder call based on its optional ``DECODER_TYPE``."""
        decoder_type = getattr(decoder, "DECODER_TYPE", "standard")
        if decoder_type == "motion_structured":
            return decoder(inputs=decoder_inputs, meta_data=meta_data)
        return decoder(x=decoder_inputs, meta_data=meta_data)

    @staticmethod
    def _build_motion_cache(
        aggregator,
        patch_tokens: List[torch.Tensor],
        B: int,
        N: int,
        meta_data: dict,
        rgb: Optional[torch.Tensor] = None,
        cam_tokens: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """Build cached motion state using an aggregator-specific hook when present."""
        if hasattr(aggregator, "build_motion_cache"):
            bmkw: Dict[str, Any] = dict(
                x=patch_tokens, B=B, N=N, meta_data=meta_data, rgb_images=rgb,
            )
            if cam_tokens is not None and MVQueryUnified._callable_accepts_kw(
                aggregator.build_motion_cache, "cam_tokens",
            ):
                bmkw["cam_tokens"] = cam_tokens
            cache = aggregator.build_motion_cache(**bmkw)
            if isinstance(cache, dict) and cam_tokens is not None and "cam_tokens" not in cache:
                cache = {**cache, "cam_tokens": cam_tokens}
            return cache if isinstance(cache, dict) else {"state": cache}

        patch_h = int(meta_data["input_height"][0]) // aggregator.patch_size
        patch_w = int(meta_data["input_width"][0]) // aggregator.patch_size
        pyramid = aggregator._build_pyramid(patch_tokens, patch_h, patch_w)
        state: Dict[str, Any] = {
            "pyramid": pyramid,
            "num_views": N,
        }
        if cam_tokens is not None:
            state["cam_tokens"] = cam_tokens
        return state

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(  # noqa: C901
        self,
        rgb: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        prompt_depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        ray_directions: Optional[torch.Tensor] = None,
        w2c: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
        ray_world: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        query_rgb: Optional[torch.Tensor] = None,
        time_idx: Optional[torch.Tensor] = None,
        external_queries: Optional[Dict[str, Any]] = None,
        meta_data: Optional[dict] = None,
        return_cached: bool = False,
        **kwargs,
    ) -> dict:
        """Run encoder once, then process every task_group.

        Args:
            external_queries: Dict mapping task_group name → pre-built query
                object/dict.  Used for pipeline-managed query banks
                (e.g. ``{"motion": motion_queries_dict}``).
            return_cached: When ``True``, build and return the spatial feature
                pyramid for every motion aggregator so that inference can call
                :meth:`decode_motion_with_cache` once per target frame without
                re-running the encoder.  Motion task groups are *not* run
                during this call; their outputs are produced later by
                :meth:`decode_motion_with_cache`.
        """
        if self.training and self.freeze_encoders_for_motion:
            self._freeze_encoders()
        elif self.training:
            self.freeze()

        B, N = rgb.shape[:2]

        if time_idx is not None and meta_data is not None:
            meta_data["time_idx"] = time_idx

        # ── Shared encoder (single forward pass) ─────────────────────────
        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth,
            intrinsics=intrinsics, ray_directions=None,
            w2c=w2c, c2w=None, ray_world=None, rgb_mask=None,
            meta_data=meta_data,
        )

        # patch_features may be 3-D [B*N, L, C] or 4-D [B, N, L, C].
        # patch_start_idx marks the first *patch* token along the L dimension.
        # We must slice dim-2 for 4-D tensors and dim-1 for 3-D tensors.
        if isinstance(patch_features, (list, tuple)):
            if patch_features[0].ndim == 4:
                patch_tokens = [f[:, :, patch_start_idx:] for f in patch_features]
            else:
                patch_tokens = [f[:, patch_start_idx:] for f in patch_features]
        else:
            if patch_features.ndim == 4:
                patch_tokens = [patch_features[:, :, patch_start_idx:]]
            else:
                patch_tokens = [patch_features[:, patch_start_idx:]]

        encoder_cam_tokens = self._extract_encoder_cam_tokens(patch_features, B, N)

        # ── Dense head outputs (camera pose, etc.) ────────────────────────
        results: dict = self.decoder(
            b=B, n=N, patch_features=patch_features, pos=pos,
            patch_start_idx=patch_start_idx, prompt_depth=prompt_depth,
            query_points=None, meta_data=meta_data,
        )

        ext = external_queries or {}
        fp32 = self.training and self.decoder_fp32

        # ── Optional: pre-build motion cache for cached inference ──────────
        # When return_cached=True, build and stash aggregator-specific motion
        # cache for all motion task groups. The groups themselves are skipped
        # during this call; per-frame decoding is done via
        # decode_motion_with_cache() without re-running the encoder.
        if return_cached and meta_data is not None:
            motion_caches: Dict[str, Any] = {}
            for tg in self.task_groups:
                agg_name = tg["aggregator"]
                aggregator = self._agg_reg.get(agg_name)
                if aggregator is None:
                    continue
                if getattr(aggregator, "AGGREGATOR_TYPE", "") != "motion":
                    continue
                motion_state = self._build_motion_cache(
                    aggregator=aggregator,
                    patch_tokens=patch_tokens,
                    B=B,
                    N=N,
                    meta_data=meta_data,
                    rgb=rgb,
                    cam_tokens=encoder_cam_tokens,
                )
                motion_caches[tg["name"]] = {
                    "_cached_motion_state": motion_state,
                    "_cached_pyramid": (
                        motion_state.get("pyramid")
                        if isinstance(motion_state, dict) else None
                    ),
                    "_cached_B":            B,
                    "_cached_N":            N,
                    "_cached_agg_name":     agg_name,
                    "_cached_dec_name":     tg["decoder_group"],
                    "_cached_tasks":        list(tg.get("tasks", [])),
                }
            results["_motion_caches"] = motion_caches

        # ── Task groups ───────────────────────────────────────────────────
        for tg in self.task_groups:
            group_name  = tg["name"]
            qb_name     = tg.get("query_bank")
            agg_name    = tg["aggregator"]
            dec_name    = tg["decoder_group"]
            tasks       = tg["tasks"]

            # When return_cached=True, skip motion groups (their pyramids are
            # already built above; per-frame decoding uses decode_motion_with_cache).
            if return_cached:
                aggregator_chk = self._agg_reg.get(tg["aggregator"])
                if aggregator_chk is not None and getattr(aggregator_chk, "AGGREGATOR_TYPE", "") == "motion":
                    continue

            # Resolve queries: external first, then model-managed query_bank.
            if group_name in ext:
                queries = ext[group_name]
            elif qb_name and qb_name in self._qb_reg:
                queries = self._qb_reg[qb_name](
                    rgb, edge_mask=edge_mask, meta_data=meta_data,
                )
            else:
                # Pipeline-managed bank but no external queries provided
                # (e.g. static scene batch with no trajectories) — skip group.
                continue

            aggregator = self._agg_reg.get(agg_name)
            decoder    = self._dec_reg.get(dec_name)
            if aggregator is None or decoder is None:
                logger.warning(
                    "[MVQueryUnified] Missing aggregator '%s' or decoder '%s'"
                    " for task_group '%s' — skipping.",
                    agg_name, dec_name, group_name,
                )
                continue

            # Aggregate features then decode.
            if fp32:
                with torch.autocast(device_type=rgb.device.type, enabled=False):
                    decoder_inputs = self._call_aggregator(
                        aggregator, patch_tokens, queries, B, N, rgb, query_rgb, meta_data,
                        cam_tokens=encoder_cam_tokens,
                    )
                    decoder_inputs, agg_aux = self._extract_aggregator_aux(decoder_inputs)
                    decoder_inputs = self._cast_decoder_inputs_fp32(decoder_inputs)
                    all_out = self._call_decoder(
                        decoder, decoder_inputs, meta_data=meta_data,
                    )
            else:
                decoder_inputs = self._call_aggregator(
                    aggregator, patch_tokens, queries, B, N, rgb, query_rgb, meta_data,
                    cam_tokens=encoder_cam_tokens,
                )
                decoder_inputs, agg_aux = self._extract_aggregator_aux(decoder_inputs)
                all_out = self._call_decoder(
                    decoder, decoder_inputs, meta_data=meta_data,
                )

            # Dense group (standard Query object): reshape [B*N, Q, D] → [B, N, Q, D].
            if hasattr(queries, "batch_uv") or hasattr(queries, "uv"):
                for task in tasks:
                    if task in all_out:
                        v = all_out[task]
                        results[task] = v.view(B, N, *v.shape[-2:])
                # Expose query object for pipeline GT sampling.
                results["query"] = queries

            else:
                # Motion / external group: features already [B, Q, D].
                group_out: dict = {}
                for task in tasks:
                    if task in all_out:
                        results[task] = all_out[task]
                        group_out[task] = all_out[task]
                    iter_key = f"{task}_iters"
                    if iter_key in all_out:
                        group_out[iter_key] = all_out[iter_key]

                # Apply residual corrections from aggregator auxiliary outputs.
                if "residual_flow_2d" in agg_aux and "flow_2d" in group_out:
                    group_out["flow_2d"] = group_out["flow_2d"] + agg_aux["residual_flow_2d"]
                    results["flow_2d"] = group_out["flow_2d"]
                if "residual_displacement" in agg_aux and "displacement" in group_out:
                    group_out["displacement"] = group_out["displacement"] + agg_aux["residual_displacement"]
                    results["displacement"] = group_out["displacement"]
                for k, v in agg_aux.items():
                    if k.startswith("coarse_") or k.startswith("pass_"):
                        group_out[k] = v

                # Alias for SparseMotionPipeline / MVFRQueryMotionPipeline.
                if "displacement" in group_out:
                    group_out["pred_3d"] = group_out["displacement"]
                if "displacement_iters" in group_out:
                    group_out["pred_3d_iters"] = group_out["displacement_iters"]
                results["sparse_motion_pred"] = group_out

        return results

    def decode_motion_with_cache(
        self,
        cached: Dict[str, Any],
        frame_queries: dict,
        rgb: Optional[torch.Tensor] = None,
    ) -> dict:
        """Decode one target frame's motion predictions using a cached pyramid.

        Called repeatedly (once per target frame) after a single
        ``forward(..., return_cached=True)`` call.  The encoder is **not**
        re-run; only the motion aggregator and its paired decoder are executed.

        Args:
            cached: One entry from ``results["_motion_caches"][group_name]``
                as returned by :meth:`forward` with ``return_cached=True``.
            frame_queries: Query dict for a single target frame (output of
                ``MotionQueryBank.motion_single_target``).
            rgb: Optional ``[B, N, 3, H, W]`` image tensor. Motion aggregators
                that use source RGB patches can consume it; older aggregators
                simply ignore this argument.

        Returns:
            Dict with key ``"sparse_motion_pred"`` → task output dict
            (same format as a normal :meth:`forward` call).
        """
        motion_state = cached.get("_cached_motion_state")
        B        = cached["_cached_B"]
        N        = cached["_cached_N"]
        agg_name = cached["_cached_agg_name"]
        dec_name = cached["_cached_dec_name"]
        tasks    = cached["_cached_tasks"]

        aggregator = self._agg_reg[agg_name]
        decoder    = self._dec_reg[dec_name]

        if motion_state is None:
            motion_state = {
                "pyramid": cached["_cached_pyramid"],
                "num_views": N,
            }

        if hasattr(aggregator, "prepare_decoder_inputs"):
            decoder_inputs = aggregator.prepare_decoder_inputs(
                cache=motion_state,
                motion_queries=frame_queries,
                B=B,
                N=N,
                meta_data=None,
                rgb_images=rgb,
            )
        else:
            pyramid = motion_state["pyramid"]
            num_views = int(motion_state.get("num_views", N))
            cam_tok = motion_state.get("cam_tokens") if isinstance(motion_state, dict) else None
            pyr_kw: Dict[str, Any] = {}
            if cam_tok is not None and self._forward_with_pyramid_accepts_cam_tokens(aggregator):
                pyr_kw["cam_tokens"] = cam_tok
            decoder_inputs = aggregator.forward_with_pyramid(
                pyramid, frame_queries, B, num_views, rgb_images=rgb, **pyr_kw,
            )

        decoder_inputs, agg_aux = self._extract_aggregator_aux(decoder_inputs)

        if self.training and self.decoder_fp32:
            decoder_inputs = self._cast_decoder_inputs_fp32(decoder_inputs)

        all_out = self._call_decoder(decoder, decoder_inputs, meta_data=None)

        group_out: dict = {}
        for task in tasks:
            if task in all_out:
                group_out[task] = all_out[task]
            iter_key = f"{task}_iters"
            if iter_key in all_out:
                group_out[iter_key] = all_out[iter_key]

        if "residual_flow_2d" in agg_aux and "flow_2d" in group_out:
            group_out["flow_2d"] = group_out["flow_2d"] + agg_aux["residual_flow_2d"]
        if "residual_displacement" in agg_aux and "displacement" in group_out:
            group_out["displacement"] = group_out["displacement"] + agg_aux["residual_displacement"]

        if "displacement" in group_out:
            group_out["pred_3d"] = group_out["displacement"]
        if "displacement_iters" in group_out:
            group_out["pred_3d_iters"] = group_out["displacement_iters"]

        return {"sparse_motion_pred": group_out}
