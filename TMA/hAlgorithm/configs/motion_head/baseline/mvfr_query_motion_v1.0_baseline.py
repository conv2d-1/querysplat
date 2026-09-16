"""2-pass refine with unshared MLP params, time token, and camera loss.

Diff vs refine_2pass_shared_params.py:

  1. fuse_encoder → DinoV2WithTimeToken
       Time token is inserted as an independent slot between camera token
       (position 0) and patch tokens, so camera head input is unaffected.

  2. aggregator → MotionAggregatorRGBPatchUVMLPRefineUnshared
       Each of the 2 refinement passes has its own out_proj and flow_head
       instead of sharing parameters across passes.

  3. Per-pass flow_2d GT supervision
       Aggregator returns aux["pass_0_flow_2d"] and aux["pass_1_flow_2d"].
       aux_loss_gamma=0.8 applies decayed flow loss to each intermediate
       prediction:
         pass_0_flow_2d  weight ∝ gamma^2 = 0.64
         pass_1_flow_2d  weight ∝ gamma^1 = 0.80
         final flow_2d   weight ∝ gamma^0 = 1.00

  4. camera_loss → Pi3CameraLoss (L1, multi-view normalised)
"""

patch_size = 14
max_size = 518

max_iter = 50000
eval_view_num = 50

rgb_patch_size = 9
rgb_patch_dim = 256
direct_uv_dim = 128
direct_uv_scale = 10.0
mlp_hidden_dim = 256
mlp_num_hidden_layers = 2

# ── Dataset selection ─────────────────────────────────────────────────────────
select_dataset = "pointodyssey,kubric4d,dynamicreplica,cotracker3kubric"
select_val_dataset = "pointodyssey,kubric4d,dynamicreplica"
select_vis_dataset = "pointodyssey,kubric4d,dynamicreplica"

data = dict(
    train="hAlgorithm/configs/motion_head/datasets_config/train/with_hasim.yaml",
    val="hAlgorithm/configs/motion_head/datasets_config/val/any4d_50frames.yaml",
    vis="hAlgorithm/configs/motion_head/datasets_config/vis/standard.yaml",
    basic=dict(
        sem_name="",
        with_pointmap_raw=True,
        interpolate_version=None,
        interpolate_k=0,
        normalize_cameras=True,
        with_global_scale=True,
        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=None,
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=max_size, patch_size=patch_size,
                is_lidar=False, low_resolution=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5],
            ),
        ],
    ),
    val_basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.sequential_sampler.SequentialClipSampler",
            view_num=eval_view_num, start_idx=0,
        ),
    ),
    vis_basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.sequential_sampler.SequentialClipSampler",
            view_num=eval_view_num, start_idx=0,
        ),
    ),
    train_basic=dict(
        with_inf_mask=False,
        with_rgb_edge_mask=True,
        with_depth_edge_mask=True,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=[2, 8], norm_pdf=True, seed=None,
        ),
        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=None,
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=max_size, patch_size=patch_size,
                is_lidar=False, low_resolution=True, backup=True,
            ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",
                to_gray_prob=0.1, distortion_prob=0.3,
            ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur",
                prob=0.05,
            ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion",
                prob=0.1, compression=[0, 50],
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5],
            ),
        ],
    ),
)

# ── Pipeline + Model ──────────────────────────────────────────────────────────
model = dict(
    type="hAlgorithm.modules.pipelines2.mvfr_query_motion.MVFRQueryMotionPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.sdk.mv_query_unified.MVQueryUnified",
        decoder_fp32=False,
        time_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.time_encoder.TimeTokenEncoder",
            embed_dim=1536,
            use_mlp_projection=True,
        ),
        # ── DinoV2WithTimeToken: time token inserted between camera token
        #    (pos 0) and patch tokens → camera head reads pos 0 cleanly.
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2WithTimeToken",
            normalize=True,
            patch_size=patch_size,
            name="vitg",
            out_layers=[19, 27, 33, 39],
            alt_start=13,
            qknorm_start=13,
            rope_start=13,
            cat_token=True,
            use_checkpoint_start=0,
            use_checkpoint_end=39,
            pretrain="/mnt/netdata/Team/AI/weights/DA3/DA3-GIANT/backbone.pth",
            pertrain_strict=True,
        ),
        camera_head=dict(
            type="hAlgorithm.modules.models2.head.camera_head.CameraHead",
            dim_in=3072,
            trunk_depth=4,
            pose_encoding_type="absT_quaR_FoV",
            num_heads=16,
            mlp_ratio=4,
            init_values=0.01,
            trans_act="linear",
            quat_act="linear",
            fl_act="relu",
        ),
        query_banks=[
            dict(
                name="motion_traj",
                type="hAlgorithm.modules.models2.query_bank.motion_query.MotionQueryBank",
                num_queries_per_frame=4096,
                src_frame_idx=0,
                min_dynamic_ratio=0.5,
                sampler_dynamic_threshold=0.01,
                use_all_points=False,
                random_src_frame=True,
                deterministic=False,
            ),
        ],
        aggregators=[
            dict(
                name="dual_frame",
                # ── Unshared params: each pass has its own out_proj + flow_head.
                type="hAlgorithm.modules.models2.query_aggregator.motion_rgb_patch_uv_mlp_refine_unshared_aggregator.MotionAggregatorRGBPatchUVMLPRefineUnshared",
                patch_size=patch_size,
                intermediate_layer_idx=[0, 1, 2, 3],
                in_chans=[3072, 3072, 3072, 3072],
                embed_dims=[256, 256, 256, 256],
                upsample_scales=[4, 2, 1, 1],
                mode="bilinear",
                time_fourier_dim=128,
                time_fourier_scale=5.0,
                rgb_patch_size=rgb_patch_size,
                rgb_patch_dim=rgb_patch_dim,
                direct_uv_dim=direct_uv_dim,
                direct_uv_scale=direct_uv_scale,
                out_dim=256,
                # 2 refine passes → 3 out_proj instances, 2 flow_head instances.
                # Intermediate flows pass_0_flow_2d / pass_1_flow_2d go to aux.
                num_refine_passes=2,
            ),
        ],
        decoder_groups=[
            dict(
                name="motion_mlp",
                decoder=dict(
                    type="hAlgorithm.modules.models2.head.mlp_head.Head",
                    in_chan=256,
                    hidden_dim=mlp_hidden_dim,
                    num_hidden_layers=mlp_num_hidden_layers,
                    names=dict(displacement=3, flow_2d=2),
                    acts=dict(displacement="", flow_2d=""),
                ),
            ),
        ],
        task_groups=[
            dict(
                name="motion",
                query_bank="motion_traj",
                aggregator="dual_frame",
                decoder_group="motion_mlp",
                tasks=["displacement", "flow_2d"],
            ),
        ],
        freeze_encoders_for_motion=False,
    ),
    training_sub_pixel_scale=1,
    testing_sub_pixel_scale=1,
    scale_align=False,
    edge_mask_name="edge_mask",
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics_reff",
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name=None,
    prompt_depth_mask_name=None,
    target_local_depth_name="pointmap_raw",
    target_global_points_name="pointmap_raw_reff",
    target_depth_mask_name="depth_raw_mask",
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name="depth_raw",
    pose_encoding_type="absT_quaR_FoV",
    motion_extrinsics_name="extrinsics",
    # gamma=0.8 → per-pass flow weights: pass_0=0.64, pass_1=0.80, final=1.00
    aux_loss_gamma=0.8,
    prediction_type="displacement",
    motion_task_weight=1.0,
    flow_2d_task_weight=0.1,
    local_depth_l1_loss=None,
    local_depth_normal_loss=None,
    local_depth_grad_loss=None,
    global_points_l1_loss=None,
    camera_loss=dict(
        type="hAlgorithm.modules.losses2.camera_loss.Pi3CameraLoss",
        loss_type="l1",
        gamma=0.6,
        pose_encoding_type="absT_quaR_FoV",
        weight_T=1.0,
        weight_R=10.0,
        weight_fl=0.5,
        loss_weight=1.0,
    ),
    task_weight=dict(tk=0.03),
    sparse_displacement_loss=dict(
        type="hAlgorithm.modules.losses2.sparse_motion_loss.SparseDisplacementLoss",
        loss_weight=100.0,
        dynamic_threshold=0.02,
        balance_weight=0.85,
        detach_log_scale=False,
    ),
    flow_2d_loss=dict(
        type="hAlgorithm.modules.losses2.sparse_motion_loss.SparseDisplacementLoss",
        loss_weight=1.0,
        dynamic_threshold=0.02,
        balance_weight=0.85,
        detach_log_scale=False,
    ),
    save_output_cfg=dict(
        save_everything=False,
        save_gaussians=False,
        save_render_results=False,
        save_glb_results=True,
        save_glb_sf_results=False,
        save_glb2local_results=False,
        save_local2glb_results=True,
        save_cameras=False,
        save_local_results=True,
        save_track_results=False,
        save_filtered_results=False,
        output_normalize_cameras=True,
        output_match_input_res=True,
        output_conf_ratio=0.2,
    ),
)

trainer = dict(
    type="hAlgorithm.trainers.moge_trainer.MogeIterTrainer",
    skip_error_step=True,
    select_dataset=select_dataset,
    select_val_dataset=select_val_dataset,
    select_vis_dataset=select_vis_dataset,
    sampler="MixedMaxIterBatchSampler",
    max_epoch=None,
    max_iter=max_iter,
    num_workers=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=1,
    lr=None,
    lr_scheduler=dict(
        type="hAlgorithm.modules.lr_schedulers.cosine_lr_updater.CosineLrUpdater",
        base_lr=1e-4,
        max_iters=max_iter,
        warmup_iters=1000,
        warmup="linear",
        warmup_ratio=1e-6,
        min_lr=1e-8,
    ),
    optimizer={
        "type": "AdamW",
        "model.motion_aggregator": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.motion_decoder": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.fuse_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics.SparseMotionEvalMetrics",
                dynamic_threshold=0.01,
                apd_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
                inlier_threshold=0.1,
                dynamic_only=False,
                metric_prefix="",
            ),
            dict(
                type="hAlgorithm.modules.metrics.sparse_flow_2d_eval_metrics.SparseFlow2DEvalMetrics",
                dynamic_threshold=0.01,
                apd_thresholds=[0.005, 0.01, 0.02, 0.05, 0.1],
                inlier_threshold=0.02,
                dynamic_only=False,
                metric_prefix="flow_",
            ),
        ],
    ),
    main_eval_metric="epe",
    main_eval_metric_goal="minimize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=2000,
    save_period=2000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",
    load_from="/mnt/netdata/Team/AI/SDK/mvfr2.0/stage5/mvfr_da3g_cam_all8stdy_251202_mvpdc_bs16_2f16f_100k_20260108-220222/checkpoint/best/ckpt.pth",
    seed=2024,
)
