"""Motion-only training config.

Uses MVQueryUnified (shared ViT-G encoder) with:
  - Motion group: MotionQueryBank → MotionAggregatorMLP → MLP Head (displacement)

Only the motion task (sparse displacement) is active.
"""

patch_size = 14
max_size   = 518

max_iter      = 50000
eval_view_num = 50

# ── Dataset selection ─────────────────────────────────────────────────────────
select_dataset     = "pointodyssey,kubric4d,dynamicreplica,cotracker3kubric"
select_val_dataset = "pointodyssey,kubric4d,dynamicreplica"
select_vis_dataset = "pointodyssey,kubric4d,dynamicreplica"

data = dict(
    train="hAlgorithm/configs/motion_head/datasets_config/train_motion_head_260301.yaml",
    val="hAlgorithm/configs/motion_head/datasets_config/val_any4d_50frames_v3.yaml",
    vis="hAlgorithm/configs/motion_head/datasets_config/vis_motion_head_260117.yaml",
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

    # ── SDK model ─────────────────────────────────────────────────────────────
    model=dict(
        type="hAlgorithm.modules.models2.sdk.mv_query_unified.MVQueryUnified",
        decoder_fp32=False,

        # Time encoder: sinusoidal embeddings (embed_dim = ViT-G pre-cat dim)
        # added onto the camera token before fuse_encoder so the backbone
        # attends to temporal ordering.  embed_dim=1536 because cat_token=True
        # doubles the hidden dim to 3072 after fuse_encoder.
        time_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.time_encoder.TimeTokenEncoder",
            embed_dim=1536,
            use_mlp_projection=True,
        ),

        # Shared encoder: DinoV2 ViT-G with DA3-style multi-view cross-attention.
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2",
            normalize=True,
            patch_size=patch_size,
            name="vitg",
            out_layers=[19, 27, 33, 39],
            alt_start=13,
            qknorm_start=13,
            rope_start=13,
            cat_token=True,
            # pretrain="results/da3_all/mvfr_da3g_rgb_all8stdy_251120_mvpdc_bs16_2f16f_100k_20251124-114829/checkpoint/latest/fuse_encoder.pth",
            pretrain="/mnt/netdata/Team/AI/SDK/mvfr2.0/stage5/mvfr_da3g_cam_all8stdy_251202_mvpdc_bs16_2f16f_100k_20260108-220222/checkpoint/best/ckpt.pth",
            pertrain_strict=False,
        ),

        # Camera head: predicts pose from encoder tokens.
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

        # ── Query bank registry ───────────────────────────────────────────────
        # "motion_traj": pipeline-managed (MotionQueryBank), instantiated here but
        #                called by the pipeline with GT trajectory data.
        query_banks=[
            dict(
                name="motion_traj",
                type="hAlgorithm.modules.models2.query_bank.motion_query.MotionQueryBank",
                # Pipeline-managed: model exposes this instance via
                # model.pipeline_managed_query_banks["motion_traj"].
                num_queries_per_frame=512,
                src_frame_idx=0,
                min_dynamic_ratio=0.5,
                sampler_dynamic_threshold=0.01,
                use_all_points=False,
                random_src_frame=True,
                deterministic=False,
            ),
        ],

        # ── Aggregator registry ───────────────────────────────────────────────
        # "dual_frame": MotionAggregatorMLP — samples src + tgt frame pair.
        aggregators=[
            dict(
                name="dual_frame",
                type="hAlgorithm.modules.models2.query_aggregator.motion_mlp_aggregator.MotionAggregatorMLP",
                patch_size=patch_size,
                intermediate_layer_idx=[0, 1, 2, 3],
                in_chans=[3072, 3072, 3072, 3072],
                embed_dims=[256, 256, 256, 256],
                upsample_scales=[4, 2, 1, 1],
                mode="bilinear",
                time_fourier_dim=128,
                time_fourier_scale=5.0,
                out_dim=256,
            ),
        ],

        # ── Decoder registry ─────────────────────────────────────────────────
        decoder_groups=[
            dict(
                name="motion_mlp",
                decoder=dict(
                    type="hAlgorithm.modules.models2.head.mlp_head.Head",
                    in_chan=256,
                    hidden_dim=64,
                    names=dict(
                        displacement=3,
                    ),
                    acts=dict(
                        displacement="",
                    ),
                ),
            ),
        ],

        # ── Task groups: wire query_bank → aggregator → decoder ───────────────
        task_groups=[
            dict(
                name="motion",
                query_bank="motion_traj",
                aggregator="dual_frame",
                decoder_group="motion_mlp",
                tasks=["displacement"],
            ),
        ],

        freeze_encoders_for_motion=False,
    ),

    # ── Pipeline field names ───────────────────────────────────────────────────
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

    # ── Motion pipeline settings (MotionQueryBank params moved to query_banks) ─
    motion_extrinsics_name="extrinsics",
    aux_loss_gamma=0.8,
    prediction_type="displacement",

    # ── Task weights ──────────────────────────────────────────────────────────
    motion_task_weight=1.0,

    # ── Dense task losses (disabled) ──────────────────────────────────────────
    local_depth_l1_loss=None,
    local_depth_normal_loss=None,
    local_depth_grad_loss=None,
    global_points_l1_loss=None,
    camera_loss=None,
    task_weight=dict(tk=0.03),

    # ── Motion task loss ──────────────────────────────────────────────────────
    sparse_displacement_loss=dict(
        type="hAlgorithm.modules.losses2.sparse_motion_loss.SparseDisplacementLoss",
        loss_weight=100.0,
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

# ── Trainer ───────────────────────────────────────────────────────────────────
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
        # Motion head: full learning rate.
        "model.motion_aggregator": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.motion_decoder":    dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        # Encoder: lower lr to preserve pre-trained representations.
        "model.fuse_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },

    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            # Sparse motion metrics
            dict(
                type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics.SparseMotionEvalMetrics",
                dynamic_threshold=0.01,
                apd_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
                inlier_threshold=0.1,
                dynamic_only=False,
                metric_prefix="",
            ),
            dict(
                type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics.SparseMotionEvalMetrics",
                dynamic_threshold=0.01,
                apd_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
                inlier_threshold=0.1,
                dynamic_only=True,
                metric_prefix="dyn_",
            ),
        ],
    ),
    main_eval_metric="apd_0.05",
    main_eval_metric_goal="minimize",

    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=2000,
    save_period=2000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    # Load pre-trained weights to initialise the encoder; motion head starts from random.
    load_from="/mnt/netdata/Team/AI/SDK/mvfr2.0/stage5/mvfr_da3g_cam_all8stdy_251202_mvpdc_bs16_2f16f_100k_20260108-220222/checkpoint/best/ckpt.pth",
    seed=2024,
)
