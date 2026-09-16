"""
VDPM Configuration for Video Depth Prediction with Multi-Frame Reconstruction.

This config integrates the VDPM model for:
- Multi-view depth estimation
- Camera pose estimation
- Scene flow prediction (computed from temporal pointmaps)
"""

patch_size = 14
max_size = 518

max_iter = 50000

view_num = 4
eval_view_num = 50
novel_view_nums = 0

select_dataset = ""
select_dataset += "pointodyssey,kubric4d,"

select_val_dataset = "pointodyssey,kubric4d,dynamicreplica,"
select_vis_dataset = "pointodyssey,kubric4d,dynamicreplica,"

data = dict(
    train="hAlgorithm/configs/motion_head/datasets_config/train_motion_head_260116.yaml",
    val="hAlgorithm/configs/motion_head/datasets_config/val_any4d_50frames_v3.yaml",  # v3: evenly spaced 50 frames
    vis="hAlgorithm/configs/motion_head/datasets_config/vis_motion_head_260117.yaml",  # Keep original for vis
    basic=dict(

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
                max_size=max_size,
                patch_size=patch_size,
                is_lidar=False,
                low_resolution=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
    ),
    # val_basic: validation-specific settings (50 consecutive frames for evaluation)
    val_basic=dict(
        clip_sampler=dict(
            # SequentialClipSampler: consecutive frames [0, 1, 2, ..., 49]
            type="hAlgorithm.datasets_mv.clip_sampler.sequential_sampler.SequentialClipSampler",
            view_num=eval_view_num,  # 50 consecutive frames
            start_idx=0,  # Start from frame 0
        ),
    ),
    train_basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
            view_num=view_num,
            seed=0
        ),

        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=None,

        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=max_size,
                patch_size=patch_size,
                is_lidar=False,
                low_resolution=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines2.vdpm.VDPMPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.sdk.vdpm.VDPM",
        # VDPM checkpoint path
        pretrained_checkpoint_path="/mnt/naspersonal/ctc/tmp/checkpoints/vdpm/model.pt",
        img_size=max_size,
        patch_size=patch_size,
        embed_dim=1024,
        decoder_depth=4,
    ),

    # Input field names
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics_reff",
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name=None,
    target_local_depth_name="pointmap",
    target_global_points_name="pointmap_reff",
    target_depth_mask_name="depth_mask",
    align_name="depth_raw",

    # Output saving configuration
    save_output_cfg=dict(
        save_everything=False,
        save_gaussians=False,
        save_render_results=False,
        save_glb_results=True,
        save_glb_sf_results=True,  # Save scene flow
        save_glb2local_results=False,
        save_local2glb_results=True,
        save_cameras=False,
        save_local_results=False,
        save_track_results=True,
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
    max_grad_norm=10,
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
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                # Use VDPMSceneFlowEvalMetrics for non-metric depth predictions with scale alignment
                type="hAlgorithm.modules.metrics.track_3d_eval_metrics.VDPMSceneFlowEvalMetrics",
                dynamic_threshold=0.01,  # Points with motion > 1cm are dynamic
                apd_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],  # Thresholds for APD
                inlier_threshold=0.1,  # 0.1m for τ per Any4D paper
                ref_frame_idx=0,  # Source frame is frame 0
                dynamic_only=True,  # Only evaluate dynamic points
                use_visibility=True,
                use_validity=True,
                scale_alignment_method="least_squares",  # Compute optimal scale for non-metric VDPM
            ),
            # Local depth metrics
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                target_name="depth_raw",
                valid_mask_name="depth_raw_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=["abs_relative_difference"],
            ),
            dict(
                type="hAlgorithm.modules.metrics.pointmap_eval_metrics.PointMapEvalMetrics",
                valid_mask_name="depth_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=["pointmap_normal_cos"],
            ),
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                target_name="depth_raw",
                valid_mask_name="depth_raw_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=["abs_relative_difference"],
                conf_ratio=0.1,
            ),
            # Global depth metrics
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
                glb2local=False,
                local2glb=False,
                target_name='pointmap',
                mv_target_name='pointmap_reff',
                valid_mask_name="depth_mask",
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                metrics=["abs_relative_difference"],
                metric_add_distance=True,
            ),
            dict(
                type="hAlgorithm.modules.metrics.pointmap_eval_metrics.GlobalPointMapEvalMetrics",
                valid_mask_name="depth_mask",
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                metrics=["pointmap_normal_cos"],
            ),
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
                glb2local=False,
                local2glb=False,
                target_name='pointmap',
                mv_target_name='pointmap_reff',
                valid_mask_name="depth_mask",
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                metrics=["abs_relative_difference"],
                metric_add_distance=True,
                conf_ratio=0.1,
            ),
            # Pose metrics
            dict(type="hAlgorithm.modules.metrics.pose_eval_metrics.PoseEvalMetricsV2"),
            # Reconstruction metrics
            dict(
                type='hAlgorithm.modules.metrics.reconstruct_eval_metrics.ReconstructEvalMetricsWithNovelView',
                target_name='image',
                metrics=['rgb_psnr', 'rgb_ssim']
            ),
        ],
    ),
    main_eval_metric="auc_30",
    main_eval_metric_goal="maximize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=1000,
    save_period=10000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",
)
