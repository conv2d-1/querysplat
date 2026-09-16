# Ablation: DA3 backbone + time token injected into backbone (no AdaLN in motion head)
#
# Experiment matrix:
#   baseline.py             | backbone time: NO  | AdaLN motion head: NO  (MotionHeadV2)
#   da3_with_adaln_only.py  | backbone time: NO  | AdaLN motion head: YES (MotionHead)
#   da3_with_backbone_time_only.py | backbone time: YES | AdaLN: NO        <-- this
#   da3_with_time_encoder.py       | backbone time: YES | AdaLN: YES
#
# Key difference from baseline.py:
#   - model type changed from MVBaseMotion -> MVBaseMotionWithTime
#   - time_encoder added: sinusoidal embeddings (embed_dim=1536) added to camera token
#     before the DA3 fuse_encoder, so the backbone sees temporal ordering
#   - motion_head unchanged: still MotionHeadV2 (no AdaLN)

patch_size = 14
max_size = 518

max_iter = 50000
train_view_num = 5
eval_view_num = 50

select_dataset = ""
select_dataset += "pointodyssey,kubric4d,dynamicreplica,cotracker3kubric"

select_val_dataset = "pointodyssey,kubric4d,dynamicreplica,"
select_vis_dataset = "pointodyssey,kubric4d,dynamicreplica,"

data = dict(
    train="hAlgorithm/configs/motion_head/datasets_config/train_motion_head_260130.yaml",
    val="hAlgorithm/configs/motion_head/datasets_config/val_any4d_50frames_v3.yaml",
    vis="hAlgorithm/configs/motion_head/datasets_config/vis_motion_head_260117.yaml",

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
    val_basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.sequential_sampler.SequentialClipSampler",
            view_num=eval_view_num,
            start_idx=0,
        ),
    ),
    train_basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=[2, 8],
            norm_pdf=True,
            seed=None,
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
    type='hAlgorithm.modules.pipelines2.mvfr_motion.MVFRMotionPipeline',
    model=dict(
        # KEY DIFFERENCE: MVBaseMotionWithTime adds time token to backbone
        type='hAlgorithm.modules.models2.sdk.base_motion.MVBaseMotionWithTime',
        camera_encoder=dict(
            type='hAlgorithm.modules.models2.encoder.camera_encoder.CameraEnc',
            dim_out=1536,
            c2w=False
        ),
        # KEY DIFFERENCE: time_encoder produces sinusoidal embeddings (embed_dim must
        # match camera_encoder dim_out=1536) that are summed onto the camera token
        time_encoder=dict(
            type='hAlgorithm.modules.models2.encoder.time_encoder.TimeTokenEncoder',
            embed_dim=1536,
            use_mlp_projection=True,
        ),
        fuse_encoder=dict(
            type='hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2',
            normalize=True,
            patch_size=14,
            name='vitg',
            out_layers=[19, 27, 33, 39],
            alt_start=13,
            qknorm_start=13,
            rope_start=13,
            cat_token=True,
            use_checkpoint_start=0,
            use_checkpoint_end=39,
            pretrain='/mnt/netdata/Team/AI/weights/DA3/DA3-GIANT/backbone.pth',
            pertrain_strict=True
        ),
        depth_head=dict(
            type='hAlgorithm.modules.models2.head.dpt_head.DPTHead',
            in_channels=[3072, 3072, 3072, 3072],
            mid_channels=[256, 256, 256, 256],
            patch_size=14,
            features=256,
            features2=32,
            depth_channel=1,
            depth_act='exp',
            interp_refinenet_cfg=dict(
                type='hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet',
                layer_num=3,
                use_bn=False
            ),
            prompt_flag=[False, False, False, False],
            pred_confidence=True,
            pred_normal=False,
            pred_motion_mask=False,
            pred_invalid_mask=False,
            pred_ray=False,
            return_features=False,
            chunk_size=0
        ),
        glb_points_head=dict(
            type='hAlgorithm.modules.models2.head.vggt_dpt_head.VGGTDPTHead',
            patch_size=14,
            dim_in=3072,
            output_dim=4,
            activation='inv_log',
            conf_activation=None,
            intermediate_layer_idx=[0, 1, 2, 3]
        ),
        camera_head=dict(
            type='hAlgorithm.modules.models2.head.camera_head.CameraHead',
            dim_in=3072,
            trunk_depth=4,
            pose_encoding_type='absT_quaR_FoV',
            num_heads=16,
            mlp_ratio=4,
            init_values=0.01,
            trans_act='linear',
            quat_act='linear',
            fl_act='relu'
        ),
        # motion_head unchanged: MotionHeadV2 (no AdaLN)
        motion_head=dict(
            type="hAlgorithm.modules.models2.head.motion_head_v2.MotionHeadV2",
            dim_in=2 * 1536,
            patch_size=14,
            intermediate_layer_idx=[0, 1, 2, 3],
            features=256,
            out_channels=[256, 512, 1024, 1024],
            output_dim=3,
            activation="identity",
            conf_activation="expp1",
            down_ratio=1,
            pos_embed=True,
            pretrain=None,
        ),
    ),
    intrinsics_name='intrinsics',
    extrinsics_name='extrinsics_reff',
    scale_name='sparse_pointmap_max_range',
    prompt_depth_name=None,
    target_local_depth_name='pointmap',
    target_global_points_name='pointmap_reff',
    target_depth_mask_name='depth_mask',
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name='depth_raw',
    local_depth_l1_loss=dict(
        type='hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss',
        loss_weight=1.0,
        with_conf=True,
        conf_loss_scale=0.1,
        valid_range=0.99,
    ),
    motion_any4d_loss=dict(
        type="hAlgorithm.modules.losses2.any4d_loss.Any4DSceneFlowLoss",
        loss_weight=100.0,
    ),
    pose_encoding_type='absT_quaR_FoV',
    save_output_cfg=dict(
        save_everything=False,
        save_gaussians=False,
        save_render_results=False,
        save_glb_results=True,
        save_glb_sf_results=False,
        save_glb2local_results=False,
        save_local2glb_results=True,
        save_cameras=False,
        save_local_results=False,
        save_track_results=False,
        save_filtered_results=False,
        output_normalize_cameras=True,
        output_match_input_res=True,
        output_conf_ratio=0.2,
        save_motion_results=True,
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
        "model.ray_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.prompt_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.rgb_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.fuse_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.track_head": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.gaussian_head": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                type="hAlgorithm.modules.metrics.track_3d_eval_metrics.Any4DSceneFlowEvalMetrics",
                dynamic_threshold=0.01,
                apd_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
                inlier_threshold=0.1,
                ref_frame_idx=0,
                dynamic_only=True,
                use_visibility=True,
                use_validity=True,
            ),
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                target_name="depth_raw",
                valid_mask_name="depth_raw_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=["abs_relative_difference"],
            ),
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
                glb2local=False,
                local2glb=True,
                target_name='pointmap',
                mv_target_name='pointmap_reff',
                valid_mask_name="depth_mask",
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                metrics=["abs_relative_difference"],
                metric_add_distance=True,
            ),
            dict(type="hAlgorithm.modules.metrics.pose_eval_metrics.PoseEvalMetricsV2"),
        ],
    ),
    main_eval_metric="epe",
    main_eval_metric_goal="minimize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=2000,
    save_period=1000,
    vis_period=2000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    load_from="/mnt/netdata/Team/AI/SDK/mvfr2.0/stage5/mvfr_da3g_cam_all8stdy_251202_mvpdc_bs16_2f16f_100k_20260108-220222/checkpoint/best/ckpt.pth",
)
