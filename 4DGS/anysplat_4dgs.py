"""4DGS with AnySplat-style direct Gaussian-attribute prediction.

Encoder 504, ``gaussian_query_scale=2`` → DGS render ~1008.
Init from the Q4RT MVQuery6 baseline (no Gaussian head) — see below.

Gaussian-head changes:
  * Reference Gaussian means are reconstructed by unprojecting identity-pair
    ``pair_depth`` with the reference camera intrinsics/extrinsics
    (``geometry_source='camera_depth'``); no xyz residual is predicted.
  * ``warp3d_delta`` supplies per-frame Gaussian displacements after the
    reference means are reconstructed.
  * Opacity, scale, rotation and SH are predicted directly from query features.
  * The additive RGB feature branch is retained without zero initialization.
  * RGB-to-SH anchoring and inference-time RGB SH bootstrapping are disabled.

Rendering settings retained from the opacity / coverage baseline:
  * ``opacity_reg_weight=0``: remove binary-entropy reg that peaked at opacity=0.5.
  * ``render_random_background=False``: train with fixed black bg, match infer compositing.
  * ``alpha_bce_weight=1.0`` (was 0.5): stronger push on render alpha → 1.
  * ``num_render_frames=4``, ``ref_frame_loss_weight=3.0``: more / heavier render supervision.

Init checkpoint (``trainer.load_from``): ``results/Q4RT1.12/latest/ckpt.pth`` (Q4RT
MVQuery6 baseline). Warm-starts ``fuse_encoder`` / ``camera_head`` /
``query_pair_feats_aggregator`` / ``query_pair_decoder{,2}``; ``query_banck`` and
``sparse_gaussian_head`` have no counterpart in that checkpoint and train from
scratch (``load_state_dict`` uses ``strict=False``).
Finetune ``max_iter=50000``, slightly lower Gaussian-head LR.
"""

patch_size = 14

max_size = 504

view_num = 4

novel_view_nums = 0

view_num_range = [
    4,
    4,
]

select_dataset = 'pointodyssey,kubric4d,dynamicreplica,cotracker3kubric,hasim,hasim_benchmark_running,hasim_character_medium,hasim_character_follow,hasim_character,hasim_character_easy,pstudio,hypersim,dl3dv,'

select_val_dataset = 'pointodyssey,kubric4d,hasim,hasim_benchmark_running'

select_vis_dataset = 'pointodyssey,kubric4d,hasim,hasim_benchmark_running'

sparse_dataset_names = [
    'dl3dv',
]

static_dataset_names = [
    'hypersim',
]

dynamic_dataset_names = [
    'pointodyssey',
    'kubric4d',
    'dynamicreplica',
    'cotracker3kubric',
    'hasim',
    'hasim_benchmark_running',
    'hasim_character_medium',
    'hasim_character_follow',
    'hasim_character',
    'hasim_character_easy',
    'waymo',
    'vkitti2',
    'stereo4d',
    'stereo4d_hesai',
]

max_iter = 50000

data = dict(
    train='hAlgorithm/configs/unified_query/dataset_configs/4d/with_hasim_0526_wild3_stereo4d_po_prob1.yaml',
    val='hAlgorithm/configs/unified_query/dataset_configs/4d/with_hasim_val_0526_wild3_stereo4d.yaml',
    vis='hAlgorithm/configs/unified_query/dataset_configs/4d/with_hasim_val_0526_wild3_stereo4d.yaml',
    basic=dict(
        sem_name='',
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
                type='hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio',
                max_size=504,
                patch_size=14,
                is_lidar=False,
                low_resolution=True,
                backup=True,
            ),
            dict(
                type='hAlgorithm.datasets.transforms.transforms.ToTensor',
            ),
            dict(
                type='hAlgorithm.datasets.transforms.transforms.Normalize',
                mean=[
                    127.5,
                    127.5,
                    127.5,
                ],
                std=[
                    127.5,
                    127.5,
                    127.5,
                ],
            ),
        ],
    ),
    train_basic=dict(
        with_inf_mask=False,
        with_rgb_edge_mask=True,
        with_depth_edge_mask=True,
        clip_maxlen=24,
        clip_sampler=dict(
            type='hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler',
            view_num=[
                4,
                4,
            ],
            seed=None,
            shuffle=False,
        ),
        novel_view_nums=0,
        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=None,
        train_transforms=[
            dict(
                type='hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio',
                max_size=504,
                patch_size=14,
                is_lidar=False,
                low_resolution=True,
                backup=True,
            ),
            dict(
                type='hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion',
                to_gray_prob=0.1,
                distortion_prob=0.3,
            ),
            dict(
                type='hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur',
                prob=0.05,
            ),
            dict(
                type='hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion',
                prob=0.1,
                compression=[
                    0,
                    50,
                ],
            ),
            dict(
                type='hAlgorithm.datasets.transforms.transforms.ToTensor',
            ),
            dict(
                type='hAlgorithm.datasets.transforms.transforms.Normalize',
                mean=[
                    127.5,
                    127.5,
                    127.5,
                ],
                std=[
                    127.5,
                    127.5,
                    127.5,
                ],
            ),
        ],
    ),
    val_basic=dict(
        with_rgb_edge_mask=True,
        clip_sampler=dict(
            type='hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory',
            view_num=4,
            seed=0,
        ),
    ),
    vis_basic=dict(
        clip_sampler=dict(
            type='hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory',
            view_num=4,
            seed=0,
        ),
    ),
)

model = dict(
    type='hAlgorithm.modules.pipelines2.mvfr_query_dual_4dgs.WFMQueryDual4DGSPipeline',
    enable_dense_gaussian_query=True,
    dense_gaussian_dynamic_only=True,
    static_dataset_names=[
        'hypersim',
    ],
    dynamic_dataset_names=[
        'pointodyssey',
        'kubric4d',
        'dynamicreplica',
        'cotracker3kubric',
        'hasim',
        'hasim_benchmark_running',
        'hasim_character_medium',
        'hasim_character_follow',
        'hasim_character',
        'hasim_character_easy',
        'waymo',
        'vkitti2',
        'stereo4d',
        'stereo4d_hesai',
    ],
    sparse_dataset_names=[
        'dl3dv',
    ],
    training_sub_pixel_scale=1,
    testing_sub_pixel_scale=1,
    scale_align=False,
    pair_mode=10,
    global_pair_mode=1,
    align_corners=True,
    model=dict(
        type='hAlgorithm.modules.models2.sdk.query_dual_4dgs.MVQueryDual4DGS',
        enable_dual_gaussian_query=True,
        gaussian_query_scale=2.0,
        gaussian_query_patch_size=14,
        normalize_predicted_cameras_for_gaussians=True,
        freeze_modules=[
            'fuse_encoder',
        ],
        chunk_size=30000,
        decoder_fp32=False,
        fuse_encoder=dict(
            type='hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2',
            normalize=True,
            patch_size=14,
            name='vitg',
            out_layers=[
                19,
                27,
                33,
                39,
            ],
            alt_start=13,
            qknorm_start=13,
            rope_start=13,
            cat_token=True,
            use_checkpoint_start=0,
            use_checkpoint_end=39,
            pretrain='/mnt/cfsdata/Team/AI/Autoresearch/MVS/da3_all/mvfr_da3g_rgb_all8stdy_251120_mvpdc_bs16_2f16f_100k_20251124-114829/checkpoint/latest/fuse_encoder.pth',
            pertrain_strict=False,
        ),
        query_banck=dict(
            type='hAlgorithm.modules.models2.query_bank.single_view.QueryBank5',
            full_ratio=0.0,
            train_sampler=dict(
                type='hAlgorithm.modules.models2.query_bank.sampler.RandomSampler',
                num_samples=30000,
            ),
            timing=False,
            with_edge_mask=True,
            offset=0,
            noise=0.5,
            noise_ratio=0.0,
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
            fl_act='relu',
        ),
        query_pair_feats_aggregator=dict(
            type='hAlgorithm.modules.models2.query_aggregator.unified_query.PairCrossAttnAggregator',
            patch_size=14,
            intermediate_layer_idx=[
                0,
                1,
                2,
                3,
            ],
            in_chans=[
                3072,
                3072,
                3072,
                3072,
            ],
            embed_dims=[
                256,
                256,
                256,
                256,
            ],
            upsample_scales=[
                4,
                2,
                1,
                1,
            ],
            mode='bilinear',
            interp_refinenet_cfg=dict(
                type='hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet',
                layer_num=1,
                use_bn=False,
            ),
            cross_attn_dim=256,
            cross_attn_heads=8,
            cross_attn_layers=4,
            cross_ff_ratio=4,
            cross_dropout=0.0,
            memory_from_encoder_last=False,
            without_src_hidden=True,
            without_tgt_hidden=True,
            align_corners=True,
            return_src_hidden=True,
        ),
        query_pair_decoder=dict(
            type='hAlgorithm.modules.models2.head.mlp_head.Head',
            in_chan=256,
            hidden_dim=64,
            names=dict(
                warp3d=3,
                warp3d_confidence=1,
                warp3d_delta_direction=3,
                warp3d_delta_magnitude=1,
                warp3d_delta_confidence=1,
                warp2d=2,
                warp2d_confidence=4,
            ),
            acts=dict(
                warp3d='inv_log',
                warp3d_confidence='',
                warp3d_delta_direction='norm',
                warp3d_delta_magnitude='softplus',
                warp3d_delta_confidence='',
                warp2d='',
                warp2d_confidence='',
            ),
        ),
        query_pair_decoder2=dict(
            type='hAlgorithm.modules.models2.head.mlp_head.Head',
            in_chan=256,
            hidden_dim=64,
            names=dict(
                pair_depth=1,
                pair_confidence=1,
            ),
            acts=dict(
                pair_depth='exp',
                pair_confidence='',
            ),
        ),
        sparse_gaussian_head=dict(
            type='hAlgorithm.modules.models2.head.sparse_pair_gaussian_head.SparsePairDynamicGaussianHead',
            in_dim=256,
            hidden_dim=64,
            sh_degree=2,
            motion_feat_dim=32,
            enhanced_motion=True,
            random_src_frame=True,
            predict_attribute_delta=False,
            use_rgb_color_anchor=False,
            use_rgb_feature_residual=True,
            rgb_feature_dim=128,
            direct_prediction=True,
            direct_scale_factor=0.001,
            direct_scale_max=0.3,
            use_sharp_zero_init=False,
            geometry_source='camera_depth',
        ),
    ),
    intrinsics_name='intrinsics',
    extrinsics_name='extrinsics_reff',
    scale_name='sparse_pointmap_max_range',
    prompt_depth_name=None,
    prompt_depth_mask_name=None,
    target_local_depth_name='pointmap_raw',
    target_global_points_name='pointmap_raw_reff',
    target_depth_mask_name='depth_raw_mask',
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name='depth_raw',
    edge_mask_name='edge_mask',
    motion_extrinsics_name='extrinsics',
    local_depth_l1_loss=dict(
        type='hAlgorithm.modules.losses2.points_l1_loss.ZWeightedLoss',
        with_conf=True,
        valid_range=0.99,
        loss_weight=dict(
            default=1.0,
            dl3dv=0.0,
            waymo=0.0,
            vkitti2=0.0,
            stereo4d=0.0,
            pstudio=0.0,
        ),
    ),
    local_depth_grad_loss=None,
    local_depth_normal_loss=None,
    local2global_loss=None,
    local_normal_loss=None,
    global_points_l1_loss=None,
    global_points_grad_loss=None,
    global_points_normal_loss=None,
    camera_loss=dict(
        type='hAlgorithm.modules.losses2.camera_loss.Pi3CameraLoss',
        loss_type='l1',
        gamma=0.6,
        pose_encoding_type='absT_quaR_FoV',
        weight_T=1.0,
        weight_R=10.0,
        weight_fl=0.5,
        frame_num=-100,
        loss_weight=1.0,
    ),
    rc_rgb_l1_loss=None,
    rc_ssim_loss=None,
    rc_lpips_loss=None,
    rc_depth_loss=None,
    rc_normal_loss=None,
    motion_3d_train_novis=True,
    motion_2d_train_novis=True,
    motion_novis_scale=0.2,
    query_motion2d_loss=dict(
        type='hAlgorithm.modules.losses2.query_match_loss.QueryMatchLoss2',
        alpha=0.5,
        scale_c=0.0001,
        conf_weight=0.01,
        precision_weight=0.01,
        precision_thresh=1.0,
        loss_weight=dict(
            default=5.0,
            pointodyssey=0.0,
            waymo=5.0,
            vkitti2=5.0,
            stereo4d=5.0,
        ),
        with_weight=True,
    ),
    query_motion3d_loss=dict(
        type='hAlgorithm.modules.losses2.dynamic_pointmap_loss.GlobalPointZWeightedLoss',
        with_conf=True,
        with_weight=True,
        loss_weight=dict(
            default=10.0,
            dl3dv=0.1,
            waymo=0.0,
            vkitti2=0.0,
            stereo4d=0.1,
        ),
    ),
    query_motion3d_delta_loss=dict(
        type='hAlgorithm.modules.losses2.sparse_motion_loss.SparseDisplacementQueryMotion3dDeltaLoss',
        loss_weight=dict(
            default=50.0,
            waymo=30.0,
            vkitti2=30.0,
            stereo4d=30.0,
        ),
        dynamic_threshold=0.02,
        balance_weight=0.85,
        detach_log_scale=False,
        batch_reduction='per_example_mean',
        pool_static_dynamic=False,
    ),
    warp3d_delta_direction_loss=dict(
        type='hAlgorithm.modules.losses2.sparse_motion_loss.SparseDirectionLoss',
        loss_weight=dict(
            default=1.0,
            waymo=1.0,
            vkitti2=1.0,
            stereo4d=1.0,
        ),
        mag_threshold=0.02,
        static_mag_weight=1.0,
    ),
    sparse_dynamic_gaussian_render_loss=dict(
        type='hAlgorithm.modules.losses2.sparse_pair_gaussian_loss.SparsePairDynamicGaussianRenderLoss',
        l1_weight=1.0,
        ssim_weight=0.2,
        num_render_frames=4,
        max_render_points=0,
        background_color=[
            0.0,
            0.0,
            0.0,
        ],
        render_in_normalized_space=True,
        scale_reg_weight=0.0,
        opacity_reg_weight=0.0,
        ffgs_alignment=True,
        supervise_covered_pixels_only=False,
        supervise_ref_full_image=True,
        use_ref_reproject_visibility=True,
        ref_frame_loss_weight=3.0,
        ref_visibility_depth_ratio_low=0.9,
        ref_visibility_depth_ratio_high=1.1,
        alpha_bce_weight=1.0,
        grad_loss_weight=0.5,
        lpips_weight=0.2,
        render_random_background=False,
        bootstrap_infer_sh_from_rgb=False,
    ),
    gs_xyz_offset_loss=None,
    task_weight=dict(
        lcl=1.0,
        glb=1.0,
        l2g=1.0,
        cm=10.0,
        tk=0.03,
        rc=1.0,
        match=1.0,
        motion=1.0,
        pair_glb=1.0,
        pair_lcl=1.0,
        dgs=2.5,
        gsreg=0.0,
    ),
    pose_encoding_type='absT_quaR_FoV',
    save_output_cfg=dict(
        save_everything=False,
        save_gaussians=False,
        save_render_results=False,
        save_glb_results=False,
        save_glb_sf_results=False,
        save_glb2local_results=False,
        save_local2glb_results=False,
        save_cameras=False,
        save_local_results=False,
        save_track_results=False,
        save_filtered_results=False,
        output_normalize_cameras=True,
        output_match_input_res=True,
        output_conf_ratio=0.2,
        save_match=False,
        save_motion=True,
        save_4dgs_results=True,
        rerun_vis_cfg=dict(
            vis_3d_view_coordinates='RDF',
            pred_points_step=1,
        ),
    ),
    motion_gs_displacement_consistency_loss=dict(
        type='hAlgorithm.modules.losses2.sparse_motion_gs_consistency_loss.SparseMotionGsDisplacementConsistencyLoss',
        loss_weight=dict(
            default=0.25,
        ),
        dynamic_threshold=0.02,
        huber_delta=0.05,
    ),
    dgs_render_normalized_only=True,
    dgs_scale_dropout_prob=0.0,
)

_LOAD_FROM = (
    "/mnt/cfsdata/Team/AI/personal/chentiancheng/workspace/Projects/TMA/"
    "results/Q4RT1.12/latest/ckpt.pth"
)

trainer = dict(
    type='hAlgorithm.trainers.moge_trainer.MogeIterTrainer',
    skip_error_step=True,
    select_dataset='pointodyssey,kubric4d,dynamicreplica,cotracker3kubric,hasim,hasim_benchmark_running,hasim_character_medium,hasim_character_follow,hasim_character,hasim_character_easy,pstudio,hypersim,dl3dv,',
    select_val_dataset='pointodyssey,kubric4d,hasim,hasim_benchmark_running',
    select_vis_dataset='pointodyssey,kubric4d,hasim,hasim_benchmark_running',
    sampler='MixedMaxIterBatchSampler',
    max_epoch=None,
    max_iter=max_iter,
    num_workers=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=1,
    lr=None,
    lr_scheduler=dict(
        type='hAlgorithm.modules.lr_schedulers.cosine_lr_updater.CosineLrUpdater',
        base_lr=5e-05,
        max_iters=max_iter,
        warmup_iters=500,
        warmup='linear',
        warmup_ratio=1e-06,
        min_lr=1e-07,
    ),
    optimizer={
        'type': 'AdamW',
        'model.sparse_gaussian_head': dict(
            lr=0.0001,
            betas=(
                0.9,
                0.999,
            ),
            weight_decay=0.001,
            eps=1e-10,
        ),
        'model': dict(
            lr=1e-05,
            betas=(
                0.9,
                0.999,
            ),
            weight_decay=0.001,
            eps=1e-10,
        ),
        'strict_match': True,
        'debug': False,
    },
    eval_metrics=dict(
        type='hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics',
        metrics=[
            dict(
                type='hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics',
                task='Global',
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                eval_groups=[
                    dict(
                        type='hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics',
                        glb2local=False,
                        local2glb=True,
                        target_name='pointmap',
                        mv_target_name='pointmap_reff',
                        valid_mask_name='depth_mask',
                        metrics=[
                            'abs_relative_difference',
                        ],
                        metric_add_distance=True,
                    ),
                    dict(
                        type='hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics',
                        glb2local=False,
                        local2glb=False,
                        target_name='pointmap',
                        mv_target_name='pointmap_reff',
                        valid_mask_name='depth_mask',
                        metrics=[
                            'abs_relative_difference',
                        ],
                        metric_add_distance=True,
                    ),
                    dict(
                        type='hAlgorithm.modules.metrics.pointmap_eval_metrics.GlobalPointMapEvalMetrics',
                        target_name='pointmap_reff',
                        valid_mask_name='depth_mask',
                        metrics=[
                            'pointmap_normal_cos',
                        ],
                    ),
                    dict(
                        type='hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics',
                        glb2local=False,
                        local2glb=False,
                        target_name='pointmap',
                        mv_target_name='pointmap_reff',
                        valid_mask_name='depth_mask',
                        metrics=[
                            'abs_relative_difference',
                        ],
                        metric_add_distance=True,
                        conf_ratio=0.1,
                    ),
                ],
            ),
            dict(
                type='hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics',
                task='Camera',
                eval_groups=[
                    dict(
                        type='hAlgorithm.modules.metrics.pose_eval_metrics.PoseEvalMetricsV2',
                    ),
                ],
            ),
            dict(
                type='hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics',
                task='Render',
                eval_groups=[
                    dict(
                        type='hAlgorithm.modules.metrics.dynamic_gaussian_eval_metrics.DynamicGaussianEvalMetrics',
                        target_name='image',
                        render_attr='dgs_render_rgb',
                        metric_prefix='',
                        metrics=[
                            'rgb_psnr',
                            'rgb_ssim',
                            'rgb_lpips',
                        ],
                        reference_frame_idx=0,
                    ),
                ],
            ),
            dict(
                type='hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics',
                task='Dyn_pts',
                eval_groups=[
                    dict(
                        type='hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics',
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[],
                        inlier_threshold_3d=None,
                        apd_thresholds_2d=[],
                        inlier_threshold_2d=None,
                        dynamic_only=True,
                        require_visible=True,
                        with_dy_ratio=False,
                        metric_prefix='dy_',
                    ),
                    dict(
                        type='hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics',
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[],
                        inlier_threshold_3d=None,
                        apd_thresholds_2d=[],
                        inlier_threshold_2d=None,
                        dynamic_only=False,
                        require_visible=True,
                        with_dy_ratio=True,
                    ),
                ],
            ),
            dict(
                type='hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics',
                task='Scene_flow',
                eval_groups=[
                    dict(
                        type='hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics',
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[],
                        inlier_threshold_3d=None,
                        apd_thresholds_2d=None,
                        dynamic_only=True,
                        require_visible=True,
                        with_motion_mask=False,
                        with_warp3d_delta=True,
                        with_dy_ratio=False,
                        metric_prefix='dy_',
                    ),
                    dict(
                        type='hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics',
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[],
                        inlier_threshold_3d=None,
                        apd_thresholds_2d=None,
                        dynamic_only=False,
                        require_visible=True,
                        with_motion_mask=False,
                        with_warp3d_delta=True,
                        with_dy_ratio=False,
                    ),
                ],
            ),
        ],
        composite_metrics=[
            dict(
                name='CameraRender|score',
                components=[
                    # Camera score: 50% total (AUC metrics are already in [0, 1]).
                    dict(metric='Camera|auc_1', weight=0.30, lower=0.0, upper=1.0),
                    dict(metric='Camera|auc_30', weight=0.20, lower=0.0, upper=1.0),
                    # Render score: 50% total. PSNR is mapped from a practical
                    # validation range; SSIM is naturally bounded; LPIPS is inverted.
                    dict(metric='Render|rgb_psnr', weight=0.225, lower=10.0, upper=40.0),
                    dict(metric='Render|rgb_ssim', weight=0.175, lower=0.0, upper=1.0),
                    dict(metric='Render|rgb_lpips', weight=0.10, lower=0.0, upper=1.0, invert=True),
                ],
            ),
        ],
    ),
    main_eval_metric='CameraRender|score',
    main_eval_metric_goal='maximize',
    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=2000,
    save_period=1000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key='image',
    load_from=_LOAD_FROM,
)

output_dir = (
    "/mnt/home/tcchen/workspace/TMA-origin-dev/results/sparse_pair_4dgs_v2/"
    "wfm_rgb_query_dual_4dgs_perpixel_trainquality_v2_wild_norm_ffgs_align_gs2x_opaque"
)
ckpt_dir = f"{output_dir}/checkpoint"
tb_dir = f"{output_dir}/tensorboard"
eval_dir = f"{output_dir}/evaluation"
vis_dir = f"{output_dir}/visualization"

trainer["output_dir"] = output_dir
trainer["ckpt_dir"] = ckpt_dir
trainer["tb_dir"] = tb_dir
trainer["eval_dir"] = eval_dir
trainer["vis_dir"] = vis_dir
