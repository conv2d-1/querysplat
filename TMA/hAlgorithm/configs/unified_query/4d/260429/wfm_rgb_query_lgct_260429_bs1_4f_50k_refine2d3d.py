patch_size = 14
max_size = 504
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

max_iter = 50000

view_num=4
novel_view_nums=0

# view_num_range=[2, 16]
# max_img_per_gpu=16

select_dataset = "pointodyssey,kubric4d,dynamicreplica,cotracker3kubric,hasim,hasim_benchmark_running,hasim_character_medium"
select_val_dataset = "pointodyssey,kubric4d,hasim,hasim_benchmark_running"
select_vis_dataset = "pointodyssey,kubric4d,hasim,hasim_benchmark_running"

select_dataset = "pointodyssey,hasim"
select_val_dataset = select_vis_dataset = "hasim_benchmark_running"

static_dataset_names = None
dynamic_dataset_names = ["pointodyssey","kubric4d","dynamicreplica","cotracker3kubric","hasim", "hasim_benchmark_running", "hasim_character_medium"]

data = dict(
    train="hAlgorithm/configs/motion_head/datasets_config/train/with_hasim_0423.yaml",
    val="hAlgorithm/configs/motion_head/datasets_config/val/with_hasim_val_0423.yaml",
    vis="hAlgorithm/configs/motion_head/datasets_config/val/with_hasim_val_0423.yaml",
    basic=dict(
        sem_name="",
        with_pointmap_raw=True,

        # clip_sampler=dict(
        #     type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
        #     view_num=4,
        #     seed=0
        # ),
        
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
    train_basic=dict(
        with_inf_mask=False,
        with_rgb_edge_mask = True,
        with_depth_edge_mask = True,
        
        clip_maxlen=None,
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=view_num,
            seed=None,
            shuffle=False,
        ),
        
        novel_view_nums=novel_view_nums,

        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=None,
        train_transforms=[
            # dict(
            #     type="hAlgorithm.datasets.transforms.transforms.FOVRandomCrop",
            #     fov_x=[67, 73], fov_y=[53, 58], prob=1.0,
            #     # names=["hablendeder", "hablendeder_objs", "habitat", "infinigen", "ase"],
            #     names=["infinigen"],
            # ),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=max_size,
                patch_size=patch_size,
                is_lidar=False, 
                low_resolution=True,
                backup=True,
            ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5
            # ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",
                to_gray_prob=0.1,
                distortion_prob=0.3,
            ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.Weather", prob=0.1  # NOTE
            # ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur",
                prob=0.05,  # NOTE
            ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion",
                prob=0.1,
                compression=[0, 50],
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            # x / 255 * 2 - 1
        ],
    ),
    val_basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
            view_num=4,
            seed=0
        ),
    ),
    vis_basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
            view_num=4,
            seed=0
        ),
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines2.mvfr_query_v4.WFMQueryPipeline",
    static_dataset_names=static_dataset_names,
    dynamic_dataset_names=dynamic_dataset_names,

    training_sub_pixel_scale=1,
    testing_sub_pixel_scale=1,
    scale_align=False,
    pair_mode=6,
    model=dict(
        type="hAlgorithm.modules.models2.sdk.query.MVQuery4",
        chunk_size=10000,
        decoder_fp32=False,
        # freeze_modules=["rgb_encoder", "prompt_encoder", "decoder", "head"],
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

            # use_checkpoint_start=0,
            # use_checkpoint_end=39,
            
            # pretrain="results/da3_all/mvfr_da3g_rgb_all8stdy_251120_mvpdc_bs16_2f16f_100k_20251124-114829/checkpoint/latest/fuse_encoder.pth",
            # pertrain_strict=False,
        ),
        query_banck=dict(
            type="hAlgorithm.modules.models2.query_bank.single_view.QueryBank5",
            full_ratio=0.2,
            train_sampler=dict(
                type="hAlgorithm.modules.models2.query_bank.sampler.RandomSampler",
                num_samples=100000,
            ),
            timing=False,
            with_edge_mask=True,
            offset=0,
            noise=0.5, 
            noise_ratio=0.0
        ),
        query_feats_aggregator=dict(
            type="hAlgorithm.modules.models2.query_aggregator.infinidepth.Aggregator24",
            patch_size=patch_size, 
            intermediate_layer_idx=[0, 1, 2, 3],
            in_chans=[3072, 3072, 3072, 3072], 
            embed_dims=[256, 256, 256, 256], 
            upsample_scales=[4, 2, 1, 1],
            mode="bilinear",
            interp_refinenet_cfg=dict(
                type="hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet",
                layer_num=1,
                use_bn=False,
            ),
        ),
        query_decoder=dict(
            type="hAlgorithm.modules.models2.head.mlp_head.Head",
            in_chan=256,
            hidden_dim=64,
            names=dict(depth=1, confidence=1, global_points=3, global_confidence=1),
            acts=dict(depth="exp", confidence="", global_points="inv_log", global_confidence="")
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
            # pretrain="results/da3_all/mvfr_da3g_rgb_all8stdy_251120_mvpdc_bs16_2f16f_100k_20251124-114829/checkpoint/latest/camera_head.pth",
        ),
        query_pair_feats_aggregator=dict(
            type="hAlgorithm.modules.models2.query_aggregator.match_query_agg.MotionQueryAggregator24",
            patch_size=patch_size,
            intermediate_layer_idx=[0, 1, 2, 3],
            in_chans=[3072, 3072, 3072, 3072],
            embed_dims=[256, 256, 256, 256],
            upsample_scales=[4, 2, 1, 1],
            mode="bilinear",
            interp_refinenet_cfg=dict(
                type="hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet",
                layer_num=1,
                use_bn=False,
            ),
            # Roma cross-view matching params
            match_layer_idx=-1,     # match on last layer
            match_dim=3072,         # same as in_chans[-1], vitl: 1024 * 2 (cat_token)
            match_temp=0.1,
            match_scale=1.0,
            enable_amp=True,
            return_pyramids=True,
        ),
        query_pair_decoder=dict(
            type="hAlgorithm.modules.models2.head.mlp_head.Head",
            in_chan=256*2,
            hidden_dim=64,
            # names=dict(warp3d=3, warp3d_confidence=3, warp3d_delta=3, warp3d_delta_confidence=3, warp2d=2, warp2d_confidence=4, motion_mask=1),
            # acts=dict(warp3d="inv_log", warp3d_confidence="", warp3d_delta="", warp3d_delta_confidence="", warp2d="", warp2d_confidence="", motion_mask=""),
            names=dict(warp3d=3, warp3d_confidence=3, warp3d_delta=3, warp3d_delta_confidence=3, warp2d=2, warp2d_confidence=4),
            acts=dict(warp3d="inv_log", warp3d_confidence="", warp3d_delta="", warp3d_delta_confidence="", warp2d="", warp2d_confidence=""),
        ),
        query_pair_refine=dict(
            type="hAlgorithm.modules.models2.query_aggregator.match_refine_aggregator_v3.QueryMatchRefine",
            refiners=dict(
                refine_scales=[4, 2, 1],
            ),
            refiner_features=dict(
                patch_size=4,
            ),
            fuse_mlp_nums=1,
            fuse_mlp_ratio=4,
        ),
        query_pair_refine2=dict(
            type="hAlgorithm.modules.models2.query_aggregator.match_refine_aggregator_3d.QueryWarp3DRefineV2",
            stage=1,
            embed_dims=[256, 256, 256, 256],
            hidden_dim=64,
            warp2d_name="refiner_1_warp2d",
            names=dict(warp3d=3, warp3d_confidence=3, warp3d_delta=3, warp3d_delta_confidence=3, warp2d=2, warp2d_confidence=4),
            acts=dict(warp3d="inv_log", warp3d_confidence="", warp3d_delta="", warp3d_delta_confidence="", warp2d="", warp2d_confidence=""),
        )
    ),
    # input names
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics_reff",
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name=None,
    prompt_depth_mask_name=None,
    target_local_depth_name="pointmap_raw",
    target_global_points_name="pointmap_raw_reff",
    # target_global_points_name=None,
    target_depth_mask_name="depth_raw_mask",
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name="depth_raw",

    edge_mask_name="edge_mask",

    motion_extrinsics_name="extrinsics",
    
    # local depth loss
    local_depth_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.ZWeightedLoss",
        loss_weight=1.0,
        with_conf=True,
        # conf_loss_scale=0.1,
        valid_range=0.99,
    ),
    local_depth_grad_loss=None,
    # local_depth_grad_loss=dict(
    #     type="hAlgorithm.modules.losses2.grad_l1_loss.GradL1Loss",
    #     loss_weight=1.0,
    #     scale_level=4,
    # ),
    local_depth_normal_loss=dict(
        type="hAlgorithm.modules.losses2.normal_cos_loss.NormalCosineLoss",
        target_is_normals=False,
        valid_range=0.95,
        loss_weight=dict(
            default=0.0,
            # 室内 场景 仿真
            hypersim=1.0,megasynth=1.0,ase=0.0,infinigen=1.0,infinigen_iphone=1.0,staticthings3d=1.0,
            # 室内 场景 MVS
            scannet=0.0,scpp=1.0,hm3d=0.0,habitat=0.0,
            # 室内 目标 仿真
            infinigen_objs=1.0,hablendeder_objs=1.0,hablendeder_iphone_objs=1.0,
            # 室外 场景 仿真
            mvssynth=1.0,unreal4k=1.0,eden=1.0,
            # 室内外 场景 MVS
            blendedmvs=0.0,
            # 动态 仿真
            kubric4d=1.0,pointodyssey=0.0,dynamicstereo=1.0,infinigen4d=1.0,tartanair=1.0,sintel=1.0,
            # 单帧
            kenburns=1.0,taskonomy=1.0,diode=1.0,vkitti=1.0,synthia=1.0,matrixcity=1.0,urbansyn=0.0,
        ),
    ),
    local2global_loss=None,
    local_normal_loss=None,

    # global points loss
    global_points_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=1.0,
        with_conf=True,
        # conf_loss_scale=0.1,
        valid_range=0.99,
    ),
    global_points_grad_loss=None,
    global_points_normal_loss=None,

    # camera loss
    camera_loss=dict(
        type="hAlgorithm.modules.losses2.camera_loss.Pi3CameraLoss",
        loss_type="l1",
        gamma=0.6,
        pose_encoding_type="absT_quaR_FoV",
        weight_T=1.0,
        weight_R=10.0,
        weight_fl=0.5,
        frame_num=-100,
        loss_weight=1.0,
    ),

    # reconstruction loss
    rc_rgb_l1_loss=None,
    rc_ssim_loss=None,
    rc_lpips_loss=None,
    rc_depth_loss=None,
    rc_normal_loss=None,

    motion_sample_mean = True,
    motion_dynamic_threshold=0.01,

    query_motion2d_loss=dict(
        type="hAlgorithm.modules.losses2.query_match_loss.QueryMatchLoss",
        # loss_weight=1,
        alpha=0.5,
        scale_c=1e-4,
        conf_weight=0.01,
        precision_weight=0.01,
        precision_thresh=1.0,
        loss_weight=dict(
            default=5.0,
            pointodyssey=0.0,
        )
    ),
    query_motion3d_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=10.0,
        with_conf=True,
        # conf_loss_scale=0.1,
        # valid_range=0.99,
    ),
    query_motion3d_delta_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=10.0,
        with_conf=True,
        # conf_loss_scale=0.1,
        # valid_range=0.99,
        zweighted=False,
    ),
    query_motion_mask_loss=dict(
        type="hAlgorithm.modules.losses2.query_track3d_loss.QueryMotionMaskLoss",
        loss_weight=1.0,
        valid_range=None,
    ),

    task_weight=dict(lcl=1.0, glb=1.0, l2g=1.0, cm=10.0, tk=0.03, rc=1.0, match=1.0, motion=1.0),

    pose_encoding_type="absT_quaR_FoV",

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
    ),
)

trainer = dict(
    type="hAlgorithm.trainers.moge_trainer.MogeIterTrainer",
    skip_error_step=True,
    select_dataset=select_dataset,
    select_val_dataset=select_val_dataset,
    select_vis_dataset=select_vis_dataset,

    sampler="MixedMaxIterBatchSampler",
    # sampler="DynamicMixedMaxIterBatchSampler",
    # max_img_per_gpu=max_img_per_gpu,
    # view_num_range=view_num_range,

    max_epoch=None,
    max_iter=max_iter,
    num_workers=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=1,
    lr=None,
    # lr_scheduler=dict(
    #     type="hAlgorithm.modules.lr_schedulers.step_lr_updater.StepLrUpdater",
    #     warmup_iters=1000,
    #     warmup="linear",
    #     warmup_ratio=1e-6,
    #     steps=[int(max_iter * k) for k in [0.8, 0.9]],
    #     ratio=0.1,
    # ),
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
        "type":"AdamW",
        "model.fuse_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            # NOTE: local
            # dict(
            #     type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
            #     task="Local",
            #     gt_min_depth=1e-3,
            #     gt_max_depth=200.0,
            #     eval_groups=[
            #         dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
            #              target_name="depth_raw", valid_mask_name="depth_raw_mask",
            #              metrics=["abs_relative_difference"]),
            #         dict(type="hAlgorithm.modules.metrics.pointmap_eval_metrics.PointMapEvalMetrics",
            #              target_name="pointmap", valid_mask_name="depth_mask",
            #              metrics=["pointmap_normal_cos"]),
            #         dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
            #              target_name="depth_raw", valid_mask_name="depth_raw_mask",
            #              metrics=["abs_relative_difference"], conf_ratio=0.1),
            #     ],
            # ),
            # NOTE: global
            # dict(
            #     type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
            #     task="Global",
            #     gt_min_depth=-200.0,
            #     gt_max_depth=200.0,
            #     eval_groups=[
            #         dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
            #              glb2local=False, local2glb=True,
            #              target_name='pointmap', mv_target_name='pointmap_reff',
            #              valid_mask_name="depth_mask",
            #              metrics=["abs_relative_difference"], metric_add_distance=True),
            #         dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
            #              glb2local=False, local2glb=False,
            #              target_name='pointmap', mv_target_name='pointmap_reff',
            #              valid_mask_name="depth_mask",
            #              metrics=["abs_relative_difference"], metric_add_distance=True),
            #         dict(type="hAlgorithm.modules.metrics.pointmap_eval_metrics.GlobalPointMapEvalMetrics",
            #              target_name="pointmap_reff",
            #              valid_mask_name="depth_mask",
            #              metrics=["pointmap_normal_cos"]),
            #         dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
            #              glb2local=False, local2glb=False,
            #              target_name='pointmap', mv_target_name='pointmap_reff',
            #              valid_mask_name="depth_mask",
            #              metrics=["abs_relative_difference"], metric_add_distance=True,
            #              conf_ratio=0.1),
            #     ],
            # ),
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Camera",
                eval_groups=[
                    dict(type="hAlgorithm.modules.metrics.pose_eval_metrics.PoseEvalMetricsV2"),
                ],
            ),
            # dict(
            #     type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
            #     task="Match",
            #     eval_groups=[
            #         dict(
            #             type="hAlgorithm.modules.metrics.densematch_eval_metrics.DenseMatchEvalMetricsV3",
            #             metrics=[
            #                 'epe_coarse', 'epe_refine_s4', 'epe_refine_s2', 'epe', 
            #                 'acc1_coarse', 'acc1_refine_s4', 'acc1_refine_s2', 'acc1', 
            #                 'acc3_coarse', 'acc3_refine_s4', 'acc3_refine_s2', 'acc3',
            #             ],
            #             with_pose_eval=False,
            #             conf_thresh=0.0,
            #         ),
            #         dict(
            #             type="hAlgorithm.modules.metrics.densematch_eval_metrics.MatchMaskEvalMetrics",
            #             gt_mask_name="warp_mask",
            #             pred_mask_name="overlap",
            #             threshold=0.0,
            #         ),
            #     ],
            # ),
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Motion",
                eval_groups=[
                    dict(
                        type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics",
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[0.05, 0.1, 0.2, 0.3, 0.5],
                        inlier_threshold_3d=0.1,
                        apd_thresholds_2d=[1.0, 3.0],
                        inlier_threshold_2d=1.0,
                        dynamic_only=False,
                        require_visible=True,
                    ),
                ],
            ),
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Motion_dy",
                eval_groups=[
                    dict(
                        type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics",
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[0.05, 0.1, 0.2, 0.3, 0.5],
                        inlier_threshold_3d=0.1,
                        apd_thresholds_2d=[1.0, 3.0],
                        inlier_threshold_2d=1.0,
                        dynamic_only=True,
                        require_visible=True,
                    ),
                ],
            ),
            # dict(
            #     type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
            #     task="Motion_mask",
            #     eval_groups=[
            #         dict(
            #             type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics",
            #             dynamic_threshold=0.01,
            #             apd_thresholds_3d=[0.05, 0.1, 0.2, 0.3, 0.5],
            #             inlier_threshold_3d=0.1,
            #             # apd_thresholds_2d=[1.0, 3.0],
            #             # inlier_threshold_2d=1.0,
            #             apd_thresholds_2d=None,
            #             dynamic_only=False,
            #             require_visible=True,
            #             with_motion_mask=False,
            #             with_delta_motion_mask=True,
            #             with_src_points=True,
            #         ),
            #     ],
            # ),
            # dict(
            #     type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
            #     task="Motion_delta",
            #     eval_groups=[
            #         dict(
            #             type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics",
            #             dynamic_threshold=0.01,
            #             apd_thresholds_3d=[0.05, 0.1, 0.2, 0.3, 0.5],
            #             inlier_threshold_3d=0.1,
            #             # apd_thresholds_2d=[1.0, 3.0],
            #             # inlier_threshold_2d=1.0,
            #             apd_thresholds_2d=None,
            #             dynamic_only=False,
            #             require_visible=True,
            #             with_motion_mask=False,
            #             with_warp3d_delta=True,
            #             with_src_points=True,
            #         ),
            #     ],
            # ),
            # dict(
            #     type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
            #     task="Motion_delta_dy",
            #     eval_groups=[
            #         dict(
            #             type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics",
            #             dynamic_threshold=0.01,
            #             apd_thresholds_3d=[0.05, 0.1, 0.2, 0.3, 0.5],
            #             inlier_threshold_3d=0.1,
            #             # apd_thresholds_2d=[1.0, 3.0],
            #             # inlier_threshold_2d=1.0,
            #             apd_thresholds_2d=None,
            #             dynamic_only=True,
            #             require_visible=True,
            #             with_motion_mask=False,
            #             with_warp3d_delta=True,
            #             with_src_points=True,
            #         ),
            #     ],
            # ),
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Scene_flow",
                eval_groups=[
                    dict(
                        type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics",
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[0.05, 0.1, 0.2, 0.3, 0.5],
                        inlier_threshold_3d=0.1,
                        # apd_thresholds_2d=[1.0, 3.0],
                        # inlier_threshold_2d=1.0,
                        apd_thresholds_2d=None,
                        dynamic_only=False,
                        require_visible=True,
                        with_motion_mask=False,
                        with_warp3d_delta=True,
                    ),
                ],
            ),
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Scene_flow_dy",
                eval_groups=[
                    dict(
                        type="hAlgorithm.modules.metrics.sparse_motion_eval_metrics_v2.SparseMotionEvalMetrics",
                        dynamic_threshold=0.01,
                        apd_thresholds_3d=[0.05, 0.1, 0.2, 0.3, 0.5],
                        inlier_threshold_3d=0.1,
                        # apd_thresholds_2d=[1.0, 3.0],
                        # inlier_threshold_2d=1.0,
                        apd_thresholds_2d=None,
                        dynamic_only=True,
                        require_visible=True,
                        with_motion_mask=False,
                        with_warp3d_delta=True,
                    ),
                ],
            ),
        ],
    ),
    main_eval_metric="Camera|auc_1",
    main_eval_metric_goal="maximize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=1000,
    save_period=1000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    # load_from="results/da3_rgb_all_251124/mvfr_da3g_rgb_all8stdy_251120_mvpdc_bs16_2f16f_100k_20251124-114829/checkpoint/latest/ckpt.pth",
    # mem=[10, 1024, 1024, 64],  # fp16 显存太少
    load_from="Autoresearch/Algorithm/Pretrain/MVS/motion_query_260427/wfm_rgb_query_lgct_260427_bs1_4f_50k_2dw5_20260428-121244/checkpoint/latest/ckpt.pth",
)
