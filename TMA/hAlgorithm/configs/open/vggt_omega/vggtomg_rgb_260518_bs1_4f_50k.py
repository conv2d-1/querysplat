patch_size = 16
max_size = 512
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

max_iter = 50000

view_num=4
novel_view_nums=0

# view_num_range=[2, 16]
# max_img_per_gpu=16


select_dataset = ""
select_dataset += "hypersim,scannet,scpp,megasynth,ase,infinigen,infinigen_iphone,hm3d," # 静态，室内，场景
select_dataset += "infinigen_objs,hablendeder_objs,hablendeder_iphone_objs," # 静态，室内，目标
select_dataset += "mvssynth,unreal4k,blendedmvs," # 静态，室外，场景
select_dataset += "kubric4d,pointodyssey,dynamicstereo,infinigen4d,tartanair,sintel," # 动态
select_dataset += "omniworld"

# select_dataset = "hypersim"
select_val_dataset = "hypersim,scannet,scpp,hablendeder_objs,nrgbd,s7"
select_vis_dataset = "hypersim,scpp,hablendeder_objs,patagonia"

# select_dataset = select_val_dataset = select_vis_dataset = "hypersim"

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs_5/train_260301_mv.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs_5/test_260301_mv.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs_5/vis_260301_mv.yaml",
    basic=dict(
        sem_name="",

        # clip_sampler=dict(
        #     type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
        #     view_num=view_num,
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
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
            view_num=view_num,
            seed=None,
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
)

model = dict(
    type="hAlgorithm.modules.pipelines2.mvfr_v1.MVFRPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.sdk.base.MVBase2",
        # freeze_modules=["rgb_encoder", "prompt_encoder", "decoder", "head"],
        decoder_fp32=True,
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.vggt_omega_encoder.Aggregator",
            patch_size=patch_size,
            embed_dim=1024,
            num_register_tokens=16,
            register_attention_block_indices= [2, 6, 9, 14, 20],
            cached_layer_indices=[4, 11, 17, 23],

            # use_checkpoint_start=0,
            # use_checkpoint_end=39,
            
            pretrain="/mnt/netdata/Team/AI/weights/VGGT-omega/aggregator.pth",
            pertrain_strict=True,

        ),
        depth_head=dict(
            type="hAlgorithm.modules.models2.head.vggt_omega_dense_head.DenseHead",
            dim_in=2 * 1024, 
            patch_size=patch_size,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            intermediate_layer_idx=[0, 1, 2, 3],
            chunk_size=0,
            pretrain="/mnt/netdata/Team/AI/weights/VGGT-omega/dense_head.pth", 
            pertrain_strict=True,
        ),
        # glb_points_head=dict(
        #     type="hAlgorithm.modules.models2.head.vggt_dpt_head.VGGTDPTHead",
        #     patch_size=patch_size,
        #     dim_in=3072, 
        #     output_dim=4, 
        #     activation="inv_log", 
        #     conf_activation=None,
        #     intermediate_layer_idx=[0, 1, 2, 3],
        # ),
        camera_head=dict(
            type="hAlgorithm.modules.models2.head.vggt_omega_camera_head.CameraHead",
            dim_in=2 * 1024,
            pretrain="/mnt/netdata/Team/AI/weights/VGGT-omega/camera_head.pth",
            pertrain_strict=True,
        ),
    ),
    # input names
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics_reff",
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name="sparse_dense_pointmap",
    target_local_depth_name="pointmap",
    target_global_points_name="pointmap_reff",
    target_depth_mask_name="depth_mask",
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name="depth_raw",

    # local depth loss
    local_depth_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=1.0,
        with_conf=True,
        conf_expp1=False,
        # conf_loss_scale=0.1,
        valid_range=0.99,
    ),
    local_depth_grad_loss=None,
    local_depth_normal_loss=None,
    local2global_loss=None,

    local_normal_loss=dict(
        type="hAlgorithm.modules.losses2.normal_angle_loss.NormalAngleL1Loss",
        valid_range=0.95,
        loss_weight=dict(
            default=0.0,
            hypersim=1.0,
            blendedmvs=0.0,
            nrgbd=0.0,
            s7=0.0,
            scannet=0.0,
            
            hablendeder=1.0,
            hablendeder_objs=1.0,
            habitat=0.0,
            
            scpp=1.0,
            megasynth=1.0,
            mvssynth=1.0,
            infinigen=1.0,
            hm3d=0.0,

            ase=0.0,

            infinigen_iphone=1.0,
            infinigen_objs=1.0,
            hablendeder_iphone_objs=1.0,

            unreal4k=1.0,

            kubric4d=1.0,
            pointodyssey=1.0,
            dynamicstereo=1.0,
            infinigen4d=1.0,
            tartanair=1.0,
            sintel=1.0,

            omniworld=1.0,
        ),
    ),

    # global points loss
    global_points_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=1.0,
        with_conf=True,
        conf_expp1=False,
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

    task_weight=dict(lcl=1.0, glb=1.0, l2g=1.0, cm=5.0, tk=0.03, rc=1.0),

    pose_encoding_type="absT_quaR_FoV",

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
        # "model.prompt_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        # "model.rgb_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.fuse_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        # "model.track_head": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        # "model.gaussian_head": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            # NOTE: local
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Local",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                eval_groups=[
                    dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                         target_name="depth_raw", valid_mask_name="depth_raw_mask",
                         metrics=["abs_relative_difference"]),
                    dict(type="hAlgorithm.modules.metrics.pointmap_eval_metrics.PointMapEvalMetrics",
                         target_name="pointmap", valid_mask_name="depth_mask",
                         metrics=["pointmap_normal_cos"]),
                    dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                         target_name="depth_raw", valid_mask_name="depth_raw_mask",
                         metrics=["abs_relative_difference"], conf_ratio=0.1),
                ],
            ),
            # NOTE: global
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Global",
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                eval_groups=[
                    dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
                         glb2local=False, local2glb=True,
                         target_name='pointmap', mv_target_name='pointmap_reff',
                         valid_mask_name="depth_mask",
                         metrics=["abs_relative_difference"], metric_add_distance=True),
                    dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
                         glb2local=False, local2glb=False,
                         target_name='pointmap', mv_target_name='pointmap_reff',
                         valid_mask_name="depth_mask",
                         metrics=["abs_relative_difference"], metric_add_distance=True),
                    dict(type="hAlgorithm.modules.metrics.pointmap_eval_metrics.GlobalPointMapEvalMetrics",
                         target_name="pointmap_reff",
                         valid_mask_name="depth_mask",
                         metrics=["pointmap_normal_cos"]),
                    dict(type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
                         glb2local=False, local2glb=False,
                         target_name='pointmap', mv_target_name='pointmap_reff',
                         valid_mask_name="depth_mask",
                         metrics=["abs_relative_difference"], metric_add_distance=True,
                         conf_ratio=0.1),
                ],
            ),
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Camera",
                eval_groups=[
                    dict(type="hAlgorithm.modules.metrics.pose_eval_metrics.PoseEvalMetricsV2"),
                ],
            ),
            dict(
                type="hAlgorithm.modules.metrics.combined_eval_metrics.CombinedEvalMetrics",
                task="Match",
                eval_groups=[
                    dict(
                        type="hAlgorithm.modules.metrics.densematch_eval_metrics.DenseMatchEvalMetricsV3",
                        metrics=[
                            'epe_coarse', 'epe_refine_s4', 'epe_refine_s2', 'epe', 
                            'acc1_coarse', 'acc1_refine_s4', 'acc1_refine_s2', 'acc1', 
                            'acc3_coarse', 'acc3_refine_s4', 'acc3_refine_s2', 'acc3',
                        ],
                        with_pose_eval=False,
                        conf_thresh=0.0,
                    ),
                    dict(
                        type="hAlgorithm.modules.metrics.densematch_eval_metrics.MatchMaskEvalMetrics",
                        gt_mask_name="warp_mask",
                        pred_mask_name="overlap",
                        threshold=0.0,
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

    # load_from="results/da3_rgb_all_251117/mvfr_da3g_rgb_all8stdy_251117_mvpdc_bs2_4f_100k_20251120-150155/checkpoint/latest/ckpt.pth",
    # mem=[10, 1024, 1024, 64],  # fp16 显存太少
)
