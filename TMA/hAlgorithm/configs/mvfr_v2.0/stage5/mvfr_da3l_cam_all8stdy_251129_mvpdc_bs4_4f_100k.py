patch_size = 14
max_size = 504
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

max_iter = 100000

view_num=4
novel_view_nums=0

# view_num_range=[2, 16]
# max_img_per_gpu=16


select_dataset = ""
select_dataset += "hypersim,scannet,scpp,megasynth,ase,infinigen,infinigen_iphone,hm3d," # 静态，室内，场景
select_dataset += "infinigen_objs,hablendeder_objs,hablendeder_iphone_objs," # 静态，室内，目标
select_dataset += "mvssynth,unreal4k,blendedmvs," # 静态，室外，场景
select_dataset += "kubric4d,pointodyssey,dynamicstereo,infinigen4d,tartanair,sintel," # 动态

# select_dataset = "hypersim,scannet,scpp,hablendeder_objs"
select_val_dataset = "hypersim,scannet,scpp,hablendeder_objs,nrgbd,s7"
select_vis_dataset = "hypersim,scpp,hablendeder_objs,patagonia"

data = dict(
    # train="hAlgorithm/configs/mv_v1.0/dataset_configs_3/train_251114_st_dy_16.yaml",
    train="hAlgorithm/configs/mv_v1.0/dataset_configs_3/train_251114_st_dy.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs_3/test_251104_st_dy.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs_2/vis_251104.yaml",
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

        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_nums=1000,
        ),

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
        # clip_sampler=dict(
        #     type="hAlgorithm.datasets_mv.clip_sampler.random_sampler.RandomClipSampler",
        #     view_num=view_num,
        #     seed=None,
        # ),
        
        novel_view_nums=novel_view_nums,

        sparse_pattern=None,
        sparse_pattern_v2=None,
        
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=None,
            sparse_nums=[500, 3000],
        ),

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
        type="hAlgorithm.modules.models2.sdk.base.MVBase",
        # freeze_modules=["rgb_encoder", "prompt_encoder", "decoder", "head"],
        camera_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.camera_encoder.CameraEnc",
            dim_out=1024,
            c2w=False,
            # pretrain="/mnt/netdata/Team/AI/weights/DA3/DA3-LARGE/cam_enc.pth",
        ),
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2",
            normalize=True,
            patch_size=patch_size,
            name="vitl",
            out_layers=[11, 15, 19, 23],
            alt_start=8,
            qknorm_start=8,
            rope_start=8,
            cat_token=True,

            use_checkpoint_start=0,
            use_checkpoint_end=7,
            
            pretrain="/mnt/netdata/Team/AI/weights/DA3/DA3-LARGE/backbone.pth",
            pertrain_strict=True,
        ),
        depth_head=dict(
            type="hAlgorithm.modules.models2.head.dpt_head.DPTHead",
            in_channels=[2048, 2048, 2048, 2048],
            mid_channels=[256, 256, 256, 256],

            patch_size=patch_size,
            features=256, # vits 64, vitb 128, vitl 256
            features2=32,
            depth_channel=1,
            depth_act="exp",

            interp_refinenet_cfg=dict(
                type="hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet",
                layer_num=3,
                use_bn=False,
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
            type="hAlgorithm.modules.models2.head.vggt_dpt_head.VGGTDPTHead",
            patch_size=patch_size,
            dim_in=2048, 
            output_dim=4, 
            activation="inv_log", 
            conf_activation=None,
            intermediate_layer_idx=[0, 1, 2, 3],
        ),
        camera_head=dict(
            type="hAlgorithm.modules.models2.head.camera_head.CameraHead",
            dim_in=2048,
            trunk_depth=4,
            pose_encoding_type="absT_quaR_FoV",
            num_heads=16,
            mlp_ratio=4,
            init_values=0.01,
            trans_act="linear",
            quat_act="linear",
            fl_act="relu",
            # pretrain="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_camera_head.pth",
        ),
    ),
    # input names
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics_reff",
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name=None,
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
        ),
    ),

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
    batch_size=4,
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
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                target_name="depth_raw",
                valid_mask_name="depth_raw_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=[
                    "abs_relative_difference",
                    # "abs_difference",
                    # "delta1_acc",
                    # "delta2_acc",
                ],
            ),
            dict(
                type="hAlgorithm.modules.metrics.pointmap_eval_metrics.PointMapEvalMetrics",
                valid_mask_name="depth_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=[
                    "pointmap_normal_cos",
                ],
            ),
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                target_name="depth_raw",
                valid_mask_name="depth_raw_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=[
                    "abs_relative_difference",
                    # "abs_difference",
                ],
                # conf_thresh=0,  # NOTE: confidence thresh
                conf_ratio=0.1,
            ),
            # NOTE: global
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.GlobalDepthEvalMetrics",
                glb2local=False,
                local2glb=True,
                target_name='pointmap',
                mv_target_name='pointmap_reff',
                valid_mask_name="depth_mask",
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                metrics=[
                    "abs_relative_difference",
                ],
                metric_add_distance=True,
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
                metrics=[
                    "abs_relative_difference",
                ],
                metric_add_distance=True,
            ),
            dict(
                type="hAlgorithm.modules.metrics.pointmap_eval_metrics.GlobalPointMapEvalMetrics",
                valid_mask_name="depth_mask",
                gt_min_depth=-200.0,
                gt_max_depth=200.0,
                metrics=[
                    "pointmap_normal_cos",
                ],
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
                metrics=[
                    "abs_relative_difference",
                ],
                metric_add_distance=True,
                # conf_thresh=0,  # NOTE: confidence thresh
                conf_ratio=0.1,
            ),
            # pose
            dict(type="hAlgorithm.modules.metrics.pose_eval_metrics.PoseEvalMetricsV2"),
            # track
            # dict(type='hAlgorithm.modules.metrics.track_eval_metrics.TAPVidMetrics', vis_thresh=0.5),
            # gs
            dict(type='hAlgorithm.modules.metrics.reconstruct_eval_metrics.ReconstructEvalMetricsWithNovelView',
                target_name='image',
                metrics=['rgb_psnr', 'rgb_ssim']
            ),
            dict(type='hAlgorithm.modules.metrics.normal_eval_metrics.NormalEvalMetrics',
                valid_mask_name='depth_mask',
                metrics=['normal_cos', 'normal_angle', 'normal_delta_30', 'normal_delta_5']
            ),
        ],
    ),
    main_eval_metric="auc_30",
    main_eval_metric_goal="maximize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=1000,
    save_period=1000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    # load_from="results/da3_ppt_251121/mvfr_da3l_cam_d4_251121_mvpdc_bs1_4f_50k_20251120-232718/checkpoint/latest/ckpt.pth",
    mem=[10, 1024, 1024, 64],  # fp16 显存太少
)
