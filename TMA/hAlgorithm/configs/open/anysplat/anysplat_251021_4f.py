patch_size = 14
max_size = 518
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

view_num=4
max_iter = 40000

select_dataset="hypersim"
select_val_dataset="hypersim"
select_vis_dataset="hypersim"

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs/train_all7.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs/test_all.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs/vis_all.yaml",
    basic=dict(
        clip_sampler=dict(
            type="hAlgorithm.datasets_mv.clip_sampler.debug_trajectory.DebugTrajectory",
            view_num=view_num,
            seed=0
        ),
        
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
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",
                to_gray_prob=0.1,
                distortion_prob=0.3,
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
        type="hAlgorithm.modules.models2.sdk.gaussian.MVFFGS",
        freeze_modules=["rgb_encoder", "fuse_encoder", "depth_head", "glb_points_head", "camera_head"],
        rgb_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.dinov2_encoder.Dinov2Encoder",
            patch_size=patch_size,
            name="vitl",
            use_clstoken=False,
            normalize=True,
            dinov2_custom_cfg=dict(
                num_register_tokens=4,
                use_checkpoint=False,
            ),
            dinov2_attention_with_sdpa=True,
            # pretrain="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_dino.pth",
        ),
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.vggt_mv_encoder.MVEncoder",
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            aa_order=["frame", "global"],
            rope_freq=100,
            num_register_tokens=4,
            hooks=[4, 11, 17, 23],

            # pretrain="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_mvdecoder.pth",
            # pertrain_strict=False,
        ),
        depth_head=dict(
            type="hAlgorithm.modules.models2.head.vggt_dpt_head.VGGTDPTHead",
            patch_size=14,
            dim_in=2 * 1024, 
            output_dim=2, 
            activation="exp", 
            conf_activation=None,
            intermediate_layer_idx=[0, 1, 2, 3],
            return_features=False,
        ),
        # glb_points_head=dict(
        #     type="hAlgorithm.modules.models2.head.vggt_dpt_head.VGGTDPTHead",
        #     patch_size=14,
        #     dim_in=2 * 1024, 
        #     output_dim=4, 
        #     activation="inv_log", 
        #     conf_activation=None,
        #     intermediate_layer_idx=[0, 1, 2, 3],
        #     return_features=False,
        # ),
        camera_head=dict(
            type="hAlgorithm.modules.models2.head.camera_head.CameraHead",
            dim_in=1024*2,
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
        ffgs=dict(
            type="hAlgorithm.modules.models2.gaussians.anysplat.FFGS",
            hooks=[0, 1, 2, 3],
            adapter=dict(
                gaussian_scale_min= None,
                gaussian_scale_max= None,
                sh_degree= 4,
            ),
            param_head=dict(
                patch_size=[14, 14],
                # activation="norm_exp", # NOTE anysplat
                # conf_activation="expp1", # NOTE anysplat
                dim_in=2048,
                features=256,
            ),
            decoder=dict(
                name= "splatting_cuda",
                background_color= [1.0, 1.0, 1.0],
                make_scale_invariant= False,
            ),
            opacity_mapping=dict(
                initial= 0.0,
                final= 0.0,
                warm_up= 1,
            ),
            voxel_size= 0.002,
            near=0.01, 
            far=100.0,

            chunk_size=8,
            pretrain="/mnt/netdata/Team/AI/weights/anysplat/ffgs.pth",
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

    # extra inputs
    # with_ray_directions=True,

    # local depth loss
    local_depth_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=1.0,
        with_conf=True,
        # conf_loss_scale=0.1,
        # valid_range=0.99,
    ),
    local_depth_grad_loss=None,
    local_depth_normal_loss=None,
    local2global_loss=None,

    local_normal_loss=None,
    local_ray_directions_loss=None,

    # global points loss
    # global_points_l1_loss=dict(
    #     type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
    #     loss_weight=1.0,
    #     with_conf=True,
    #     # conf_loss_scale=0.1,
    #     # valid_range=0.99,
    # ),
    global_points_grad_loss=None,
    global_points_normal_loss=None,

    # camera loss
    # camera_loss=dict(
    #     type="hAlgorithm.modules.losses2.camera_loss.VGGTCameraLoss",
    #     loss_type="l1",
    #     gamma=0.6,
    #     pose_encoding_type="absT_quaR_FoV",
    #     weight_T=1.0,
    #     weight_R=1.0,
    #     weight_fl=0.5,
    #     frame_num=-100,
    #     loss_weight=1.0,
    # ),

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
        save_gaussians=True,
        save_render_results=True,
        save_render_video=True,
        save_render_video_with_normalize_c2w=True,
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
        # output_intrinsics_from_ray=True,
        render_video_with_pred_camera=True,
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
        "type":"AdamW",
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
            )
        ],
    ),
    main_eval_metric="auc_30",
    main_eval_metric_goal="maximize",
    in_evaluation=True,
    in_visualize=False,
    backup_period=100000,
    val_period=1000,
    save_period=10000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    load_from="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_to_pipe2.pth",
    # load_from="/mnt/netdata/Team/AI/weights/anysplat/tma_pipe2.pth", # anysplat 原始权重，测试显示指标低于 vggt 权重
    mem=[100, 1024, 1024, 64],  # fp16 显存太少
)
