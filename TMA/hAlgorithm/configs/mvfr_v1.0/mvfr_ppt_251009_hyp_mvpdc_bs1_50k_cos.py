patch_size = 14
max_size = 518
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

max_iter = 50000

select_dataset="hypersim"
select_val_dataset="hypersim"
select_vis_dataset="hypersim"

# select_dataset="hypersim,scannet,scpp,hablendeder_objs"
# select_val_dataset="hypersim,scannet,scpp,hablendeder_objs"
# select_vis_dataset="hypersim,scannet,scpp,hablendeder_objs,iphone,patagonia"

# select_dataset="hypersim,blendedmvs,nrgbd,s7,scannet,scpp,megasynth,mvssynth,hablendeder,infinigen,hablendeder_objs,ase"
# select_val_dataset="hypersim,scannet,scpp,hablendeder_objs"
# select_vis_dataset="hypersim,scannet,scpp,hablendeder_objs,patagonia"

novel_view_nums=0

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs/train_all7.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs/test_all.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs/vis_all.yaml",
    basic=dict(
        # track_points_nums=64,
        # track_seed=0,

        point_normalize_mode="distance_quantile_0.9",
        
        interpolate_version="v2",
        interpolate_k=3,

        normalize_cameras=True,
        with_global_scale=False,

        sparse_pattern=None,
        sparse_pattern_v2=None,

        # sparse_max_size=max_size,
        # sparse_patch_size=patch_size,
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_ratio=0.05,
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
        max_scale_noise=0.2,

        novel_view_nums=novel_view_nums,

        # track_points_nums=256,
        # track_seed=None,
        # track_neg_ratio=0.4,

        sparse_pattern=None,
        sparse_pattern_v2=None,

        # sparse_max_size=max_size,
        # sparse_patch_size=patch_size,
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=None,
            sparse_ratio=0.05,
            
            project_noise_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.RandomProjectNoise",
                noise_mean=0.0,
                noise_std=0.005, # meter
                rot_noise_std=0.008, # rad, 0.5 deg
                prob=0.2,
            ),
            pts_jitter_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.Random3DJitter",
                noise_std=0.01,
                p=0.5,
            ),
            patch_crop_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.PatchCropMask",
                patch_range=[0.05, 0.5],
                p=0.3,
            ),
            
        ),
        train_transforms=[
            # dict(
            #     type="hAlgorithm.datasets.transforms.transforms.FOVRandomCrop",
            #     fov_x=[35, 90], fov_y=[30, 90], prob=0.5,
            #     names=["hablendeder", "habitat", "infinigen", "ase"],
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
        # freeze_modules=["rgb_encoder", "depth_encoder", "decoder", "head"],
        rgb_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.dinov2_encoder.Dinov2Encoder",
            patch_size=patch_size,
            name="vitb",
            use_clstoken=False,
            normalize=True,
            dinov2_custom_cfg=dict(
                num_register_tokens=4
            ),
            dinov2_attention_with_sdpa=True,
            pretrain="/mnt/netdata/Team/AI/personal/ts/weights/dino/dinov2_vitb14_reg4_pretrain.pth",
        ),
        depth_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet",
            input_channel=3,
            layer_num=1,
            use_bn=False,
            output_channel=[32, 64, 128],
            kernel_size=3,
        ),
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.prompt_mv_encoder.MVEncoder",
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=16,
            mlp_ratio=4.0,
            aa_order=["frame", "global"],
            rope_freq=100,
            num_register_tokens=4,
            hooks=[2, 5, 8, 11],

            prompt_in_chans=128,
            prompt_embed_dim=256,
            fuse_embed_dim=768,

            # pretrain="/mnt/netdata/Team/AI/personal/ts/projects/hdepth/total_datas/MV_v0.1/lite/rc_250519_all3_mvpdc_bs1_wopre_b_step1_20250603-115305/checkpoint/mvvit_1000k.pth",
            # pertrain_strict=False,
        ),
        depth_head=dict(
            type="hAlgorithm.modules.models2.head.dpt_head.DPTHead",
            in_channels=[768*2, 768*2, 768*2, 768*2],
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
            type="hAlgorithm.modules.models2.head.dpt_head.DPTHead",
            in_channels=[768*2, 768*2, 768*2, 768*2],
            mid_channels=[256, 256, 256, 256],

            patch_size=patch_size,
            features=256, # vits 64, vitb 128, vitl 256
            features2=32,
            depth_channel=3,
            depth_act="inverse_log",

            interp_refinenet_cfg=None,

            prompt_flag=[False, False, False, False],

            pred_confidence=True,
            pred_normal=False,
            pred_motion_mask=False,
            pred_invalid_mask=False,
            pred_ray=False,
            
            return_features=False,
            chunk_size=0
        ),

        camera_head=dict(
            type="hAlgorithm.modules.models2.head.camera_head.CameraHead",
            dim_in=768*2,
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
        conf_loss_scale=0.1,
        valid_range=0.99,
    ),
    local_depth_grad_loss=None,
    local_depth_normal_loss=None,
    local2global_loss=None,

    # global points loss
    global_points_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=1.0,
        with_conf=True,
        conf_loss_scale=0.1,
        valid_range=0.99,
    ),
    global_points_grad_loss=None,
    global_points_normal_loss=None,

    # camera loss
    camera_loss=dict(
        type="hAlgorithm.modules.losses2.camera_loss.VGGTCameraLoss",
        loss_type="l1",
        gamma=0.6,
        pose_encoding_type="absT_quaR_FoV",
        weight_T=1.0,
        weight_R=1.0,
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

    task_weight=dict(lcl=1.0, glb=1.0, l2g=1.0, cm=3.0, tk=0.03, rc=1.0),

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
    max_epoch=None,
    max_iter=max_iter,
    num_workers=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=100,
    skip_grad_norm=500,
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
        "model.rgb_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model.depth_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
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
    in_evaluation=False,
    in_visualize=False,
    backup_period=50000,
    val_period=1000,
    save_period=10000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    load_from="/mnt/netdata/Team/AI/SDK/mvfr1.0/rc_lite_250905_all_mvpdc_bs1_400k_lclmax_step_clip100_skip500_dres_20250915-111910/checkpoint/latest/ckpt_pipe2_wo_head.pth",
    # load_from="/mnt/netdata/Team/AI/SDK/mvfr1.0/rc_lite_250905_all_mvpdc_bs1_400k_lclmax_step_clip100_skip500_dres_20250915-111910/checkpoint/latest/ckpt_pipe2.pth",
    mem=[150, 1024, 1024, 64],  # fp16 显存太少
)
