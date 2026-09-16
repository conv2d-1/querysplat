patch_size=14

max_size=784
# max_size_list = [i for i in range(630, 840+1, 28)]

max_iter=50000

batch_size=4

view_num=2

novel_view_nums=0

select_dataset = None
select_val_dataset = None
select_vis_dataset = None

data=dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs_matching_pair/fisheye/train_pinhole90_fisheye.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs_matching_pair/fisheye/test_pinhole_fisheye_260224.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs_matching_pair/fisheye/vis_fisheye_kosmo_260224.yaml",
    basic=dict(
        sem_name='',
        
        interpolate_version=None,
        interpolate_k=0,

        normalize_cameras=False,
        with_global_scale=False,

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
        novel_view_nums=0,
        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=None,
        train_transforms=[
            dict(type='hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio',
                max_size=max_size,
                patch_size=patch_size,
                is_lidar=False,
                low_resolution=True),
            dict(type='hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion',
                to_gray_prob=0.1,
                distortion_prob=0.3),
            dict(type='hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur',
                prob=0.05),
            dict(type='hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion',
                prob=0.1,
                compression=[0, 50]),
            dict(type='hAlgorithm.datasets.transforms.transforms.ToTensor'),
            dict(type='hAlgorithm.datasets.transforms.transforms.Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5])]
    )
)

model=dict(
    type='hAlgorithm.modules.pipelines2.mvfr_v2.MVFRPipeline',
    model=dict(
        type='hAlgorithm.modules.models2.sdk.match.MatchBase',
        # freeze_modules=["camera_encoder"],
        fuse_encoder=dict(
            type='hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2',
            normalize=True,
            patch_size=patch_size,
            name='vitb',
            out_layers=[11],
            alt_start=4,
            qknorm_start=4,
            rope_start=4,
            cat_token=True,
            use_checkpoint_start=0,
            use_checkpoint_end=3,
            # pretrain='/mnt/netdata/Team/AI/weights/DA3/DA3-BASE/backbone.pth',
            pertrain_strict=False
        ),
        match_head=dict(
            type="hAlgorithm.modules.models2.head.romav2_match_head.RoMaV2MatchHead",
            matcher=dict(
                mv_vit=None,
                dim=1536, # dpt dim
            ),
            refiners=dict(
                confidence_type="pixel_conf",
            ),
            pretrained=None,
            use_feat_layers=[-1],
            patch_size=patch_size,
        ),
    ),
    match_precision_type="pixel_conf",
    target_match_gt_depth_name="depth",
    target_match_gt_extrinsics_name="extrinsics",
    target_match_gt_intrinsics_name="intrinsics",
    intrinsics_name='intrinsics',
    extrinsics_name='extrinsics_reff',
    scale_name='sparse_pointmap_max_range',
    prompt_depth_name=None,
    target_local_depth_name=None,
    target_global_points_name=None,
    target_depth_mask_name=None,
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name='depth_raw',
    match_scales=[1, 2, 3.5, 4],
    dense_match_loss=dict(
        type="hAlgorithm.modules.losses2.dense_match_loss.DenseMatchLossV2",
        loss_weight=dict(
            default=0.5,
            hablendeder_fisheye_objs_v2=1.0,
        ),
        scale_weight=1,
        alpha=0.5,
        scale_c=1e-4,
        conf_weight=0.01,
        precision_weight=0.01,
        precision_thresh=1.0,
        coarse_scale=3.5,
    ),
    local_depth_l1_loss=None,
    local_depth_grad_loss=None,
    local_depth_normal_loss=None,
    local2global_loss=None,
    local_normal_loss=None,
    global_points_l1_loss=None,
    global_points_grad_loss=None,
    global_points_normal_loss=None,
    camera_loss=None,
    rc_rgb_l1_loss=None,
    rc_ssim_loss=None,
    rc_lpips_loss=None,
    rc_depth_loss=None,
    rc_normal_loss=None,
    task_weight=dict(
        lcl=1.0,
        glb=1.0,
        l2g=1.0,
        cm=5.0,
        tk=0.03,
        rc=1.0,
        match=1.0,
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
        save_match=True,
    ),
    seed=2024
)

trainer=dict(
    type='hAlgorithm.trainers.moge_trainer.MogeIterTrainer',
    skip_error_step=True,
    select_dataset=select_dataset,
    select_val_dataset=select_val_dataset,
    select_vis_dataset=select_vis_dataset,
    sampler='MixedMaxIterBatchSampler',
    # max_size_list=max_size_list,

    max_epoch=None,
    max_iter=max_iter,
    num_workers=16,
    batch_size=batch_size,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=1e-2,
    lr=None,
    lr_scheduler=dict(
        type='hAlgorithm.modules.lr_schedulers.cosine_lr_updater.CosineLrUpdater',
        base_lr=0.0001,
        max_iters=max_iter,
        warmup_iters=1000,
        warmup='linear',
        warmup_ratio=1e-06,
        min_lr=1e-08
    ),
    optimizer=dict(
        {
        'type': 'AdamW',
        'model.fuse_encoder': dict(lr=1e-05, betas=(0.9, 0.999), weight_decay=0.001, eps=1e-10),
        'model': dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.001, eps=1e-10),
        'strict_match': True,
        'debug': False
        }
    ),
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                type="hAlgorithm.modules.metrics.densematch_eval_metrics.DenseMatchEvalMetricsV2",
                with_coarse=True,
                with_pose_eval=True,
                conf_thresh=0,
            ),
            dict(
                type="hAlgorithm.modules.metrics.densematch_eval_metrics.MatchMaskEvalMetrics",
                gt_mask_name="warp_mask",
                pred_mask_name="overlap",
                threshold=0.5,
            ),
        ],
    ),
    main_eval_metric="auc@1",
    main_eval_metric_goal="maximize",
    in_evaluation=True,
    in_visualize=False,
    backup_period=100000,
    val_period=10000,
    save_period=1000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key='image',
    seed=2024,
    num_processes=4,
    select_max_depth=None,
    load_from="/mnt/netdata/Team/AI/personal/wsz/project/hDepth/total_datas/2602/matching/mvfr_match/total_exp_0311_conf/mvfr_da3b_260316_match_bs4_2f_all0224_with_conf_load0311_200k_20260316-132318/checkpoint/latest/ckpt.pth",
    # mem=[10, 1024, 1024, 64],
)
