patch_size=14

max_size=784
# max_size_list = [i for i in range(630, 840+1, 28)]

max_iter=50000

batch_size=4

view_num=2

novel_view_nums=0

select_dataset = "megadepth_35,megadepth_1,hypersim,hypersim_5"
select_val_dataset = "megadepth_35,megadepth_1,hypersim,hypersim_5"
select_vis_dataset = "megadepth_35,megadepth_1,hypersim"

# select_dataset = select_val_dataset = select_vis_dataset = "megadepth_35"

data=dict(
    train="hAlgorithm/configs/unified_query/dataset_configs/match/train_all_overlap_5_260224.yaml",
    val="hAlgorithm/configs/unified_query/dataset_configs/match/test_260214.yaml",
    vis="hAlgorithm/configs/unified_query/dataset_configs/match/vis_260104.yaml",

    basic=dict(
        sem_name='',
        with_depth_raw=False,

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
        # edge mask for query sampling
        with_rgb_edge_mask=True,
        with_depth_edge_mask=True,

        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=None,
        train_transforms=[
            dict(type='hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio',
                max_size=max_size,
                patch_size=patch_size,
                is_lidar=False,
                low_resolution=True,
                backup=True,
            ),
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
    type='hAlgorithm.modules.pipelines2.mvfr_match_query_v2.MVFRMatchQueryPipeline',
    training_sub_pixel_scale=1,
    testing_sub_pixel_scale=1,

    model=dict(
        type='hAlgorithm.modules.models2.sdk.match_query.MatchQuery',
        decoder_fp32=True,
        fuse_encoder=dict(
            type='hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2',
            normalize=True,
            patch_size=patch_size,
            name='vitl',
            out_layers=[19, 23], # da3=[11, 15, 19, 23], dinov2=[4, 11, 17, 23]
            alt_start=8,
            qknorm_start=8,
            rope_start=8,
            cat_token=True,
            use_checkpoint_start=0,
            use_checkpoint_end=7,
            pretrain='/mnt/netdata/Team/AI/SDK/mvfr2.0/stage4/mvfr_da3l_rgb_all8stdy_251120_mvpdc_bs16_2f16f_100k_20251120-230915/checkpoint/latest/fuse_encoder.pth',
            pertrain_strict=False
        ),
        query_banck=dict(
            type="hAlgorithm.modules.models2.query_bank.single_view.QueryBank5",
            full_ratio=0.0,
            train_sampler=dict(
                type="hAlgorithm.modules.models2.query_bank.sampler.RandomSampler",
                num_samples=100000,
            ),
            timing=False,
            with_edge_mask=True,
            offset=0,
            noise=0.0,
            noise_ratio=0.0,
        ),
        query_feats_aggregator=dict(
            type="hAlgorithm.modules.models2.query_aggregator.match_query_agg.MatchQueryAggregator3",
            patch_size=patch_size,
            intermediate_layer_idx=[0, 1],
            in_chans=[2048, 2048],
            embed_dims=[256, 256],
            upsample_scales=[4, 1],
            mode="bilinear",
            interp_refinenet_cfg=dict(
                type="hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet",
                layer_num=1,
                use_bn=False,
            ),
            # Roma cross-view matching params
            match_layer_idx=-1,     # match on last layer
            match_dim=2048,         # same as in_chans[-1], vitl: 1024 * 2 (cat_token)
            match_temp=0.1,
            match_scale=1.0,
            enable_amp=True,
        ),
        query_decoder=dict(
            type="hAlgorithm.modules.models2.head.mlp_head.Head",
            in_chan=256,
            hidden_dim=64,
            names=dict(warp=2, confidence=4),
            # acts=dict(warp="tanh", confidence=""),
            acts=dict(warp="", confidence=""),
        ),
        query_match_refine=dict(
            type="hAlgorithm.modules.models2.query_aggregator.match_refine_aggregator_v2.QueryMatchRefine",
            refiners=dict(
                refine_scales=[4, 2, 1],
            ),
            refiner_features=dict(
                patch_size=4,
            ),
            fuse_mlp_nums=1,
            fuse_mlp_ratio=4,
        )
    ),

    query_match_loss=dict(
        type="hAlgorithm.modules.losses2.query_match_loss.QueryMatchLoss",
        loss_weight=1,
        alpha=0.5,
        scale_c=1e-4,
        conf_weight=0.01,
        precision_weight=0.01,
        precision_thresh=1.0,
    ),
    target_match_gt_depth_name="depth",
    target_match_gt_depth_intrinsics_name="intrinsics",
    target_match_gt_extrinsics_name="extrinsics",
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

    edge_mask_name="edge_mask",

    dense_match_loss=None,
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

    max_epoch=None,
    max_iter=max_iter,
    num_workers=16,
    batch_size=batch_size,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=1.0,
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
    # >>> Same eval metrics as original roma (match-based)
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
    in_evaluation=False,
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
    load_from=None,
)
