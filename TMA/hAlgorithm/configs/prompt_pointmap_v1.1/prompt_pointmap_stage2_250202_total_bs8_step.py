data = dict(
    train="hAlgorithm/configs/prompt_pointmap_v1.1/stage2_dataset_configs/total_train_k1_syn_mvs_wodiml.yaml",
    val="hAlgorithm/configs/prompt_pointmap_v1.1/stage2_dataset_configs/base_test_k1.yaml",
    vis="hAlgorithm/configs/prompt_pointmap_v1.1/stage2_dataset_configs/total_vis_k1.yaml",
    basic=dict(
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_ratio=0.05,
        )
    ),
    # val_basic=dict(max_depth=200),
    train_basic=dict(
        # max_depth=200,
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=None,
            sparse_ratio=0.05,
            blur_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.BlurPointmap",
                scale_range=[0.3, 0.8],
                p=0.8,
            ),
            pts_jitter_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.Random3DJitter",
                noise_std=0.01,
                p=0.5,
            ),
            uv_jitter_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.RandomUVJitter",
                max_jitter=5,
                jitter_ratio=[0.05, 0.5],
                p=0.5,
            ),
            patch_crop_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.PatchCropMask",
                patch_range=[0.05, 0.25],
                p=0.5,
            ),
        )
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines.depth_prompt_pointmap_pipeline_v2.DepthPromptPointMapPipeline",
    head=dict(
        type="hAlgorithm.modules.models.promptda.pointmap_dpt.PointMapDPTHead",
        nclass=1,
        use_bn=False,
        use_clstoken=False,
        output_act="",  # NOTE
        with_uv=False,
        # prompt_fusion="cat",
    ),
    encoder_pretrain="/mnt/netdata/Team/AI/personal/ts/weights/moge/dinov2_vitb14_pretrain.pth",
    head_pretrain=None,
    encoder="vitb",
    patch_size=14,
    align_name="depth_raw",
    align_mask_name="depth_raw_mask",
    match_input_res=True,
    target_name="pointmap",
    target_mask_name="depth_mask",
    prompt_name="sparse_dense_pointmap",  # NOTE: interpolate
    prompt_mask_name=None,
    prompt_scale_name="sparse_pointmap_max_range",
    prompt_center_name=None,
    l1_loss=dict(
        type="hAlgorithm.modules.losses.global_point_z_weighted_loss.GlobalPointZWeightedLossV2",
        loss_weight=1.0,
        # zweighted=False, # NOTE
        # threshold=None,
    ),
    grad_loss=dict(
        type="hAlgorithm.modules.losses.grad_l1_loss.GradL1Loss",
        loss_weight=8.0,
        scale_level=4,
    ),
    normal_loss=dict(
        type="hAlgorithm.modules.losses.normal_cosine_Loss.NormalCosineLossV2",
        # loss_weight=0.1,
        # start_iter=10000,
        loss_weight=dict(
            default=0.0,
            hypersim=0.1,
            blendedmvs=0.1,
            kenburns=0.1,
            diml=0.1,
            dynamicstereo=0.1,
            scannet=0.1,
        ),
    ),
    depth_loss=dict(
        type="hAlgorithm.modules.losses.gradient_loss_v3.GradientLoss",
        laplace=0.0,
        scharr=1.0,
        scharr_xy=0.0,
        sobel=0.0,
        sobel_xy=0.0,
        scharr_p=2,
        scharr_xy_p=1,
        sobel_p=1,
        sobel_xy_p=1,
        # loss_weight=1.0,
        M=4,
        loss_weight=dict(
            default=0.0,
            hypersim=1.0,
            blendedmvs=1.0,
            kenburns=1.0,
            diml=1.0,
            dynamicstereo=1.0,
            scannet=1.0,
        ),
    ),
    # msa_loss=dict(
    #     type="hAlgorithm.modules.losses.multi_scale_anchor_loss.MultiScaleAnchorLoss",
    #     loss_weight=1.0,
    #     nums=100,
    #     alpha=[1/4, 1/16, 1/64],
    #     alpha_weights=[2, 4, 8]
    # ),
    warmup_iters=-1,
    target_clip=None,
    post_align=False,
    prompt_set_none=False,  # NOTE: prompt depth = None
)

trainer = dict(
    type="hAlgorithm.modules.trainers.moge_trainer.MogeTrainer",
    max_epoch=None,
    max_iter=200000,
    num_workers=8,
    batch_size=8,
    gradient_accumulation_steps=1,
    set_random_size=True,
    max_grad_norm=10,
    lr=None,
    # lr_scheduler=dict(
    #     type="hAlgorithm.modules.lr_schedulers.poly_lr_updater.PolyLrUpdater",
    #     base_lr=1e-4,
    #     max_iters=200000,
    #     warmup_iters=1000,
    #     warmup="linear",
    #     warmup_ratio=1e-6,
    #     power=0.9,
    #     min_lr=1e-8,
    # ),
    lr_scheduler=dict(
        type="hAlgorithm.modules.lr_schedulers.step_lr_updater.StepLrUpdater",
        warmup_iters=1000,
        warmup="linear",
        warmup_ratio=1e-6,
        steps=[int(200000 * 0.8)],
        ratio=0.1,
    ),
    # max_grad_norm=1,
    optimizer=dict(
        type="AdamW",
        pretrained=dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        depth_head=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-10),
        strict_match=True,
    ),
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
                target_name="depth_raw",
                valid_mask_name="depth_raw_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=[
                    "abs_relative_difference",
                    "squared_relative_difference",
                    "rmse_linear",
                    "rmse_log",
                    "log10",
                    "delta1_acc",
                    "delta2_acc",
                    "delta3_acc",
                    "i_rmse",
                    "silog_rmse",
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
        ],
    ),
    main_eval_metric="abs_relative_difference",
    main_eval_metric_goal="minimize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=0,
    val_period=1000,
    save_period=1000,
    vis_period=1000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",
    mem=[200, 1024, 1024, 64],  # fp16 显存太少
    # load_from="results/dense_prompt_pointmap_250127_total/prompt_pointmap_stage1_d30_250127_total_20250127-011953/checkpoint/latest/ckpt.pth",
)
