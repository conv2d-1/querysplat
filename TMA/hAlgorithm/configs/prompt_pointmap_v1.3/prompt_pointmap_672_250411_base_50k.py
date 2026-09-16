patch_size = 14
max_size = 672
sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

data = dict(
    train="hAlgorithm/configs/prompt_pointmap_v1.3/stage2_dataset_configs/base_train_60valid_loss.yaml",
    val="hAlgorithm/configs/prompt_pointmap_v1.3/stage2_dataset_configs/base_test.yaml",
    vis="hAlgorithm/configs/prompt_pointmap_v1.3/stage2_dataset_configs/total_vis_v2.yaml",
    basic=dict(
        interpolate_version=None,
        interpolate_k=0,
        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_max_size=sparse_max_size,
        sparse_patch_size=patch_size,
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_nums=6000,
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
    # val_basic=dict(max_depth=200),
    train_basic=dict(
        # max_depth=200,
        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_max_size=sparse_max_size,
        sparse_patch_size=patch_size,
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=None,
            sparse_nums=[3000, 15000],
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
                patch_range=[0.05, 0.5],
                p=0.3,
            ),
            random_noise=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.RandomNoise",
                rand_edge_noise="0.0~0.2",
                rand_global_noise="0.0",
            ),
        ),
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
                max_size=max_size,
                patch_size=patch_size,
                is_lidar=False,
                low_resolution=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
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
                prob=0.05,
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
    type="hAlgorithm.modules.pipelines.prompt_pointmap_pipeline_v2.PromptPointMapPipeline",
    model=dict(
        type="hAlgorithm.modules.models.combined_model.base.CombinedModel",
        freeze_modules=[],
        rgb_encoder=dict(
            type="hAlgorithm.modules.models.encoder.prompt_dinov2_encoder.PromptDinov2Encoder",
            patch_size=patch_size,
            name="vitb",
            use_clstoken=False,
            out_channels=[96, 192, 384, 768],
            dinov2_attention_with_sdpa=True,
            pretrain="/mnt/netdata/Team/AI/personal/ts/weights/moge/dinov2_vitb14_pretrain.pth",
            normalize=True,
        ),
        prompt_encoder=dict(
            type="hAlgorithm.modules.models.encoder.aux_encoder_fixact.AuxResnetModuleV4",
            use_dims=[-1],  # Prompt depth
            layer_num=2,
            use_bn=False,
            output_channel=[32, 64, 128, 128],
            kernel_size=3,
        ),
        decoder=dict(
            type="hAlgorithm.modules.models.decoder.prompt_dinov2_decoder.PromptDinov2Decoder",
            features=128,  # vits 64, vitb 128, vitl 256
            prompt_inchannel=128,
            in_channels=[96, 192, 384, 768],
            use_bn=False,
            prompt_enc_deg=None,
            prompt_fusion="cat",
            prompt_cfg=None,
            prompt_flag=None,
            fusion_out_cfg=dict(
                type="hAlgorithm.modules.models.promptda.blocks.AuxResnetModule",
                layer_num=7,
                use_bn=False,
            ),
        ),
        head=dict(
            type="hAlgorithm.modules.models.head.pointmap_head.PointMapHead",
            features=128,  # vits 64, vitb 128, vitl 256
            features2=32,
            output_act="exp",
            pred_confidence=True,
            pred_mask=False,
            pred_gradient=False,
            pred_prompt_confidence=False,
            final_outchannel=1,
            interpolate_out_cfg=dict(
                type="hAlgorithm.modules.models.encoder.aux_encoder_fixact.AuxResnetModule",
                layer_num=7,
                use_bn=False,
            ),
            interpolate_out_size=None,
        ),
    ),
    output_conf_thresh=0,  # NOTE: confidence thresh
    align_name="depth_raw",
    align_mask_name="depth_raw_mask",
    match_input_res=True,
    target_name="pointmap",
    target_mask_name="depth_mask",
    prompt_name="sparse_pointmap",
    prompt_mask_name=None,
    prompt_scale_name="sparse_pointmap_max_range",
    prompt_center_name=None,
    l1_loss=dict(
        type="hAlgorithm.modules.losses.global_point_z_weighted_loss.GlobalPointZWeightedLossV3",
        loss_weight=1.0,
        # zweighted=False, # NOTE
        # threshold=None,
        with_conf=False,
    ),
    grad_loss=dict(
        type="hAlgorithm.modules.losses.grad_l1_loss.GradL1Loss",
        loss_weight=8.0,
        scale_level=4,
    ),
    normal_loss=dict(
        type="hAlgorithm.modules.losses.normal_cosine_Loss.NormalCosineLossV2",
        loss_weight=dict(
            default=0.0,
            hypersim=2.0,
            blendedmvs=2.0,
            kenburns=2.0,
            diml=2.0,
            dynamicstereo=2.0,
            # scannet=0.0,
            habitat=2.0,
        ),
    ),
    depth_loss=dict(
        type="hAlgorithm.modules.losses.gradient_loss_v4.GradientLoss",
        laplace=0.0,
        scharr=1.0,
        scharr_xy=0.0,
        sobel=0.0,
        sobel_xy=0.0,
        scharr_p=2,
        scharr_xy_p=1,
        sobel_p=1,
        sobel_xy_p=1,
        M=4,
        loss_weight=dict(
            default=0.0,
            hypersim=1.0,
            blendedmvs=1.0,
            kenburns=1.0,
            diml=1.0,
            dynamicstereo=1.0,
            scannet=1.0,
            habitat=1.0,
        ),
    ),
    conf_loss=dict(
        type="hAlgorithm.modules.losses.confidence_branch_bce_iou_loss.ConfOutputBceIouLoss",
        loss_weight=0.1,
        norm_type="L1",
        soft_weight=False,
        threshold=0.05,
    ),
    warmup_iters=-1,
    target_clip=None,
    post_align=False,
    prompt_set_none=False,
)

max_iter = 50000
trainer = dict(
    type="hAlgorithm.trainers.moge_trainer.MogeTrainer",
    max_epoch=None,
    max_iter=max_iter,
    num_workers=8,
    batch_size=4,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=100,
    lr=None,
    lr_scheduler=dict(
        type="hAlgorithm.modules.lr_schedulers.step_lr_updater.StepLrUpdater",
        warmup_iters=1000,
        warmup="linear",
        warmup_ratio=1e-6,
        # steps=[200000, 400000, 500000],
        # ratio=0.2,
        steps=[int(max_iter * k) for k in [0.8, 0.9]],
        ratio=0.1,
    ),
    optimizer={
        "type": "AdamW",
        "model.rgb_encoder.dinov2": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
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
                    "abs_difference",
                    "delta1_acc",
                    "delta2_acc",
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
                    "abs_difference",
                ],
                conf_thresh=0,  # NOTE: confidence thresh
            ),
        ],
    ),
    main_eval_metric="abs_relative_difference",
    main_eval_metric_goal="minimize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=50000,
    val_period=2000,
    save_period=1000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",
    # load_from="/mnt/netdata/Team/AI/personal/ts/projects/hAlgorithm/total_datas/Prompt_PointMap_v1.2/prompt_pointmap_stage2_250225_total_1m_bs4_20250226-165941/v1.2_to_v1.3_depth.pth",
    mem=[200, 1024, 1024, 64],  # fp16 显存太少
)
