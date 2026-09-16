data = dict(
    train="hAlgorithm/configs/prompt_pointmap_v1.3/stage2_dataset_configs/total_train_range_whabitat.yaml",
    val="hAlgorithm/configs/prompt_pointmap_v1.3/stage2_dataset_configs/total_train_range_whabitat.yaml",
    vis="hAlgorithm/configs/prompt_pointmap_v1.3/stage2_dataset_configs/total_vis_v2.yaml",
    basic=dict(
        interpolate_version=None,
        mf_to_sf=True,
        interpolate_k=0,
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_ratio=0.02,
        ),
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=224 * 3,
                height=224 * 2,
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
        sparse_pattern=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_ratio=0.02,
            # blur_t=dict(
            #     type="hAlgorithm.datasets.patterns.pattern_transform.BlurPointmap",
            #     scale_range=[0.3, 0.8],
            #     p=0.8
            # ),
            # pts_jitter_t=dict(
            #     type="hAlgorithm.datasets.patterns.pattern_transform.Random3DJitter",
            #     noise_std=0.01,
            #     p=0.5,
            # ),
            # uv_jitter_t=dict(
            #     type="hAlgorithm.datasets.patterns.pattern_transform.RandomUVJitter",
            #     max_jitter=5,
            #     jitter_ratio=[0.05, 0.5],
            #     p=0.5,
            # ),
            # patch_crop_t=dict(
            #     type="hAlgorithm.datasets.patterns.pattern_transform.PatchCropMask",
            #     patch_range=[0.05, 0.25],
            #     p=0.5,
            # ),
        ),
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=224 * 3,
                height=224 * 2,
            ),
            # dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",
            #     to_gray_prob=0.1,
            #     distortion_prob=0.05,
            # ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.Weather", prob=0.1  # NOTE
            # ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur",
            #     prob=0.05,
            # ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion",
            #     prob=0.1,
            #     compression=[0, 50],
            # ),
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
        freeze_modules=["rgb_encoder", "prompt_encoder", "decoder", "head"],
        rgb_encoder=dict(
            type="hAlgorithm.modules.models.encoder.prompt_dinov2_encoder.PromptDinov2Encoder",
            patch_size=14,
            name="vitb",
            use_clstoken=False,
            out_channels=[96, 192, 384, 768],
            dinov2_attention_with_sdpa=True,
            pretrain="/mnt/netdata/Team/AI/personal/ts/weights/moge/dinov2_vitb14_pretrain.pth",
            normalize=True,
        ),
        prompt_encoder=dict(
            type="hAlgorithm.modules.models.encoder.aux_encoder.AuxResnetModuleV3",
            input_channel=3,
            layer_num=2,
            use_bn=False,
            output_channel=[32, 64, 128, 128],
            kernel_size=7,
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
                type="hAlgorithm.modules.models.encoder.aux_encoder.AuxResnetModule",
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
        # loss_weight=0.1,
        # start_iter=10000,
        loss_weight=dict(
            default=0.5,
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
        # loss_weight=1.0,
        M=4,
        loss_weight=dict(
            default=1.0,
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
    type="hAlgorithm.script.data_filter.data_trainer.DataTrainer",
    max_epoch=1,
    max_iter=None,
    num_workers=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=10,
    lr=None,
    lr_scheduler=dict(
        type="hAlgorithm.modules.lr_schedulers.step_lr_updater.StepLrUpdater",
        warmup_iters=1000,
        warmup="linear",
        warmup_ratio=1e-6,
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
    backup_period=0,
    val_period=1000,
    save_period=1000,
    vis_period=1000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",
    load_from="/mnt/netdata/Team/AI/personal/ts/projects/hAlgorithm/total_datas/Prompt_PointMap_v1.2/prompt_pointmap_stage2_250225_total_1m_bs4_20250226-165941/v1.2_to_v1.3_depth.pth",
    # mem=[200, 1024, 1024, 64],  # fp16 显存太少
)
