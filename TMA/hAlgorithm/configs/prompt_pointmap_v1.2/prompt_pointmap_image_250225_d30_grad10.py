data = dict(
    train="hAlgorithm/configs/prompt_pointmap_v1.2/stage2_dataset_configs/base_train.yaml",
    val="hAlgorithm/configs/prompt_pointmap_v1.2/stage2_dataset_configs/base_test.yaml",
    vis="hAlgorithm/configs/prompt_pointmap_v1.2/stage2_dataset_configs/total_vis.yaml",
    basic=dict(
        interpolate_version=None,
        interpolate_k=0,
        sparse_depth_ratio=0.05,  # NOTE: image
        sparse_pattern=None,  # NOTE: image
        test_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=672,
                height=448,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=672,
                height=448,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",
                to_gray_prob=0.1,
                distortion_prob=0.05,
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
    train_basic=dict(max_depth=30),  # NOTE: image
    val_basic=dict(max_depth=30),  # NOTE: image
)

model = dict(
    type="hAlgorithm.modules.pipelines.prompt_pointmap_pipeline.PromptPointMapPipeline",
    head=dict(
        type="hAlgorithm.modules.models.promptda.pointmap_dpt.MTPointMapDPTHead",
        nclass=1,
        use_bn=False,
        use_clstoken=False,
        output_act="exp",  # NOTE
        with_uv=False,
        # prompt_fusion="cat", # NOTE: image
        final_outchannel=1,
        # prompt_pre_cfg=dict( # NOTE: image
        #     type="hAlgorithm.modules.models.promptda.blocks.AuxResnetModuleV3",
        #     input_channel=3,
        #     layer_num=2,
        #     use_bn=False,
        #     output_channel=[32, 64, 128, 128],
        #     kernel_size=7,
        # ),
        # prompt_inchannel=128,
        # fusion_out_cfg=dict(
        #     type="hAlgorithm.modules.models.promptda.blocks.AuxResnetModule",
        #     layer_num=7,
        #     use_bn=False,
        # ),
        interpolate_out_cfg=dict(
            type="hAlgorithm.modules.models.promptda.blocks.AuxResnetModule",
            layer_num=7,
            use_bn=False,
        ),
        pred_confidence=True,
        return_features=False,
        prompt_flag=[False, False, False, False],  # NOTE: image
    ),
    encoder_pretrain="/mnt/netdata/Team/AI/personal/ts/weights/moge/dinov2_vitb14_pretrain.pth",
    head_pretrain=None,
    encoder="vitb",
    patch_size=14,
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
    # l1_loss=dict(
    #     type="hAlgorithm.modules.losses.global_point_z_weighted_loss.GlobalPointZWeightedLossV3",
    #     loss_weight=1.0,
    #     # zweighted=False, # NOTE
    #     # threshold=None,
    #     with_conf=False,
    # ),
    l1_loss=dict(  # NOTE: image
        type="hAlgorithm.modules.losses.global_point_z_weighted_loss.GlobalPointZWeightedLossV3",
        loss_weight=1.0,
        zweighted=False,  # NOTE
        threshold=None,
        with_conf=True,
        conf_loss_scale=0.5,
    ),
    grad_loss=dict(
        type="hAlgorithm.modules.losses.grad_l1_loss.GradL1Loss",
        loss_weight=8.0,
        scale_level=4,
    ),
    normal_loss=dict(
        type="hAlgorithm.modules.losses.normal_cosine_Loss.NormalCosineLossV2",
        # loss_weight=0.1,
        start_iter=10000,
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
    # depth_loss=dict(
    #     type="hAlgorithm.modules.losses.gradient_loss_v4.GradientLoss",
    #     laplace=0.0,
    #     scharr=1.0,
    #     scharr_xy=0.0,
    #     sobel=0.0,
    #     sobel_xy=0.0,
    #     scharr_p=2,
    #     scharr_xy_p=1,
    #     sobel_p=1,
    #     sobel_xy_p=1,
    #     # loss_weight=1.0,
    #     M=4,
    #     loss_weight=dict(
    #         default=0.0,
    #         hypersim=1.0,
    #         blendedmvs=1.0,
    #         kenburns=1.0,
    #         diml=1.0,
    #         dynamicstereo=1.0,
    #         scannet=1.0,
    #     ),
    # ),
    # conf_loss=dict(
    #     type="hAlgorithm.modules.losses.confidence_branch_bce_iou_loss.ConfOutputBceIouLoss",
    #     loss_weight=0.1,
    #     norm_type="L1",
    #     soft_weight=False,
    #     threshold=0.05,
    # ),
    warmup_iters=-1,
    target_clip=None,
    post_align=True,
    prompt_set_none=True,
)

max_iter = 50000
trainer = dict(
    type="hAlgorithm.modules.trainers.moge_trainer.MogeTrainer",
    max_epoch=None,
    max_iter=max_iter,
    num_workers=8,
    batch_size=4,
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
    # load_from="results/dense_prompt_pointmap_250127_total/prompt_pointmap_stage1_d30_250127_total_20250127-011953/checkpoint/latest/ckpt.pth",
    mem=[200, 1024, 1024, 64],  # fp16 显存太少
)
