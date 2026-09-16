patch_size = 14
max_iter = 15000

base = "hAlgorithm/configs/finetune_gs_v1.0/dataset_configs/gs_mv_ios.yaml"
data = dict(
    train=base, val=base, vis=base, 
    basic=dict(
        data_path=None,
        normalize_cameras=True,
        with_depth=True,
        mf_scene=None,
    )
)

model = dict(
    type="hAlgorithm.modules.pipelines.gaussians_pipeline_v2.Gaussian_Finetuning_PipelineV2",
    model=dict(
        type="hAlgorithm.modules.models.combined_model.finetune_gs_v2.FinetuneGSModel",
        gs_near=0.01,
        gs_far=100.0,
        gaussian_parameters=dict(
            type="hAlgorithm.modules.utils.gaussians.types_finetune_gs.Gaussians",
            pretrain_ply=None,
        ),
        gaussian_render=dict(
            type="hAlgorithm.modules.models.gaussiansv2.decoder_pgsr_v2.decoder_splatting_cuda_finetunegs.DecoderSplattingCUDA",
        ),
        extrinsics_c2w=False,
    ),

    target_name="pointmap",
    target_mask_name=None,
    object_mask_name="object_mask",

    rgb_l1_loss=dict(
        type="hAlgorithm.modules.losses.rgb_l1_loss.RGBL1Loss",
        loss_weight=1.0,
    ),
    ssim_loss=dict(
        type="hAlgorithm.modules.losses.ssim.SSIMLoss",
        loss_weight=0.2,
    ),
    # lpips_loss=dict(
    #     type="hAlgorithm.modules.losses.lpips.LpipsLoss",
    #     loss_weight=0.2,
    # ),

    render_depth_loss=dict(
        type="hAlgorithm.modules.losses.global_point_z_weighted_loss.GlobalPointZWeightedLossV3",
        loss_weight=1.0,
        # zweighted=False, # NOTE
        # threshold=None,
        with_conf=False,
    ),
    # render_normal_loss=dict(
    #     type="hAlgorithm.modules.losses.normal_cosine_Loss.NormalCosineLossV2",
    #     predict_is_normals=True, # NOTE
    #     loss_weight=0.1,
    # ),
    render_normal_match_loss=True,
    normal_match_start=5000,
)


trainer = dict(
    type="hAlgorithm.trainers.gs_trainer.GSTrainerV2",
    logging_step=100,
    max_epoch=None,
    max_iter=max_iter,
    num_workers=8,
    batch_size=1,

    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=10,
    cam_optimizer_end=10000,
    densify_grad_threshold=0.0002,
    scene_extent=5.0,
    densify_until_iter=5000,
    densification_interval=100,
    pruning_interval=500,
    sampler="MixedMaxIterBatchSampler",

    lr=None,
    lr_scheduler=dict(max_iters=max_iter),
    optimizer={
        "type": "AdamW",
        "model.gaussian_parameters._means": dict(
            lr=0.00001,
            eps=1e-15,
            weight_decay=0,
            parameter_name="means",
            lambda_str=lambda step: 1.0,
        ),  # 0.00016 # lambda_str=lambda step: (1 - step / max_iter) ** 0.9,   # xyz_scheduler_args(step)
        "model.gaussian_parameters._features_dc": dict(
            lr=0.0025,
            eps=1e-15,
            weight_decay=0,
            parameter_name="features_dc",
            lambda_str=lambda step: 1.0,
        ),  # 0.0025
        "model.gaussian_parameters._features_rest": dict(
            lr=0.0025 / 20.0,
            eps=1e-15,
            weight_decay=0,
            parameter_name="features_rest",
            lambda_str=lambda step: 1.0,
        ),  # 0.0025
        "model.gaussian_parameters._opacity": dict(
            lr=0.05,
            eps=1e-15,
            weight_decay=0,
            parameter_name="opacity",
            lambda_str=lambda step: 1.0,
        ),
        "model.gaussian_parameters._scaling": dict(
            lr=0.005,
            eps=1e-15,
            weight_decay=0,
            parameter_name="scaling",
            lambda_str=lambda step: 1.0,
        ),
        "model.gaussian_parameters._rotation": dict(
            lr=0.001,
            eps=1e-15,
            weight_decay=0,
            parameter_name="rotation",
            lambda_str=lambda step: 1.0,
        ),
        "strict_match": True,
        "debug": False,
    },
    cam_optimizer={
        "type":"AdamW",
        "cameras": dict(lr=0.0005, betas=(0.9, 0.999), weight_decay=0, eps=1e-10, parameter_name="extrinsics", lambda_str=lambda step: 1.0),

        "strict_match": False,
        "debug": False,
    },
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                type="hAlgorithm.modules.metrics.reconstruct_eval_metrics.ReconstructEvalMetricsWithFinetuneGS",
                target_name="image",
                metrics=["rgb_psnr", "rgb_ssim", "rgb_lpips"],
            ),
            dict(
                type="hAlgorithm.modules.metrics.reconstruct_eval_metrics.ReconstructEvalMetricsWithFinetuneGS",
                target_name="image",
                metrics=["rgb_psnr", "rgb_ssim", "rgb_lpips"],
                with_mask=True,
            ),
            # dict(type="hAlgorithm.modules.metrics.pose_eval_metrics.PoseEvalMetrics"),
        ],
    ),
    main_eval_metric="rgb_psnr",
    main_eval_metric_goal="maximize",
    in_evaluation=True,
    in_visualize=True,
    backup_period=0,
    val_period=1000,
    save_period=10000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",
    # load_from="results/debug/rc_250328_kb_mv_pose_poly_pconf0.1_20250401-105446/checkpoint/latest/ckpt.pth",
    # mem=[200, 1024, 1024, 128],  # fp16 显存太少
)
