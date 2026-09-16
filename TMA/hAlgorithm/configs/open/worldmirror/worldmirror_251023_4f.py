patch_size = 14
max_size = 518
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

max_iter = 50000
select_dataset="hypersim,blendedmvs,nrgbd,s7,scannet,scpp,megasynth,mvssynth,hablendeder,infinigen,hablendeder_objs,ase"
select_val_dataset="hypersim,scannet,scpp,hablendeder_objs"
select_vis_dataset="hypersim,scannet,scpp,hablendeder_objs,patagonia"

view_num=4
novel_view_nums=0

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs/train_all7.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs/test_all.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs/vis_all.yaml",
    basic=dict(
        # track_points_nums=64,
        # track_seed=0,

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
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
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
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
            ),
        ],
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines2.worldmirror.MVFRPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.external.worldmirror.models.models.worldmirror.WorldMirror",
        img_size=518, 
        patch_size=14, 
        embed_dim=1024, 
        gs_dim=256, 
        enable_cond=False, 
        enable_cam=True, 
        enable_pts=True, 
        enable_depth=True, 
        enable_norm=True, 
        enable_gs=True,
        patch_embed="dinov2_vitl14_reg", 
        fixed_patch_embed=False, 
        sampling_strategy="uniform",
        dpt_gradient_checkpoint=False, 
        condition_strategy=["token", "pow3r", "token"],
        enable_interpolation=False, 
        max_resolution=2044
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
    align_name="depth_raw",

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
        # output_intrinsics_from_ray=True,
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
    max_grad_norm=10,
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
                target_normalize=False, # NOTE: match Normalize(mean=0, std=255.0)
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
    save_period=10000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    load_from="/mnt/netdata/Team/AI/weights/HunyuanWorld-Mirror/tma_pipe2.pth",
    # mem=[100, 1024, 1024, 64],  # fp16 显存太少
)
