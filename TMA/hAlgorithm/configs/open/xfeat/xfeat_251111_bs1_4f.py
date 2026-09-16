patch_size = 32
max_size = 518
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

max_iter = 160000

view_num=4

downsample = 8

select_dataset = ""
select_dataset += "hypersim,scannet,scpp,megasynth,ase,infinigen,infinigen_iphone,hm3d," # 静态，室内，场景
select_dataset += "infinigen_objs,hablendeder_objs,hablendeder_iphone_objs," # 静态，室内，目标
select_dataset += "mvssynth,unreal4k,blendedmvs," # 静态，室外，场景

select_val_dataset="hypersim,scpp,hablendeder_objs"
select_vis_dataset="hypersim,scpp,hablendeder_objs,patagonia"

select_dataset = None
select_val_dataset = None
select_vis_dataset = None

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs_matching/train_251106.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs_matching/test_251106.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs_matching/vis_251106.yaml",
    basic=dict(
        downsample=downsample,

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
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=800,
                height=608,
                is_lidar=False,
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
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=800,
                height=608,
                is_lidar=False,
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
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0],
            ),
            # x / 255 * 2 - 1
        ],
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines2.xfeat_v1.XFeatPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.sdk.xfeat.XFeat",
        weights="/mnt/nasTeam/AI/lx/code/TM/droid_slam_local/matching/xfeat/weights/xfeat.pt", 
        top_k=4096, 
        detection_threshold=0.05,
        downsample=downsample,
    ),
    downsample=downsample,
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics",
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
        "model": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                type="hAlgorithm.modules.metrics.match_eval_metrics.MatchEvalMetrics",
                ransac_threshold=2.5,
                # matches_name="matches_gt",
            ),
        ],
    ),
    main_eval_metric=None,
    main_eval_metric_goal=None,
    in_evaluation=True,
    in_visualize=False,
    backup_period=100000,
    val_period=10000,
    save_period=1000,
    vis_period=50000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    mem=[10, 1024, 1024, 64],  # fp16 显存太少
)
