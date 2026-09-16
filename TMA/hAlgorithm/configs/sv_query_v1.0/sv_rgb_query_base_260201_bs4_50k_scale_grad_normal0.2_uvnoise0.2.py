patch_size = 14
max_size = 672
# sparse_max_size = max_size // 2
# prompt_patch_size = patch_size // 2

max_iter = 50000

view_num=1
novel_view_nums=0

# select_dataset = ""
# select_dataset += "hypersim,scannet,scpp,megasynth,ase,infinigen,hm3d," # 静态，室内，场景
# select_dataset += "infinigen_objs,hablendeder_iphone_objs," # 静态，室内，目标
# select_dataset += "mvssynth,unreal4k,blendedmvs," # 静态，室外，场景
# select_dataset += "kubric4d,pointodyssey,dynamicstereo,tartanair,sintel," # 动态
# select_dataset += "kenburns,taskonomy,diode,vkitti,staticthings3d,synthia,matrixcity,urbansyn,eden" # 单帧

# select_dataset = "hypersim,scannet,scpp,hablendeder_objs,mvssynth"
# select_val_dataset = "hypersim,scannet,scpp,hablendeder_objs"
# select_vis_dataset = "hypersim,scpp,mvssynth,iphone,patagonia,kosmo_1,kosmo_2"

select_dataset = select_val_dataset = select_vis_dataset = "hypersim"

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs_4/train_hyperim.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs_3/test_251104_st_dy_1.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs_4/vis_251205_1_sky.yaml",
    basic=dict(
        with_pointmap_raw=True,

        interpolate_version=None,
        interpolate_k=0,

        # normalize_cameras=True,
        # with_global_scale=True,
        point_normalize_mode="depth_quantile_0.98",

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
        with_inf_mask=False,
        with_rgb_edge_mask = True,
        with_depth_edge_mask = True,

        clip_maxlen=1,
        clip_step=None,

        # sparse_pattern_ratio=0.9,
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
                backup=True,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",
                to_gray_prob=0.1,
                distortion_prob=0.8,
            ),
            # dict(
            #     type="hAlgorithm.datasets.transforms.metric3d_transforms.Weather", prob=0.1  # NOTE
            # ),
            dict(
                type="hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur",
                prob=0.1,  # NOTE
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
    type="hAlgorithm.modules.pipelines2.mvfr_query_v1.MVFRQueryPipeline",
    training_sub_pixel_scale=1,
    testing_sub_pixel_scale=-1,
    scale_align=True,
    model=dict(
        type="hAlgorithm.modules.models2.sdk.query.MVQuery",
        timing=False,
        decoder_fp32=False,
        # freeze_modules=["depth_encoder", "fuse_encoder"],
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2",
            normalize=True,
            patch_size=patch_size,
            name="vitl",
            out_layers=[4, 11, 23],
            alt_start=-1,
            qknorm_start=-1,
            rope_start=-1,
            cat_token=False, 
            pretrain="/mnt/netdata/Team/AI/weights/DA3/DA3METRIC-LARGE/backbone.pth",
            pertrain_strict=False,
        ),
        query_banck=dict(
            type="hAlgorithm.modules.models2.query_bank.single_view.QueryBank5",
            full_ratio=0.2,
            train_sampler=dict(
                type="hAlgorithm.modules.models2.query_bank.sampler.RandomSampler",
                num_samples=100000,
            ),
            timing=False,
            with_edge_mask=True,
            offset=0,
            noise=0.5, 
            noise_ratio=0.2
        ),
        query_feats_aggregator=dict(
            type="hAlgorithm.modules.models2.query_aggregator.infinidepth.Aggregator2",
            patch_size=patch_size, 
            in_chans=[1024, 1024, 1024], 
            embed_dims=[256, 512, 1024], 
            upsample_scales=[4, 2, 1],
            mode="bilinear",
        ),
        query_decoder=dict(
            type="hAlgorithm.modules.models2.head.mlp_head.Head",
            in_chan=1024,
            hidden_dim=256,
            names=dict(depth=1, confidence=1),
            acts=dict(depth="exp", confidence="")
        ),
    ),
    # input names
    intrinsics_name="intrinsics",
    extrinsics_name=None,
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name=None,
    prompt_depth_mask_name=None,
    target_local_depth_name="pointmap_raw",
    target_global_points_name=None,
    target_depth_mask_name="depth_raw_mask",
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name="depth_raw",

    edge_mask_name="edge_mask",

    # local depth loss
    local_depth_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.ZWeightedLoss",
        loss_weight=1.0,
        with_conf=False,
        conf_loss_scale=0.5,
        valid_range=0.99,
    ),
    local_depth_grad_loss=dict(
        type="hAlgorithm.modules.losses2.grad_l1_loss.GradL1Loss",
        loss_weight=1.0,
        scale_level=4,
    ),
    local_depth_normal_loss=dict(
        type="hAlgorithm.modules.losses2.normal_cos_loss.NormalCosineLoss",
        target_is_normals=False,
        valid_range=0.95,
        loss_weight=dict(
            default=0.0,
            # 室内 场景 仿真
            hypersim=1.0,megasynth=1.0,ase=0.0,infinigen=1.0,infinigen_iphone=1.0,staticthings3d=1.0,
            # 室内 场景 MVS
            scannet=0.0,scpp=1.0,hm3d=0.0,habitat=0.0,
            # 室内 目标 仿真
            infinigen_objs=1.0,hablendeder_objs=1.0,hablendeder_iphone_objs=1.0,
            # 室外 场景 仿真
            mvssynth=1.0,unreal4k=1.0,eden=1.0,
            # 室内外 场景 MVS
            blendedmvs=0.0,
            # 动态 仿真
            kubric4d=1.0,pointodyssey=0.0,dynamicstereo=1.0,infinigen4d=1.0,tartanair=1.0,sintel=1.0,
            # 单帧
            kenburns=1.0,taskonomy=1.0,diode=1.0,vkitti=1.0,synthia=1.0,matrixcity=1.0,urbansyn=0.0,
        ),
    ),

    task_weight=dict(lcl=1.0, glb=1.0, l2g=1.0, cm=5.0, tk=0.03, rc=1.0),

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
        save_local_results=True,
        save_extra_local_results=True,
        save_track_results=False,
        save_filtered_results=False,
        output_normalize_cameras=True,
        output_match_input_res=True,
        output_conf_ratio=0.2,
    ),
)

trainer = dict(
    type="hAlgorithm.trainers.moge_trainer.MogeIterTrainer",
    timing=False,
    skip_error_step=True,
    select_dataset=select_dataset,
    select_val_dataset=select_val_dataset,
    select_vis_dataset=select_vis_dataset,
    sampler="MixedMaxIterBatchSampler",
    max_epoch=None,
    max_iter=max_iter,
    num_workers=16,
    batch_size=4,
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=1,
    # skip_grad_norm=400,
    lr=None,
    # lr_scheduler=dict(
    #     type="hAlgorithm.modules.lr_schedulers.step_lr_updater.StepLrUpdater",
    #     warmup_iters=1000,
    #     warmup="linear",
    #     warmup_ratio=1e-6,
    #     steps=[46000, 100000] + [int(max_iter * k) for k in [0.5, 0.8, 0.9]],
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
        "model.fuse_encoder": dict(lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
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
                    "rmse_linear",
                ],
            ),
            dict(
                type="hAlgorithm.modules.metrics.pointmap_eval_metrics.PointMapEvalMetrics",
                valid_mask_name="depth_raw_mask",
                gt_min_depth=1e-3,
                gt_max_depth=200.0,
                metrics=[
                    "pointmap_normal_cos",
                ],
            ),
            # dict(
            #     type="hAlgorithm.modules.metrics.depthmap_eval_metrics.DepthEvalMetrics",
            #     target_name="depth_raw",
            #     valid_mask_name="depth_raw_mask",
            #     gt_min_depth=1e-3,
            #     gt_max_depth=200.0,
            #     metrics=[
            #         "abs_relative_difference",
            #         # "abs_difference",
            #     ],
            #     # conf_thresh=0,  # NOTE: confidence thresh
            #     conf_ratio=0.1,
            # ),
            # dict(
            #     type='hAlgorithm.modules.metrics.normal_eval_metrics.NormalEvalMetrics',
            #     metrics=['normal_cos', 'normal_angle', 'normal_delta_30', 'normal_delta_5'],
            #     valid_mask_name='depth_mask',
            # ),
        ],
    ),
    main_eval_metric="abs_relative_difference",
    main_eval_metric_goal="minimize",
    in_evaluation=False,
    in_visualize=False,
    backup_period=100000,
    val_period=1000,
    save_period=1000,
    vis_period=10000,
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",

    # resume="results/svgt_all_260105/svgt_da3g2_all_260105_bs4_500k_20260113-095508/checkpoint/best/ckpt.pth",
    # load_lr_scheduler=False,
    # load_optimizer=False,
    # mem=[50, 1024, 1024, 64],  # fp16 显存太少
)
