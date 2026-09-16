patch_size = 14
max_size = 504
# sparse_max_size = max_size // 2
# prompt_patch_size = patch_size // 2

max_iter = 50000

view_num=1
novel_view_nums=0

select_dataset = ""
select_dataset += "hypersim,scannet,scpp,megasynth,ase,infinigen,infinigen_iphone,hm3d," # 静态，室内，场景
select_dataset += "infinigen_objs,hablendeder_objs,hablendeder_iphone_objs," # 静态，室内，目标
select_dataset += "mvssynth,unreal4k,blendedmvs," # 静态，室外，场景
select_dataset += "kubric4d,pointodyssey,dynamicstereo,infinigen4d,tartanair,sintel," # 动态

select_dataset = "hypersim,scannet,scpp,hablendeder_objs,mvssynth"
select_val_dataset = "hypersim,scannet,scpp,hablendeder_objs,nrgbd,s7,sintel"
select_vis_dataset = "hypersim,scpp,mvssynth,iphone,patagonia,kosmo_0,kosmo_1,kosmo_2,kosmo_3"

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs_4/train_251205_st_dy_sv_1_sky.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs_3/test_251104_st_dy_1.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs_4/vis_251205_1_sky.yaml",
    basic=dict(
        interpolate_version=None,
        interpolate_k=0,

        # normalize_cameras=True,
        # with_global_scale=True,

        sparse_pattern=None,
        sparse_pattern_v2=None,

        # sparse_max_size=sparse_max_size,
        sparse_patch_size=patch_size,
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=0,
            sparse_nums=2000
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
    train_basic=dict(
        sparse_pattern=None,
        sparse_pattern_v2=None,
        sparse_pattern_v3=dict(
            type="hAlgorithm.datasets.patterns.random_pattern.RandomPattern",
            seed=None,
            sparse_nums=[500, 5000],

            project_noise_t=dict(
                type='hAlgorithm.datasets.patterns.pattern_transform.RandomProjectNoise',
                noise_mean=0.0,
                noise_std=0.005,
                rot_noise_std=0.008,
                prob=0.2
            ),
            pts_jitter_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.Random3DJitter",
                noise_std=0.01,
                p=0.5,
            ),
            patch_crop_t=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.PatchCropMask",
                patch_range=[0.0, 1.0],
                p=1.0,
                reverse_p=0.5,
            ),
            random_noise=dict(
                type="hAlgorithm.datasets.patterns.pattern_transform.RandomNoise",
                rand_edge_noise="0.0~0.2",
                rand_global_noise="0.0",
            ),
        ),

        train_transforms=[
            # dict(
            #     type="hAlgorithm.datasets.transforms.transforms.FOVRandomCrop",
            #     fov_x=[67, 73], fov_y=[53, 58], prob=1.0,
            #     # names=["hablendeder", "hablendeder_objs", "habitat", "infinigen", "ase"],
            #     names=["infinigen"],
            # ),
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
                distortion_prob=0.5,
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
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            # x / 255 * 2 - 1
        ],
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines2.mvfr_v1.MVFRPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.sdk.base.MVBase",
        # freeze_modules=["rgb_encoder", "prompt_encoder", "decoder", "head"],
        depth_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet",
            use_dims=[-2, -1],
            layer_num=2,
            use_bn=False,
            output_channel=[32, 64, 128, 128],
            kernel_size=3,
        ),
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.da3_mv_encoder.DinoV2",
            normalize=True,
            patch_size=patch_size,
            name="vitb",
            out_layers=[5, 7, 9, 11],
            alt_start=4,
            qknorm_start=4,
            rope_start=4,
            cat_token=True,

            prompt_start=0,
            prompt_index=[0, 6, 8, 10],
            prompt_patch_size=patch_size,
            prompt_in_chans=128,
            prompt_embed_dim=768,
            # prompt_zero_adapt=True,

            use_checkpoint_start=-1,
            use_checkpoint_end=-1,
            
            pretrain="/mnt/netdata/Team/AI/weights/DA3/DA3-BASE/backbone.pth",
            pertrain_strict=False,
        ),
        depth_head=dict(
            type="hAlgorithm.modules.models2.head.da3_dpt.DPT",
            dim_in=768*2,
            output_dim=2,
            features=128,
            out_channels=[96, 192, 384, 768],
            conf_activation="",
            pretrain="/mnt/netdata/Team/AI/weights/DA3/DA3-BASE/head.pth",
            pretrain_strict=False,

            pred_normal=True,
            pred_invalid_mask=True,
        ),
        # normal_head=dict(
        #     type="hAlgorithm.modules.models2.head.normal_dpt_head.NormalDPTHead",
        #     patch_size=14,
        #     dim_in=2 * 3072, 
        #     output_dim=3, 
        #     activation=None, 
        #     conf_activation=None,
        #     intermediate_layer_idx=[0, 1, 2, 3],
        # ),
    ),
    # input names
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics_reff",
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name="sparse_pointmap",
    prompt_depth_mask_name="sparse_pointmap_mask",
    target_local_depth_name="pointmap",
    target_global_points_name="pointmap_reff",
    target_depth_mask_name="depth_mask",
    target_normal_name=None,
    target_motion_mask_name=None,
    target_invalid_mask_name="inf_mask",
    align_name="depth_raw",

    # local depth loss
    local_depth_l1_loss=dict(
        type="hAlgorithm.modules.losses2.points_l1_loss.GlobalPointZWeightedLoss",
        loss_weight=1.0,
        with_conf=True,
        # conf_loss_scale=0.1,
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

    local_normal_loss=dict(
        type="hAlgorithm.modules.losses2.normal_angle_loss.NormalAngleL1Loss",
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
    local_invalid_mask_loss=dict(
        type="hAlgorithm.modules.losses2.mask_bce_loss.MaskLoss",
        loss_weight=1.0,
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

    # load_from="/mnt/netdata/Team/AI/weights/DA3/DA3-LARGE/model.safetensors",
    mem=[100, 1024, 1024, 64],  # fp16 显存太少
)
