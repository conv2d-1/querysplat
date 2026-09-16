patch_size = 14
max_size = 518
# sparse_max_size = round(max_size / patch_size * 8 / patch_size) * patch_size

max_iter = 50000
view_num=4

select_dataset = ""
select_dataset += "hypersim,scannet,scpp,megasynth,ase,infinigen,infinigen_iphone,hm3d," # 静态，室内，场景
select_dataset += "infinigen_objs,hablendeder_objs,hablendeder_iphone_objs," # 静态，室内，目标
select_dataset += "mvssynth,unreal4k,blendedmvs," # 静态，室外，场景

select_val_dataset="hypersim,scannet,scpp,hablendeder_objs,nrgbd,s7"
select_vis_dataset="hypersim,scpp,hablendeder_objs,patagonia"

data = dict(
    train="hAlgorithm/configs/mv_v1.0/dataset_configs_2/train_251104.yaml",
    val="hAlgorithm/configs/mv_v1.0/dataset_configs_2/test_251104.yaml",
    vis="hAlgorithm/configs/mv_v1.0/dataset_configs_2/vis_251104.yaml",
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
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines2.mvfr_v1.MVFRPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.sdk.base.MemoryEfficientMV",
        rgb_encoder=dict(
            type="hAlgorithm.modules.models2.encoder.dinov2_encoder.Dinov2Encoder",
            patch_size=patch_size,
            name="vitl",
            use_clstoken=False,
            normalize=True,
            dinov2_custom_cfg=dict(
                num_register_tokens=4,
                use_checkpoint=False,
            ),
            dinov2_attention_with_sdpa=True,
            # pretrain="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_dino.pth",
        ),
        fuse_encoder=dict(
            type="hAlgorithm.modules.models2.mv_encoder.vggt_mv_encoder.RayMVEncoder",
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            aa_order=["frame", "global"],
            rope_freq=100,
            num_register_tokens=4,
            hooks=[4, 11, 17, 23],

            # pretrain="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_mvdecoder.pth",
            # pertrain_strict=False,
        ),
        depth_head=dict(
            type="hAlgorithm.modules.models2.head.vggt_dpt_head.VGGTDPTHead",
            patch_size=14,
            dim_in=2 * 1024, 
            output_dim=2, 
            activation="exp", 
            conf_activation=None,
            intermediate_layer_idx=[0, 1, 2, 3],
            return_features=False,

            chunk_size=1,
        ),
        glb_points_head=dict(
            type="hAlgorithm.modules.models2.head.vggt_dpt_head.VGGTDPTHead",
            patch_size=14,
            dim_in=2 * 1024, 
            output_dim=4, 
            activation="inv_log", 
            conf_activation=None,
            intermediate_layer_idx=[0, 1, 2, 3],

            chunk_size=1,
        ),
        camera_head=dict(
            type="hAlgorithm.modules.models2.head.camera_head.CameraHead",
            dim_in=1024*2,
            trunk_depth=4,
            pose_encoding_type="absT_quaR_FoV",
            num_heads=16,
            mlp_ratio=4,
            init_values=0.01,
            trans_act="linear",
            quat_act="linear",
            fl_act="relu",
            # pretrain="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_camera_head.pth",
        ),
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
    target_motion_mask_name=None,
    target_invalid_mask_name=None,
    align_name=None,

    # extra inputs
    # with_ray_directions=True,

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
        output_colmap_format=True,
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
    lr_scheduler=None,
    optimizer=None,
    eval_metrics=None,
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

    load_from="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_to_pipe2.pth",
    # mem=[100, 1024, 1024, 64],  # fp16 显存太少
)
