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
    # train="hAlgorithm/configs/mv_v1.0/dataset_configs_obj6/train_251013.yaml",
    # val="hAlgorithm/configs/mv_v1.0/dataset_configs_obj6/test.yaml",
    # vis="hAlgorithm/configs/mv_v1.0/dataset_configs_obj6/vis.yaml",
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
            # dict(type="hAlgorithm.datasets.transforms.transforms.MapAnyThingResizeCrop"),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
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
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            # x / 255 * 2 - 1
        ],
    ),
)

model = dict(
    type="hAlgorithm.modules.pipelines2.mapanything.MVFRPipeline",
    model=dict(
        type="hAlgorithm.modules.models2.sdk.mapanything.MapAnything",
        pretrained_checkpoint_path="/mnt/netdata/Team/AI/weights/map-anything/model.safetensors",
        ignore_calibration_inputs = True,
        ignore_depth_inputs = True,
        ignore_pose_inputs = True,
        ignore_depth_scale_inputs = True,
        ignore_pose_scale_inputs = True,
        enc_embed_dim=1024,
        encoder_config={
            "type": "hAlgorithm.modules.models2.external.uniception.models.encoders.dinov2.DINOv2Encoder",
            "patch_size": 14,
            "data_norm_type": "dinov2",
            "gradient_checkpointing": True,
            "name": "dinov2_large",
            "size": "large",
            "torch_hub_force_reload": False,
            "with_registers": False,
            "pretrained_checkpoint_path": "/mnt/netdata/Team/AI/personal/ts/weights/moge/dinov2_vitl14_pretrain.pth" 
        },
        geometric_input_config=dict(
            ray_dirs_encoder_config={
                "type": "hAlgorithm.modules.models2.encoder.rep_2d_encoder.DenseRepresentationEncoder",
                "apply_pe": False,
                "enc_embed_dim": 1024,
                "in_chans": 3,
                "name": "ray_dirs_encoder",
                "patch_size": 14
            },
            depth_encoder_config={
                "type": "hAlgorithm.modules.models2.encoder.rep_2d_encoder.DenseRepresentationEncoder",
                "apply_pe": False,
                "enc_embed_dim": 1024,
                "in_chans": 1,
                "name": "depth_encoder",
                "patch_size": 14
            },
            cam_rot_encoder_config={
                "type": "hAlgorithm.modules.models2.encoder.rep_1d_encoder.GlobalRepresentationEncoder",
                "enc_embed_dim": 1024,
                "in_chans": 4,
                "name": "cam_rot_quats_encoder"
            },
            cam_trans_encoder_config={
                "type": "hAlgorithm.modules.models2.encoder.rep_1d_encoder.GlobalRepresentationEncoder",
                "enc_embed_dim": 1024,
                "in_chans": 3,
                "name": "cam_trans_encoder"
            },
            scale_encoder_config={
                "type": "hAlgorithm.modules.models2.encoder.rep_1d_encoder.GlobalRepresentationEncoder",
                "enc_embed_dim": 1024,
                "in_chans": 1,
                "name": "scale_encoder"
            },
        ),
        info_sharing_config={
            "module_args": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.info_sharing.alternating_attention_transformer.MultiViewAlternatingAttentionTransformerIFR",
                "custom_positional_encoding": None,
                "depth": 24,
                "distinguish_ref_and_non_ref_views": True,
                "gradient_checkpointing": False,
                "indices": [
                    11,
                    17
                ],
                "input_embed_dim": 1024,
                "name": "aat_24_layers_ifr",
                "norm_intermediate": True,
                "size": "24_layers"
            }
        },
        pred_head_config={
            "gradient_checkpointing": False,
            "feature_head": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.prediction_heads.dpt.DPTFeature",
                "checkpoint_gradient": False,
                "feature_dim": 256,
                "hooks": [
                    0,
                    1,
                    2,
                    3
                ],
                "input_feature_dims": [
                    1024,
                    768,
                    768,
                    768
                ],
                "patch_size": 14
            },
            "dpt_adaptor": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.prediction_heads.adaptors.RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor",
                "confidence_type": "exp",
                "confidence_vmax": float("inf"),
                "confidence_vmin": 1,
                "depth_mode": "exp",
                "depth_vmax": float("inf"),
                "depth_vmin": 0,
                "name": "raydirs+depth+pose+confidence+mask+scale",
                "ray_directions_clamp_min_of_z_dir": False,
                "ray_directions_mode": "linear",
                "ray_directions_normalize_to_unit_image_plane": False,
                "ray_directions_normalize_to_unit_sphere": True,
                "ray_directions_vmax": float("inf"),
                "ray_directions_vmin": -float("inf"),
                "ray_directions_z_dir_min": -float("inf")
            },
            "pose_head": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.prediction_heads.pose_head.PoseHead",
                "input_feature_dim": 768,
                "num_resconv_block": 2,
                "patch_size": 14,
                "rot_representation_dim": 4
            },
            "pose_adaptor": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.prediction_heads.adaptors.CamTranslationPlusQuatsAdaptor",
                "cam_trans_mode": "linear",
                "cam_trans_vmax": float("inf"),
                "cam_trans_vmin": -float("inf"),
                "name": "raydirs+depth+pose+confidence+mask+scale",
                "quaternions_mode": "linear",
                "quaternions_normalize": True,
                "quaternions_vmax": float("inf"),
                "quaternions_vmin": -float("inf")
            },
            "regressor_head": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.prediction_heads.dpt.DPTRegressionProcessor",
                "checkpoint_gradient": False,
                "input_feature_dim": 256,
                "output_dim": 6
            },
            "scale_head": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.prediction_heads.mlp_head.MLPHead",
                "input_feature_dim": 768,
                "output_dim": 1
            },
            "scale_adaptor": {
                "type": "hAlgorithm.modules.models2.external.uniception.models.prediction_heads.adaptors.ScaleAdaptor",
                "mode": "exp",
                "name": "raydirs+depth+pose+confidence+mask+scale",
                "vmax": float("inf"),
                "vmin": 1e-08
            },
        },
    ),
    # input names
    intrinsics_name="intrinsics",
    extrinsics_name="extrinsics_reff",
    scale_name="sparse_pointmap_max_range",
    prompt_depth_name=None,
    target_local_depth_name="pointmap",
    target_global_points_name="pointmap_reff",
    target_depth_mask_name="depth_mask",
    align_name="depth_raw",

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
                metrics=['rgb_psnr', 'rgb_ssim']
            )
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

    # load_from="/mnt/netdata/Team/AI/personal/ts/weights/vggt/vggt_to_pipe2.pth",
    # mem=[100, 1024, 1024, 64],  # fp16 显存太少
)
