# Segmentation & Detection Config using DINOv2 + DPT
# Reuses existing Base model structure, only changes head and losses

patch_size = 14
max_size = 840
max_iter = 50000
# Classes: 0=background, 1=person, 2=window, 3=sky
num_classes = 4  # Including background class

# ============== Data Configuration ==============
data = dict(
    train="hAlgorithm/configs/seg/data/train_kosmo_seg.yaml",
    val="hAlgorithm/configs/seg/data/val_kosmo_seg.yaml",
    
    basic=dict(
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
                distortion_prob=0.5,
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

# ============== Model Configuration ==============
# Reuse Base model structure, use normal_head slot for segmentation
model = dict(
    type="hAlgorithm.modules.pipelines2.seg_pipeline.SegPipeline",  # Simple pipeline inheriting from Pipeline
    model=dict(
        type="hAlgorithm.modules.models2.sdk.base.Base",  # Reuse existing Base class
        decoder_fp32=True,
        
        # DINOv2 Encoder (same as reference config)
        rgb_encoder=dict(
            type='hAlgorithm.modules.models2.encoder.dinov2_encoder.Dinov2Encoder',
            patch_size=14,
            name='vitl',
            use_clstoken=False,
            dinov2_attention_with_sdpa=True,
            normalize=True,
            with_register=False,
        ),
        
        # Use normal_head slot for semantic segmentation
        # DPTHead outputs [B, num_classes, H, W] instead of [B, 3, H, W] for normal
        normal_head=dict(
            type='hAlgorithm.modules.models2.head.dpt_stack_head.DPTHead',
            in_channels=[1024, 1024, 1024, 1024],
            mid_channels=[256, 256, 256, 256],
            patch_size=14,
            features=256,
            interp_refinenet_cfg=dict(
                type='hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet',
                layer_num=2,
                use_bn=False
            ),
            prompt_flag=[False, False, False, False],
            depth_stack=None,
            # Segmentation stack: output num_classes channels
            normal_stack=dict(
                type='hAlgorithm.modules.models2.head.dpt_stack_head.DPTStack',
                features=256,
                features2=64,
                out_channel=num_classes,  # 3 classes: person, window, sky
                out_act=None,  # Raw logits, apply softmax in loss
                prompt_flag=None,
                interp_refinenet_cfg=dict(
                    type='hAlgorithm.modules.models2.encoder.prompt_encoder.Resnet',
                    layer_num=1,
                    use_bn=False
                ),
                pred_confidence=False
            ),
        ),
    ),
    
    # Input/Target name mapping
    target_seg_name="masks",      # Instance masks from dataset
    target_labels_name="labels",  # Class labels
    
    # Segmentation loss: CrossEntropy + optional Dice
    seg_loss=dict(
        type="hAlgorithm.modules.losses2.seg_loss.SegmentationLoss",
        loss_weight=1.0,
        use_dice=False,  # Disable Dice for faster training
        dice_weight=1.0,
        ce_weight=1.0,
        ignore_index=255,
        class_weights=None,  # Optional: [w_person, w_window, w_sky]
    ),
    
    task_weight=dict(seg=1.0),
)

# ============== Trainer Configuration ==============
trainer = dict(
    type="hAlgorithm.trainers.moge_trainer.MogeIterTrainer",
    skip_error_step=True,
    sampler="MixedMaxIterBatchSampler",
    
    max_epoch=None,
    max_iter=max_iter,
    num_workers=4,  # Reduce workers for small batch_size
    batch_size=1,   # Increase batch size for better GPU utilization
    gradient_accumulation_steps=1,
    set_random_size=False,
    max_grad_norm=1.0,
    
    enable_profile=True,
    
    lr_scheduler=dict(
        type="hAlgorithm.modules.lr_schedulers.cosine_lr_updater.CosineLrUpdater",
        base_lr=1e-4,
        max_iters=max_iter,
        warmup_iters=1000,
        warmup="linear",
        warmup_ratio=1e-6,
        min_lr=1e-6,
    ),
    
    optimizer={
        "type": "AdamW",
        "model.rgb_encoder": dict(lr=1e-6, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-10),
        "model": dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-10),
        "strict_match": True,
        "debug": False,
    },
    
    eval_metrics=dict(
        type="hAlgorithm.modules.metrics.multi_eval_metrics.MultiEvalMetrics",
        metrics=[
            dict(
                type='hAlgorithm.modules.metrics.seg_eval_metrics.SegEvalMetrics',
                num_classes=num_classes,
                metrics=['miou', 'pixel_acc']
            ),
        ],
    ),
    main_eval_metric="miou",
    main_eval_metric_goal="maximize",
    
    in_evaluation=False,
    in_visualize=False,
    backup_period=10000,
    val_period=1000,
    save_period=5000,
    vis_period=500,
    
    accelerator_dynamic_batch=True,
    accelerator_dynamic_batch_key="image",
)
