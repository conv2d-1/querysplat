# Myworkspace TMA dual-4DGS

本目录包含与 `iter_050000` checkpoint 配套的 TMA dual-4DGS 代码、可移植配置和运行脚本。

## 目录

```text
Myworkspace/
├── TMA/                         # exp/4dgs-gs2x-opaque-20260707 代码
├── 4DGS/
│   ├── local_4dgs.py            # 已修正本机路径的统一配置
│   ├── smoke_4dgs.py            # 224px / 2 帧低成本验证配置
│   ├── checkpoint/latest/       # 模型权重和 trainer 状态
│   └── configs/
│       ├── local_train.yaml
│       ├── local_val.yaml
│       ├── smoke_train.yaml
│       └── smoke_val.yaml
├── scripts/                  # 运行、训练与评测入口
│   ├── run_4dgs.sh
│   ├── run_4dgs_full.sh
│   └── run_4dgs_overfit_10.sh
└── setup_4dgs_env.sh
```

## 环境

环境安装在项目内部，不修改已有 `cu128`：

```bash
cd /mnt/cfsdata/Team/AI/personal/zhangweiqi/workspace/Myworkspace
./setup_4dgs_env.sh
./scripts/run_4dgs.sh check
```

默认环境路径：

```text
Myworkspace/.envs/tma4dgs
```

## Smoke 流程

```bash
# 无 GT 视频/图像序列推理，默认使用 DAVIS bear，4 帧、280px
./scripts/run_4dgs.sh infer

# 稠密 Gaussian 渲染和 Web viewer 导出
./scripts/run_4dgs.sh render

# 单场景验证、指标及 4DGS 可视化
./scripts/run_4dgs.sh test

# 两步微调，验证数据、前向、反向、优化器和 checkpoint
./scripts/run_4dgs.sh train-smoke

# 完整微调或从 iter_50000 连续恢复
./scripts/run_4dgs.sh train
MAX_ITER=50010 ./scripts/run_4dgs.sh resume
```

自定义输入：

```bash
./scripts/run_4dgs.sh infer /path/to/video.mp4
./scripts/run_4dgs.sh render /path/to/image_sequence
```

可通过环境变量覆盖：

```bash
MAX_FRAMES=8 PROCESS_RES=504 QUERY_STRIDE=2 ./scripts/run_4dgs.sh infer /path/to/input
TRAIN_STEPS=10 CUDA_VISIBLE_DEVICES=0 ./scripts/run_4dgs.sh train-smoke
```

## 完整训练

`local_4dgs.py` 默认从 `4DGS/checkpoint/latest/ckpt.pth` 加载网络权重并开始新的微调。
完整训练可从 TMA 根目录运行：

```bash
cd TMA
../.envs/tma4dgs/bin/python hAlgorithm/script/train/train.py \
  --config ../4DGS/local_4dgs.py \
  --output_dir ../results \
  --exp dual_4dgs_finetune \
  --load_from ../4DGS/checkpoint/latest/ckpt.pth \
  --resume None \
  --mixed_precision bf16
```

若要延续原训练迭代和优化器状态，使用 `--resume ../4DGS/checkpoint/latest/ckpt.pth`，
并把 `--trainer.max_iter` 设置为大于 50000 的值。

## 说明

- `local_4dgs.py` 将失效的 `/mnt/home/tcchen` 和 `/mnt/nasTeam` 模型路径改为本地路径。
- `local_train.yaml`、`local_val.yaml` 将数据根目录映射到 `/mnt/cfsdata/Team/AI/datasets/TMD`。
- `smoke_4dgs.py` 使用 224px、2 帧；`smoke_*.yaml` 仅包含一个 Hasim benchmark
  场景，用于快速验证。
- dual-4DGS 主流程使用 `gsplat`，并因现有模块的顶层导入同时需要
  `fused-ssim`、`pytorch3d`；安装脚本使用兼容的公开实现。
- 旧 FFGS 专用的私有 `diff-gaussian-rasterization-modified` 尚未安装，不影响本
  README 中的 dual-4DGS 流程。
- 当前 Linux 5.4 低于 Accelerate 建议的 5.5；全部 smoke 流程已实测通过，但长时间
  多卡训练仍需留意潜在进程挂起。
- 未安装可选的 `rerun-sdk`，因此仅跳过 Rerun 轨迹导出，不影响指标、Gaussian
  图片、MP4、PLY 和 Web viewer。
