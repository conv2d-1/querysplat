import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import logging
import math
import pprint

import numpy as np
import open3d as o3d
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    config_logging,
    config_merge_args,
    dict_to_file,
    file2dict,
    instantiate_from_config,
    parse_unknown,
)
from hAlgorithm.utils.depth_util import compute_zero_regions


def parse_args():
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file.",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="Path of checkpoint to be load.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="exp",
        help="name of experiment”",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show test results.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to test dataset file.",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Path to test results.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0,
        help="Confidence threshold.",
    )

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.output_dir = os.path.join(args.output_dir, args.exp, config_name + "_" + now)

    if (args.test or args.vis) and args.save_outputs:
        args.output_dir = os.path.realpath(args.output_dir)

    if args.load_from in ["None", "none", "null"]:
        args.load_from = None

    return args, unknown_args


def main():
    args, unknown_args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(
        timeout=datetime.timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", 1200)))
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[dist_kwargs, init_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        config_logging(out_dir=logging_dir)
    else:
        config_logging()

    logging.info(f"device: {accelerator.device}")
    logging.info(f"seed: {args.seed}")

    # Set the training seed now.
    set_seed(args.seed)

    # Initialize directories
    eval_dir = os.path.join(args.output_dir, "evaluation")
    vis_dir = os.path.join(args.output_dir, "visualization")
    configs_dir = os.path.join(args.output_dir, "configs")
    reconstruct_dir = os.path.join(args.output_dir, "reconstruct")

    if accelerator.is_main_process:
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(reconstruct_dir, exist_ok=True)

    # Load configuration
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    cfg["data"]["vis"] = args.data

    cfg["trainer"]["seed"] = args.seed
    cfg["model"]["seed"] = args.seed

    cfg["trainer"]["resume"] = None
    cfg["trainer"]["load_from"] = args.load_from

    cfg["trainer"]["output_dir"] = args.output_dir
    cfg["trainer"]["eval_dir"] = eval_dir
    cfg["trainer"]["vis_dir"] = vis_dir

    vis_datasets, vis_dataloaders, vis_nums = prepare_data_loaders(
        "vis",
        cfg,
        configs_dir,
        is_main_process=accelerator.is_main_process,
    )

    if accelerator.is_main_process:
        logging.info("")
        logging.info("***** Running Reconstruction *****")
        logging.info(f"  Num examples = {sum([len(dataset) for dataset in vis_datasets])}")
        logging.info(f"  logging_dir = {args.output_dir}")
        logging.info("")

        # -------------------- Snapshot of config --------------------
        cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
        assert cfg_save_path != args.config
        dict_to_file(cfg, cfg_save_path)

    # Initialize model
    model_cfg = cfg["model"].copy()
    model = instantiate_from_config(model_cfg)
    assert model is not None
    # setattr(model, "output_global_pointmap", True)
    setattr(model, "match_input_res", False)

    # Initialize trainer
    trainer_cfg = cfg["trainer"].copy()
    module_dict = dict(
        accelerator=accelerator,
        model=model,
        train_dataset=None,
        train_dataloader=None,
        val_datasets=None,
        val_dataloaders=None,
        vis_datasets=vis_datasets,
        vis_dataloaders=vis_dataloaders,
    )
    trainer_cfg.update(module_dict)
    trainer = instantiate_from_config(trainer_cfg)
    assert trainer is not None

    for data_loader in trainer.vis_dataloaders:
        # Ensure batch size is 1 for this visualization
        assert data_loader.batch_size == 1, "Batch size must be 1 for this visualization."
        frame_outputs, eval_results, eval_text_save_path = trainer.validate_single_dataset(
            data_loader=data_loader,
            eval_metrics=trainer.eval_metrics if args.test else None,
            vis=args.vis,
            save=args.save_outputs,
        )

        pointmap_global_list = []
        pointmap_gt_global_list = []
        pointmap_color_list = []
        confidence_list = []

        for frame_output in frame_outputs:
            extrinsics_inv = np.linalg.inv(frame_output.extrinsics)
            R = extrinsics_inv[:3, :3]
            T = extrinsics_inv[:3, 3]

            pointmap_globa = np.dot(R, frame_output.pointmap.T).T + T
            pointmap_global_list.append(pointmap_globa)

            if frame_output.pointmap_gt is not None:
                pointmap_gt = frame_output.pointmap_gt
                pointmap_gt_global = np.dot(R, pointmap_gt.T).T + T
                pointmap_gt_global_list.append(pointmap_gt_global)

            if frame_output.pointmap_color is not None:
                pointmap_color_list.append(frame_output.pointmap_color)

            if frame_output.confidence is not None:
                confidence = frame_output.confidence

                if pointmap_gt is not None:
                    pointmap_gt_mask = (
                        pointmap_gt.reshape([confidence.shape[0], confidence.shape[1], 3])[:, :, 2]
                        > 0.001
                    )
                    top_zero, bottom_zero, left_zero, right_zero = compute_zero_regions(
                        pointmap_gt_mask, thresh=10
                    )
                    if top_zero > 0:
                        confidence[:top_zero, :] = args.conf_thresh - 1
                    if bottom_zero > 0:
                        confidence[-bottom_zero:, :] = args.conf_thresh - 1
                    if left_zero > 0:
                        confidence[:, :left_zero] = args.conf_thresh - 1
                    if right_zero > 0:
                        confidence[:, -right_zero:] = args.conf_thresh - 1

                confidence = confidence.reshape(-1)
                confidence_list.append(confidence)

        pointmap_global = np.concatenate(pointmap_global_list, axis=0)
        pointmap_gt_global = pointmap_color = confidence = None

        if len(pointmap_gt_global_list) > 0:
            pointmap_gt_global = np.concatenate(pointmap_gt_global_list, axis=0)
            assert pointmap_gt_global.shape[0] == pointmap_global.shape[0]

        if len(pointmap_color_list) > 0:
            pointmap_color = np.concatenate(pointmap_color_list, axis=0)
            assert pointmap_color.shape[0] == pointmap_global.shape[0]

        if len(confidence_list) > 0:
            confidence = np.concatenate(confidence_list, axis=0)
            assert confidence.shape[0] == pointmap_global.shape[0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointmap_global)
        if pointmap_color is not None:
            pcd.colors = o3d.utility.Vector3dVector(pointmap_color / 255.0)
        save_path = os.path.join(reconstruct_dir, f"global_point.ply")
        o3d.io.write_point_cloud(save_path, pcd)
        logging.info(f"save: {save_path}")

        if pointmap_gt_global is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointmap_gt_global)
            if pointmap_color is not None:
                pcd.colors = o3d.utility.Vector3dVector(pointmap_color / 255.0)
            save_path = os.path.join(reconstruct_dir, f"global_point_gt.ply")
            o3d.io.write_point_cloud(save_path, pcd)

        if confidence is not None and args.conf_thresh is not None:
            mask = confidence >= args.conf_thresh
            filtered_pointmap = pointmap_global[mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_pointmap)
            if pointmap_color is not None:
                filtered_pointmap_color = pointmap_color[mask]
                pcd.colors = o3d.utility.Vector3dVector(filtered_pointmap_color / 255.0)
            save_path = os.path.join(reconstruct_dir, f"global_point_filter{args.conf_thresh}.ply")
            o3d.io.write_point_cloud(save_path, pcd)


if __name__ == "__main__":
    main()
