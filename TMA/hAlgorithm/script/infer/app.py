import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import logging

import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
import trimesh
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from gradio_rerun import Rerun
from tqdm import tqdm

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    colorize_depth_maps,
    config_logging,
    config_merge_args,
    dict_to_file,
    file2dict,
    instantiate_from_config,
    parse_unknown,
)
from hAlgorithm.utils.depth_util import compute_zero_regions

# 初始化 rerun 日志记录
rr.init("hAlgorithm App")


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
        "--server_name",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=1234,
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


def points_to_mesh(pts, col, outfile, save_ply=False):
    if os.path.exists(outfile):
        return

    pts = pts.copy()
    if save_ply:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
        if col is not None:
            pcd.colors = o3d.utility.Vector3dVector(col.reshape(-1, 3) / 255.0)
        save_path = outfile + ".ply"
        o3d.io.write_point_cloud(save_path, pcd)

    pts[:, 0] *= -1
    pts[:, 1] *= -1
    scene = trimesh.Scene()
    if col is not None:
        pct = trimesh.PointCloud(
            pts.reshape(-1, 3), colors=col.reshape(-1, 3).astype(np.uint8) / 255.0
        )
    else:
        pct = trimesh.PointCloud(pts.reshape(-1, 3))
    scene.add_geometry(pct)
    scene.export(file_obj=outfile)


class Open3DFuser:
    """Open3D-based implementation of TSDF fusion for depth maps.

    This class provides functionality to fuse depth maps and RGB images into a TSDF volume
    using Open3D's integration pipeline, and extract triangle meshes from the fused volume.

    Args:
        fusion_resolution (float, optional): Resolution of the TSDF volume in meters. Defaults to 0.04.

    Attributes:
        fusion_max_depth (float): Maximum depth threshold for fusion.
        volume (o3d.pipelines.integration.ScalableTSDFVolume): Open3D TSDF volume for integration.
    """

    def __init__(self, fusion_resolution: float = 0.02):
        super().__init__()
        self.fusion_resolution: float = fusion_resolution
        self.fusion_max_depth = -1

        voxel_size: float = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def fuse_frames(
        self,
        depth_hw,
        K_33,
        cam_T_world_44,
        rgb_hw3,
        fusion_max_depth,
    ) -> None:
        height: int = depth_hw.shape[0]
        width: int = depth_hw.shape[1]

        fusion_max_depth = max(self.fusion_max_depth, fusion_max_depth)

        rgbd: o3d.geometry.RGBDImage = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb_hw3.astype(np.uint8))),
            o3d.geometry.Image(np.ascontiguousarray(depth_hw.astype(np.float32))),
            depth_scale=1.0,
            depth_trunc=fusion_max_depth,
            convert_rgb_to_intensity=False,
        )

        self.volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=K_33[0, 0],
                fy=K_33[1, 1],
                cx=K_33[0, 2],
                cy=K_33[1, 2],
            ),
            cam_T_world_44,
        )

    def export_mesh(self, path) -> None:
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())

    def get_mesh(
        self, export_single_mesh=None, convert_to_trimesh=False
    ) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()

        return mesh


def get_layer(model, path):
    parts = path.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def get_layers(model, config_dict):
    non_quant_parts = []
    for layer_path, indices in config_dict.items():
        layer = get_layer(model, layer_path)
        if len(indices) > 0:
            selected_layers = [layer[i] for i in indices]
        else:
            selected_layers = [layer]
        non_quant_parts.extend(selected_layers)
    return non_quant_parts


class Event:
    def __init__(self) -> None:
        self.frame_outputs = dict()
        self.dataset_index = None

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

        try:
            mem = torch.ones(cfg["trainer"]["mem"]).cuda()
        except Exception:
            pass

        self.args = args
        self.cfg = cfg
        self.accelerator = accelerator

        self.vis_datasets = vis_datasets
        self.vis_dataloaders = vis_dataloaders
        self.vis_nums = vis_nums

        self.vis_dir = vis_dir
        self.reconstruct_dir = reconstruct_dir

        data_name_list = [dataset.name for dataset in vis_datasets]
        data_path_list = [dataset.data_path for dataset in vis_datasets]
        data_nums_list = [len(dataset) for dataset in vis_datasets]
        data_path_list = ["Dataset List:"] + [
            f"{i}. {name}, frames num: {num}"
            for i, (name, num) in enumerate(zip(data_name_list, data_nums_list))
        ]
        self.data_path_list = "<b>" + "<br>".join(data_path_list) + "<b>"

        self.build_trainer()
        self.base_config = args.config
        self.base_weight = args.load_from

    def build_trainer(self):
        if self.accelerator.is_main_process:
            logging.info("")
            logging.info("***** Running Reconstruction *****")
            logging.info(f"  Num examples = {sum([len(dataset) for dataset in self.vis_datasets])}")
            logging.info(f"  logging_dir = {self.args.output_dir}")
            logging.info("")

            # -------------------- Snapshot of config --------------------
            cfg_save_path = os.path.join(self.args.output_dir, os.path.basename(self.args.config))
            assert cfg_save_path != self.args.config
            dict_to_file(self.cfg, cfg_save_path)

        # Initialize model
        model_cfg = self.cfg["model"].copy()
        trainer_cfg = self.cfg["trainer"].copy()

        if "quantization" not in model_cfg:
            model = instantiate_from_config(model_cfg)
        else:
            from hAlgorithm.script.quantization.ptq_quantizer import PTQuantizer

            quant_cfg = model_cfg["quantization"]
            if "non_quant_part" not in quant_cfg:
                quant_cfg["non_quant_part"] = {}
            ckpt_path = self.cfg["trainer"]["load_from"]
            self.cfg["trainer"]["load_from"] = None
            ptq_quantizer = PTQuantizer()
            ptq_quantizer.load_model(model_cfg, ckpt_path=ckpt_path, forward_func=None)
            non_quant_parts = get_layers(ptq_quantizer.model, quant_cfg["non_quant_part"])
            ptq_quantizer.set_non_quant_modules(non_quant_parts)
            model = ptq_quantizer.model

        assert model is not None
        setattr(model, "match_input_res", False)

        # Initialize trainer
        module_dict = dict(
            accelerator=self.accelerator,
            model=model,
            train_dataset=None,
            train_dataloader=None,
            val_datasets=None,
            val_dataloaders=None,
            vis_datasets=self.vis_datasets,
            vis_dataloaders=self.vis_dataloaders,
        )
        trainer_cfg.update(module_dict)
        trainer = instantiate_from_config(trainer_cfg)
        assert trainer is not None

        self.trainer = trainer

    def change_model(self, new_config, new_weight):
        if self.base_config != new_config or self.base_weight != new_weight:

            self.args.config = new_config
            self.args.load_from = new_weight

            self.cfg["model"] = file2dict(new_config)["model"]
            self.cfg["trainer"]["load_from"] = new_weight

            self.build_trainer()

    def restore_model(self):
        if self.base_config != self.args.config or self.base_weight != self.args.load_from:
            self.cfg["model"] = file2dict(self.base_config)["model"]
            self.cfg["trainer"]["load_from"] = self.base_weight

            self.build_trainer()

    def process(
        self,
        dataset_index: int,
        frame_index: int,
        conf_thresh: float,
        boundary: int,
    ):
        dataset_index = int(dataset_index)
        frame_index = int(frame_index)
        conf_thresh = float(conf_thresh)
        boundary = int(boundary)

        self.dataset_index = dataset_index
        if dataset_index in self.frame_outputs:
            frame_outputs = self.frame_outputs[dataset_index]
        else:
            args, trainer = self.args, self.trainer
            data_loader = trainer.vis_dataloaders[dataset_index]
            # Ensure batch size is 1 for this visualization
            assert data_loader.batch_size == 1, "Batch size must be 1 for this visualization."
            try:
                frame_outputs, eval_results, eval_text_save_path = trainer.validate_single_dataset(
                    data_loader=data_loader,
                    eval_metrics=trainer.eval_metrics if args.test else None,
                    vis=args.vis,
                    save=args.save_outputs,
                    return_frame_output=True,
                )
            except Exception:
                frame_outputs, eval_results, eval_text_save_path = trainer.validate_single_dataset(
                    data_loader=data_loader,
                    eval_metrics=trainer.eval_metrics if args.test else None,
                    vis=args.vis,
                    save=args.save_outputs,
                )
            self.frame_outputs[dataset_index] = frame_outputs

        pointmap_list = []
        pointmap_reconstruct_list = []
        prompt_pointmap_list = []
        prompt_pointmap_reconstruct_list = []
        pointmap_color_list = []
        image_show_list = []
        confidence_list = []
        confidence_mask_list = []
        depth_show_list = []
        confidence_show_list = []
        intrinsics_list = []
        extrinsics_list = []
        first_extrinsics = None

        for frame_output in frame_outputs:
            if isinstance(frame_output, (list, tuple)):
                frame_output = frame_output[-1]
            pointmap = frame_output.pointmap.copy()
            pointmap_list.append(pointmap)

            if frame_output.intrinsics is not None:
                intrinsics_list.append(frame_output.intrinsics.copy())

            if frame_output.extrinsics is not None:
                extrinsics = frame_output.extrinsics.copy()
                if first_extrinsics is None:
                    first_extrinsics = extrinsics

                cur2first = first_extrinsics @ np.linalg.inv(extrinsics)
                R = cur2first[:3, :3]
                T = cur2first[:3, 3]
                pointmap_reconstruct = np.dot(R, pointmap.T).T + T
                pointmap_reconstruct_list.append(pointmap_reconstruct)

                extrinsics = np.linalg.inv(cur2first)
                extrinsics_list.append(extrinsics)

            else:
                pointmap_reconstruct_list.append(pointmap)

            depth = pointmap_list[-1][:, 2].reshape(
                [frame_output.pointmap_h, frame_output.pointmap_w]
            )
            depth_show = colorize_depth_maps(depth, depth.min(), depth.max(), cmap="turbo")
            depth_show_list.append(np.array(depth_show))

            if frame_output.prompt_pointmap is not None:
                prompt_pointmap = frame_output.prompt_pointmap.copy()
                prompt_pointmap_list.append(prompt_pointmap)

                if frame_output.extrinsics is not None:
                    R = cur2first[:3, :3]
                    T = cur2first[:3, 3]
                    prompt_pointmap = np.dot(R, prompt_pointmap.T).T + T
                    prompt_pointmap_reconstruct_list.append(prompt_pointmap)
                else:
                    prompt_pointmap_reconstruct_list.append(prompt_pointmap)

            if frame_output.pointmap_color is not None:
                pointmap_color_list.append(frame_output.pointmap_color)
                image_show_list.append(
                    frame_output.pointmap_color.reshape(
                        [frame_output.pointmap_h, frame_output.pointmap_w, 3]
                    ).astype(np.uint8)
                )

            if frame_output.pointmap_gt is not None:
                pointmap_gt = frame_output.pointmap_gt.copy()
                pointmap_gt_path = os.path.join(
                    self.reconstruct_dir,
                    f"points_gt_d{dataset_index}_f{frame_index}.glb",
                )
                points_to_mesh(
                    pointmap_gt, frame_output.pointmap_color, pointmap_gt_path, save_ply=False
                )

            if frame_output.confidence is not None:
                confidence = frame_output.confidence.copy()
                if conf_thresh is not None:
                    confidence_mask = confidence > conf_thresh
                else:
                    confidence_mask = np.ones(confidence.shape, dtype=bool)

                if boundary is not None and boundary > 0 and frame_output.pointmap_gt is not None:
                    pointmap_gt = frame_output.pointmap_gt.copy()
                    pointmap_gt_mask = (
                        pointmap_gt.reshape([confidence.shape[0], confidence.shape[1], 3])[:, :, 2]
                        > 0.001
                    )
                    top_zero, bottom_zero, left_zero, right_zero = compute_zero_regions(
                        pointmap_gt_mask, thresh=boundary
                    )
                    if top_zero > 0:
                        confidence_mask[:top_zero, :] = False
                    if bottom_zero > 0:
                        confidence_mask[-bottom_zero:, :] = False
                    if left_zero > 0:
                        confidence_mask[:, :left_zero] = False
                    if right_zero > 0:
                        confidence_mask[:, -right_zero:] = False

                confidence_show = colorize_depth_maps(
                    confidence, confidence.min(), confidence.max(), cmap="turbo"
                )
                confidence_list.append(confidence.reshape(-1))
                confidence_mask_list.append(confidence_mask.reshape(-1))
                confidence_show_list.append(confidence_show)

        self.pointmap_list = pointmap_list
        self.pointmap_reconstruct_list = pointmap_reconstruct_list
        self.prompt_pointmap_list = prompt_pointmap_list
        self.prompt_pointmap_reconstruct_list = prompt_pointmap_reconstruct_list
        self.pointmap_color_list = pointmap_color_list
        self.image_show_list = image_show_list
        self.depth_show_list = depth_show_list
        self.confidence_list = confidence_list
        self.confidence_mask_list = confidence_mask_list
        self.confidence_show_list = confidence_show_list
        self.extrinsics_list = extrinsics_list
        self.intrinsics_list = intrinsics_list

        if frame_index >= 0:
            (
                image_show,
                depth_show,
                confidence_show,
                prompt_pointmap_path,
                pointmap_path,
                frame_index,
                confidence_filter_path,
                filter_points_num,
            ) = self.vis_frame(
                dataset_index=dataset_index,
                frame_index=frame_index,
                conf_thresh=conf_thresh,
                boundary=boundary,
            )

            return (
                image_show,
                depth_show,
                confidence_show,
                prompt_pointmap_path,
                pointmap_path,
                frame_index,
                confidence_filter_path,
                filter_points_num,
            )

    def vis_next_frame(
        self, dataset_index: int, frame_index: int, conf_thresh: float, boundary: int
    ):
        frame_index = int(frame_index) + 1
        return self.vis_frame(
            dataset_index=dataset_index,
            frame_index=frame_index,
            conf_thresh=conf_thresh,
            boundary=boundary,
        )

    def vis_frame(self, dataset_index: int, frame_index: int, conf_thresh: float, boundary: int):
        dataset_index = int(dataset_index)
        frame_index = int(frame_index)
        conf_thresh = float(conf_thresh)
        boundary = int(boundary)

        if self.dataset_index != dataset_index:
            return self.process(
                dataset_index=dataset_index,
                frame_index=frame_index,
                conf_thresh=conf_thresh,
                boundary=boundary,
            )

        frame_index = frame_index % len(self.pointmap_list)

        pointmap = self.pointmap_list[frame_index]
        image_show = self.image_show_list[frame_index]
        depth_show = self.depth_show_list[frame_index]
        if len(self.confidence_show_list) > 0:
            confidence_show = self.confidence_show_list[frame_index]
        else:
            confidence_show = None

        prompt_pointmap = pointmap_color = None

        if len(self.prompt_pointmap_list) > 0:
            prompt_pointmap = self.prompt_pointmap_list[frame_index]
            assert prompt_pointmap.shape[0] == pointmap.shape[0]

        if len(self.pointmap_color_list) > 0:
            pointmap_color = self.pointmap_color_list[frame_index]
            assert pointmap_color.shape[0] == pointmap.shape[0]

        pointmap_path = os.path.join(
            self.reconstruct_dir,
            f"points_d{dataset_index}_f{frame_index}.glb",
        )
        points_to_mesh(pointmap, pointmap_color, pointmap_path, save_ply=False)

        if prompt_pointmap is not None:
            prompt_pointmap_path = os.path.join(
                self.reconstruct_dir,
                f"points_prompt_d{dataset_index}_f{frame_index}.glb",
            )
            points_to_mesh(prompt_pointmap, pointmap_color, prompt_pointmap_path, save_ply=False)
        else:
            prompt_pointmap_path = None

        confidence_filter_path, filter_points_num = self.confidence_filter(
            dataset_index=dataset_index,
            frame_index=frame_index,
            conf_thresh=conf_thresh,
            boundary=boundary,
        )

        return (
            image_show,
            depth_show,
            confidence_show,
            prompt_pointmap_path,
            pointmap_path,
            frame_index,
            confidence_filter_path,
            filter_points_num,
        )

    def reconstruct(self, dataset_index: int, conf_thresh: float, boundary: int):
        frame_index = -1

        dataset_index = int(dataset_index)
        conf_thresh = float(conf_thresh)
        boundary = int(boundary)

        if self.dataset_index != dataset_index:
            self.process(
                dataset_index=dataset_index,
                frame_index=frame_index,
                conf_thresh=conf_thresh,
                boundary=boundary,
            )

        pointmap = np.concatenate(self.pointmap_reconstruct_list, axis=0)
        prompt_pointmap = pointmap_color = None

        if len(self.prompt_pointmap_reconstruct_list) > 0:
            prompt_pointmap = np.concatenate(self.prompt_pointmap_reconstruct_list, axis=0)
            assert prompt_pointmap.shape[0] == pointmap.shape[0]

        if len(self.pointmap_color_list) > 0:
            pointmap_color = np.concatenate(self.pointmap_color_list, axis=0)
            assert pointmap_color.shape[0] == pointmap.shape[0]

        pointmap_path = os.path.join(self.reconstruct_dir, f"points_d{dataset_index}.glb")
        points_to_mesh(pointmap, pointmap_color, pointmap_path, save_ply=False)

        prompt_pointmap_path = os.path.join(
            self.reconstruct_dir, f"points_prompt_d{dataset_index}.glb"
        )
        if prompt_pointmap is not None:
            points_to_mesh(prompt_pointmap, pointmap_color, prompt_pointmap_path, save_ply=False)

        confidence_filter_path, filter_points_num = self.confidence_filter(
            dataset_index=dataset_index,
            frame_index=frame_index,
            conf_thresh=conf_thresh,
            boundary=boundary,
        )

        return (
            prompt_pointmap_path,
            pointmap_path,
            frame_index,
            confidence_filter_path,
            filter_points_num,
        )

    def confidence_filter(
        self, dataset_index: int, frame_index: int, conf_thresh: float, boundary: int
    ):
        dataset_index = int(dataset_index)
        frame_index = int(frame_index)
        conf_thresh = float(conf_thresh)
        boundary = int(boundary)

        if self.dataset_index != dataset_index:
            self.process(
                dataset_index=dataset_index,
                frame_index=-1,
                conf_thresh=conf_thresh,
                boundary=boundary,
            )

        if len(self.confidence_list) == 0:
            return None, 0

        frame_outputs = self.frame_outputs[dataset_index]

        if frame_index < 0:
            pointmap = np.concatenate(self.pointmap_reconstruct_list, axis=0)
            pointmap_color = np.concatenate(self.pointmap_color_list, axis=0)

            confidence_mask_list = []
            for frame_output in frame_outputs:
                if isinstance(frame_output, (list, tuple)):
                    frame_output = frame_output[-1]
                confidence = frame_output.confidence.copy()
                if conf_thresh is not None:
                    confidence_mask = confidence > conf_thresh
                else:
                    confidence_mask = np.ones(confidence.shape, dtype=bool)

                if boundary is not None and boundary > 0 and frame_output.pointmap_gt is not None:
                    pointmap_gt = frame_output.pointmap_gt.copy()
                    pointmap_gt_mask = (
                        pointmap_gt.reshape([confidence.shape[0], confidence.shape[1], 3])[:, :, 2]
                        > 0.001
                    )
                    top_zero, bottom_zero, left_zero, right_zero = compute_zero_regions(
                        pointmap_gt_mask, thresh=boundary
                    )
                    if top_zero > 0:
                        confidence_mask[:top_zero, :] = False
                    if bottom_zero > 0:
                        confidence_mask[-bottom_zero:, :] = False
                    if left_zero > 0:
                        confidence_mask[:, :left_zero] = False
                    if right_zero > 0:
                        confidence_mask[:, -right_zero:] = False
                confidence_mask_list.append(confidence_mask.reshape(-1))
            confidence_mask = np.concatenate(confidence_mask_list, axis=0)

        else:
            pointmap = self.pointmap_list[frame_index]
            pointmap_color = self.pointmap_color_list[frame_index]

            frame_output = frame_outputs[frame_index]
            if isinstance(frame_output, (list, tuple)):
                frame_output = frame_output[-1]
            confidence = frame_output.confidence.copy()
            if conf_thresh is not None:
                confidence_mask = confidence > conf_thresh
            else:
                confidence_mask = np.ones(confidence.shape, dtype=bool)

            if boundary is not None and boundary > 0 and frame_output.pointmap_gt is not None:
                pointmap_gt = frame_output.pointmap_gt.copy()
                pointmap_gt_mask = (
                    pointmap_gt.reshape([confidence.shape[0], confidence.shape[1], 3])[:, :, 2]
                    > 0.001
                )
                top_zero, bottom_zero, left_zero, right_zero = compute_zero_regions(
                    pointmap_gt_mask, thresh=boundary
                )
                if top_zero > 0:
                    confidence_mask[:top_zero, :] = False
                if bottom_zero > 0:
                    confidence_mask[-bottom_zero:, :] = False
                if left_zero > 0:
                    confidence_mask[:, :left_zero] = False
                if right_zero > 0:
                    confidence_mask[:, -right_zero:] = False

            confidence_mask = confidence_mask.reshape(-1)

        if confidence_mask.sum() == 0:
            return None, 0

        filtered_pointmap = pointmap[confidence_mask]
        filtered_pointmap_color = pointmap_color[confidence_mask]
        points_num = filtered_pointmap.shape[0]

        save_path = os.path.join(
            self.reconstruct_dir,
            f"points_d{dataset_index}_f{frame_index}_conf{conf_thresh}_b{boundary}.glb",
        )
        points_to_mesh(filtered_pointmap, filtered_pointmap_color, save_path, save_ply=False)

        return save_path, points_num


def setupUI(event: Event):
    custom_css = """
    #model3d-container {
        background: linear-gradient(to bottom, #cccccc, #1e3c72) !important;
        padding: 20px; /* 可选：增加内边距 */
        border-radius: 10px; /* 可选：圆角效果 */
    }
    """

    with gr.Blocks(css=custom_css) as UI:

        with gr.Row():
            image = gr.Image(type="numpy", label="RGB")
            model3d_prompt = gr.Model3D(label="Prompt", elem_id="model3d-container")

        with gr.Row():
            image_depth = gr.Image(type="numpy", label="Depth")
            image_confidence = gr.Image(type="numpy", label="Confidence")

        with gr.Row():
            model3d_pred = gr.Model3D(label="Prediction", elem_id="model3d-container")

        with gr.Row():
            model3d_pred_filter = gr.Model3D(label="Prediction Clean", elem_id="model3d-container")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    data_path_list = gr.HTML(  # noqa: F841
                        value=event.data_path_list, label="dataset list"
                    )

                with gr.Row():
                    dataset_index = gr.Textbox(interactive=True, value=0, label="dataset index")
                with gr.Row():
                    button_run = gr.Button("Run")

                with gr.Row():
                    frame_index = gr.Textbox(interactive=True, value=0, label="frame index")

                with gr.Row():
                    button_infer = gr.Button("Infer")
                    button_next = gr.Button("Next")

                with gr.Row():
                    button_reconstruct = gr.Button("Reconstruct")

            with gr.Column():
                conf_thresh = gr.Slider(-10, 10, value=0, label="confidence threshold", step=0.5)
                boundary = gr.Slider(0, 100, value=0, label="boundary points num", step=1)
                points_num = gr.Textbox(interactive=True, value=0, label="points num")
                button_filter = gr.Button("Confidence Filter")

            with gr.Column():
                data_yaml = gr.Textbox(  # noqa: F841
                    interactive=True, value=event.args.data, label="data yaml"
                )
                model_config = gr.Textbox(  # noqa: F841
                    interactive=True, value=event.args.config, label="model config"
                )
                model_weight = gr.Textbox(  # noqa: F841
                    interactive=True, value=event.args.load_from, label="model weight"
                )

        with gr.Row():
            stream_button = gr.Button("Viewer Display")
            stream_reconstruct_start = gr.Textbox(
                interactive=True, value=0, label="Viewer Reconstruct Start"
            )
            stream_reconstruct_end = gr.Textbox(
                interactive=True, value=0, label="Viewer Reconstruct End"
            )
            stream_reconstruct_voxel = gr.Slider(
                0.00, 0.1, value=0.04, label="Viewer Voxel Size", step=0.01
            )

        with gr.Row():
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
            )

        button_run.click(
            event.process,
            inputs=[dataset_index, frame_index, conf_thresh, boundary],
            outputs=[
                image,
                image_depth,
                image_confidence,
                model3d_prompt,
                model3d_pred,
                frame_index,
                model3d_pred_filter,
                points_num,
            ],
        )
        button_infer.click(
            event.vis_frame,
            inputs=[dataset_index, frame_index, conf_thresh, boundary],
            outputs=[
                image,
                image_depth,
                image_confidence,
                model3d_prompt,
                model3d_pred,
                frame_index,
                model3d_pred_filter,
                points_num,
            ],
        )
        button_next.click(
            event.vis_next_frame,
            inputs=[dataset_index, frame_index, conf_thresh, boundary],
            outputs=[
                image,
                image_depth,
                image_confidence,
                model3d_prompt,
                model3d_pred,
                frame_index,
                model3d_pred_filter,
                points_num,
            ],
        )
        button_reconstruct.click(
            event.reconstruct,
            inputs=[dataset_index, conf_thresh, boundary],
            outputs=[
                model3d_prompt,
                model3d_pred,
                frame_index,
                model3d_pred_filter,
                points_num,
            ],
        )
        button_filter.click(
            event.confidence_filter,
            inputs=[dataset_index, frame_index, conf_thresh, boundary],
            outputs=[model3d_pred_filter, points_num],
        )

        def streaming_mesh(
            dataset_index: int,
            conf_thresh: float,
            boundary: int,
            reconstruct_start: int,
            reconstruct_end: int,
            stream_reconstruct_voxel: float,
        ):
            dataset_index = int(dataset_index)
            conf_thresh = float(conf_thresh)
            boundary = int(boundary)
            reconstruct_start = int(reconstruct_start)
            reconstruct_end = int(reconstruct_end)
            stream_reconstruct_voxel = float(stream_reconstruct_voxel)

            if dataset_index >= len(event.trainer.vis_datasets):
                return

            stream = rr.binary_stream()

            mf_view_ids = event.trainer.vis_datasets[dataset_index].mf_view_ids
            if mf_view_ids is None:
                mf_view_ids = []

            if dataset_index != event.dataset_index:
                event.process(
                    dataset_index=dataset_index,
                    frame_index=-1,
                    conf_thresh=conf_thresh,
                    boundary=boundary,
                )

            pointmap_list = event.pointmap_list
            pointmap_color_list = event.pointmap_color_list
            depth_show_list = event.depth_show_list
            confidence_show_list = event.confidence_show_list
            confidence_mask_list = event.confidence_mask_list
            extrinsics_list = event.extrinsics_list
            intrinsics_list = event.intrinsics_list

            reconstruct_end = min(reconstruct_end, len(pointmap_list) - 1)

            if reconstruct_end - reconstruct_start > 0:
                pointmap_list = pointmap_list[reconstruct_start : reconstruct_end + 1]
                pointmap_color_list = pointmap_color_list[reconstruct_start : reconstruct_end + 1]
                depth_show_list = depth_show_list[reconstruct_start : reconstruct_end + 1]
                confidence_show_list = confidence_show_list[reconstruct_start : reconstruct_end + 1]
                confidence_mask_list = confidence_mask_list[reconstruct_start : reconstruct_end + 1]
                extrinsics_list = extrinsics_list[reconstruct_start : reconstruct_end + 1]
                intrinsics_list = intrinsics_list[reconstruct_start : reconstruct_end + 1]

            reconstruct = (reconstruct_end - reconstruct_start > 0) or (len(mf_view_ids) > 1)
            fuser = None
            if reconstruct:
                blueprint = rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Spatial3DView(),
                        rrb.Vertical(
                            rrb.Spatial2DView(origin="app/camera/pinhole"),
                            rrb.Spatial2DView(origin="app/camera/pinhole/depth"),
                            rrb.Spatial2DView(origin="app/camera/confidence"),
                        ),
                        column_shares=[20, 5],
                    ),
                    collapse_panels=False,
                )
                rr.send_blueprint(blueprint)

                if stream_reconstruct_voxel > 0:
                    fuser = Open3DFuser(fusion_resolution=stream_reconstruct_voxel)
            else:
                blueprint = rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Spatial3DView(),
                        rrb.Vertical(
                            rrb.Spatial2DView(origin="app/camera/image"),
                            rrb.Spatial2DView(origin="app/camera/depth"),
                            rrb.Spatial2DView(origin="app/camera/confidence"),
                        ),
                        column_shares=[20, 5],
                    ),
                    collapse_panels=False,
                )
                rr.send_blueprint(blueprint)

            rotation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            frames_num = len(pointmap_list)
            if frames_num == 0:
                return

            for i in tqdm(range(frames_num + 2), total=frames_num + 2, desc="viewer stream"):
                rr.set_time_sequence("frame", i)

                if i < frames_num:
                    pointmap = pointmap_list[i]
                    pointmap_color = pointmap_color_list[i]
                    depth_color = depth_show_list[i]

                    if reconstruct:
                        intrinsics = intrinsics_list[i]
                        extrinsics = extrinsics_list[i]

                        extrinsics_inv = np.linalg.inv(extrinsics)
                        R = extrinsics_inv[:3, :3]
                        T = extrinsics_inv[:3, 3]
                        pointmap_global = np.dot(R, pointmap.T).T + T
                    else:
                        pointmap_global = pointmap

                    if len(confidence_show_list) > 0:
                        confidence_show = confidence_show_list[i]
                        rr.log("app/camera/confidence", rr.Image(confidence_show))

                    if len(confidence_mask_list) > 0:
                        mask = confidence_mask_list[i]
                        filter_pointmap = pointmap_global[mask]
                        filter_pointmap_color = pointmap_color[mask].astype(int)
                    else:
                        filter_pointmap = pointmap_global
                        filter_pointmap_color = pointmap_color.astype(int)

                    if reconstruct:
                        extrinsics = extrinsics @ np.linalg.inv(rotation_matrix)
                    pointmap = pointmap @ rotation_matrix[:3, :3].T
                    filter_pointmap = filter_pointmap @ rotation_matrix[:3, :3].T

                    height, width = depth_color.shape[:2]
                    depth_color = depth_color.astype(np.uint8)
                    pinhole_image = pointmap_color.reshape([height, width, 3]).astype(np.uint8)

                    if reconstruct:
                        rr.log(
                            "app/camera",
                            rr.Transform3D(
                                translation=extrinsics[:3, 3],
                                mat3x3=extrinsics[:3, :3],
                                from_parent=True,
                            ),
                            static=False,
                        )
                        rr.log(
                            f"app/camera/pinhole",
                            rr.Pinhole(
                                image_from_camera=intrinsics,
                                height=height,
                                width=width,
                                camera_xyz=getattr(
                                    rr.ViewCoordinates,
                                    "RDF",
                                ),
                                image_plane_distance=0.2,
                            ),
                            static=False,
                        )

                        rr.log(
                            f"app/camera/pinhole/image",
                            rr.Image(pinhole_image).compress(jpeg_quality=75),
                        )

                        depth_pred = pointmap[:, 1].reshape([height, width])
                        rr.log(
                            f"app/camera/pinhole/depth",
                            rr.DepthImage(depth_pred, meter=1.0, colormap="turbo"),
                        )

                        if fuser is not None:
                            if len(confidence_mask_list) > 0:
                                mask = confidence_mask_list[i].reshape([height, width])
                                depth_pred[~mask] = 0

                            fuser.fuse_frames(
                                depth_hw=depth_pred,
                                K_33=intrinsics,
                                # cam_T_world_44=np.linalg.inv(extrinsics),
                                cam_T_world_44=extrinsics,
                                rgb_hw3=pinhole_image,
                                fusion_max_depth=depth_pred.max(),
                            )
                            pred_mesh = fuser.get_mesh()
                            pred_mesh.compute_vertex_normals()
                            rr.log(
                                "app/mesh",
                                rr.Mesh3D(
                                    vertex_positions=pred_mesh.vertices,
                                    triangle_indices=pred_mesh.triangles,
                                    vertex_normals=pred_mesh.vertex_normals,
                                    vertex_colors=pred_mesh.vertex_colors,
                                ),
                            )
                    else:
                        rr.log(f"app/camera/image", rr.Image(pinhole_image))
                        rr.log(f"app/camera/depth", rr.Image(depth_color))
                else:
                    rr.log("app/camera", rr.Clear(recursive=True))

                if fuser is not None:
                    pred_mesh = fuser.get_mesh()
                    pred_mesh.compute_vertex_normals()
                    rr.log(
                        "app/mesh",
                        rr.Mesh3D(
                            vertex_positions=pred_mesh.vertices,
                            triangle_indices=pred_mesh.triangles,
                            vertex_normals=pred_mesh.vertex_normals,
                            vertex_colors=pred_mesh.vertex_colors,
                        ),
                    )
                else:
                    rr.log(
                        "app/pointmap",
                        rr.Points3D(
                            filter_pointmap,
                            colors=filter_pointmap_color,
                        ),
                    )

                yield stream.read()

        stream_event = stream_button.click(
            streaming_mesh,
            inputs=[
                dataset_index,
                conf_thresh,
                boundary,
                stream_reconstruct_start,
                stream_reconstruct_end,
                stream_reconstruct_voxel,
            ],
            outputs=[viewer],
        )

    return UI


def main():
    event = Event()
    UI = setupUI(event)
    # UI.launch(server_name=event.args.server_name, server_port=event.args.server_port, share=True)
    UI.launch(server_name=event.args.server_name, server_port=event.args.server_port)


if __name__ == "__main__":
    main()
