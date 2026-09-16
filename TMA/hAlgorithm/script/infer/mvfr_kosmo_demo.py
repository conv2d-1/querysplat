import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import torch
import numpy as np
import open3d as o3d

from hAlgorithm.utils import (
    config_merge_args,
    file2dict,
    instantiate_from_config,
    get_obj_from_str,
    parse_unknown,
)
from hAlgorithm.modules.pipelines2.utils.kosmo_util import get_kosmo_mask


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
        "--data",
        type=str,
        default=None,
        help="Kosmo json Path.",
    )
    parser.add_argument(
        "--cam",
        nargs='+',
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--main_cam",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--depth_name",
        type=str,
        default="lidar_depth",
        help="",
    )
    parser.add_argument(
        "--conf_name",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--pose_name",
        type=str,
        default="cam2world_pose",
    )
    parser.add_argument(
        "--conf_ratio",
        type=float,
        default=0.3,
        help="conf_ratio",
    )
    parser.add_argument(
        "--sv_conf_ratio",
        type=float,
        default=None,
        help="sv conf_ratio",
    )
    parser.add_argument(
        "--fake_sv",
        action="store_true",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=504,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--nums",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--no_mv_points",
        action="store_true",
    )
    parser.add_argument(
        "--sv_points",
        action="store_true",
    )
    parser.add_argument(
        "--no_sv_normal",
        action="store_true",
    )
    parser.add_argument(
        "--no_time",
        action="store_true",
        help="output dirname without time",
    )
    parser.add_argument(
        "--onnx_type",
        type=str,
        default=None,
        help="Type of onnx to run",
    )
    parser.add_argument(
        "--mask_mode",
        type=int,
        default=None,
        help="Kosmo mask mode, 20251231:6 view",
    )
    
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.no_time:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output_dir = os.path.join(args.output_dir, config_name)
    else:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output_dir = os.path.join(args.output_dir, config_name + "_" + now)
    os.makedirs(args.output_dir, exist_ok=True)

    if isinstance(args.cam, int):
        args.cam = [args.cam]

    if args.load_from in ["None", "none", "null"]:
        args.load_from = None

    return args, unknown_args



def main():
    args, unknown_args = parse_args()

    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    print(f"Loading kosmo_mask, {args.mask_mode}.")
    kosmo_mask = get_kosmo_mask(mask_mode=args.mask_mode)

    print("Loading model...")
    model = instantiate_from_config(cfg["model"]).cuda()
    model.eval()

    if args.load_from is not None:
        if args.load_from == "latest":
            args.load_from = os.path.join(os.path.dirname(args.config), "checkpoint/latest/ckpt.pth")
        elif args.load_from == "best":
            args.load_from = os.path.join(os.path.dirname(args.config), "checkpoint/best/ckpt.pth")

        model.load_checkpoint(ckpt_path=args.load_from)
    elif "trainer" in cfg and "load_from" in cfg["trainer"] and cfg["trainer"]["load_from"] is not None:
        model.load_checkpoint(ckpt_path=cfg["trainer"]["load_from"])
    
    with open(f'{args.output_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    trt_model = None
    if args.onnx_type is not None:
        trt_model_cfg = cfg.get("trt_model", {})
        print(f"Loading ONNX, args: {trt_model_cfg}")
        trt_model = get_obj_from_str(args.onnx_type)(**trt_model_cfg, model=model, use_fp16=True)

    print("Loading data...")
    with open(args.data, "r") as f:
        datas = json.load(f)["mf_files"]
        scene = sorted(list(datas.keys()))[0]
        datas = datas[scene]
    
    cam_params = datas["cam_params"]

    main_w2c_outputs = None
    main_pcd_path = None

    total_w2c_outputs = dict()
    total_pcd_path = []
    total_other_outputs = dict()
    for cam in args.cam:
        frame_ids = []
        view_ids = []
        img_list = []
        lidar_list = []
        conf_list = []
        invalid_mask_list = []
        extrinsics_list = []
        intrinsics_list = []

        for i, frame in enumerate(datas["frames"]):
            if not os.path.exists(frame["rgb"]):
                continue
            frame_id = frame["frame_id"]
            view_id = frame["view_id"]
            if view_id != cam:
                continue

            k = np.array(cam_params[f"cam{view_id}"]["K"])
            if len(k) > 4:
                continue

            intrinsics = np.eye(3)
            intrinsics[0, 0] = k[0]
            intrinsics[1, 1] = k[1]
            intrinsics[0, 2] = k[2]
            intrinsics[1, 2] = k[3]

            if args.pose_name not in frame:
                continue
            w2c = np.linalg.inv(np.array(frame[args.pose_name]))

            frame_ids.append(frame_id)
            view_ids.append(view_id)
            img_list.append(frame["rgb"])

            lidar_list.append(frame.get(args.depth_name, None))
            
            invalid_mask_list.append(frame.get("pred_invalid_mask_mvfr", None))
            # invalid_mask_list.append(None)
            
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(w2c)

            if args.conf_name not in [None, "None", "none"]:
                conf_list.append(frame[args.conf_name])
            else:
                conf_list.append(None)

            if args.nums is not None and len(frame_ids) >= args.nums:
                break

        # assert len(frame_ids) > 0
        if len(frame_ids) == 0:
            continue

        print(f"Model inference camera{cam}...")
        if args.fake_sv:
            if trt_model is not None:
                w2c_outputs, pcd_path, other_outputs = model.trt_inference_fake_sv(
                    trt_model=trt_model,
                    frame_ids=frame_ids[::args.step],
                    view_ids=cam,
                    img_list=img_list[::args.step],
                    lidar_list=lidar_list[::args.step],
                    conf_list=conf_list[::args.step],
                    invalid_mask_list=invalid_mask_list[::args.step],
                    intrinsics_list=intrinsics_list[::args.step],
                    extrinsics_list=extrinsics_list[::args.step],
                    chunk_size=args.chunk,
                    overlap=args.overlap,
                    process_res=args.process_res,
                    output_dir=args.output_dir,
                    save_points=args.sv_points,
                    save_normal=not args.no_sv_normal,
                    conf_ratio=args.sv_conf_ratio,
                )
            else:
                w2c_outputs, pcd_path, other_outputs = model.simple_inference_fake_sv(
                    frame_ids=frame_ids[::args.step],
                    view_ids=cam,
                    img_list=img_list[::args.step],
                    lidar_list=lidar_list[::args.step],
                    conf_list=conf_list[::args.step],
                    invalid_mask_list=invalid_mask_list[::args.step],
                    intrinsics_list=intrinsics_list[::args.step],
                    extrinsics_list=extrinsics_list[::args.step],
                    chunk_size=args.chunk,
                    overlap=args.overlap,
                    process_res=args.process_res,
                    output_dir=args.output_dir,
                    save_points=args.sv_points,
                    save_normal=not args.no_sv_normal,
                    conf_ratio=args.sv_conf_ratio,
                    kosmo_mask=kosmo_mask,
                )
        else:
            w2c_outputs, pcd_path, other_outputs = model.simple_inference(
                frame_ids=frame_ids[::args.step],
                view_ids=cam,
                img_list=img_list[::args.step],
                lidar_list=lidar_list[::args.step],
                conf_list=conf_list,
                invalid_mask_list=invalid_mask_list[::args.step],
                intrinsics_list=intrinsics_list[::args.step],
                extrinsics_list=extrinsics_list[::args.step],
                chunk_size=args.chunk,
                overlap=args.overlap,
                process_res=args.process_res,
                output_dir=args.output_dir,
                conf_ratio=args.conf_ratio,
                save_points=not args.no_mv_points,
            )

        if args.main_cam is not None and cam == args.main_cam:
            main_w2c_outputs = w2c_outputs
            main_pcd_path = pcd_path
        
        if w2c_outputs is not None:
            total_w2c_outputs.update(w2c_outputs)
        if pcd_path is not None:
            total_pcd_path.append(pcd_path)
        
        for key in other_outputs:
            if key not in total_other_outputs:
                total_other_outputs[key] = dict()
            total_other_outputs[key].update(other_outputs[key])
    
    # fix other cam pose
    if not args.fake_sv:
        for frame in datas["frames"]:
            frame_id = frame["frame_id"]
            view_id = frame["view_id"]

            if "depth" in total_other_outputs and frame["rgb"] in total_other_outputs["depth"]:
                frame["pred_depth_mvfr"] = total_other_outputs["depth"][frame["rgb"]]
            if "conf" in total_other_outputs and frame["rgb"] in total_other_outputs["conf"]:
                frame["pred_confidence_mvfr"] = total_other_outputs["conf"][frame["rgb"]]
            if "normal" in total_other_outputs and frame["rgb"] in total_other_outputs["normal"]:
                frame["pred_normal_mvfr"] = total_other_outputs["normal"][frame["rgb"]]
            if "invalid_mask" in total_other_outputs and frame["rgb"] in total_other_outputs["invalid_mask"]:
                frame["pred_invalid_mask_mvfr"] = total_other_outputs["invalid_mask"][frame["rgb"]]

            if args.main_cam is not None and view_id == args.main_cam:
                if frame["rgb"] in main_w2c_outputs:
                    frame["world2cam_pose_mvfr"] = main_w2c_outputs[frame["rgb"]].tolist()
                    frame["cam2world_pose_mvfr"] = np.linalg.inv(main_w2c_outputs[frame["rgb"]]).tolist()
            elif args.main_cam is not None:
                match_name = frame["rgb"].replace(f"/camera_{view_id}/", f"/camera_{args.main_cam}/")
                if match_name in main_w2c_outputs:
                    main_cam_w2c = main_w2c_outputs[match_name]
                    main_cam_c2w = np.linalg.inv(main_cam_w2c)
                    main_cam2cam0 = np.array(datas["extrinsics"][f"T_cam{args.main_cam}_2_cam0"])
                    cur_cam2cam0 = np.array(datas["extrinsics"][f"T_cam{view_id}_2_cam0"])
                    main_cam_2_cur_cam = np.linalg.inv(cur_cam2cam0) @ main_cam2cam0
                    cur_cam_c2w = np.array(main_cam_c2w) @ np.linalg.inv(main_cam_2_cur_cam)

                    frame["cam2world_pose_mvfr"] = cur_cam_c2w.tolist()
                    frame["world2cam_pose_mvfr"] = np.linalg.inv(cur_cam_c2w).tolist()
            else:
                if frame["rgb"] in total_w2c_outputs:
                    frame["world2cam_pose_mvfr"] = total_w2c_outputs[frame["rgb"]].tolist()
                    frame["cam2world_pose_mvfr"] = np.linalg.inv(total_w2c_outputs[frame["rgb"]]).tolist()
        
        # merge mvfr points and lidar points
        if not args.no_mv_points:
            if "mask_whole_map_colored" in datas:
                if args.main_cam is not None:
                    pcd = o3d.io.read_point_cloud(main_pcd_path)
                    lidar_pcd = o3d.io.read_point_cloud(datas["mask_whole_map_colored"])
                    pcd += lidar_pcd
                else:
                    pcd = o3d.io.read_point_cloud(datas["mask_whole_map_colored"])
                    for path in total_pcd_path:
                        if path is not None:
                            pcd += o3d.io.read_point_cloud(path)
                
                pcd_path = os.path.join(args.output_dir, "lidar_and_mvfr.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)

                datas["merged_colorized_pointcloud_mvfr"] = pcd_path
            else:
                if args.main_cam is not None:
                    pcd_path = main_pcd_path
                else:
                    pcd = o3d.io.read_point_cloud(total_pcd_path[0])
                    for path in total_pcd_path[1:]:
                        if path is not None:
                            pcd += o3d.io.read_point_cloud(path)
                
                    pcd_path = os.path.join(args.output_dir, "mvfr.ply")
                    o3d.io.write_point_cloud(pcd_path, pcd)

                datas["colorized_pointcloud_mvfr"] = pcd_path

        frames = datas.pop("frames")
        datas["frames"] = frames

    else:
        for frame in datas["frames"]:
            if "depth" in total_other_outputs and frame["rgb"] in total_other_outputs["depth"]:
                frame["pred_depth_mvfr"] = total_other_outputs["depth"][frame["rgb"]]
                frame["pred_depth"] = frame["pred_depth_mvfr"]
            if "conf" in total_other_outputs and frame["rgb"] in total_other_outputs["conf"]:
                frame["pred_confidence_mvfr"] = total_other_outputs["conf"][frame["rgb"]]
                frame["pred_confidence"] = frame["pred_confidence_mvfr"]
            if "normal" in total_other_outputs and frame["rgb"] in total_other_outputs["normal"]:
                frame["pred_normal_mvfr"] = total_other_outputs["normal"][frame["rgb"]]
                frame["pred_normal"] = frame["pred_normal_mvfr"]
            if "invalid_mask" in total_other_outputs and frame["rgb"] in total_other_outputs["invalid_mask"]:
                frame["pred_invalid_mask_mvfr"] = total_other_outputs["invalid_mask"][frame["rgb"]]
                frame["pred_invalid_mask"] = frame["pred_invalid_mask_mvfr"]

    new_path = os.path.join(args.output_dir, os.path.basename(args.data))
    assert new_path != args.data
    with open(new_path, "w") as f:
        json.dump({"mf_files": {scene: datas}}, f, indent=2, ensure_ascii=False)

    meta_path = args.data[:-5] + "_metadata.json"
    new_meta_path = new_path[:-5] + "_metadata.json"
    if os.path.exists(meta_path):
        os.system(f"cp {meta_path} {new_meta_path}")

if __name__ == "__main__":
    main()
