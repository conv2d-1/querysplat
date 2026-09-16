import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import torch
import numpy as np

from hAlgorithm.modules.pipelines2.utils.kosmo_util import trt_inference_fake_sv_chunk

from hAlgorithm.utils import (
    file2dict,
    instantiate_from_config,
    get_obj_from_str,
    parse_unknown,
)

from hAlgorithm.script.kosmo.utils import get_device_type

if get_device_type()=="4090":
    settings = dict(
        config="/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4.py",
        load_from="/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/checkpoint/latest/ckpt.pth",
        onnx="/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/vggt_normal.onnx",
        engine={
            "4090":"/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_4090.engine",
        },
        trt_model_type="hAlgorithm.script.kosmo.torch_trt.vggt_normal.VGGTNormalTRT",
        process_res=840,
        chunk=1,
    )
    
else:
    settings = dict(
        config="/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4.py",
        load_from="/mnt/netdata/Team/AI/SDK/Normal/MVFR/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/checkpoint/latest/ckpt.pth",
        onnx="/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/vggt_normal.onnx",
        engine={
            "A800":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_a800.engine",
            "A100":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_a100.engine",
            "3090":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_3090.engine",
            "H20":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_h20.engine",
        },
        trt_model_type="hAlgorithm.script.kosmo.torch_trt.vggt_normal.VGGTNormalTRT",
        process_res=840,
        chunk=1,
    )

class MVTrtInference(object):
    
    def __init__(self, args):
        
        if get_device_type()=="4090":
            settings = dict(
                    config="/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4.py",
                    load_from="/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/checkpoint/latest/ckpt.pth",
                    onnx="/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/vggt_normal.onnx",
                    engine={
                        "4090":"/mnt/cpfs/kosmo/model/mv/20260115/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_4090.engine",
                    },
                    trt_model_type="hAlgorithm.script.kosmo.torch_trt.vggt_normal.VGGTNormalTRT",
                    process_res=840,
                    chunk=20,
                )
                
        else:
            settings = dict(
                config="/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4.py",
                load_from="/mnt/netdata/Team/AI/SDK/Normal/MVFR/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/checkpoint/latest/ckpt.pth",
                onnx="/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/vggt_normal.onnx",
                engine={
                    "A800":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_a800.engine",
                    "A100":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_a100.engine",
                    "3090":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_3090.engine",
                    "H20":"/mnt/netdata/Team/AI/SDK/Normal/deploy/20251105/mvfr_rgb_normal_251028_bs1_1f4f_100k_vggtpre_trainall_840_nums4_20251031-101922/engine/vggt_normal_h20.engine",
                },
                trt_model_type="hAlgorithm.script.kosmo.torch_trt.vggt_normal.VGGTNormalTRT",
                process_res=840,
                chunk=20,
            )
            
        self.setting = settings
        self.args = args
        cfg = file2dict(settings["config"])
        print("Loading model...")
        self.model = instantiate_from_config(cfg["model"]).cuda()
        self.model.eval()
        
        self.model.load_checkpoint(ckpt_path=settings["load_from"])
        
        onnx = settings["onnx"]
        engine = settings["engine"].get(get_device_type(), None)
        trt_model_type = settings["trt_model_type"]
        assert engine is not None
        print(f"Loading ONNX, onnx={onnx}, engine={engine}")
        self.trt_model = get_obj_from_str(trt_model_type)(onnx=onnx, engine=engine, model=self.model, use_fp16=True)
        
        if args.no_time:
            config_name = os.path.splitext(os.path.basename(settings["config"]))[0]
            self.args.mv_output_dir = os.path.join(self.args.mv_output_dir, config_name)
        else:
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config_name = os.path.splitext(os.path.basename(settings["config"]))[0]
            self.args.mv_output_dir = os.path.join(self.args.mv_output_dir, config_name + "_" + now)
        os.makedirs(self.args.mv_output_dir, exist_ok=True)
    
    def run_mv_inference(self,cam,frame_ids, img_list, intrinsics_list, extrinsics_list, images=None):
        
        w2c_outputs, pcd_path, other_outputs = trt_inference_fake_sv_chunk(
            self=self.model,
            trt_model=self.trt_model,
            frame_ids=frame_ids[::self.args.mv_step],
            view_ids=cam,
            img_list=img_list[::self.args.mv_step],
            lidar_list=None,
            conf_list=None,
            invalid_mask_list=None,
            intrinsics_list=intrinsics_list[::self.args.mv_step],
            extrinsics_list=extrinsics_list[::self.args.mv_step],
            chunk_size=self.setting["chunk"],
            overlap=self.args.mv_overlap,
            process_res=self.setting["process_res"],
            output_dir=self.args.mv_output_dir,
            save_points=self.args.sv_points,
            save_normal=not self.args.no_sv_normal,
            conf_ratio=self.args.mv_conf_ratio,
            images_list=images[::self.args.mv_step] if images is not None else None,
        )

        return w2c_outputs, pcd_path, other_outputs

    def save_json_res(self, datas, total_other_outputs, seq_name):
        
            ## update mv info
        for frame in datas["frames"]:
            if "depth" in total_other_outputs and frame["rgb"] in total_other_outputs["depth"]:
                frame["pred_depth_mvfr"] = total_other_outputs["depth"][frame["rgb"]]
            if "conf" in total_other_outputs and frame["rgb"] in total_other_outputs["conf"]:
                frame["pred_confidence_mvfr"] = total_other_outputs["conf"][frame["rgb"]]
            if "normal" in total_other_outputs and frame["rgb"] in total_other_outputs["normal"]:
                frame["pred_normal_mvfr"] = total_other_outputs["normal"][frame["rgb"]]
                frame["pred_normal"] = frame["pred_normal_mvfr"]
            if "invalid_mask" in total_other_outputs and frame["rgb"] in total_other_outputs["invalid_mask"]:
                frame["pred_invalid_mask_mvfr"] = total_other_outputs["invalid_mask"][frame["rgb"]]
        
        new_path = os.path.join(self.args.mv_output_dir, os.path.basename(self.args.json_path))
        assert new_path != self.args.json_path
        with open(new_path, "w") as f:
            json.dump({"mf_files": {seq_name: datas}}, f, indent=2, ensure_ascii=False)

        meta_path = self.args.json_path[:-5] + "_metadata.json"
        new_meta_path = new_path[:-5] + "_metadata.json"
        if os.path.exists(meta_path):
            os.system(f"cp {meta_path} {new_meta_path}")
        
    def release_mv_model(self):
        
        if self.model is not None:
            if hasattr(self.model, "model"):
                del self.model.model
            del self.model
            self.model = None
            
        if self.trt_model is not None:
            del self.trt_model
            self.trt_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("mv model released.")
        

def parse_args():
    parser = argparse.ArgumentParser(description="Train your cute model!")
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
        "--conf_ratio",
        type=float,
        default=0.3,
        help="sv conf_ratio",
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
    )
    
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    if args.no_time:
        config_name = os.path.splitext(os.path.basename(settings["config"]))[0]
        args.output_dir = os.path.join(args.output_dir, config_name)
    else:
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        config_name = os.path.splitext(os.path.basename(settings["config"]))[0]
        args.output_dir = os.path.join(args.output_dir, config_name + "_" + now)
    os.makedirs(args.output_dir, exist_ok=True)

    if isinstance(args.cam, int):
        args.cam = [args.cam]

    return args, unknown_args


def main():
    args, unknown_args = parse_args()
    cfg = file2dict(settings["config"])
    print("Loading model...")
    model = instantiate_from_config(cfg["model"]).cuda()
    model.eval()
    
    model.load_checkpoint(ckpt_path=settings["load_from"])
    
    onnx = settings["onnx"]
    engine = settings["engine"].get(get_device_type(), None)
    trt_model_type = settings["trt_model_type"]
    assert engine is not None
    print(f"Loading ONNX, onnx={onnx}, engine={engine}")
    trt_model = get_obj_from_str(trt_model_type)(onnx=onnx, engine=engine, model=model, use_fp16=True)
    
    
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

            w2c = np.linalg.inv(np.array(frame["cam2world_pose"]))

            frame_ids.append(frame_id)
            view_ids.append(view_id)
            img_list.append(frame["rgb"])

            lidar_list.append(frame.get(args.depth_name, None))
            
            invalid_mask_list.append(frame.get("pred_invalid_mask_mvfr", None))
            # invalid_mask_list.append(None)
            
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(w2c)

            if args.conf_name is not None:
                conf_list.append(frame[args.conf_name])
            else:
                conf_list.append(None)

            if args.nums is not None and len(frame_ids) >= args.nums:
                break

        assert len(frame_ids) > 0
    
        from hAlgorithm.modules.pipelines2.utils.kosmo_util import trt_inference_fake_sv
        
        w2c_outputs, pcd_path, other_outputs = trt_inference_fake_sv(
            self=model,
            trt_model=trt_model,
            frame_ids=frame_ids[::args.step],
            view_ids=cam,
            img_list=img_list[::args.step],
            lidar_list=None,
            conf_list=None,
            invalid_mask_list=None,
            intrinsics_list=intrinsics_list[::args.step],
            extrinsics_list=extrinsics_list[::args.step],
            chunk_size=settings["chunk"],
            overlap=args.overlap,
            process_res=settings["process_res"],
            output_dir=args.output_dir,
            save_points=args.sv_points,
            save_normal=not args.no_sv_normal,
            conf_ratio=args.conf_ratio,
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
        
    for frame in datas["frames"]:
        if "depth" in total_other_outputs and frame["rgb"] in total_other_outputs["depth"]:
            frame["pred_depth_mvfr"] = total_other_outputs["depth"][frame["rgb"]]
        if "conf" in total_other_outputs and frame["rgb"] in total_other_outputs["conf"]:
            frame["pred_confidence_mvfr"] = total_other_outputs["conf"][frame["rgb"]]
        if "normal" in total_other_outputs and frame["rgb"] in total_other_outputs["normal"]:
            frame["pred_normal_mvfr"] = total_other_outputs["normal"][frame["rgb"]]
            frame["pred_normal"] = frame["pred_normal_mvfr"]
        if "invalid_mask" in total_other_outputs and frame["rgb"] in total_other_outputs["invalid_mask"]:
            frame["pred_invalid_mask_mvfr"] = total_other_outputs["invalid_mask"][frame["rgb"]]
    
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