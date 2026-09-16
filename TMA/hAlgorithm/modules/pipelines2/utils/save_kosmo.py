import logging
import os

import cv2
import numpy as np


def get_basename(path):
    tmp = os.path.basename(path).split(".")[:-1]
    return tmp[0] if len(tmp) == 1 else ".".join(tmp)


def save_kosmo_outputs(cfg, mv_outputs, meta_data, out_dir, output_meta_dict=None):
    save_normal_vis = cfg.get("save_normal_vis", False)
    output_conf_ratio = cfg.get("output_conf_ratio", False)

    kosmo_scene_infos = meta_data["kosmo_scene_infos"]
    kosmo_frames = meta_data["kosmo_frames"]

    data_info = meta_data["data_info"]
    output_dtype = np.float16

    if not isinstance(mv_outputs, (list, tuple)):
        mv_outputs = [mv_outputs]

    for i, outputs in enumerate(mv_outputs):
        out_dir = os.path.abspath(out_dir)
        scene = data_info[0].get("scene", None)
        rgb_path = data_info[0]["rgb"]
        rgb_name = get_basename(rgb_path)
        frame_id = int(data_info[0]["frame_id"])
        view_id = int(data_info[0]["view_id"])
        depth_scale = float(data_info[0]["depth_scale"])

        if scene is not None:
            cur_out_dir = os.path.join(out_dir, scene)
        else:
            cur_out_dir = os.path.join(out_dir)
        os.makedirs(cur_out_dir, exist_ok=True)

        invalid_mask = None
        if outputs.invalid_mask is not None:
            invalid_mask = outputs.invalid_mask > 0
        
        # NOTE: local depth * depth_scale
        depth_save_path = None
        if outputs.depth_align is not None:
            depth_save_path = os.path.join(cur_out_dir, f"depth/camera_{view_id}/{rgb_name}.npy")
            os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
            depth = outputs.depth_align
            if invalid_mask is not None:
                depth[invalid_mask] = 0
            np.save(depth_save_path, (depth * depth_scale).astype(output_dtype))
            logging.info(f"output save: {depth_save_path}")

        normal_save_path = None
        if outputs.normal is not None:
            normal_save_path = os.path.join(cur_out_dir, f"normal/camera_{view_id}/{rgb_name}.npy")
            os.makedirs(os.path.dirname(normal_save_path), exist_ok=True)
            normal = outputs.normal
            if invalid_mask is not None:
                normal[invalid_mask] = 0
            np.save(normal_save_path, normal.astype(output_dtype))

            if save_normal_vis:
                normal_colored = normal.copy() * [0.5, -0.5, -0.5] + 0.5
                normal_colored = (normal_colored.clip(0, 1) * 255).astype(np.uint8)
                save_path = os.path.join(cur_out_dir, f"normal_vis/camera_{view_id}/{rgb_name}.jpg")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR))

        confidence_save_path = None
        if outputs.confidence is not None:
            confidence_save_path = os.path.join(cur_out_dir, f"conf/camera_{view_id}/{rgb_name}.npy")
            os.makedirs(os.path.dirname(confidence_save_path), exist_ok=True)
            np.save(confidence_save_path, outputs.confidence.astype(output_dtype))

        invalid_mask_path = None
        # if outputs.invalid_mask is not None:
        #     invalid_mask_path = os.path.join(cur_out_dir, "invalid_mask", f"{rgb_name}.png")
        #     os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
        #     invalid_mask_np = np.zeros_like(outputs.invalid_mask, dtype=np.uint8)
        #     invalid_mask_np[outputs.invalid_mask > 0] = 127
        #     cv2.imwrite(invalid_mask_path, invalid_mask_np)

        info = dict(rgb=rgb_path, depth_scale=depth_scale)
        if depth_save_path is not None:
            info["pred_depth"] = depth_save_path
        if confidence_save_path is not None:
            info["pred_confidence"] = confidence_save_path
        if normal_save_path is not None:
            info["pred_normal"] = normal_save_path
        if invalid_mask_path is not None:
            info["pred_invalid_mask"] = invalid_mask_path

        if "mf_files" not in output_meta_dict:
            output_meta_dict["mf_files"] = dict()

        if scene not in output_meta_dict["mf_files"]:
            assert len(kosmo_scene_infos) == 1
            output_meta_dict["mf_files"][scene] = kosmo_scene_infos[0]
            output_meta_dict["mf_files"][scene]["frames"] = []

        assert len(kosmo_frames) == 1
        kosmo_frame = kosmo_frames[0][frame_id][view_id]
        kosmo_frame.update(info)
        kosmo_frame.pop("version", None)
        output_meta_dict["mf_files"][scene]["frames"].append(kosmo_frame)