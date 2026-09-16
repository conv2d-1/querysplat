import json
import logging
import os

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh

from hAlgorithm.modules.models2.external.vggtx.utils.colmap import read_cameras_binary, read_images_binary, rename_colmap_recons_and_rescale_camera, write_cameras_text, write_images_text
from hAlgorithm.modules.models.vggt.ba.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
from hAlgorithm.modules.models.vggt.utils.geometry import unproject_depth_map_to_point_map
from hAlgorithm.modules.models.vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from hAlgorithm.modules.pipelines2.utils.points_filter import filtered_points_with_confidence
from hAlgorithm.modules.pipelines2.utils.visualize import (
    save_point_cloud,
)


def get_basename(path):
    tmp = os.path.basename(path).split(".")[:-1]
    return tmp[0] if len(tmp) == 1 else ".".join(tmp)


def save_mv_outputs(cfg, mv_outputs, meta_data, out_dir, output_meta_dict=None):
    save_output_ply = cfg["save_everything"]
    save_output_conf = cfg["save_output_conf"]
    save_output_only_local_glb = cfg["save_output_only_local_glb"]
    save_normal = cfg["save_normal"]
    save_normal_vis = cfg["save_normal_vis"]
    save_invalid_mask = cfg["save_invalid_mask"]
    save_name_match_rgb = cfg["save_name_match_rgb"]
    output_conf_ratio = cfg["output_conf_ratio"]
    output_match_input_res = cfg["output_match_input_res"]
    output_colmap_format = cfg["output_colmap_format"]
    output_colmap_gt = cfg.get("output_colmap_gt", False)

    if output_colmap_gt:
        save_colmap_gt(cfg, mv_outputs, meta_data, out_dir)

    if output_colmap_format:
        return save_colmap_outputs(cfg, mv_outputs, meta_data, out_dir)

    kosmo_scene_infos = meta_data.get("kosmo_scene_infos", None)
    kosmo_frames = meta_data.get("kosmo_frames", None)

    data_info = meta_data["data_info"][0]

    for i, outputs in enumerate(mv_outputs):
        out_dir = os.path.abspath(out_dir)
        scene = data_info[i].get("scene", None)
        rgb_path = data_info[i]["rgb"]
        rgb_name = get_basename(rgb_path)
        frame_id = int(data_info[i]["frame_id"])
        view_id = int(data_info[i]["view_id"])
        depth_scale = float(data_info[i]["depth_scale"])

        if scene is not None:
            cur_out_dir = os.path.join(out_dir, scene)
        else:
            cur_out_dir = out_dir
        os.makedirs(cur_out_dir, exist_ok=True)

        invalid_mask_path = None
        if save_invalid_mask and outputs.invalid_mask is not None:
            if save_name_match_rgb:
                invalid_mask_path = os.path.join(cur_out_dir, f"invalid_mask/camera_{view_id}", f"{rgb_name}.npy")
                os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
            else:
                invalid_mask_path = os.path.join(cur_out_dir, f"invalid_mask_{frame_id:06d}_{view_id:06d}.npy")
            invalid_mask_np = np.zeros_like(outputs.invalid_mask, dtype=np.uint8)
            invalid_mask_np[outputs.invalid_mask > 0] = 127
            invalid_mask_path = invalid_mask_path[:-4] + '.png'
            cv2.imwrite(invalid_mask_path, invalid_mask_np)

        # NOTE: local depth * depth_scale
        depth_save_path = None
        if not save_output_only_local_glb and outputs.depth_align is not None:
            if save_name_match_rgb:
                depth_save_path = os.path.join(cur_out_dir, f"depth/camera_{view_id}/{rgb_name}.npy")
                os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
            else:
                depth_save_path = os.path.join(cur_out_dir, f"depth_{frame_id:06d}_{view_id:06d}.npy")
            if outputs.invalid_mask is not None:
                outputs.depth_align *= (outputs.invalid_mask <= 0)
            np.save(depth_save_path, outputs.depth_align * depth_scale)
            logging.info(f"output save: {depth_save_path}")

        normal_save_path = None
        if save_normal and outputs.normal is not None:
            if save_name_match_rgb:
                normal_save_path = os.path.join(cur_out_dir, f"normal/camera_{view_id}/{rgb_name}.npy")
                os.makedirs(os.path.dirname(normal_save_path), exist_ok=True)
            else:
                normal_save_path = os.path.join(cur_out_dir, f"normal_{frame_id:06d}_{view_id:06d}.npy")
            if outputs.invalid_mask is not None:
                outputs.normal *= (outputs.invalid_mask <= 0)[..., None]
            np.save(normal_save_path, outputs.normal)

            if save_normal_vis:
                normal_colored = outputs.normal.copy() * [0.5, -0.5, -0.5] + 0.5
                normal_colored = (normal_colored.clip(0, 1) * 255).astype(np.uint8)
                if save_name_match_rgb:
                    save_path = os.path.join(cur_out_dir, f"normal_vis/camera_{view_id}/{rgb_name}.jpg")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                else:
                    save_path = os.path.join(cur_out_dir, f"normal_vis_{frame_id:06d}_{view_id:06d}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR))

        points_save_dir = cur_out_dir
        points_save_index = view_id

        points_save_path = glb_points_save_path = None
        if save_output_ply:
            pointmap_color = outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None
            if outputs.pointmap is not None:
                if save_name_match_rgb:
                    tmp_points_save_dir = os.path.join(points_save_dir, f"points/camera_{view_id}/")
                    os.makedirs(tmp_points_save_dir, exist_ok=True)
                    points_save_path = save_point_cloud(
                        outputs.pointmap.copy(),
                        pointmap_color,
                        tmp_points_save_dir,
                        rgb_name,
                        None,
                        info=False,
                    )
                else:
                    points_save_path = save_point_cloud(
                        outputs.pointmap.copy(),
                        pointmap_color,
                        points_save_dir,
                        f"points_{frame_id:06d}",
                        points_save_index,
                        info=False,
                    )

            filtered_point_save_path = None
            # if not save_output_only_local_glb and outputs.filtered_pointmap is not None:
            #     filtered_point_save_path = save_point_cloud(
            #         outputs.filtered_pointmap.copy(),
            #         (
            #             outputs.filtered_pointmap_color.copy()
            #             if outputs.filtered_pointmap_color is not None
            #             else None
            #         ),
            #         points_save_dir,
            #         f"filtered_point_{frame_id:06d}",
            #         points_save_index,
            #     )

        glb_points_save_path = filtered_glb_points_path = None
        if not save_output_only_local_glb and save_output_ply and outputs.glb_mv_pointmap is not None:
            glb_mv_pointmap = outputs.glb_mv_pointmap.copy().reshape(-1, 3)
            if save_name_match_rgb:
                tmp_points_save_dir = os.path.join(points_save_dir, f"glb_points/camera_{view_id}/")
                os.makedirs(tmp_points_save_dir, exist_ok=True)
                glb_points_save_path = save_point_cloud(
                    glb_mv_pointmap,
                    pointmap_color,
                    tmp_points_save_dir,
                    rgb_name,
                    None,
                    info=False,
                )
            else:
                glb_points_save_path = save_point_cloud(
                    glb_mv_pointmap,
                    pointmap_color,
                    points_save_dir,
                    f"glb_points_{frame_id:06d}",
                    points_save_index,
                    info=False,
                )

            # if outputs.glb_mv_confidence is not None:
            #     confidence = outputs.glb_mv_confidence.copy()
            #     filtered_points, filtered_colors = filtered_points_with_confidence(
            #         points=glb_mv_pointmap,
            #         confidence=confidence,
            #         h=outputs.pointmap_h,
            #         w=outputs.pointmap_w,
            #         colors=pointmap_color,
            #         output_conf_ratio=output_conf_ratio,
            #     )
            #     filtered_glb_points_path = save_point_cloud(
            #         filtered_points,
            #         filtered_colors,
            #         points_save_dir,
            #         f"filtered_glb_points_{frame_id:06d}",
            #         points_save_index,
            #         info=False,
            #     )

        intrinsics = outputs.intrinsics_pred
        extrinsics = outputs.extrinsics_pred
        intrinsics_save_path = extrinsics_save_path = local2glb_points_save_path = None
        if intrinsics is not None:
            # infer 时 intrinsics 对应 model input 尺寸
            # save output 时 intrinsics 对应输出的 depth 尺寸
            if output_match_input_res:
                h, w = outputs.depth_align.shape[:2]
                ratio_h, ratio_w = h / outputs.pointmap_h, w / outputs.pointmap_w
                intrinsics_align = intrinsics.copy()
                intrinsics_align[0, 0] = intrinsics_align[0, 0] * ratio_w
                intrinsics_align[1, 1] = intrinsics_align[1, 1] * ratio_h
                intrinsics_align[0, 2] = intrinsics_align[0, 2] * ratio_w
                intrinsics_align[1, 2] = intrinsics_align[1, 2] * ratio_h
                intrinsics = intrinsics_align

            if save_name_match_rgb:
                intrinsics_save_path = os.path.join(cur_out_dir, f"intrinsics/camera_{view_id}/{rgb_name}.json")
                os.makedirs(os.path.dirname(intrinsics_save_path), exist_ok=True)
            else:
                intrinsics_save_path = os.path.join(cur_out_dir, f"intrinsics_{frame_id:06d}_{view_id:06d}.json")
            with open(intrinsics_save_path, "w") as f:
                json.dump(intrinsics.tolist(), f, indent=2)

        local2glb_points_save_path = filtered_local2glb_points_path = None
        if extrinsics is not None:
            if save_name_match_rgb:
                tmp_extrinsics_save_dir = os.path.join(cur_out_dir,f"extrinsics/camera_{view_id}/")
                os.makedirs(tmp_extrinsics_save_dir, exist_ok=True)
                extrinsics_save_path = os.path.join(tmp_extrinsics_save_dir, f"{rgb_name}.json")
            else:
                extrinsics_save_path = os.path.join(cur_out_dir, f"extrinsics_{frame_id:06d}_{view_id:06d}.json")
            with open(extrinsics_save_path, "w") as f:
                json.dump(extrinsics.tolist(), f, indent=2)

            local2glb_pointmap = outputs.local2glb_pointmap
            if not save_output_only_local_glb and save_output_ply and local2glb_pointmap is not None:
                if save_name_match_rgb:
                    tmp_points_save_dir = os.path.join(points_save_dir, f"lcl2glb_points/camera_{view_id}/")
                    os.makedirs(tmp_points_save_dir, exist_ok=True)
                    local2glb_points_save_path = save_point_cloud(
                        local2glb_pointmap,
                        (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                        tmp_points_save_dir,
                        rgb_name,
                        None,
                        info=False,
                    )
                else:
                    local2glb_points_save_path = save_point_cloud(
                        local2glb_pointmap,
                        (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                        points_save_dir,
                        f"lcl2glb_points_{frame_id:06d}",
                        points_save_index,
                        info=False,
                    )

                # if outputs.confidence is not None:
                #     confidence = outputs.confidence.copy()
                #     filtered_points, filtered_colors = filtered_points_with_confidence(
                #         points=local2glb_pointmap,
                #         confidence=confidence,
                #         h=outputs.pointmap_h,
                #         w=outputs.pointmap_w,
                #         colors=pointmap_color,
                #         output_conf_ratio=output_conf_ratio,
                #     )
                #     filtered_local2glb_points_path = save_point_cloud(
                #         filtered_points,
                #         filtered_colors,
                #         points_save_dir,
                #         f"filtered_lcl2glb_points_{frame_id:06d}",
                #         points_save_index,
                #         info=False,
                #     )

        confidence_save_path = None
        if not save_output_only_local_glb and save_output_conf and outputs.confidence is not None:
            if save_name_match_rgb:
                confidence_save_path = os.path.join(cur_out_dir, f"conf/camera_{view_id}/{rgb_name}.npy")
                os.makedirs(os.path.dirname(confidence_save_path), exist_ok=True)
            else:
                confidence_save_path = os.path.join(cur_out_dir, f"conf_{frame_id:06d}_{view_id:06d}.npy")
            np.save(confidence_save_path, outputs.confidence)

        glb_confidence_save_path = None
        # if not save_output_only_local_glb and save_output_conf and outputs.glb_mv_confidence is not None:
        #     glb_confidence_save_path = os.path.join(
        #         cur_out_dir, f"glb_conf_{frame_id:06d}_{view_id:06d}.npy"
        #     )
        #     np.save(glb_confidence_save_path, outputs.glb_mv_confidence)

        if output_meta_dict is not None:
            info = dict(rgb=rgb_path, depth_scale=depth_scale)
            if depth_save_path is not None:
                info["pred_depth"] = depth_save_path
            if intrinsics_save_path is not None:
                info["pred_intrinsic"] = intrinsics_save_path
            if extrinsics_save_path is not None:
                info["pred_extrinsic"] = extrinsics_save_path
                info["world2cam_pose_mvfr"] = extrinsics.tolist()
                info["cam2world_pose_mvfr"] = np.linalg.inv(extrinsics).tolist()
            if points_save_path is not None:
                info["pred_points"] = points_save_path
            if confidence_save_path is not None:
                info["pred_confidence"] = confidence_save_path
            if glb_points_save_path is not None:
                info["pred_glb_points"] = glb_points_save_path
            if glb_confidence_save_path is not None:
                info["pred_glb_confidence"] = glb_confidence_save_path
            if filtered_glb_points_path is not None:
                info["pred_filtered_glb_points"] = filtered_glb_points_path
            if local2glb_points_save_path is not None:
                info["pred_local2glb_points"] = local2glb_points_save_path
            if filtered_local2glb_points_path is not None:
                info["pred_filtered_local2glb_points"] = filtered_local2glb_points_path
            if normal_save_path is not None:
                info["pred_normal"] = normal_save_path
            if invalid_mask_path is not None:
                info["invalid_mask"] = invalid_mask_path

            if kosmo_scene_infos is None:
                if "cam_in" in data_info[i]:
                    info["cam_in"] = data_info[i]["cam_in"]
                if "extrinsics" in data_info[i]:
                    info["extrinsics"] = data_info[i]["extrinsics"]
                if "depth" in data_info[i]:
                    info["depth"] = data_info[i]["depth"]
                if "lidar_depth" in data_info[i]:
                    info["lidar_depth"] = data_info[i]["lidar_depth"]
                if "confidence" in data_info[i]:
                    info["confidence"] = data_info[i]["confidence"]
                if "normal_svd" in data_info[i]:
                    info["normal_svd"] = data_info[i]["normal_svd"]
                if "normal_mask_svd" in data_info[i]:
                    info["normal_mask_svd"] = data_info[i]["normal_mask_svd"]

                if "mf_files" not in output_meta_dict:
                    output_meta_dict["mf_files"] = dict()
                if scene not in output_meta_dict["mf_files"]:
                    output_meta_dict["mf_files"][scene] = []
                frame_ids = [d["frame_id"] for d in output_meta_dict["mf_files"][scene]]
                if len(frame_ids) == 0:
                    index = None
                else:
                    if frame_id in set(frame_ids):
                        index = frame_ids.index(frame_id)
                    else:
                        index = None

                info["view_id"] = view_id
                if index is None:
                    output_meta_dict["mf_files"][scene].append(
                        dict(
                            frame_id=frame_id,
                            views=[info],
                        )
                    )
                else:
                    output_meta_dict["mf_files"][scene][index]["views"].append(info)
            else:
                if scene is None:
                    if "files" not in output_meta_dict:
                        output_meta_dict["files"] = list()
                    output_meta_dict["files"].append(info)
                else:
                    if "mf_files" not in output_meta_dict:
                        output_meta_dict["mf_files"] = dict()

                    if scene not in output_meta_dict["mf_files"]:
                        assert len(kosmo_scene_infos) == 1
                        output_meta_dict["mf_files"][scene] = kosmo_scene_infos[0]
                        output_meta_dict["mf_files"][scene]["frames"] = []

                    assert len(kosmo_frames) == 1
                    kosmo_frame = kosmo_frames[0][frame_id][view_id]
                    kosmo_frame.update(info)
                    output_meta_dict["mf_files"][scene]["frames"].append(kosmo_frame)
        else:
            raise NotImplementedError


def save_sv_outputs(cfg, mv_outputs, meta_data, out_dir, output_meta_dict=None):
    save_output_ply = cfg["save_everything"]
    save_output_conf = cfg["save_output_conf"]
    save_output_only_local_glb = cfg["save_output_only_local_glb"]
    save_normal = cfg["save_normal"]
    save_normal_vis = cfg["save_normal_vis"]
    save_name_match_rgb = cfg["save_name_match_rgb"]
    output_conf_ratio = cfg["output_conf_ratio"]
    output_match_input_res = cfg["output_match_input_res"]
    save_invalid_mask = cfg.get("save_invalid_mask", False)

    kosmo_scene_infos = meta_data.get("kosmo_scene_infos", None)
    kosmo_frames = meta_data.get("kosmo_frames", None)

    if not isinstance(mv_outputs, (list, tuple)):
        mv_outputs = [mv_outputs]
    data_info = meta_data["data_info"][0]

    for i, outputs in enumerate(mv_outputs):
        out_dir = os.path.abspath(out_dir)
        scene = data_info.get("scene", None)
        rgb_name = get_basename(data_info["rgb"])
        frame_id = int(data_info.get("frame_id", 0))
        view_id = int(data_info.get("view_id", meta_data["data_idx"]))
        depth_scale = data_info["depth_scale"]

        if scene is not None:
            cur_out_dir = os.path.join(out_dir, scene)
        else:
            cur_out_dir = out_dir
        os.makedirs(cur_out_dir, exist_ok=True)

        # NOTE: local depth * depth_scale
        depth_save_path = None
        if not save_output_only_local_glb and outputs.depth_align is not None:
            if save_name_match_rgb:
                depth_save_path = os.path.join(cur_out_dir, "depth", f"{rgb_name}.npy")
                os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
            else:
                depth_save_path = os.path.join(cur_out_dir, f"depth_{frame_id:06d}_{view_id:06d}.npy")
            np.save(depth_save_path, outputs.depth_align * depth_scale)
            logging.info(f"output save: {depth_save_path}")

        normal_save_path = None
        if save_normal and outputs.normal is not None:
            if save_name_match_rgb:
                normal_save_path = os.path.join(cur_out_dir, "normal", f"{rgb_name}.npy")
                os.makedirs(os.path.dirname(normal_save_path), exist_ok=True)
            else:
                normal_save_path = os.path.join(cur_out_dir, f"normal_{frame_id:06d}_{view_id:06d}.npy")
            np.save(normal_save_path, outputs.normal)

            if save_normal_vis:
                normal_colored = outputs.normal.copy() * [0.5, -0.5, -0.5] + 0.5
                normal_colored = (normal_colored.clip(0, 1) * 255).astype(np.uint8)
                if save_name_match_rgb:
                    save_path = os.path.join(cur_out_dir, "normal_vis", f"{rgb_name}.jpg")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                else:
                    save_path = os.path.join(cur_out_dir, f"normal_vis_{frame_id:06d}_{view_id:06d}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR))

        points_save_dir = cur_out_dir
        points_save_index = view_id

        points_save_path = glb_points_save_path = None
        if save_output_ply:
            pointmap_color = outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None
            if outputs.pointmap is not None:
                if save_name_match_rgb:
                    tmp_points_save_dir = os.path.join(points_save_dir, "points")
                    os.makedirs(tmp_points_save_dir, exist_ok=True)
                    points_save_path = save_point_cloud(
                        outputs.pointmap.copy(),
                        pointmap_color,
                        tmp_points_save_dir,
                        rgb_name,
                        None,
                        info=False,
                    )
                else:
                    points_save_path = save_point_cloud(
                        outputs.pointmap.copy(),
                        pointmap_color,
                        points_save_dir,
                        f"points_{frame_id:06d}",
                        points_save_index,
                        info=False,
                    )
        glb_points_save_path = filtered_glb_points_path = None
        intrinsics_save_path = extrinsics_save_path = local2glb_points_save_path = None
        local2glb_points_save_path = filtered_local2glb_points_path = None
        confidence_save_path = None
        if not save_output_only_local_glb and save_output_conf and outputs.confidence is not None:
            if save_name_match_rgb:
                confidence_save_path = os.path.join(cur_out_dir, "conf", f"{rgb_name}.npy")
                os.makedirs(os.path.dirname(confidence_save_path), exist_ok=True)
            else:
                confidence_save_path = os.path.join(cur_out_dir, f"conf_{frame_id:06d}_{view_id:06d}.npy")
            np.save(confidence_save_path, outputs.confidence)

        invalid_mask_path = None
        if save_invalid_mask and outputs.invalid_mask is not None:
            if save_name_match_rgb:
                invalid_mask_path = os.path.join(cur_out_dir, "invalid_mask", f"{rgb_name}.npy")
                os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
            else:
                invalid_mask_path = os.path.join(cur_out_dir, f"invalid_mask_{frame_id:06d}_{view_id:06d}.npy")
            invalid_mask_np = np.zeros_like(outputs.invalid_mask, dtype=np.uint8)
            invalid_mask_np[outputs.invalid_mask > 0] = 127
            invalid_mask_path = invalid_mask_path[:-4] + '.png'
            cv2.imwrite(invalid_mask_path, invalid_mask_np)
            # np.save(invalid_mask_path, (outputs.invalid_mask > 0).bool())

        glb_confidence_save_path = None

        if output_meta_dict is not None:
            info = dict(
                rgb=data_info["rgb"],
                depth_scale=depth_scale,
            )
            if depth_save_path is not None:
                info["pred_depth"] = depth_save_path
            if intrinsics_save_path is not None:
                info["pred_intrinsic"] = intrinsics_save_path
            if extrinsics_save_path is not None:
                info["pred_extrinsic"] = extrinsics_save_path
            if points_save_path is not None:
                info["pred_points"] = points_save_path
            if confidence_save_path is not None:
                info["pred_confidence"] = confidence_save_path
            if glb_points_save_path is not None:
                info["pred_glb_points"] = glb_points_save_path
            if glb_confidence_save_path is not None:
                info["pred_glb_confidence"] = glb_confidence_save_path
            if filtered_glb_points_path is not None:
                info["pred_filtered_glb_points"] = filtered_glb_points_path
            if local2glb_points_save_path is not None:
                info["pred_local2glb_points"] = local2glb_points_save_path
            if filtered_local2glb_points_path is not None:
                info["pred_filtered_local2glb_points"] = filtered_local2glb_points_path
            if normal_save_path is not None:
                info["pred_normal"] = normal_save_path
            if invalid_mask_path is not None:
                info["pred_invalid_mask"] = invalid_mask_path

            if kosmo_scene_infos is None:

                def update_info(key):
                    if key in data_info:
                        info[key] = data_info[key]

                update_info("cam_in")
                update_info("depth")
                update_info("lidar_depth")
                update_info("confidence")
                update_info("normal_svd")
                update_info("normal_mask_svd")
                update_info("extrinsics")

                if "scene" in data_info:
                    if "mf_files" not in output_meta_dict:
                        output_meta_dict["mf_files"] = dict()
                    if scene not in output_meta_dict["mf_files"]:
                        output_meta_dict["mf_files"][scene] = []
                    frame_ids = [d["frame_id"] for d in output_meta_dict["mf_files"][scene]]
                    if len(frame_ids) == 0:
                        index = None
                    else:
                        if frame_id in set(frame_ids):
                            index = frame_ids.index(frame_id)
                        else:
                            index = None

                    info["view_id"] = view_id

                    if index is None:
                        output_meta_dict["mf_files"][scene].append(
                            dict(
                                frame_id=frame_id,
                                views=[info],
                            )
                        )
                    else:
                        output_meta_dict["mf_files"][scene][index]["views"].append(info)
                else:
                    if "files" not in output_meta_dict:
                        output_meta_dict["files"] = list()
                    output_meta_dict["files"].append(info)
            else:
                if "mf_files" not in output_meta_dict:
                    output_meta_dict["mf_files"] = dict()

                if scene not in output_meta_dict["mf_files"]:
                    assert len(kosmo_scene_infos) == 1
                    output_meta_dict["mf_files"][scene] = kosmo_scene_infos[0]
                    output_meta_dict["mf_files"][scene]["frames"] = []

                assert len(kosmo_frames) == 1
                kosmo_frame = kosmo_frames[0][frame_id][view_id]
                kosmo_frame.update(info)
                output_meta_dict["mf_files"][scene]["frames"].append(kosmo_frame)


def save_colmap_outputs(cfg, mv_outputs, meta_data, out_dir):
    post_opt = cfg.get("output_post_optimizer", None)
    output_conf_ratio = cfg["output_conf_ratio"]
    max_points_for_colmap = cfg.get("max_points_for_colmap", 100000)
    shared_camera = cfg.get("colmap_shared_camera", False)

    camera_type = "PINHOLE"
    if post_opt is not None:
        post_opt_mode = post_opt["mode"]
        max_query_pts = post_opt.get("max_query_pts", 4096)

        if max_query_pts is None:
            max_query_pts = 4096 if len(mv_outputs) < 500 else 2048
    else:
        post_opt_mode = ""

    extrinsic = np.stack([outputs.extrinsics_pred[:3, :4] for outputs in mv_outputs], axis=0)
    intrinsic = np.stack([outputs.intrinsics_pred for outputs in mv_outputs], axis=0)
    depth_map = np.stack([outputs.depth_align for outputs in mv_outputs], axis=0)
    depth_conf = np.stack([outputs.confidence for outputs in mv_outputs], axis=0)
    images = np.stack([outputs.rgb for outputs in mv_outputs], axis=0)

    input_width = int(meta_data["input_width"][0])
    input_height = int(meta_data["input_height"][0])
    origin_width = int(meta_data["origin_width"][0])
    origin_height = int(meta_data["origin_height"][0])

    data_root = meta_data["data_root"][0]
    data_infos = meta_data["data_info"][0]
    image_path_list = [os.path.join(data_root, data_info["rgb"]) for data_info in data_infos]
    # base_image_path_list = [os.path.basename(rgb_path) for rgb_path in image_path_list]
    base_image_path_list = [f"{i:06}" for i, rgb_path in enumerate(image_path_list)]

    image_size = [depth_map.shape[2], depth_map.shape[1]]  # W H
    original_coords = np.array([[0, 0, input_width, input_height, origin_width, origin_height]])
    original_coords = np.tile(original_coords, (len(mv_outputs), 1))

    if post_opt_mode == "ba":
        raise NotImplementedError
        # from hAlgorithm.modules.models.vggt.ba.track_predict import predict_tracks

        # query_frame_num = post_opt.get("query_frame_num", 8)
        # fine_tracking = post_opt.get("fine_tracking", True)
        # max_reproj_error = post_opt.get("max_reproj_error", 8)
        # vis_thresh = post_opt.get("vis_thresh", 0.2)

        # scale = [origin_width[0] / input_width[0], origin_height[0] / input_height[0]]

        # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        # with torch.cuda.amp.autocast(dtype=dtype):
        #     # Predicting Tracks
        #     # Using VGGSfM tracker instead of VGGT tracker for efficiency
        #     # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
        #     # Will be fixed in VGGT v2

        #     # You can also change the pred_tracks to tracks from any other methods
        #     # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
        #     pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
        #         images,
        #         conf=depth_conf,
        #         points_3d=points_3d,
        #         masks=None,
        #         max_query_pts=max_query_pts,
        #         query_frame_num=query_frame_num,
        #         keypoint_extractor="aliked+sp",
        #         fine_tracking=fine_tracking,
        #     )

        #     torch.cuda.empty_cache()

        # # rescale the intrinsic matrix from 518 to 1024
        # intrinsic[:, 0, :] *= scale[0]
        # intrinsic[:, 1, :] *= scale[1]
        # track_mask = pred_vis_scores > vis_thresh

        # # TODO: radial distortion, iterative BA, masks
        # reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        #     points_3d,
        #     extrinsic,
        #     intrinsic,
        #     pred_tracks,
        #     image_size,
        #     masks=track_mask,
        #     max_reproj_error=max_reproj_error,
        #     shared_camera=shared_camera,
        #     camera_type=camera_type,
        #     points_rgb=points_rgb,
        # )

        # if reconstruction is None:
        #     raise ValueError("No reconstruction can be built with BA")

        # # Bundle Adjustment
        # ba_options = pycolmap.BundleAdjustmentOptions()
        # pycolmap.bundle_adjustment(reconstruction, ba_options)

        # reconstruction_resolution = [origin_width[0], origin_height[0]]

    elif post_opt_mode == "ga":
        from hAlgorithm.modules.models2.external.vggtx.utils.opt import extract_conf_mask, extract_matches, pose_optimization

        logging.info("Extracting matches for global alignment")

        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().cuda()

        match_outputs = extract_matches(extrinsic, intrinsic, images, depth_conf, base_image_path_list, max_query_pts)
        match_outputs["original_width"] = images.shape[-1]
        match_outputs["original_height"] = images.shape[-2]
        torch.save(match_outputs, os.path.join(out_dir, "matches.pt"))
        logging.info(f"Saved matches to {os.path.join(out_dir, 'matches.pt')}")

        extrinsic, intrinsic = pose_optimization(
            match_outputs,
            extrinsic,
            intrinsic,
            images,
            depth_map,
            depth_conf,
            base_image_path_list,
            target_scene_dir=out_dir,
            shared_intrinsics=shared_camera,
        )

        points_3d = unproject_depth_map_to_point_map(depth_map[..., None], extrinsic, intrinsic)
        num_frames, height, width, _ = points_3d.shape

        if images.shape[-1] != image_size[0] or images.shape[-2] != image_size[1]:
            points_rgb = F.interpolate(images, size=(image_size[-2], image_size[-1]), mode="bilinear", align_corners=False)
            points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
            points_rgb = points_rgb.transpose(0, 2, 3, 1)
        else:
            points_rgb = (images.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = extract_conf_mask(match_outputs, depth_conf, base_image_path_list)
        conf_thres_value = np.percentile(depth_conf.reshape(-1), output_conf_ratio * 100)
        conf_mask = conf_mask & (depth_conf >= conf_thres_value)
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        logging.info("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )
        reconstruction_resolution = image_size

    else:
        points_3d = unproject_depth_map_to_point_map(depth_map[..., None], extrinsic, intrinsic)
        num_frames, height, width, _ = points_3d.shape

        if images.shape[2] != image_size[0] or images.shape[1] != image_size[1]:
            points_rgb = F.interpolate(torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), size=(image_size[1], image_size[0]), mode="bilinear", align_corners=False)
            points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
            points_rgb = points_rgb.transpose(0, 2, 3, 1)
        else:
            points_rgb = (images * 255).astype(np.uint8)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_thres_value = np.percentile(depth_conf.reshape(-1), output_conf_ratio * 100)
        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        logging.info("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )
        reconstruction_resolution = image_size

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords,
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    sparse_reconstruction_dir = os.path.join(out_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    logging.info(f"Saving reconstruction to {sparse_reconstruction_dir}")

    reconstruction.write(sparse_reconstruction_dir)
    # Save for fast visualization
    cameras = read_cameras_binary(os.path.join(sparse_reconstruction_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_reconstruction_dir, "images.bin"))
    write_cameras_text(cameras, os.path.join(sparse_reconstruction_dir, "cameras.txt"))
    write_images_text(images, os.path.join(sparse_reconstruction_dir, "images.txt"))
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(out_dir, "sparse/0/points.ply"))

    images_dir = os.path.abspath(os.path.join(out_dir, "images"))
    os.makedirs(images_dir, exist_ok=True)
    logging.info(f"Symlink images to {images_dir}")

    for image_path, base_image_path in zip(image_path_list, base_image_path_list):
        os.symlink(os.path.abspath(image_path), os.path.join(images_dir, base_image_path))


def save_colmap_gt(cfg, mv_outputs, meta_data, out_dir):
    out_dir = out_dir + "_gt"
    shared_camera = cfg.get("colmap_shared_camera", False)

    camera_type = "PINHOLE"

    extrinsic = np.stack([outputs.extrinsics[:3, :4] for outputs in mv_outputs], axis=0)
    intrinsic = np.stack([outputs.intrinsics for outputs in mv_outputs], axis=0)

    origin_width = int(meta_data["origin_width"][0])
    origin_height = int(meta_data["origin_height"][0])

    data_root = meta_data["data_root"][0]
    data_infos = meta_data["data_info"][0]
    image_path_list = [os.path.join(data_root, data_info["rgb"]) for data_info in data_infos]
    # base_image_path_list = [os.path.basename(rgb_path) for rgb_path in image_path_list]
    base_image_path_list = [f"{i:06}" for i, rgb_path in enumerate(image_path_list)]

    image_size = [origin_width, origin_height]  # W H
    original_coords = np.array([[0, 0, origin_width, origin_height, origin_width, origin_height]])
    original_coords = np.tile(original_coords, (len(mv_outputs), 1))

    logging.info("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        None,
        None,
        None,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )
    reconstruction_resolution = image_size

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords,
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    sparse_reconstruction_dir = os.path.join(out_dir, "sparse/0")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    logging.info(f"Saving reconstruction to {sparse_reconstruction_dir}")

    reconstruction.write(sparse_reconstruction_dir)
    # Save for fast visualization
    cameras = read_cameras_binary(os.path.join(sparse_reconstruction_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_reconstruction_dir, "images.bin"))
    write_cameras_text(cameras, os.path.join(sparse_reconstruction_dir, "cameras.txt"))
    write_images_text(images, os.path.join(sparse_reconstruction_dir, "images.txt"))
