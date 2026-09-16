import logging
import os
import sys

sys.path.append(os.getcwd())

import cv2
import numpy as np
import open3d as o3d

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from hAlgorithm.datasets.transforms.transforms import resize_depth_preserve
from hAlgorithm.modules.models.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from hAlgorithm.modules.pipelines2.utils.save_points import save_confident_pointcloud_batch
from hAlgorithm.modules.utils.parallel_utils import parallel_execution
from hAlgorithm.modules.utils.pose_align import align_poses_umeyama


def get_kosmo_mask(mask_mode=None, debug=False):
    if mask_mode is not None:
        if mask_mode == 20251231:
            camera_0_mask = np.ones((1024, 1024)).astype(bool)
            for i in range(510, 1024):
                for j in range(0, i - 510 + 1):
                    camera_0_mask[i, j] = 0

            camera_1_mask = np.ones((1024, 1024))
            for i in range(510, 1024):
                for j in range(1024 - (i - 510 + 1), 1024):
                    camera_1_mask[i, j] = 0

            camera_2_mask = np.ones((1024, 1024))
            for i in range(0, 510):
                for j in range(0, 510 - i + 2):
                    camera_2_mask[i, j] = 0

            camera_3_mask = np.ones((1024, 1024))
            for i in range(510, 1024):
                for j in range(0, i - 510 + 1):
                    camera_3_mask[i, j] = 0

            camera_4_mask = np.ones((1024, 1024))
            for i in range(510, 1024):
                for j in range(1024 - (i - 510 + 1), 1024):
                    camera_4_mask[i, j] = 0

            camera_5_mask = np.ones((1024, 1024))
            for i in range(0, 510):
                for j in range(0, 510 - i + 2):
                    camera_5_mask[i, j] = 0

            if debug:
                image = cv2.imread("/mnt/nasTeam/Kosmo/processed_data/kosmo/20260104_5_part2/0744_新华路建筑3/split_images/camera_0/1767511937.940524.png")
                image[~camera_0_mask[:, :].astype(bool)] = 255
                cv2.imwrite("camera_0.png", image)

                image = cv2.imread("/mnt/nasTeam/Kosmo/processed_data/kosmo/20260104_5_part2/0744_新华路建筑3/split_images/camera_1/1767511937.940524.png")
                image[~camera_1_mask[:, :].astype(bool)] = 255
                cv2.imwrite("camera_1.png", image)

                image = cv2.imread("/mnt/nasTeam/Kosmo/processed_data/kosmo/20260104_5_part2/0744_新华路建筑3/split_images/camera_2/1767511937.940524.png")
                image[~camera_2_mask[:, :].astype(bool)] = 255
                cv2.imwrite("camera_2.png", image)

                image = cv2.imread("/mnt/nasTeam/Kosmo/processed_data/kosmo/20260104_5_part2/0744_新华路建筑3/split_images/camera_3/1767511937.940524.png")
                image[~camera_3_mask[:, :].astype(bool)] = 255
                cv2.imwrite("camera_3.png", image)

                image = cv2.imread("/mnt/nasTeam/Kosmo/processed_data/kosmo/20260104_5_part2/0744_新华路建筑3/split_images/camera_4/1767511937.940524.png")
                image[~camera_4_mask[:, :].astype(bool)] = 255
                cv2.imwrite("camera_4.png", image)

                image = cv2.imread("/mnt/nasTeam/Kosmo/processed_data/kosmo/20260104_5_part2/0744_新华路建筑3/split_images/camera_5/1767511937.940524.png")
                image[~camera_5_mask[:, :].astype(bool)] = 255
                cv2.imwrite("camera_5.png", image)

                breakpoint()

        return {0: camera_0_mask, 1: camera_1_mask, 2: camera_2_mask, 3: camera_3_mask, 4: camera_4_mask, 5: camera_5_mask}
    else:
        return None

def get_images_base_name(img_name):
    return os.path.splitext(os.path.basename(img_name))[0]

def rotation_matrix_to_quaternion(R):
    # Convert rotation matrix to quaternion (w, x, y, z)
    # Using method from COLMAP (ensuring positive w)
    R = R[:3, :3]
    q = np.empty(4)
    t = np.trace(R)
    if t > 0.0:
        t = np.sqrt(t + 1.0)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        j = (i + 1) % 3
        k = (i + 2) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[j, i] + R[i, j]) * t
        q[k + 1] = (R[k, i] + R[i, k]) * t
    # Ensure w >= 0 (COLMAP convention)
    if q[0] < 0:
        q = -q
    return q


def _resize_longest_side(img: Image.Image, target_size: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest == target_size:
        return img
    scale = target_size / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if new_w % 14 != 0:
        new_w = int(new_w / 14) * 14
    if new_h % 14 != 0:
        new_h = int(new_h / 14) * 14

    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
    return Image.fromarray(arr)


def _resize_ixt(
    intrinsic: np.ndarray | None,
    orig_w: int,
    orig_h: int,
    w: int,
    h: int,
):
    if intrinsic is None:
        return None
    K = intrinsic.copy()
    # scale fx, cx by w ratio; fy, cy by h ratio
    K[:1] *= w / float(orig_w)
    K[1:2] *= h / float(orig_h)
    return K


def _normalize_image(img):
    img_tensor = torch.from_numpy(np.asarray(img).copy())
    mean = torch.tensor([127.5, 127.5, 127.5])
    std = torch.tensor([127.5, 127.5, 127.5])
    img_tensor = torch.div((img_tensor - mean[None, None]), std[None, None])
    img_tensor = img_tensor.permute(2, 0, 1).contiguous()
    return img_tensor


def accumulate_sim3_transforms(transforms):
    """
    Accumulate adjacent SIM(3) transforms into transforms from the initial frame to each subsequent frame.

    Args:
    transforms: list, each element is a tuple (R, s, t)
        R: 3x3 rotation matrix (np.array)
        s: scale factor (scalar)
        t: 3x1 translation vector (np.array)

    Returns:
    Cumulative transforms list, each element is (R_cum, s_cum, t_cum)
        representing the transform from frame 0 to frame k
    """
    if not transforms:
        return []

    cumulative_transforms = [transforms[0]]

    for i in range(1, len(transforms)):
        s_cum_prev, R_cum_prev, t_cum_prev = cumulative_transforms[i - 1]
        s_next, R_next, t_next = transforms[i]
        R_cum_new = R_cum_prev @ R_next
        s_cum_new = s_cum_prev * s_next
        t_cum_new = s_cum_prev * (R_cum_prev @ t_next) + t_cum_prev
        cumulative_transforms.append((s_cum_new, R_cum_new, t_cum_new))

    return cumulative_transforms


def apply_sim3_direct(point_maps, s, R, t):
    # point_maps: (b, h, w, 3) -> (b, h, w, 3, 1)
    point_maps_expanded = point_maps[..., np.newaxis]  # (b, h, w, 3, 1)

    # R: (3, 3) -> (b, h, w, 3, 1) = (3, 3) @ (3, 1)
    rotated = np.matmul(R, point_maps_expanded)  # (b, h, w, 3, 1)
    rotated = rotated.squeeze(-1)  # (b, h, w, 3)
    transformed = s * rotated + t  # (b, h, w, 3)

    return transformed


@torch.inference_mode()
def simple_inference(
    self,
    frame_ids,
    view_ids,
    img_list,
    lidar_list=None,
    conf_list=None,
    invalid_mask_list=None,
    extrinsics_list=None,
    intrinsics_list=None,
    chunk_size=None,
    overlap=0,
    process_res=504,
    output_dir=None,
    conf_ratio=0.2,
    save_points=True,
    **kwargs,
):

    w2c_outputs = dict()
    other_outputs = dict()
    ply_path_list = []

    print_progress = chunk_size is not None and chunk_size > 500

    # step1: chunk
    if chunk_size is None:
        num_chunks = 1
        chunk_indices = [(0, len(img_list))]
    else:
        if overlap >= chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
        if len(img_list) <= chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            step = chunk_size - overlap
            num_chunks = (len(img_list) - overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + chunk_size, len(img_list))

                if i > 0 and i == num_chunks - 1:
                    cur_chunk_size = end_idx - start_idx
                    if cur_chunk_size < chunk_size // 2:
                        start_idx = end_idx - chunk_size // 2

                chunk_indices.append((start_idx, end_idx))

    print(f"[MVFRPipeline] Processing {len(img_list)} images in {num_chunks} chunks of size {chunk_size} with {overlap} overlap")
    print(f"[MVFRPipeline] output_dir: {output_dir}")

    for chunk_idx in tqdm(range(len(chunk_indices))):
        start_idx, end_idx = chunk_indices[chunk_idx]
        chunk_image_paths = img_list[start_idx:end_idx]
        chunk_view_ids = view_ids[start_idx:end_idx] if isinstance(view_ids, (list, tuple)) else view_ids
        chunk_invalid_mask_paths = invalid_mask_list[start_idx:end_idx] if invalid_mask_list is not None else None
        chunk_extrinsics = extrinsics_list[start_idx:end_idx] if extrinsics_list is not None else None
        chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None
        chunk_lidar_paths = lidar_list[start_idx:end_idx] if lidar_list is not None else None
        chunk_conf_paths = conf_list[start_idx:end_idx] if conf_list is not None else None

        def process_one(idx):
            image_path = chunk_image_paths[idx]
            invalid_mask = chunk_invalid_mask_paths[idx] if chunk_invalid_mask_paths is not None else None
            extrinsic = chunk_extrinsics[idx] if chunk_extrinsics is not None else None
            intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None
            lidar_path = chunk_lidar_paths[idx] if chunk_lidar_paths is not None else None
            conf_path = chunk_conf_paths[idx] if chunk_conf_paths is not None else None

            pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            # Boundary resize
            pil_img = _resize_longest_side(pil_img, process_res)
            w, h = pil_img.size
            intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

            # Convert to tensor & normalize
            img_tensor = _normalize_image(pil_img)
            _, H, W = img_tensor.shape

            assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

            if invalid_mask is not None and invalid_mask[0] is not None:
                invalid_mask = np.load(invalid_mask).astype(np.float32)
                invalid_mask = cv2.resize(
                    invalid_mask,
                    dsize=(W, H),
                    interpolation=cv2.INTER_LINEAR,
                )
                invalid_mask = invalid_mask > 0

            img_show = np.asarray(pil_img)

            if lidar_path is not None:
                lidar_depth = np.load(lidar_path)

                if conf_path is not None:
                    conf = np.load(conf_path)
                    conf_threshold = np.percentile(conf.reshape(-1), int(conf_ratio * 100))
                    mask = conf > conf_threshold
                    lidar_depth *= mask

                lidar_depth = resize_depth_preserve(lidar_depth, img_tensor.shape[-2:])
            else:
                lidar_depth = None

            return img_show, img_tensor, intrinsic, extrinsic, invalid_mask, lidar_depth

        # step2: process
        outputs = parallel_execution(
            list(range(len(chunk_image_paths))),
            action=process_one,
            num_processes=8,
            print_progress=print_progress,
            sequential=False,
            desc=f"read chunk {chunk_idx}",
        )
        images_show, images, intrinsics, extrinsics, invalid_masks, lidar_depths = zip(*outputs)

        if chunk_intrinsics is None:
            intrinsics = None
        if chunk_extrinsics is None:
            extrinsics = None
        if chunk_invalid_mask_paths is None:
            invalid_masks = None
        if chunk_lidar_paths is None:
            lidar_depths = None

        images_show = np.stack(images_show, axis=0)
        images = torch.stack(images).unsqueeze(0).float()
        extrinsics = np.asarray(extrinsics)[None].astype(np.float32) if extrinsics is not None and extrinsics[0] is not None else None
        intrinsics = np.asarray(intrinsics)[None].astype(np.float32) if intrinsics is not None and intrinsics[0] is not None else None

        if lidar_depths is not None:
            lidar_depths = np.stack(lidar_depths)[None].astype(np.float32)
            lidar_depths_mask = (lidar_depths > 0.001) & (lidar_depths < 200)
            lidar_depths = self.depth_to_points(torch.from_numpy(lidar_depths).unsqueeze(2), K=torch.from_numpy(intrinsics), device="cpu", cache=False)

        if extrinsics is not None:
            ori_extrinsics = extrinsics.copy()
            w2c = extrinsics
            base_c2w_pred = np.linalg.inv(w2c[:, 0:1])
            extrinsics = w2c @ base_c2w_pred

        # step3: scale
        scale = 1.0
        if lidar_depths is not None:
            b, n = extrinsics.shape[:2]
            h, w = lidar_depths.shape[-2:]
            lidar_depths_h = np.concatenate([lidar_depths.reshape(b, n, 3, -1), np.ones([b, n, 1, h * w])], axis=2).transpose(0, 1, 3, 2)

            world_lidar_depths = np.einsum("bnij,bnkj->bnki", np.linalg.inv(extrinsics), lidar_depths_h)[..., :3]

            world_lidar_depths = world_lidar_depths[lidar_depths_mask.reshape(b, n, -1)]
            scale = np.mean(np.linalg.norm(world_lidar_depths, axis=-1)).astype(np.float32)

        # step4: inputs to cuda
        images = images.cuda()
        if intrinsics is not None:
            intrinsics = torch.from_numpy(intrinsics).float().cuda()

        if lidar_depths is not None:
            lidar_depths /= scale
            lidar_depths_mask = torch.from_numpy(lidar_depths_mask).float().unsqueeze(2)
            lidar_depths = torch.cat([lidar_depths, lidar_depths_mask], dim=-3)
            lidar_depths = lidar_depths.float().cuda()

        if extrinsics is not None:
            extrinsics[:, :, :3, 3] /= scale
            extrinsics = torch.from_numpy(extrinsics).float().cuda()

        if lidar_depths is not None:
            scale = torch.tensor([scale])  # .cuda()

        meta_data = dict(frames=[1], views=[images.shape[1]], input_width=[images.shape[-1]], input_height=[images.shape[-2]])
        meta_data["data_info"] = {"scene": [["kosmo"]]}

        # step4: model infer
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            results = self.model(
                images,
                scale=None,
                prompt_depth=lidar_depths,
                intrinsics=intrinsics,
                ray_directions=None,
                w2c=extrinsics,
                ray_world=None,
                query_points=None,
                meta_data=meta_data,
            )
            torch.cuda.empty_cache()

        depth = results["depth"].cpu() * scale
        confidence = results["confidence"].cpu()

        if "pose_enc" in results:
            pose_enc = results["pose_enc"][-1].cpu()

            if self.pose_encoding_type == "absT_quaR_FoV":
                pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc.float(),
                    image_size_hw=images.shape[-2:],
                    build_intrinsics=True,
                )
                pred_extrinsics[..., :3, 3] *= scale
                w2c_pred = pred_extrinsics
                base_c2w_pred = w2c_pred[:, 0:1].inverse()
                pred_extrinsics = w2c_pred @ base_c2w_pred
            else:
                raise NotImplementedError

        # step5: align
        if extrinsics is not None:
            _, _, scale, aligned_extrinsics = align_poses_umeyama(
                ori_extrinsics[0],
                pred_extrinsics.numpy()[0],
                ransac=False,
                return_aligned=True,
                random_state=42,
            )
        else:
            aligned_extrinsics = pred_extrinsics.numpy()[0]

        depth *= scale
        points = self.depth_to_points(depth, K=pred_intrinsics, device=depth.device)

        depth = depth[0, :, 0].contiguous().numpy()
        points = points[0].permute(0, 2, 3, 1).contiguous().numpy()

        confidence = confidence[0, :, 0].numpy()

        # step6: save
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            if save_points:
                points = points.reshape(points.shape[0], -1, 3)
                points = np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=-1)

                world_points = np.einsum("bij,bkj->bki", np.linalg.inv(aligned_extrinsics), points)[..., :3]
                if isinstance(view_ids, int):
                    ply_path = os.path.join(output_dir, f"pcd/camera_{view_ids}/{chunk_idx}_pcd.ply")
                else:
                    ply_path = os.path.join(output_dir, f"pcd/{chunk_idx}_pcd.ply")
                os.makedirs(os.path.dirname(ply_path), exist_ok=True)

                conf_threshold = np.percentile(confidence, 30)

                if invalid_masks is not None and invalid_masks[0] is not None:
                    invalid_masks = np.stack(invalid_masks, axis=0)
                    save_confident_pointcloud_batch(
                        points=world_points.reshape(-1, 3),  # shape: (H, W, 3)
                        colors=images_show.reshape(-1, 3),  # shape: (H, W, 3)
                        confs=confidence.reshape(-1),  # shape: (H, W)
                        output_path=ply_path,
                        conf_threshold=conf_threshold,
                        sample_ratio=0.01,
                        valid_mask=~invalid_masks.reshape(-1),
                    )
                else:
                    save_confident_pointcloud_batch(
                        points=world_points.reshape(-1, 3),  # shape: (H, W, 3)
                        colors=images_show.reshape(-1, 3),  # shape: (H, W, 3)
                        confs=confidence.reshape(-1),  # shape: (H, W)
                        output_path=ply_path,
                        conf_threshold=conf_threshold,
                        sample_ratio=0.01,
                    )
                ply_path_list.append(ply_path)

            def process_one(idx):
                image_basename = get_images_base_name(chunk_image_paths[idx])
                view_id = chunk_view_ids[idx] if isinstance(chunk_view_ids, (list, tuple)) else chunk_view_ids
                cur_depth = depth[idx]
                cur_conf = confidence[idx]

                depth_path = os.path.join(output_dir, f"depth/camera_{view_id}/{image_basename}.npy")
                if not os.path.exists(depth_path):
                    os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                    np.save(depth_path, cur_depth.astype(np.float16))

                conf_path = os.path.join(output_dir, f"conf/camera_{view_id}/{image_basename}.npy")
                if not os.path.exists(conf_path):
                    os.makedirs(os.path.dirname(conf_path), exist_ok=True)
                    np.save(conf_path, cur_conf.astype(np.float16))

                return depth_path, conf_path

            assert depth.shape[0] == len(chunk_image_paths)
            outputs = parallel_execution(
                list(range(depth.shape[0])),
                action=process_one,
                num_processes=8,
                print_progress=print_progress,
                sequential=False,
                desc="save chunk",
            )
            depth_paths, conf_paths = zip(*outputs)

            if "depth" not in other_outputs:
                other_outputs["depth"] = dict()
            if "conf" not in other_outputs:
                other_outputs["conf"] = dict()

            other_outputs["depth"].update(dict(zip(chunk_image_paths, depth_paths)))
            other_outputs["conf"].update(dict(zip(chunk_image_paths, conf_paths)))
            w2c_outputs.update({image_path: w2c for image_path, w2c in zip(chunk_image_paths, aligned_extrinsics)})

    aligned_extrinsics = np.stack([w2c_outputs[path] for path in img_list], axis=0)
    all_c2w = np.linalg.inv(aligned_extrinsics)

    if isinstance(view_ids, int):
        output_dir = os.path.join(output_dir, f"colmap/sparse/{view_ids}")
    else:
        output_dir = os.path.join(output_dir, f"colmap/sparse/0")
    os.makedirs(output_dir, exist_ok=True)

    # ----------- 生成 merge points ------------
    if len(ply_path_list) > 0:
        pcd = o3d.io.read_point_cloud(ply_path_list[0])
        for ply_path in ply_path_list[1:]:
            pcd += o3d.io.read_point_cloud(ply_path)
        pcd_path = os.path.join(output_dir, "points3D.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
    else:
        pcd_path = None

    # ----------- 生成 camera points ------------
    ply_path = os.path.join(output_dir, "camera_poses.ply")
    with open(ply_path, "w") as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_c2w)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        color = [255, 0, 0]
        for pose in all_c2w:
            position = pose[:3, 3]
            f.write(f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n")

    print(f"[MVFRPipeline] Camera poses visualization saved to {ply_path}")

    # ----------- 生成 images.txt ------------
    images_path = os.path.join(output_dir, "images.txt")
    os.makedirs(os.path.dirname(images_path), exist_ok=True)
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, (w2c, img_path) in enumerate(zip(aligned_extrinsics, img_list)):
            if pose is None:
                continue
            R_w2c = w2c[:3, :3]
            t_w2c = w2c[:3, 3]

            # Convert R to quaternion (w, x, y, z)
            q = rotation_matrix_to_quaternion(R_w2c)
            qw, qx, qy, qz = q

            # Image file name (relative or basename)
            img_name = os.path.basename(img_path)

            # CAMERA_ID = i+1 (same as in cameras.txt)
            if isinstance(view_ids, int):
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids)}/{img_name}\n")
            else:
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids[i])}/{img_name}\n")
            f.write("\n")  # 第二行为特征点，留空

    print(f"[MVFRPipeline] COLMAP images.txt saved to {images_path}")

    return w2c_outputs, pcd_path, other_outputs


@torch.inference_mode()
def simple_inference_fake_sv(
    self,
    frame_ids,
    view_ids,
    img_list,
    lidar_list=None,
    conf_list=None,
    invalid_mask_list=None,
    extrinsics_list=None,
    intrinsics_list=None,
    chunk_size=None,
    overlap=0,
    process_res=504,
    output_dir=None,
    save_points=False,
    save_normal=True,
    conf_ratio=None,
    kosmo_mask=None,
    **kwargs,
):
    other_outputs = dict()

    print_progress = chunk_size is not None and chunk_size > 500

    assert isinstance(view_ids, int)
    view_id = view_ids

    # step1: chunk
    if chunk_size is None:
        num_chunks = 1
        chunk_indices = [(0, len(img_list))]
    else:
        if overlap >= chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
        if len(img_list) <= chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            step = chunk_size - overlap
            num_chunks = (len(img_list) - overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + chunk_size, len(img_list))
                chunk_indices.append((start_idx, end_idx))

    print(f"[MVFRPipeline] Processing {len(img_list)} images in {num_chunks} chunks of size {chunk_size} with {overlap} overlap")
    print(f"[MVFRPipeline] output_dir: {output_dir}")

    for chunk_idx in tqdm(range(len(chunk_indices))):
        start_idx, end_idx = chunk_indices[chunk_idx]
        chunk_image_paths = img_list[start_idx:end_idx]
        chunk_invalid_mask_paths = invalid_mask_list[start_idx:end_idx] if invalid_mask_list is not None else None
        chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None
        chunk_lidar_paths = lidar_list[start_idx:end_idx] if lidar_list is not None else None
        chunk_conf_paths = conf_list[start_idx:end_idx] if conf_list is not None else None

        def process_one(idx):
            image_path = chunk_image_paths[idx]
            intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None
            lidar_path = chunk_lidar_paths[idx] if chunk_lidar_paths is not None else None
            # conf_path = chunk_conf_paths[idx] if chunk_conf_paths is not None else None

            pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            # Boundary resize
            pil_img = _resize_longest_side(pil_img, process_res)
            w, h = pil_img.size
            intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

            # Convert to tensor & normalize
            img_tensor = _normalize_image(pil_img)
            _, H, W = img_tensor.shape

            assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

            img_show = np.asarray(pil_img)

            if lidar_path is not None:
                lidar_depth = np.load(lidar_path)
                lidar_depth = resize_depth_preserve(lidar_depth, img_tensor.shape[-2:])
            else:
                lidar_depth = None

            return img_show, img_tensor, intrinsic, None, None, lidar_depth

        # step2: process
        outputs = parallel_execution(
            list(range(len(chunk_image_paths))),
            action=process_one,
            num_processes=8,
            print_progress=print_progress,
            sequential=False,
            desc=f"read chunk {chunk_idx}",
        )
        images_show, images, intrinsics, extrinsics, masks, lidar_depths = zip(*outputs)

        images_show = np.stack(images_show, axis=0)
        images = torch.stack(images).unsqueeze(0).float()
        intrinsics = np.asarray(intrinsics)[None].astype(np.float32) if intrinsics is not None and intrinsics[0] is not None else None

        if lidar_depths is not None and lidar_depths[0] is not None:
            lidar_depths = np.stack(lidar_depths)[None].astype(np.float32)
            lidar_depths_mask = (lidar_depths > 0.001) & (lidar_depths < 200)
            lidar_depths = self.depth_to_points(torch.from_numpy(lidar_depths).unsqueeze(2), K=torch.from_numpy(intrinsics), device="cpu", cache=False)

            # step3: max scale
            scale = []
            for bi in range(lidar_depths.shape[1]):
                cur_lidar_depths = lidar_depths[0, bi].permute(1, 2, 0)[lidar_depths_mask[0, bi]]
                if cur_lidar_depths.numel() > 0:
                    scale.append(torch.max(torch.norm(lidar_depths[0, bi].permute(1, 2, 0)[lidar_depths_mask[0, bi]], dim=-1)))
                else:
                    scale.append(torch.tensor(1.0))
            scale = torch.tensor(scale).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            lidar_depths = scale = None

        # step4: inputs to cuda
        images = images.cuda()
        intrinsics = torch.from_numpy(intrinsics).float().cuda()

        # if save_points:
        #     points_path = os.path.join(output_dir, f"prompt/camera_{view_id}/ppt.ply")
        #     os.makedirs(os.path.dirname(points_path), exist_ok=True)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(lidar_depths[0, 0].permute(1,2,0).reshape(-1, 3))
        #     pcd.colors = o3d.utility.Vector3dVector(images_show[0].reshape(-1, 3) / 255.0)
        #     o3d.io.write_point_cloud(points_path, pcd)

        if lidar_depths is not None:
            lidar_depths /= scale
            lidar_depths_mask = torch.from_numpy(lidar_depths_mask).float().unsqueeze(2)
            lidar_depths = torch.cat([lidar_depths, lidar_depths_mask], dim=-3)
            lidar_depths = lidar_depths.float().cuda()

        meta_data = dict(frames=[1], views=[images.shape[1]], input_width=[images.shape[-1]], input_height=[images.shape[-2]])
        meta_data["data_info"] = {"scene": [["kosmo"]]}

        # step4: model infer
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            results = self.model(
                images,
                scale=None,
                prompt_depth=lidar_depths,
                intrinsics=intrinsics,
                ray_directions=None,
                w2c=None,
                ray_world=None,
                query_points=None,
                meta_data=meta_data,
            )
            # torch.cuda.empty_cache()
        # end.record()
        # torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        # elapsed_ms = start.elapsed_time(end)
        # print(f"Torch Inference time: {elapsed_ms}ms")

        depth = confidence = points = normal = invalid_mask = None

        if "depth" in results:
            if scale is not None:
                depth = results["depth"].cpu() * scale
            else:
                depth = results["depth"].cpu()
            confidence = results["confidence"].cpu()

            if save_points:
                points = self.depth_to_points(depth, K=intrinsics.cpu(), device=depth.device)

            depth = depth[0, :, 0].contiguous().numpy()
            confidence = confidence[0, :, 0].numpy()

            if save_points:
                points = points[0].permute(0, 2, 3, 1).contiguous().numpy()

        if "normal" in results:
            normal = F.normalize(results["normal"], dim=-3).cpu()
            normal = normal[0].permute(0, 2, 3, 1).contiguous().numpy()

        if "invalid_mask" in results:
            invalid_mask = results["invalid_mask"].cpu().numpy()
            invalid_mask = invalid_mask[0, :, 0] > 0

            if normal is not None:
                normal *= ~invalid_mask[..., None]

        # step6: save
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            if kosmo_mask is not None:
                cur_kosmo_mask = kosmo_mask[view_id]
                if images_show.shape[-2:] != cur_kosmo_mask.shape:
                    cur_kosmo_mask = cv2.resize(cur_kosmo_mask.astype(np.float32), (images_show.shape[-2], images_show.shape[-3])) > 0
                else:
                    cur_kosmo_mask = cur_kosmo_mask.astype(np.bool)

                cur_kosmo_mask = cur_kosmo_mask | (images_show[0][..., 0] > 0) | (images_show[0][..., 1] > 0) | (images_show[0][..., 1] > 0)
            else:
                cur_kosmo_mask = None

            def process_one(idx):
                depth_path = conf_path = normal_path = invalid_mask_path = None

                image_basename = get_images_base_name(chunk_image_paths[idx])
                if depth is not None:
                    cur_depth = depth[idx]
                    cur_conf = confidence[idx]

                    if conf_ratio is not None:
                        valid_mask = cur_conf >= np.quantile(cur_conf, conf_ratio)
                        cur_depth[valid_mask] = 0
                    
                    if cur_kosmo_mask is not None:
                        cur_depth[~cur_kosmo_mask] = 0

                    depth_path = os.path.join(output_dir, f"depth/camera_{view_id}/{image_basename}.npy")
                    if not os.path.exists(depth_path):
                        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                        np.save(depth_path, cur_depth.astype(np.float16))

                    # depth_vis_path = os.path.join(output_dir, f"depth_vis/camera_{view_id}/{image_basename}.png")
                    # if not os.path.exists(depth_vis_path):
                    #     os.makedirs(os.path.dirname(depth_vis_path), exist_ok=True)
                    #     depth_norm = (cur_depth - cur_depth.min()) / (cur_depth.max() - cur_depth.min() + 1e-9)
                    #     from hAlgorithm.utils import colorize_depth_maps
                    #     depth_colored = colorize_depth_maps(depth_norm, 0, 1, cmap="turbo")
                    #     depth_colored.save(depth_vis_path)

                    conf_path = os.path.join(output_dir, f"conf/camera_{view_id}/{image_basename}.npy")
                    if not os.path.exists(conf_path):
                        os.makedirs(os.path.dirname(conf_path), exist_ok=True)
                        np.save(conf_path, cur_conf.astype(np.float16))

                    if save_points:
                        cur_points = points[idx]
                        cur_images_show = images_show[idx]

                        if cur_kosmo_mask is not None:
                            cur_points *= cur_kosmo_mask[..., None].astype(cur_points.dtype)

                        points_path = os.path.join(output_dir, f"points/camera_{view_id}/{image_basename}.ply")
                        if not os.path.exists(points_path):

                            if conf_ratio is not None:
                                cur_points = cur_points[valid_mask]
                                cur_images_show = cur_images_show[valid_mask]

                            os.makedirs(os.path.dirname(points_path), exist_ok=True)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(cur_points.reshape(-1, 3))
                            pcd.colors = o3d.utility.Vector3dVector(cur_images_show.reshape(-1, 3) / 255.0)
                            o3d.io.write_point_cloud(points_path, pcd)

                if save_normal and normal is not None:
                    cur_normal = normal[idx]

                    if cur_kosmo_mask is not None:
                        cur_normal[~cur_kosmo_mask] = 0

                    normal_path = os.path.join(output_dir, f"normal/camera_{view_id}/{image_basename}.npy")
                    if not os.path.exists(normal_path):
                        os.makedirs(os.path.dirname(normal_path), exist_ok=True)
                        np.save(normal_path, cur_normal.astype(np.float16))

                    # cur_normal = cur_normal * [0.5, -0.5, -0.5] + 0.5

                    # if cur_kosmo_mask is not None:
                    #     cur_normal[~cur_kosmo_mask] = 0

                    # diff = (cur_normal * 255).astype(np.uint8)
                    # save_path = os.path.join(output_dir, f"normal_vis/camera_{view_id}/{image_basename}.jpg")
                    # if not os.path.exists(save_path):
                    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #     cv2.imwrite(save_path, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))

                if invalid_mask is not None:
                    cur_invalid_mask = invalid_mask[idx].astype(bool)
                    cur_images_show = images_show[idx].astype(float)[:, :, ::-1]

                    if cur_kosmo_mask is not None:
                        cur_invalid_mask[~cur_kosmo_mask] = 0

                    invalid_mask_path = os.path.join(output_dir, f"invalid_mask/camera_{view_id}/{image_basename}.npy")
                    if not os.path.exists(invalid_mask_path):
                        os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
                        np.save(invalid_mask_path, cur_invalid_mask)
                    
                    # cur_images_show[cur_invalid_mask, :] *= 0.5
                    # cur_images_show[cur_invalid_mask, 2] += 255 * 0.5
                    # cur_images_show = cur_images_show.astype(np.uint8)
                    # invalid_mask_vis_path = os.path.join(output_dir, f"invalid_mask_vis/camera_{view_id}/{image_basename}.png")
                    # if not os.path.exists(invalid_mask_vis_path):
                    #     os.makedirs(os.path.dirname(invalid_mask_vis_path), exist_ok=True)
                    #     cv2.imwrite(invalid_mask_vis_path, cur_images_show)

                return depth_path, conf_path, normal_path, invalid_mask_path

            outputs = parallel_execution(
                list(range(images.shape[1])),
                action=process_one,
                num_processes=8,
                print_progress=print_progress,
                sequential=False,
                desc="save chunk",
            )
            depth_paths, conf_paths, normal_paths, invalid_mask_paths = zip(*outputs)

            if depth_paths[0] is not None:
                if "depth" not in other_outputs:
                    other_outputs["depth"] = dict()
                other_outputs["depth"].update(dict(zip(chunk_image_paths, depth_paths)))

            if conf_paths[0] is not None:
                if "conf" not in other_outputs:
                    other_outputs["conf"] = dict()
                other_outputs["conf"].update(dict(zip(chunk_image_paths, conf_paths)))

            if normal_paths[0] is not None:
                if "normal" not in other_outputs:
                    other_outputs["normal"] = dict()
                other_outputs["normal"].update(dict(zip(chunk_image_paths, normal_paths)))

            if invalid_mask_paths[0] is not None:
                if "invalid_mask" not in other_outputs:
                    other_outputs["invalid_mask"] = dict()
                other_outputs["invalid_mask"].update(dict(zip(chunk_image_paths, invalid_mask_paths)))

    return None, None, other_outputs


@torch.inference_mode()
def simple_inference_align(
    self,
    frame_ids,
    view_ids,
    img_list,
    lidar_list=None,
    conf_list=None,
    invalid_mask_list=None,
    extrinsics_list=None,
    intrinsics_list=None,
    chunk_size=None,
    overlap=0,
    process_res=504,
    output_dir=None,
    conf_ratio=0.2,
    save_points=True,
    **kwargs,
):
    import roma

    w2c_outputs = dict()
    c2w_outputs = dict()
    other_outputs = dict()
    ply_path_list = []
    chunk_results = []

    assert overlap > 0, "overlap must be > 0 for inter-chunk alignment"
    print_progress = chunk_size is not None and chunk_size > 500

    # step1: chunk with overlap
    if chunk_size is None:
        num_chunks = 1
        chunk_indices = [(0, len(img_list))]
    else:
        if overlap >= chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
        if len(img_list) <= chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            step = chunk_size - overlap
            num_chunks = (len(img_list) - overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + chunk_size, len(img_list))

                if i > 0 and i == num_chunks - 1:
                    cur_chunk_size = end_idx - start_idx
                    if cur_chunk_size < chunk_size // 2:
                        start_idx = end_idx - chunk_size // 2

                chunk_indices.append((start_idx, end_idx))

    print(f"[MVFRPipeline] simple_inference_align: {len(img_list)} images, {num_chunks} chunks, "
          f"chunk_size={chunk_size}, overlap={overlap}")
    print(f"[MVFRPipeline] output_dir: {output_dir}")

    # step2: per-chunk inference
    for chunk_idx in tqdm(range(len(chunk_indices)), desc="Chunk inference"):
        start_idx, end_idx = chunk_indices[chunk_idx]
        chunk_image_paths = img_list[start_idx:end_idx]
        chunk_view_ids = view_ids[start_idx:end_idx] if isinstance(view_ids, (list, tuple)) else view_ids
        chunk_invalid_mask_paths = invalid_mask_list[start_idx:end_idx] if invalid_mask_list is not None else None
        chunk_extrinsics = extrinsics_list[start_idx:end_idx] if extrinsics_list is not None else None
        chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None
        chunk_lidar_paths = lidar_list[start_idx:end_idx] if lidar_list is not None else None
        chunk_conf_paths = conf_list[start_idx:end_idx] if conf_list is not None else None

        def process_one(idx):
            image_path = chunk_image_paths[idx]
            invalid_mask = chunk_invalid_mask_paths[idx] if chunk_invalid_mask_paths is not None else None
            extrinsic = chunk_extrinsics[idx] if chunk_extrinsics is not None else None
            intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None
            lidar_path = chunk_lidar_paths[idx] if chunk_lidar_paths is not None else None
            conf_path = chunk_conf_paths[idx] if chunk_conf_paths is not None else None

            pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            pil_img = _resize_longest_side(pil_img, process_res)
            w, h = pil_img.size
            intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

            img_tensor = _normalize_image(pil_img)
            _, H, W = img_tensor.shape

            assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

            if invalid_mask is not None and invalid_mask[0] is not None:
                invalid_mask = np.load(invalid_mask).astype(np.float32)
                invalid_mask = cv2.resize(invalid_mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
                invalid_mask = invalid_mask > 0

            img_show = np.asarray(pil_img)

            if lidar_path is not None:
                lidar_depth = np.load(lidar_path)
                if conf_path is not None:
                    conf = np.load(conf_path)
                    conf_threshold = np.percentile(conf.reshape(-1), int(conf_ratio * 100))
                    mask = conf > conf_threshold
                    lidar_depth *= mask
                lidar_depth = resize_depth_preserve(lidar_depth, img_tensor.shape[-2:])
            else:
                lidar_depth = None

            return img_show, img_tensor, intrinsic, extrinsic, invalid_mask, lidar_depth

        outputs = parallel_execution(
            list(range(len(chunk_image_paths))),
            action=process_one,
            num_processes=8,
            print_progress=print_progress,
            sequential=False,
            desc=f"read chunk {chunk_idx}",
        )
        images_show, images, intrinsics, extrinsics, invalid_masks, lidar_depths = zip(*outputs)

        if chunk_intrinsics is None:
            intrinsics = None
        if chunk_extrinsics is None:
            extrinsics = None
        if chunk_invalid_mask_paths is None:
            invalid_masks = None
        if chunk_lidar_paths is None:
            lidar_depths = None

        images_show = np.stack(images_show, axis=0)
        images = torch.stack(images).unsqueeze(0).float()
        extrinsics = np.asarray(extrinsics)[None].astype(np.float32) if extrinsics is not None and extrinsics[0] is not None else None
        intrinsics = np.asarray(intrinsics)[None].astype(np.float32) if intrinsics is not None and intrinsics[0] is not None else None

        if lidar_depths is not None:
            lidar_depths = np.stack(lidar_depths)[None].astype(np.float32)
            lidar_depths_mask = (lidar_depths > 0.001) & (lidar_depths < 200)
            lidar_depths = self.depth_to_points(torch.from_numpy(lidar_depths).unsqueeze(2), K=torch.from_numpy(intrinsics), device="cpu", cache=False)

        scale = 1.0
        ori_extrinsics = None
        if extrinsics is not None:
            ori_extrinsics = extrinsics.copy()
            w2c = extrinsics
            base_c2w_pred = np.linalg.inv(w2c[:, 0:1])
            extrinsics = w2c @ base_c2w_pred

        if lidar_depths is not None and extrinsics is not None:
            b, n = extrinsics.shape[:2]
            h, w = lidar_depths.shape[-2:]
            lidar_depths_h = np.concatenate([lidar_depths.reshape(b, n, 3, -1), np.ones([b, n, 1, h * w])], axis=2).transpose(0, 1, 3, 2)
            world_lidar_depths = np.einsum("bnij,bnkj->bnki", np.linalg.inv(extrinsics), lidar_depths_h)[..., :3]
            world_lidar_depths = world_lidar_depths[lidar_depths_mask.reshape(b, n, -1)]
            scale = np.mean(np.linalg.norm(world_lidar_depths, axis=-1)).astype(np.float32)

        images = images.cuda()
        if intrinsics is not None:
            intrinsics_cuda = torch.from_numpy(intrinsics).float().cuda()
        else:
            intrinsics_cuda = None

        if lidar_depths is not None:
            lidar_depths /= scale
            lidar_depths_mask_t = torch.from_numpy(lidar_depths_mask).float().unsqueeze(2)
            lidar_depths = torch.cat([lidar_depths, lidar_depths_mask_t], dim=-3)
            lidar_depths = lidar_depths.float().cuda()

        extrinsics_cuda = None
        if extrinsics is not None:
            extrinsics[:, :, :3, 3] /= scale
            extrinsics_cuda = torch.from_numpy(extrinsics).float().cuda()

        if lidar_depths is not None:
            scale = torch.tensor([scale])

        meta_data = dict(
            frames=[1], views=[images.shape[1]],
            input_width=[images.shape[-1]], input_height=[images.shape[-2]],
        )
        meta_data["data_info"] = {"scene": [["kosmo"]]}

        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            results = self.model(
                images,
                scale=None,
                prompt_depth=lidar_depths,
                intrinsics=intrinsics_cuda,
                ray_directions=None,
                w2c=extrinsics_cuda,
                ray_world=None,
                query_points=None,
                meta_data=meta_data,
            )
            torch.cuda.empty_cache()

        depth = results["depth"].cpu() * scale
        confidence = results["confidence"].cpu()

        if "pose_enc" in results:
            pose_enc = results["pose_enc"][-1].cpu()
            if self.pose_encoding_type == "absT_quaR_FoV":
                pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                    pose_encoding=pose_enc.float(),
                    image_size_hw=images.shape[-2:],
                    build_intrinsics=True,
                )
                pred_extrinsics[..., :3, 3] *= scale
                w2c_pred = pred_extrinsics
                base_c2w_pred = w2c_pred[:, 0:1].inverse()
                pred_extrinsics = w2c_pred @ base_c2w_pred
                pred_extrinsics = pred_extrinsics[0]
            else:
                raise NotImplementedError
        else:
            pred_extrinsics = torch.from_numpy(ori_extrinsics[0]).float() if ori_extrinsics is not None else torch.eye(4).unsqueeze(0).repeat(images.shape[1], 1, 1)
            pred_intrinsics = intrinsics_cuda

        points = self.depth_to_points(depth, K=pred_intrinsics, device=depth.device)

        n = images.shape[1]
        h_pts, w_pts = points.shape[-2:]
        depth_np = depth[0, :, 0].contiguous().numpy()
        points_np = points[0].permute(0, 2, 3, 1).contiguous().numpy()
        confidence_np = confidence[0, :, 0].numpy()

        points_h = np.concatenate([points_np.reshape(n, -1, 3), np.ones([n, h_pts * w_pts, 1])], axis=-1)
        world_points = np.einsum("nij,nkj->nki", np.linalg.inv(pred_extrinsics.numpy()), points_h)[..., :3]

        chunk_results.append(dict(
            image_paths=chunk_image_paths,
            view_ids=chunk_view_ids,
            depth=depth_np,
            confidence=confidence_np,
            extrinsics=pred_extrinsics.numpy(),
            intrinsics=pred_intrinsics,
            world_points=world_points,
            images_show=images_show,
            invalid_masks=invalid_masks,
        ))

    # step3: inter-chunk alignment via overlap point registration
    sim3_list = []
    for chunk_idx in range(len(chunk_indices) - 1):
        logging.info(f"[MVFRAlign] Aligning chunk {chunk_idx} <-> {chunk_idx + 1} "
                     f"(Total {len(chunk_indices) - 1})")

        chunk_data1 = chunk_results[chunk_idx]
        chunk_data2 = chunk_results[chunk_idx + 1]

        start_idx1, end_idx1 = chunk_indices[chunk_idx]
        start_idx2, end_idx2 = chunk_indices[chunk_idx + 1]
        ovlp = end_idx1 - start_idx2

        point_map1 = chunk_data1["world_points"][-ovlp:]
        point_map2 = chunk_data2["world_points"][:ovlp]
        conf1 = chunk_data1["confidence"][-ovlp:]
        conf2 = chunk_data2["confidence"][:ovlp]

        # conf_threshold = min(np.median(conf1), np.median(conf2))

        # mask1 = conf1 > conf_threshold
        # mask2 = conf2 > conf_threshold

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        mask1 = sigmoid(conf1) > 0.1
        mask2 = sigmoid(conf2) > 0.1

        valid_mask = (mask1 & mask2).reshape(ovlp, -1)

        idx = np.where(valid_mask)
        all_pts1 = point_map1[idx]
        all_pts2 = point_map2[idx]

        R, t, s = roma.rigid_points_registration(
            torch.from_numpy(all_pts2),
            torch.from_numpy(all_pts1),
            compute_scaling=True,
        )
        R = R.numpy()
        t = t.numpy()
        s = float(s)
        sim3_list.append((s, R, t))

    sim3_list = accumulate_sim3_transforms(sim3_list)

    # step4: apply SIM3 transforms and save
    for chunk_idx in range(len(chunk_indices)):
        logging.info(f"[MVFRAlign] Applying transform for chunk {chunk_idx}")

        chunk_data = chunk_results[chunk_idx]
        world_points = chunk_data["world_points"]

        if chunk_idx > 0 and chunk_idx - 1 < len(sim3_list):
            s, R, t = sim3_list[chunk_idx - 1]
            chunk_data["world_points"] = apply_sim3_direct(world_points, s, R, t)
            chunk_data["depth"] *= s

            c2w = np.linalg.inv(chunk_data["extrinsics"])
            transformed_c2w_list = []
            for i in range(len(c2w)):
                t_global = s * R @ c2w[i, :3, 3] + t
                r_global = R @ c2w[i, :3, :3]
                transformed_c2w = np.eye(4)
                transformed_c2w[:3, :3] = r_global
                transformed_c2w[:3, 3] = t_global
                transformed_c2w_list.append(transformed_c2w)
            transformed_c2w = np.stack(transformed_c2w_list)

            chunk_data["extrinsics"] = np.linalg.inv(transformed_c2w)
            chunk_data["c2w"] = transformed_c2w
        else:
            chunk_data["c2w"] = np.linalg.inv(chunk_data["extrinsics"])

        chunk_data["w2c"] = chunk_data["extrinsics"]

        if save_points:
            pts = chunk_data["world_points"].reshape(-1, 3)
            colors = chunk_data["images_show"].reshape(-1, 3).astype(np.uint8)
            confs = chunk_data["confidence"].reshape(-1)
            conf_threshold_save = np.percentile(confs, 30)

            if isinstance(view_ids, int):
                ply_path = os.path.join(output_dir, f"pcd/camera_{view_ids}/{chunk_idx}_pcd.ply")
            else:
                ply_path = os.path.join(output_dir, f"pcd/{chunk_idx}_pcd.ply")
            os.makedirs(os.path.dirname(ply_path), exist_ok=True)

            invalid_masks_chunk = chunk_data.get("invalid_masks")
            if invalid_masks_chunk is not None and invalid_masks_chunk[0] is not None:
                inv_masks = np.stack(invalid_masks_chunk, axis=0)
                save_confident_pointcloud_batch(
                    points=pts, colors=colors, confs=confs,
                    output_path=ply_path,
                    conf_threshold=conf_threshold_save,
                    sample_ratio=0.01,
                    valid_mask=~inv_masks.reshape(-1),
                )
            else:
                save_confident_pointcloud_batch(
                    points=pts, colors=colors, confs=confs,
                    output_path=ply_path,
                    conf_threshold=conf_threshold_save,
                    sample_ratio=0.01,
                )
            ply_path_list.append(ply_path)

        def process_one_save(idx):
            image_basename = get_images_base_name(chunk_data["image_paths"][idx])
            view_id = chunk_data["view_ids"][idx] if isinstance(chunk_data["view_ids"], (list, tuple)) else chunk_data["view_ids"]
            cur_depth = chunk_data["depth"][idx]
            cur_conf = chunk_data["confidence"][idx]

            depth_path = os.path.join(output_dir, f"depth/camera_{view_id}/{image_basename}.npy")
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            np.save(depth_path, cur_depth.astype(np.float16))

            conf_path = os.path.join(output_dir, f"conf/camera_{view_id}/{image_basename}.npy")
            os.makedirs(os.path.dirname(conf_path), exist_ok=True)
            np.save(conf_path, cur_conf.astype(np.float16))

            return depth_path, conf_path

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            assert chunk_data["depth"].shape[0] == len(chunk_data["image_paths"])
            outputs = parallel_execution(
                list(range(chunk_data["depth"].shape[0])),
                action=process_one_save,
                num_processes=8,
                print_progress=print_progress,
                sequential=False,
                desc=f"save chunk {chunk_idx}",
            )
            depth_paths, conf_paths = zip(*outputs)

            if "depth" not in other_outputs:
                other_outputs["depth"] = dict()
            if "conf" not in other_outputs:
                other_outputs["conf"] = dict()

            other_outputs["depth"].update(dict(zip(chunk_data["image_paths"], depth_paths)))
            other_outputs["conf"].update(dict(zip(chunk_data["image_paths"], conf_paths)))
            w2c_outputs.update({p: w for p, w in zip(chunk_data["image_paths"], chunk_data["w2c"])})
            c2w_outputs.update({p: c for p, c in zip(chunk_data["image_paths"], chunk_data["c2w"])})

    # step5: save colmap format
    if output_dir is not None:
        aligned_extrinsics = np.stack([w2c_outputs[path] for path in img_list], axis=0)
        all_c2w = np.stack([c2w_outputs[path] for path in img_list], axis=0)

        if isinstance(view_ids, int):
            colmap_dir = os.path.join(output_dir, f"colmap/sparse/{view_ids}")
        else:
            colmap_dir = os.path.join(output_dir, "colmap/sparse/0")
        os.makedirs(colmap_dir, exist_ok=True)

        if len(ply_path_list) > 0:
            pcd = o3d.io.read_point_cloud(ply_path_list[0])
            for ply_path in ply_path_list[1:]:
                pcd += o3d.io.read_point_cloud(ply_path)
            pcd_path = os.path.join(colmap_dir, "points3D.ply")
            o3d.io.write_point_cloud(pcd_path, pcd)
        else:
            pcd_path = None

        ply_path = os.path.join(colmap_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_c2w)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            color = [255, 0, 0]
            for pose in all_c2w:
                position = pose[:3, 3]
                f.write(f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n")
        print(f"[MVFRPipeline] Camera poses visualization saved to {ply_path}")

        images_path = os.path.join(colmap_dir, "images.txt")
        with open(images_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for i, (w2c, img_path) in enumerate(zip(aligned_extrinsics, img_list)):
                R_w2c = w2c[:3, :3]
                t_w2c = w2c[:3, 3]
                q = rotation_matrix_to_quaternion(R_w2c)
                qw, qx, qy, qz = q
                img_name = os.path.basename(img_path)
                if isinstance(view_ids, int):
                    f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids)}/{img_name}\n")
                else:
                    f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids[i])}/{img_name}\n")
                f.write("\n")
        print(f"[MVFRPipeline] COLMAP images.txt saved to {images_path}")
    else:
        pcd_path = None

    return w2c_outputs, pcd_path, other_outputs


@torch.inference_mode()
def simple_inference_long(
    self,
    frame_ids,
    view_ids,
    img_list,
    lidar_list=None,
    conf_list=None,
    invalid_mask_list=None,
    extrinsics_list=None,
    intrinsics_list=None,
    chunk_size=None,
    overlap=0,
    process_res=504,
    output_dir=None,
):
    import roma

    w2c_outputs = dict()
    c2w_outputs = dict()
    other_outputs = dict()
    ply_path_list = []
    chunk_results = []

    assert overlap > 0
    print_progress = chunk_size is not None and chunk_size > 500

    # step1: chunk
    if chunk_size is None:
        num_chunks = 1
        chunk_indices = [(0, len(img_list))]
    else:
        if overlap >= chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
        if len(img_list) <= chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            step = chunk_size - overlap
            num_chunks = (len(img_list) - overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + chunk_size, len(img_list))

                if i > 0 and i == num_chunks - 1:
                    cur_chunk_size = end_idx - start_idx
                    if cur_chunk_size < chunk_size // 2:
                        start_idx = end_idx - chunk_size // 2

                chunk_indices.append((start_idx, end_idx))

    print(f"[MVFRPipeline] Processing {len(img_list)} images in {num_chunks} chunks of size {chunk_size} with {overlap} overlap")
    print(f"[MVFRPipeline] output_dir: {output_dir}")

    for chunk_idx in tqdm(range(len(chunk_indices))):
        start_idx, end_idx = chunk_indices[chunk_idx]
        chunk_image_paths = img_list[start_idx:end_idx]
        chunk_view_ids = view_ids[start_idx:end_idx] if isinstance(view_ids, (list, tuple)) else view_ids
        chunk_invalid_mask_paths = invalid_mask_list[start_idx:end_idx] if invalid_mask_list is not None else None
        chunk_extrinsics = extrinsics_list[start_idx:end_idx] if extrinsics_list is not None else None
        chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None
        chunk_lidar_paths = lidar_list[start_idx:end_idx] if lidar_list is not None else None
        chunk_conf_paths = conf_list[start_idx:end_idx] if conf_list is not None else None

        def process_one(idx):
            image_path = chunk_image_paths[idx]
            invalid_mask = chunk_invalid_mask_paths[idx] if chunk_invalid_mask_paths is not None else None
            extrinsic = chunk_extrinsics[idx] if chunk_extrinsics is not None else None
            intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None
            lidar_path = chunk_lidar_paths[idx] if chunk_lidar_paths is not None else None
            conf_path = chunk_conf_paths[idx] if chunk_conf_paths is not None else None

            pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            # Boundary resize
            pil_img = _resize_longest_side(pil_img, process_res)
            w, h = pil_img.size
            intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

            # Convert to tensor & normalize
            img_tensor = _normalize_image(pil_img)
            _, H, W = img_tensor.shape

            assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

            if invalid_mask is not None and invalid_mask[0] is not None:
                invalid_mask = np.load(invalid_mask).astype(np.float32)
                invalid_mask = cv2.resize(
                    invalid_mask,
                    dsize=(W, H),
                    interpolation=cv2.INTER_LINEAR,
                )
                invalid_mask = invalid_mask > 0

            img_show = np.asarray(pil_img)

            if lidar_path is not None:
                lidar_depth = np.load(lidar_path)

                if conf_path is not None:
                    conf = np.load(conf_path)
                    conf_threshold = np.percentile(conf.reshape(-1), 20)
                    mask = conf > conf_threshold
                    lidar_depth *= mask

                lidar_depth = resize_depth_preserve(lidar_depth, img_tensor.shape[-2:])
            else:
                lidar_depth = None

            return img_show, img_tensor, intrinsic, extrinsic, invalid_mask, lidar_depth

        # step2: process
        outputs = parallel_execution(
            list(range(len(chunk_image_paths))),
            action=process_one,
            num_processes=8,
            print_progress=print_progress,
            sequential=False,
            desc=f"read chunk {chunk_idx}",
        )
        images_show, images, intrinsics, extrinsics, invalid_masks, lidar_depths = zip(*outputs)

        images_show = np.stack(images_show, axis=0)
        images = torch.stack(images).unsqueeze(0).float()
        extrinsics = np.asarray(extrinsics)[None].astype(np.float32) if extrinsics is not None and extrinsics[0] is not None else None
        intrinsics = np.asarray(intrinsics)[None].astype(np.float32) if intrinsics is not None and intrinsics[0] is not None else None

        lidar_depths = np.stack(lidar_depths)[None].astype(np.float32)
        lidar_depths_mask = (lidar_depths > 0.001) & (lidar_depths < 200)
        lidar_depths = self.depth_to_points(torch.from_numpy(lidar_depths).unsqueeze(2), K=torch.from_numpy(intrinsics), device="cpu", cache=False)

        ori_extrinsics = extrinsics.copy()
        if extrinsics is not None:
            w2c = extrinsics
            base_c2w_pred = np.linalg.inv(w2c[:, 0:1])
            extrinsics = w2c @ base_c2w_pred

        # step3: scale
        b, n = extrinsics.shape[:2]
        h, w = lidar_depths.shape[-2:]
        lidar_depths_h = np.concatenate([lidar_depths.reshape(b, n, 3, -1), np.ones([b, n, 1, h * w])], axis=2).transpose(0, 1, 3, 2)

        world_lidar_depths = np.einsum("bnij,bnkj->bnki", np.linalg.inv(extrinsics), lidar_depths_h)[..., :3]

        world_lidar_depths = world_lidar_depths[lidar_depths_mask.reshape(b, n, -1)]
        scale = np.mean(np.linalg.norm(world_lidar_depths, axis=-1)).astype(np.float32)

        # step4: inputs to cuda
        images = images.cuda()
        intrinsics = torch.from_numpy(intrinsics).float().cuda()

        lidar_depths /= scale
        lidar_depths_mask = torch.from_numpy(lidar_depths_mask).float().unsqueeze(2)
        lidar_depths = torch.cat([lidar_depths, lidar_depths_mask], dim=-3)
        lidar_depths = lidar_depths.float().cuda()

        extrinsics[:, :, :3, 3] /= scale
        extrinsics = torch.from_numpy(extrinsics).float().cuda()

        scale = torch.tensor([scale])  # .cuda()

        meta_data = dict(frames=[1], views=[images.shape[1]], input_width=[images.shape[-1]], input_height=[images.shape[-2]])
        meta_data["data_info"] = {"scene": [["kosmo"]]}

        # step4: model infer
        with torch.autocast("cuda", enabled=True, dtype=torch.float16):
            results = self.model(
                images,
                scale=None,
                prompt_depth=lidar_depths,
                intrinsics=intrinsics,
                ray_directions=None,
                w2c=extrinsics,
                ray_world=None,
                query_points=None,
                meta_data=meta_data,
            )
            torch.cuda.empty_cache()

        depth = results["depth"].cpu() * scale
        confidence = results["confidence"].cpu()
        pose_enc = results["pose_enc"][-1].cpu()

        if self.pose_encoding_type == "absT_quaR_FoV":
            pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                pose_encoding=pose_enc.float(),
                image_size_hw=images.shape[-2:],
                build_intrinsics=True,
            )
            pred_extrinsics[..., :3, 3] *= scale
            w2c_pred = pred_extrinsics
            base_c2w_pred = w2c_pred[:, 0:1].inverse()
            pred_extrinsics = w2c_pred @ base_c2w_pred
            pred_extrinsics = pred_extrinsics[0]
        else:
            raise NotImplementedError

        points = self.depth_to_points(depth, K=pred_intrinsics, device=depth.device)

        depth = depth[0, :, 0].contiguous().numpy()
        points = points[0].permute(0, 2, 3, 1).contiguous().numpy()
        confidence = confidence[0, :, 0].numpy()

        points_h = np.concatenate([points.reshape(n, -1, 3), np.ones([n, h * w, 1])], axis=-1)
        world_points = np.einsum("nij,nkj->nki", np.linalg.inv(pred_extrinsics), points_h)[..., :3]

        chunk_results.append(
            dict(
                image_paths=chunk_image_paths,
                view_ids=chunk_view_ids,
                depth=depth,
                confidence=confidence,
                extrinsics=pred_extrinsics,
                intrinsics=pred_intrinsics,
                world_points=world_points,
                images_show=images_show,
            )
        )

    sim3_list = []
    for chunk_idx in range(len(chunk_indices) - 1):

        logging.info(f"[MVFRAlign] Aligning {chunk_idx} and {chunk_idx+1} (Total {len(chunk_indices)-1})")
        chunk_data1 = chunk_results[chunk_idx]
        chunk_data2 = chunk_results[chunk_idx + 1]

        start_idx1, end_idx1 = chunk_indices[chunk_idx]
        start_idx2, end_idx2 = chunk_indices[chunk_idx + 1]
        overlap = end_idx1 - start_idx2
        print(f"chunk_idx {chunk_idx}, overlap {overlap}")

        point_map1 = chunk_data1["world_points"][-overlap:]
        point_map2 = chunk_data2["world_points"][:overlap]
        conf1 = chunk_data1["confidence"][-overlap:]
        conf2 = chunk_data2["confidence"][:overlap]

        conf_threshold = min(np.median(conf1), np.median(conf2))  # * 0.1

        mask1 = conf1 > conf_threshold
        mask2 = conf2 > conf_threshold
        valid_mask = mask1 & mask2
        valid_mask = valid_mask.reshape(overlap, -1)

        idx = np.where(valid_mask)
        all_pts1 = point_map1[idx]
        all_pts2 = point_map2[idx]

        R, t, s = roma.rigid_points_registration(
            torch.from_numpy(all_pts2),
            torch.from_numpy(all_pts1),
            compute_scaling=True,
        )
        R = R.numpy()
        t = t.numpy()
        s = float(s)
        sim3_list.append((s, R, t))

    sim3_list = accumulate_sim3_transforms(sim3_list)

    for chunk_idx in range(len(chunk_indices)):
        print(f"[MVFRAlign] Applying {chunk_idx} -> {chunk_idx-1} (Total {len(chunk_indices)})")

        chunk_data = chunk_results[chunk_idx]
        world_points = chunk_data["world_points"]

        if chunk_idx > 0 and chunk_idx - 1 < len(sim3_list):
            s, R, t = sim3_list[chunk_idx - 1]
            chunk_data["world_points"] = apply_sim3_direct(world_points, s, R, t)
            chunk_data["depth"] *= s

            c2w = np.linalg.inv(chunk_data["extrinsics"])

            transformed_c2w_list = []
            for i in range(len(c2w)):
                t_global = s * R @ c2w[i, :3, 3] + t
                r_global = R @ c2w[i, :3, :3]
                transformed_c2w = np.eye(4)
                transformed_c2w[:3, :3] = r_global
                transformed_c2w[:3, 3] = t_global
                transformed_c2w_list.append(transformed_c2w)
            transformed_c2w = np.stack(transformed_c2w_list)

            chunk_data["extrinsics"] = np.linalg.inv(transformed_c2w)
            chunk_data["c2w"] = transformed_c2w
        else:
            chunk_data["c2w"] = np.linalg.inv(chunk_data["extrinsics"])

        chunk_data["w2c"] = chunk_data["extrinsics"]

        points = chunk_data["world_points"].reshape(-1, 3)
        colors = chunk_data["images_show"].reshape(-1, 3).astype(np.uint8)
        confs = chunk_data["confidence"].reshape(-1)
        conf_threshold = np.percentile(confs, 30)

        if isinstance(view_ids, int):
            ply_path = os.path.join(output_dir, f"pcd/camera_{view_ids}/{chunk_idx}_pcd.ply")
        else:
            ply_path = os.path.join(output_dir, f"pcd/{chunk_idx}_pcd.ply")
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)

        save_confident_pointcloud_batch(
            points=points,  # shape: (H, W, 3)
            colors=colors,  # shape: (H, W, 3)
            confs=confs,  # shape: (H, W)
            output_path=ply_path,
            conf_threshold=conf_threshold,
            sample_ratio=0.01,
        )
        ply_path_list.append(ply_path)

        def process_one(idx):
            image_basename = get_images_base_name(chunk_data["image_paths"][idx])
            view_id = chunk_data["view_ids"][idx] if isinstance(chunk_data["view_ids"], (list, tuple)) else chunk_data["view_ids"]
            cur_depth = chunk_data["depth"][idx]
            cur_conf = chunk_data["confidence"][idx]

            depth_path = os.path.join(output_dir, f"depth/camera_{view_id}/{image_basename}.npy")
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            np.save(depth_path, cur_depth.astype(np.float16))

            conf_path = os.path.join(output_dir, f"conf/camera_{view_id}/{image_basename}.npy")
            os.makedirs(os.path.dirname(conf_path), exist_ok=True)
            np.save(conf_path, cur_conf.astype(np.float16))

            return depth_path, conf_path

        assert chunk_data["depth"].shape[0] == len(chunk_data["image_paths"])
        outputs = parallel_execution(
            list(range(chunk_data["depth"].shape[0])),
            action=process_one,
            num_processes=8,
            print_progress=print_progress,
            sequential=False,
            desc="save chunk",
        )
        depth_paths, conf_paths = zip(*outputs)

        if "depth" not in other_outputs:
            other_outputs["depth"] = dict()
        if "conf" not in other_outputs:
            other_outputs["conf"] = dict()

        other_outputs["depth"].update(dict(zip(chunk_data["image_paths"], depth_paths)))
        other_outputs["conf"].update(dict(zip(chunk_data["image_paths"], conf_paths)))
        w2c_outputs.update({image_path: w2c for image_path, w2c in zip(chunk_data["image_paths"], chunk_data["w2c"])})
        c2w_outputs.update({image_path: w2c for image_path, w2c in zip(chunk_data["image_paths"], chunk_data["c2w"])})

    all_w2c = np.stack([w2c_outputs[path] for path in img_list], axis=0)
    all_c2w = np.stack([c2w_outputs[path] for path in img_list], axis=0)

    if isinstance(view_ids, int):
        output_dir = os.path.join(output_dir, f"colmap/sparse/{view_ids}")
    else:
        output_dir = os.path.join(output_dir, f"colmap/sparse/0")
    os.makedirs(output_dir, exist_ok=True)

    # ----------- 生成 merge points ------------
    if len(ply_path_list) > 0:
        pcd = o3d.io.read_point_cloud(ply_path_list[0])
        for ply_path in ply_path_list[1:]:
            pcd += o3d.io.read_point_cloud(ply_path)
        pcd_path = os.path.join(output_dir, "points3D.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
    else:
        pcd_path = None

    # ----------- 生成 camera points ------------
    ply_path = os.path.join(output_dir, "camera_poses.ply")
    with open(ply_path, "w") as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_c2w)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        color = [255, 0, 0]
        for pose in all_c2w:
            position = pose[:3, 3]
            f.write(f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n")

    print(f"[MVFRPipeline] Camera poses visualization saved to {ply_path}")

    # ----------- 生成 images.txt ------------
    images_path = os.path.join(output_dir, "images.txt")
    os.makedirs(os.path.dirname(images_path), exist_ok=True)
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, (w2c, img_path) in enumerate(zip(all_w2c, img_list)):
            if pose is None:
                continue
            R_w2c = w2c[:3, :3]
            t_w2c = w2c[:3, 3]

            # Convert R to quaternion (w, x, y, z)
            q = rotation_matrix_to_quaternion(R_w2c)
            qw, qx, qy, qz = q

            # Image file name (relative or basename)
            img_name = os.path.basename(img_path)

            # CAMERA_ID = i+1 (same as in cameras.txt)
            if isinstance(view_ids, int):
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids)}/{img_name}\n")
            else:
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {i + 1} camera_{int(view_ids[i])}/{img_name}\n")
            f.write("\n")  # 第二行为特征点，留空

    print(f"[MVFRPipeline] COLMAP images.txt saved to {images_path}")

    return w2c_outputs, pcd_path, other_outputs


import time
from contextlib import contextmanager


@contextmanager
def cpu_timer(msg="Operation"):
    """
    用法：
        with timer("Data loading"):
            data = load_data()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{msg}: {end - start:.4f} seconds")


@torch.inference_mode()
def trt_inference_fake_sv(
    self,
    trt_model,
    frame_ids,
    view_ids,
    img_list,
    lidar_list=None,
    conf_list=None,
    invalid_mask_list=None,
    extrinsics_list=None,
    intrinsics_list=None,
    chunk_size=None,
    overlap=0,
    process_res=504,
    output_dir=None,
    save_points=False,
    save_normal=True,
    images_list=None,
    **kwargs
):
    other_outputs = dict()

    print_progress = chunk_size is not None and chunk_size > 500

    assert isinstance(view_ids, int)
    view_id = view_ids

    # step1: chunk
    if chunk_size is None:
        num_chunks = 1
        chunk_indices = [(0, len(img_list))]
    else:
        if overlap >= chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
        if len(img_list) <= chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            step = chunk_size - overlap
            num_chunks = (len(img_list) - overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + chunk_size, len(img_list))
                chunk_indices.append((start_idx, end_idx))

    print(f"[MVFRPipeline] Processing {len(img_list)} images in {num_chunks} chunks of size {chunk_size} with {overlap} overlap")
    print(f"[MVFRPipeline] output_dir: {output_dir}")

    for chunk_idx in tqdm(range(len(chunk_indices))):
        # with cpu_timer("预处理"):
        start_idx, end_idx = chunk_indices[chunk_idx]
        chunk_image_paths = img_list[start_idx:end_idx]
        chunk_invalid_mask_paths = invalid_mask_list[start_idx:end_idx] if invalid_mask_list is not None else None
        chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None
        chunk_lidar_paths = lidar_list[start_idx:end_idx] if lidar_list is not None else None
        chunk_conf_paths = conf_list[start_idx:end_idx] if conf_list is not None else None
        chunk_images = images_list[start_idx:end_idx] if images_list is not None else None

        def process_one(idx):
            image_path = chunk_image_paths[idx]
            intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None
            lidar_path = chunk_lidar_paths[idx] if chunk_lidar_paths is not None else None
            # conf_path = chunk_conf_paths[idx] if chunk_conf_paths is not None else None
            if chunk_images is not None:
                pil_img = chunk_images[idx]
                pil_img = Image.fromarray(cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            # Boundary resize
            pil_img = _resize_longest_side(pil_img, process_res)
            w, h = pil_img.size
            intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

            # Convert to tensor & normalize
            img_tensor = _normalize_image(pil_img)
            _, H, W = img_tensor.shape

            assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

            img_show = np.asarray(pil_img)

            if lidar_path is not None:
                lidar_depth = np.load(lidar_path)
                lidar_depth = resize_depth_preserve(lidar_depth, img_tensor.shape[-2:])
            else:
                lidar_depth = None

            return img_show, img_tensor, intrinsic, None, None, lidar_depth

        # step2: process
        outputs = parallel_execution(
            list(range(len(chunk_image_paths))),
            action=process_one,
            num_processes=8,
            print_progress=print_progress,
            sequential=False,
            desc=f"read chunk {chunk_idx}",
        )
        images_show, images, intrinsics, extrinsics, masks, lidar_depths = zip(*outputs)

        images_show = np.stack(images_show, axis=0)
        images = torch.stack(images).unsqueeze(0).float()
        intrinsics = np.asarray(intrinsics)[None].astype(np.float32) if intrinsics is not None and intrinsics[0] is not None else None

        if lidar_depths is not None and lidar_depths[0] is not None:
            lidar_depths = np.stack(lidar_depths)[None].astype(np.float32)
            lidar_depths_mask = (lidar_depths > 0.001) & (lidar_depths < 200)
            lidar_depths = self.depth_to_points(torch.from_numpy(lidar_depths).unsqueeze(2), K=torch.from_numpy(intrinsics), device="cpu", cache=False)

            # step3: max scale
            scale = []
            for bi in range(lidar_depths.shape[1]):
                cur_lidar_depths = lidar_depths[0, bi].permute(1, 2, 0)[lidar_depths_mask[0, bi]]
                if cur_lidar_depths.numel() > 0:
                    scale.append(torch.max(torch.norm(lidar_depths[0, bi].permute(1, 2, 0)[lidar_depths_mask[0, bi]], dim=-1)))
                else:
                    scale.append(torch.tensor(1.0))
            scale = torch.tensor(scale).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            lidar_depths = scale = None

        # step4: inputs to cuda
        images = images.cuda()
        intrinsics = torch.from_numpy(intrinsics).float().cuda()

        # if save_points:
        #     points_path = os.path.join(output_dir, f"prompt/camera_{view_id}/ppt.ply")
        #     os.makedirs(os.path.dirname(points_path), exist_ok=True)
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(lidar_depths[0, 0].permute(1,2,0).reshape(-1, 3))
        #     pcd.colors = o3d.utility.Vector3dVector(images_show[0].reshape(-1, 3) / 255.0)
        #     o3d.io.write_point_cloud(points_path, pcd)

        if lidar_depths is not None:
            lidar_depths = lidar_depths.float().cuda()
            # lidar_depths /= scale
            # lidar_depths_mask = torch.from_numpy(lidar_depths_mask).float().unsqueeze(2)
            # lidar_depths = torch.cat([lidar_depths, lidar_depths_mask], dim=-3)
            # lidar_depths = lidar_depths.float().cuda()

        meta_data = dict(frames=[1], views=[images.shape[1]], input_width=[images.shape[-1]], input_height=[images.shape[-2]])
        meta_data["data_info"] = {"scene": [["kosmo"]]}

        # step4: model infer
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        results = trt_model(
            images,
            prompt_scale=scale,
            prompt_depth=lidar_depths,
            intrinsics=intrinsics,
            ray_directions=None,
            w2c=None,
            ray_world=None,
            query_points=None,
            meta_data=meta_data,
        )
        # end.record()
        # torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        # elapsed_ms = start.elapsed_time(end)
        # print(f"Tensor RT Inference time: {elapsed_ms}ms")

        # with cpu_timer("后处理"):
        depth = confidence = points = normal = invalid_mask = None

        if "pred_local_depth" in results:
            if scale is not None:
                depth = results["pred_local_depth"].cpu().clone().unsqueeze(0) * scale
            else:
                depth = results["pred_local_depth"].cpu().clone().unsqueeze(0)
            confidence = results["pred_local_conf"].cpu().clone().unsqueeze(0)

            if save_points:
                points = self.depth_to_points(depth, K=intrinsics.cpu(), device=depth.device)

            depth = depth[0, :, 0].contiguous().numpy()
            confidence = confidence[0, :, 0].numpy()

            if save_points:
                points = points[0].permute(0, 2, 3, 1).contiguous().numpy()

        if "pred_local_normal" in results:
            normal = results["pred_local_normal"].cpu().clone().unsqueeze(0)
            # normal = F.normalize(normal, dim=-3)
            normal = normal[0].permute(0, 2, 3, 1).contiguous().numpy()

        if "pred_local_invalid_mask" in results:
            invalid_mask = results["pred_local_invalid_mask"].cpu().clone().unsqueeze(0).numpy()
            invalid_mask = invalid_mask[0, :, 0] > 0

            if normal is not None:
                normal *= ~invalid_mask[..., None]

        # step6: save
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            def process_one(idx):
                depth_path = conf_path = normal_path = invalid_mask_path = None

                image_basename = get_images_base_name(chunk_image_paths[idx])
                if depth is not None:
                    cur_depth = depth[idx]
                    cur_conf = confidence[idx]

                    depth_path = os.path.join(output_dir, f"depth/camera_{view_id}/{image_basename}.npy")
                    os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                    if not os.path.exists(depth_path):
                        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                        np.save(depth_path, cur_depth.astype(np.float16))

                    conf_path = os.path.join(output_dir, f"conf/camera_{view_id}/{image_basename}.npy")
                    if not os.path.exists(conf_path):
                        os.makedirs(os.path.dirname(conf_path), exist_ok=True)
                        np.save(conf_path, cur_conf.astype(np.float16))

                    if save_points:
                        cur_points = points[idx]
                        points_path = os.path.join(output_dir, f"points/camera_{view_id}/{image_basename}.ply")
                        if not os.path.exists(points_path):
                            os.makedirs(os.path.dirname(points_path), exist_ok=True)
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(cur_points.reshape(-1, 3))
                            pcd.colors = o3d.utility.Vector3dVector(images_show[idx].reshape(-1, 3) / 255.0)
                            o3d.io.write_point_cloud(points_path, pcd)

                if save_normal and normal is not None:
                    cur_normal = normal[idx]

                    normal_path = os.path.join(output_dir, f"normal/camera_{view_id}/{image_basename}.npy")
                    os.makedirs(os.path.dirname(normal_path), exist_ok=True)
                    if not os.path.exists(normal_path):
                        os.makedirs(os.path.dirname(normal_path), exist_ok=True)
                        np.save(normal_path, cur_normal.astype(np.float16))

                    cur_normal = cur_normal * [0.5, -0.5, -0.5] + 0.5
                    diff = (cur_normal * 255).astype(np.uint8)
                    save_path = os.path.join(output_dir, f"normal/camera_{view_id}/{image_basename}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))

                if invalid_mask is not None:
                    cur_invalid_mask = invalid_mask[idx]

                    invalid_mask_path = os.path.join(output_dir, f"invalid_mask/camera_{view_id}/{image_basename}.npy")
                    os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
                    if not os.path.exists(invalid_mask_path):
                        os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
                        np.save(invalid_mask_path, cur_invalid_mask.astype(bool))

                return depth_path, conf_path, normal_path, invalid_mask_path

            outputs = parallel_execution(
                list(range(images.shape[1])),
                action=process_one,
                num_processes=8,
                print_progress=print_progress,
                sequential=False,
                desc="save chunk",
            )
            depth_paths, conf_paths, normal_paths, invalid_mask_paths = zip(*outputs)

            if depth_paths[0] is not None:
                if "depth" not in other_outputs:
                    other_outputs["depth"] = dict()
                other_outputs["depth"].update(dict(zip(chunk_image_paths, depth_paths)))

            if conf_paths[0] is not None:
                if "conf" not in other_outputs:
                    other_outputs["conf"] = dict()
                other_outputs["conf"].update(dict(zip(chunk_image_paths, conf_paths)))

            if normal_paths[0] is not None:
                if "normal" not in other_outputs:
                    other_outputs["normal"] = dict()
                other_outputs["normal"].update(dict(zip(chunk_image_paths, normal_paths)))

            if invalid_mask_paths[0] is not None:
                if "invalid_mask" not in other_outputs:
                    other_outputs["invalid_mask"] = dict()
                other_outputs["invalid_mask"].update(dict(zip(chunk_image_paths, invalid_mask_paths)))

    return None, None, other_outputs


@torch.inference_mode()
def trt_inference_fake_sv_chunk(
    self,
    trt_model,
    frame_ids,
    view_ids,
    img_list,
    lidar_list=None,
    conf_list=None,
    invalid_mask_list=None,
    extrinsics_list=None,
    intrinsics_list=None,
    chunk_size=None,
    overlap=0,
    process_res=504,
    output_dir=None,
    save_points=False,
    save_normal=True,
    images_list=None,
    **kwargs
):
    other_outputs = dict()

    print_progress = chunk_size is not None and chunk_size > 500

    assert isinstance(view_ids, int)
    view_id = view_ids

    # step1: chunk
    if chunk_size is None:
        num_chunks = 1
        chunk_indices = [(0, len(img_list))]
    else:
        if overlap >= chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({overlap}) must be less than chunk size ({chunk_size})")
        if len(img_list) <= chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(img_list))]
        else:
            step = chunk_size - overlap
            num_chunks = (len(img_list) - overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + chunk_size, len(img_list))
                chunk_indices.append((start_idx, end_idx))

    print(f"[MVFRPipeline] Processing {len(img_list)} images in {num_chunks} chunks of size {chunk_size} with {overlap} overlap")
    print(f"[MVFRPipeline] output_dir: {output_dir}")
    data_preprocess_time = 0
    inference_time = 0
    post_time = 0
    for chunk_idx in tqdm(range(len(chunk_indices))):
        # with cpu_timer("预处理"):
        start_idx, end_idx = chunk_indices[chunk_idx]
        chunk_image_paths = img_list[start_idx:end_idx]
        chunk_invalid_mask_paths = invalid_mask_list[start_idx:end_idx] if invalid_mask_list is not None else None
        chunk_intrinsics = intrinsics_list[start_idx:end_idx] if intrinsics_list is not None else None
        chunk_lidar_paths = lidar_list[start_idx:end_idx] if lidar_list is not None else None
        chunk_conf_paths = conf_list[start_idx:end_idx] if conf_list is not None else None
        chunk_images = images_list[start_idx:end_idx] if images_list is not None else None
        
        def process_one(idx):
            image_path = chunk_image_paths[idx]
            intrinsic = chunk_intrinsics[idx] if chunk_intrinsics is not None else None
            lidar_path = chunk_lidar_paths[idx] if chunk_lidar_paths is not None else None
            # conf_path = chunk_conf_paths[idx] if chunk_conf_paths is not None else None
            if chunk_images is not None:
                pil_img = chunk_images[idx]
                pil_img = Image.fromarray(cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = pil_img.size

            # Boundary resize
            pil_img = _resize_longest_side(pil_img, process_res)
            w, h = pil_img.size
            intrinsic = _resize_ixt(intrinsic, orig_w, orig_h, w, h)

            # Convert to tensor & normalize
            img_tensor = _normalize_image(pil_img)
            _, H, W = img_tensor.shape

            assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."

            img_show = np.asarray(pil_img)

            if lidar_path is not None:
                lidar_depth = np.load(lidar_path)
                lidar_depth = resize_depth_preserve(lidar_depth, img_tensor.shape[-2:])
            else:
                lidar_depth = None

            return img_show, img_tensor, intrinsic, None, None, lidar_depth

        preprocess_start_time = time.time()
        # step2: process
        outputs = parallel_execution(
            list(range(len(chunk_image_paths))),
            action=process_one,
            num_processes=8,
            print_progress=print_progress,
            sequential=False,
            desc=f"read chunk {chunk_idx}",
        )
        images_show, images, intrinsics, extrinsics, masks, lidar_depths = zip(*outputs)
        data_preprocess_time += (time.time() - preprocess_start_time)
        images_show = np.stack(images_show, axis=0)
        images = torch.stack(images).unsqueeze(0).float()
        # intrinsics = np.asarray(intrinsics)[None].astype(np.float32) if intrinsics is not None and intrinsics[0] is not None else None
        intrinsics = np.stack(intrinsics, axis=0).astype(np.float32) if intrinsics is not None  and intrinsics[0] is not None else None

        # step4: inputs to cuda
        images = images.cuda()
        intrinsics = torch.from_numpy(intrinsics).float().cuda().unsqueeze(0) if intrinsics is not None  and intrinsics[0] is not None else None


        meta_data = dict(frames=[1], views=[1], input_width=[images.shape[-1]], input_height=[images.shape[-2]])
        meta_data["data_info"] = {"scene": [["kosmo"]]}

        # step4: model infer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        total_depth = total_confidence = total_points = total_normal = total_invalid_mask = []
        # breakpoint()
        for i in range(images.shape[1]):
            results = trt_model(
                images[:, i:i+1],
                prompt_scale=None,
                prompt_depth=None,
                intrinsics=intrinsics[:, i:i+1] if intrinsics is not None else None,
                ray_directions=None,
                w2c=None,
                ray_world=None,
                query_points=None,
                meta_data=meta_data,
            )
        # end.record()
        # torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        # elapsed_ms = start.elapsed_time(end)
        # print(f"Tensor RT Inference time: {elapsed_ms}ms")

        # with cpu_timer("后处理"):
            # depth = confidence = points = normal = invalid_mask = None
            
            if "pred_local_normal" in results:
                normal = results["pred_local_normal"].cpu().clone()
                # normal = F.normalize(normal, dim=-3)
                normal = normal[0].permute(1,2,0).contiguous().numpy()
                total_normal.append(normal)
            if "pred_local_invalid_mask" in results:
                invalid_mask = results["pred_local_invalid_mask"].cpu().clone().numpy()
                invalid_mask = invalid_mask[:, 0] > 0
                total_invalid_mask.append(invalid_mask)
                if normal is not None:
                    normal *= ~invalid_mask[..., None]
                    total_normal[-1] = normal
        end.record()
        torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        elapsed_ms = start.elapsed_time(end)
        # print(f"Tensor RT Inference time: {elapsed_ms}ms")
        inference_time += elapsed_ms / 1000
        
        post_start_time = time.time()
        # step6: save
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            def process_one(idx):
                depth_path = conf_path = normal_path = invalid_mask_path = None

                image_basename = get_images_base_name(chunk_image_paths[idx])

                if save_normal and total_normal:
                    cur_normal = total_normal[idx]
                    normal_path = os.path.join(output_dir, f"normal/camera_{view_id}/{image_basename}.npy")
                    os.makedirs(os.path.dirname(normal_path), exist_ok=True)
                    if not os.path.exists(normal_path):
                        os.makedirs(os.path.dirname(normal_path), exist_ok=True)
                        np.save(normal_path, cur_normal.astype(np.float16))
                        # iio.imwrite(normal_path, cur_normal.astype(np.float32), extension='.exr')
                        
                    # cur_normal = cur_normal * [0.5, -0.5, -0.5] + 0.5
                    # diff = (cur_normal * 255).astype(np.uint8)
                    # save_path = os.path.join(output_dir, f"normal/camera_{view_id}/{image_basename}.jpg")
                    # cv2.imwrite(save_path, cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))

                # if total_invalid_mask:
                #     cur_invalid_mask = total_invalid_mask[idx]

                #     invalid_mask_path = os.path.join(output_dir, f"invalid_mask/camera_{view_id}/{image_basename}.npy")
                #     os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
                #     if not os.path.exists(invalid_mask_path):
                #         os.makedirs(os.path.dirname(invalid_mask_path), exist_ok=True)
                #         np.save(invalid_mask_path, cur_invalid_mask.astype(bool))

                return depth_path, conf_path, normal_path, invalid_mask_path

            outputs = parallel_execution(
                list(range(images.shape[1])),
                action=process_one,
                num_processes=8,
                print_progress=print_progress,
                sequential=False,
                desc="save chunk",
            )
            depth_paths, conf_paths, normal_paths, invalid_mask_paths = zip(*outputs)

            # if depth_paths[0] is not None:
            #     if "depth" not in other_outputs:
            #         other_outputs["depth"] = dict()
            #     other_outputs["depth"].update(dict(zip(chunk_image_paths, depth_paths)))

            # if conf_paths[0] is not None:
            #     if "conf" not in other_outputs:
            #         other_outputs["conf"] = dict()
            #     other_outputs["conf"].update(dict(zip(chunk_image_paths, conf_paths)))

            if normal_paths[0] is not None:
                if "normal" not in other_outputs:
                    other_outputs["normal"] = dict()
                other_outputs["normal"].update(dict(zip(chunk_image_paths, normal_paths)))

            # if invalid_mask_paths[0] is not None:
            #     if "invalid_mask" not in other_outputs:
            #         other_outputs["invalid_mask"] = dict()
            #     other_outputs["invalid_mask"].update(dict(zip(chunk_image_paths, invalid_mask_paths)))
        post_time += (time.time() - post_start_time)
    
    logging.info(f"frame: {len(img_list)}, preprocess_time: {data_preprocess_time}")
    logging.info(f"inference_time: {inference_time}")
    logging.info(f"post_time: {post_time}")
    return None, None, other_outputs


if __name__ == "__main__":
    get_kosmo_mask(mask_mode=20251231, debug=True)