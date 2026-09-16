import collections
import csv
import json
import os
import struct

import numpy as np
import open3d as o3d

try:
    from .geo_helper import mat_to_quat_trans_np, quat_trans_to_mat_np
except:
    from geo_helper import mat_to_quat_trans_np, quat_trans_to_mat_np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


##############  colmap  ##############
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def write_intrinsics_text(path, cameras):
    """
    Save cameras dictionary (as loaded by read_intrinsics_text)
    into a COLMAP-compatible cameras.txt file.
    """
    with open(path, "w") as fid:
        fid.write("# Camera list with one line of data per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fid.write(f"# Number of cameras: {len(cameras)}\n")

        for cam in cameras:
            # Flatten parameters and format them cleanly
            params_str = " ".join(f"{p:.6f}" for p in np.ravel(cam.params))
            fid.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params_str}\n")


def load_poses_txt_colmap(input_file, get_index_method=None):
    match_line_exist = False
    poses_dict = {}
    camera_id_dict = {}
    with open(input_file, "r") as fid:
        i = 0
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] == "#":
                elems = line.split()
                if elems[1] == "POINTS2D[]":
                    match_line_exist = True
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(float(elems[0]))
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                if get_index_method is None:
                    idx = image_name
                else:
                    try:
                        idx = int(get_index_method(image_name))
                    except:
                        idx = i
                w2c_mat = quat_trans_to_mat_np(qvec, tvec)
                c2w_mat = np.linalg.inv(w2c_mat)
                c2w_quat, c2w_trans = mat_to_quat_trans_np(c2w_mat)
                # row_data = [c2w_trans[0], c2w_trans[1], c2w_trans[2],
                #             c2w_quat[1], c2w_quat[2], c2w_quat[3], c2w_quat[0]]
                row_data = [float(c2w_trans[0]), float(c2w_trans[1]), float(c2w_trans[2]), float(c2w_quat[0]), float(c2w_quat[1]), float(c2w_quat[2]), float(c2w_quat[3])]
                poses_dict[idx] = row_data
                camera_id_dict[idx] = camera_id
                i += 1
                if match_line_exist:
                    fid.readline()
    return poses_dict, camera_id_dict


def load_poses_bin_colmap(input_file, get_index_method=None):
    poses_dict = {}
    camera_id_dict = {}
    with open(input_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for i in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            if get_index_method is None:
                idx = image_name
            else:
                idx = int(get_index_method(image_name))
            w2c_mat = quat_trans_to_mat_np(qvec, tvec)
            c2w_mat = np.linalg.inv(w2c_mat)
            c2w_quat, c2w_trans = mat_to_quat_trans_np(c2w_mat)
            # row_data = [c2w_trans[0], c2w_trans[1], c2w_trans[2],
            #             c2w_quat[1], c2w_quat[2], c2w_quat[3], c2w_quat[0]]
            row_data = [float(c2w_trans[0]), float(c2w_trans[1]), float(c2w_trans[2]), float(c2w_quat[0]), float(c2w_quat[1]), float(c2w_quat[2]), float(c2w_quat[3])]
            poses_dict[idx] = row_data
            camera_id_dict[idx] = camera_id
    return poses_dict, camera_id_dict


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        count = 0
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            if error > 2.0 or track_length < 3:
                continue
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1
    xyzs = np.delete(xyzs, np.arange(count, num_points), axis=0)
    rgbs = np.delete(rgbs, np.arange(count, num_points), axis=0)
    errors = np.delete(errors, np.arange(count, num_points), axis=0)
    return xyzs, rgbs


def save_colmap_poses(poses_dict, output_file, camera_id_dict=None):
    with open(output_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(poses_dict)))
        idx = 1
        for name, values in poses_dict.items():
            cam_id = 1
            if camera_id_dict is not None:
                cam_id = camera_id_dict[name]
            f.write(f"{idx} {values[3]:.6f} {values[4]:.6f} {values[5]:.6f} {values[6]:.6f} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {cam_id} {name}\n\n")
            idx += 1


def load_colmap(sparse_folder, get_idx_method=None):
    if os.path.exists(os.path.join(sparse_folder, "points3D.bin")):
        xyz, rgb = read_points3D_binary(os.path.join(sparse_folder, "points3D.bin"))
    else:
        ply_path = os.path.join(sparse_folder, "points3D.ply")
        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points).astype(np.float32)
        rgb = np.asarray(pcd.colors).astype(np.float32) * 255

    if os.path.exists(os.path.join(sparse_folder, "images.bin")):
        poses, camera_ids = load_poses_bin_colmap(os.path.join(sparse_folder, "images.bin"), get_index_method=get_idx_method)
    else:
        poses, camera_ids = load_poses_txt_colmap(os.path.join(sparse_folder, "images.txt"), get_index_method=get_idx_method)

    return poses, xyz, rgb, camera_ids


############### other #################
def write_tum_txt(poses, output_file):
    with open(output_file, "w") as f:
        for item in poses:
            f.write("{} {} {} {} {} {} {} {}\n".format(item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7]))


def write_2ddict2csv(output_dict, csv_file, key_name="Key"):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头
        header = [key_name] + list(next(iter(output_dict.values())).keys())
        writer.writerow(header)
        # 每一行
        for name, values in output_dict.items():
            row = [name] + [f"{values[k]:.3f}" for k in header[1:]]
            writer.writerow(row)


def write_1ddict2csv(output_dict, csv_file, key_name="Key"):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头
        header = [key_name]
        writer.writerow(header)
        # 每一行
        for name, values in output_dict.items():
            row = [name] + [values]
            writer.writerow(row)


################ load Kosmo json ################
def load_kosmo_json(json_path, scene, view_id, pose_key, get_index_method=None):
    json_all = json.load(open(json_path))["mf_files"]
    if scene is None:
        scene = list(json_all.keys())[0]
    scene_json = json_all[scene]
    view2main_key = f"T_cam{view_id}_2_cam0"
    extrinsics = scene_json["extrinsics"]
    T_view2main = np.eye(4)
    # if view2main_key in extrinsics:
    #     T_view2main = np.array(extrinsics[view2main_key])

    frames_json = scene_json["frames"]

    poses_dict = {}
    camera_id_dict = {}
    for frame_j in frames_json:
        if frame_j["view_id"] == view_id:
            image_path = frame_j["rgb"]
            image_name = f"camera_{view_id}/" + os.path.basename(image_path)

            if get_index_method is None:
                idx = image_name
            else:
                idx = int(get_index_method(image_name))

            ts = frame_j["timestamp"]
            if pose_key not in frame_j:
                continue

            main_cam2world = np.array(frame_j[pose_key])
            T_view2world = main_cam2world @ T_view2main
            c2w_quat, c2w_trans = mat_to_quat_trans_np(T_view2world)
            row_data = [float(c2w_trans[0]), float(c2w_trans[1]), float(c2w_trans[2]), float(c2w_quat[0]), float(c2w_quat[1]), float(c2w_quat[2]), float(c2w_quat[3])]
            poses_dict[idx] = row_data
            camera_id_dict[idx] = view_id

    return poses_dict, camera_id_dict
