import os
import struct
import numpy as np
from plyfile import PlyData, PlyElement
from collections import defaultdict
from tqdm import tqdm


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


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_images_bin(path):
    """
    读取 COLMAP images.bin 返回
    - 图像ID → {name, r, t, camera_id, ...}
    - 所有相机中心点 positions (Nx3)
    """
    images = {}
    positions = []

    images = []
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            R = qvec2rotmat(qvec)
            T = tvec

            rotation = qvec2rotmat(qvec)
            translation = tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
            c2w = np.linalg.inv(w2c)

            # images[image_id] = 
            img = {
                'name': image_name,
                'qvec': qvec,
                'tvec': tvec,
                # 'R': R,
                # 'C': C,
                'camera_id': camera_id,
                "image_id": image_id,
            }
            images.append(img)
            positions.append(c2w[:3, 3])

    return images, np.array(positions)


def read_images_txt(path):
    """
    读取 COLMAP images.txt
    返回:
        images: list of dict, 每个元素包含 'name', 'qvec', 'tvec', 'camera_id', 'image_id'
        positions: (N, 3) 相机在世界坐标系中的位置
    """
    images = []
    positions = []

    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if len(line) == 0 or line.startswith('#'):
            i += 1
            continue

        # 第1行：图像参数
        parts = line.split()
        if len(parts) < 10:
            i += 2  # 跳过无效行 + 下一行
            continue

        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]])  # QW, QX, QY, QZ
        tvec = np.array([float(x) for x in parts[5:8]])  # TX, TY, TZ
        camera_id = int(parts[8])
        image_name = " ".join(parts[9:])  # 支持文件名带空格

        # 计算相机位置（世界坐标系）
        R = qvec2rotmat(qvec)
        t = tvec.reshape(3, 1)
        w2c = np.hstack([R, t])  # 3x4
        w2c = np.vstack([w2c, [0, 0, 0, 1]])  # 4x4
        c2w = np.linalg.inv(w2c)
        camera_position = c2w[:3, 3]  # (3,)

        img = {
            'name': image_name,
            'qvec': qvec,
            'tvec': tvec,
            'camera_id': camera_id,
            'image_id': image_id,
        }
        images.append(img)
        positions.append(camera_position)

        # 跳过第2行（2D 点观测）
        i += 2

    return images, np.array(positions)


def read_points3D_bin(path):
    """
    读取 COLMAP 的 points3D.bin 文件
    返回：点云位置 (Nx3), 颜色 (Nx3), 点ID (N,)
    """
    points3D_imgs = defaultdict(list)
    points3D = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]  # uint64
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = np.array(struct.unpack("<ddd", fid.read(24)))  # double x3
            rgb = np.array(struct.unpack("<BBB", fid.read(3)))   # uchar x3
            error = struct.unpack("<d", fid.read(8))[0]
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("<I", fid.read(4))[0]
                point2D_idx = struct.unpack("<I", fid.read(4))[0]
                track.append((image_id, point2D_idx))
            
            if error < 2:
                points3D[point3D_id] = {
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error,
                    'track': track,
                }
                for tk in track:
                    image_id, point2D_idx = tk
                    points3D_imgs[image_id].append([xyz, rgb])
            # else:
            #     print(point3D_id)

    xyzs = np.array([v['xyz'] for v in points3D.values()])
    rgbs = np.array([v['rgb'] for v in points3D.values()])
    ids = np.array(list(points3D.keys()))

    # imgs_xyzs = np.array([points3D_imgs[image_id][0] ])
    # imgs_rgbs = np.array([points3D_imgs[image_id][1] for image_id in points3D_imgs.keys()])

    imgs_xyzs = dict()
    imgs_rgbs = dict()
    # for image_id in points3D_imgs.keys():
    #     cur_xyzs = np.array([d[0] for d in points3D_imgs[image_id]])
    #     cur_rgbs = np.array([d[1] for d in points3D_imgs[image_id]])

    #     imgs_xyzs[image_id] = cur_xyzs
    #     imgs_rgbs[image_id] = cur_rgbs

    return xyzs, rgbs, ids, imgs_xyzs, imgs_rgbs

def write_ply(points, colors, filename):
    """
    将点云写入 .ply 文件
    """
    vertex = np.array([tuple(p) + tuple(c) for p, c in zip(points, colors)],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(filename)
    print(f"Saved {len(points)} points to {filename}")

# ===== 使用示例 =====
if __name__ == "__main__":
    points3D_path = "nas_debug/pycolmap2/sparse/0/points3D.bin"  # 替换为你的路径
    
    ply_path = "./debug.ply"
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    xyzs, rgbs, ids, imgs_xyzs, imgs_rgbs = read_points3D_bin(points3D_path)
    write_ply(xyzs, rgbs, ply_path)

