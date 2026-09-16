from torch.utils.data import Dataset
import json

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat
import cv2
import re
import os
import copy
import torch
import torch.nn.functional as F

class HypersimPreprocess(Dataset):
    def __init__(
        self, 
        json_path, 
        data_root,
        output_root,
        depth_scale = 1.0,
        rbg_name: str = "rgb",
        depth_name: str = "depth",
        mask_name: str = "mask",
        sem_name: str = "sem_id",
        normal_name: str = "normal",
        intrinsics_name: str = "cam_in",
        extrinsics_name: str = "extrinsics",
    ):
        self.data_root = data_root
        self.output_root = output_root
        
        self.rbg_name = rbg_name
        self.depth_name = depth_name
        self.mask_name = mask_name
        self.sem_name = sem_name
        self.normal_name = normal_name
        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.depth_scale = depth_scale
        
        self.data_infos = self.load_json(json_path)
    
    def read_image(self, image_path):
        """
        Reads an image from the specified file path and returns it as a NumPy array.

        Args:
            image_path (str or None): The file path to the image. If None, returns None.

        Returns:
            np.ndarray or None: The image as a NumPy array in RGB format, or None if image_path is None.

        Raises:
            RuntimeError: If the file extension is not supported.
        """
        # Return None if the image_path is None.
        if image_path is None:
            return None

        # Extract the file extension from the image_path.
        data_type = os.path.splitext(image_path)[-1].lower()

        # List of supported image file types.
        img_file_type = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

        # Handle different file types.
        if data_type in img_file_type:
            # Open the image using PIL and convert it to RGB format.
            data = Image.open(image_path).convert("RGB")  # [H, W, rgb]
            # Convert the PIL Image to a NumPy array.
            data = np.asarray(data)
        elif data_type in [".hdf5", ".h5"]:
            # Open the HDF5 file and read the dataset.
            with h5py.File(image_path, "r") as f:
                data = np.array(f["dataset"])
        else:
            # Raise an error if the file type is not supported.
            raise RuntimeError(f"File type {data_type} is not supported in the current version.")

        return data

    def read_file(self, file_path, image=None, const=0, dtype=np.float32):
        """
        Reads data from the specified file path or creates a constant-valued array based on the given image dimensions.

        Args:
            file_path (str or None): The file path to the data file. If None, uses the dimensions from the provided image.
            image (np.ndarray or None): An optional image to infer dimensions if file_path is None.
            const (float): The constant value to fill the array if file_path is None.
            dtype (type): The data type for the output array.

        Returns:
            np.ndarray or None: The data as a NumPy array, or None if both file_path and image are None.

        Raises:
            RuntimeError: If the file extension is not supported.
        """
        # Return None if both file_path and image are None.
        if file_path is None and image is None:
            return None

        # Create a constant-valued array based on the image dimensions if file_path is None.
        if file_path is None or not os.path.exists(file_path):
            data = np.zeros(image.shape, dtype=dtype) + const
        else:
            # Extract the file extension from the file_path.
            data_type = os.path.splitext(file_path)[-1].lower()

            # List of supported image file types.
            img_file_type = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

            # Handle different file types.
            if data_type in img_file_type:
                # Open the image using PIL and convert it to a NumPy array.
                data = Image.open(file_path)
                data = np.asarray(data)
            elif data_type in [".npz", ".npy"]:
                # Load the data from a .npz or .npy file.
                data = np.load(file_path)
                # If it's a .npz file, extract the first array (assuming single array storage).
                if isinstance(data, np.lib.npyio.NpzFile):
                    data = data[data.files[0]]
            elif data_type in [".hdf5", ".h5"]:
                # Open the HDF5 file and read the dataset.
                with h5py.File(file_path, "r") as f:
                    data = np.array(f["dataset"])
            elif data_type in [".pfm"]:
                data = self.load_pfm(file_path)
            elif data_type == ".exr":
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                data = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            elif data_type in [".mat"]:
                data = loadmat(file_path)
                keys = list(data.keys())
                for key in ["__header__", "__version__", "__globals__"]:
                    if key in keys:
                        keys.remove(key)
                data = data[keys[0]].squeeze()
                assert isinstance(data, np.ndarray)
            elif data_type in [".dpt"]:
                with open(file_path, "rb") as f:
                    _ = np.fromfile(f, dtype=np.float32, count=1)[0]
                    width = np.fromfile(f, dtype=np.int32, count=1)[0]
                    height = np.fromfile(f, dtype=np.int32, count=1)[0]
                    size = width * height
                    assert (
                        width > 0 and height > 0 and size > 1 and size < 100000000
                    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
                        width, height
                    )
                    data = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
            elif data_type in [".float3"]:
                with open(file_path, "rb") as f:
                    if (f.readline().decode("utf-8")) != "float\n":
                        raise Exception("float file %s did not contain <float> keyword" % file_path)

                    dim = int(f.readline())

                    dims = []
                    count = 1
                    for i in range(0, dim):
                        d = int(f.readline())
                        dims.append(d)
                        count *= d

                    dims = list(reversed(dims))
                    data = np.fromfile(f, np.float32, count).reshape(dims)
            else:
                # Raise an error if the file type is not supported.
                raise RuntimeError(
                    f"File type {data_type} is not supported in the current version."
                )

        # Ensure the data is of the specified data type.
        data = data.astype(dtype)

        return data

    def load_pfm(self, depth_path):
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        with open(depth_path, "rb") as file:
            header = file.readline().decode("UTF-8").rstrip()
            if header == "PF":
                color = True
            elif header == "Pf":
                color = False
            else:
                raise Exception("Not a PFM file.")
            dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception("Malformed PFM header.")
            # scale = float(file.readline().rstrip())
            scale = float((file.readline()).decode("UTF-8").rstrip())
            if scale < 0:  # little-endian
                data_type = "<f"
            else:
                data_type = ">f"  # big-endian
            data_string = file.read()
            data = np.fromstring(data_string, data_type)
            shape = (height, width, 3) if color else (height, width)
            data = np.reshape(data, shape)
            data = cv2.flip(data, 0)

        return data

    def load_data_path(self, data_info):
        curr_rgb_path = os.path.join(self.data_root, data_info[self.rbg_name])
        curr_depth_path = (
            os.path.join(self.data_root, data_info[self.depth_name])
            if data_info.get(self.depth_name, None) is not None
            else None
        )
        curr_depth_mask_path = (
            os.path.join(self.data_root, data_info[self.mask_name])
            if data_info.get(self.mask_name, None) is not None
            else None
        )
        curr_sem_path = (
            os.path.join(self.data_root, data_info[self.sem_name])
            if data_info.get(self.sem_name, None) is not None
            else None
        )
        curr_norm_path = (
            os.path.join(self.data_root, data_info[self.normal_name])
            if data_info.get(self.normal_name, None) is not None
            else None
        )
        data_path = dict(
            rgb_path=curr_rgb_path,
            depth_path=curr_depth_path,
            depth_mask_path=curr_depth_mask_path,
            sem_path=curr_sem_path,
            normal_path=curr_norm_path,
        )
        return data_path

    def load_normal(self, normal_path, image, const, dtype):
        return self.read_file(normal_path, image=image, const=const, dtype=dtype)

    def load_depth(self, depth_path, image, const, dtype):
        return self.read_file(depth_path, image=image, const=const, dtype=dtype)

    def load_data(self, data_info):
        curr_rgb = curr_depth = curr_depth_mask = None
        curr_normal = None
        data_path = self.load_data_path(data_info)
        curr_intrinsics = data_info.get(self.intrinsics_name, None)
        
        # 加载RGB图像，形状为[h, w, 3]，即高度、宽度和RGB通道
        if data_path["rgb_path"] is not None:
            curr_rgb = self.read_image(data_path["rgb_path"]).copy()
        if data_path["depth_path"] is not None:
            curr_depth = self.load_depth(
                data_path["depth_path"], image=None, const=None, dtype=np.float32
            )
            if len(curr_depth.shape) == 3:
                curr_depth = curr_depth[..., -1]
            if len(curr_depth.shape) == 1:
                curr_depth = curr_depth.reshape(*curr_rgb.shape[:2])
        if curr_intrinsics is not None and isinstance(curr_intrinsics, str):
            if curr_intrinsics.endswith(".npy"):
                curr_intrinsics = np.load(curr_intrinsics).reshape(3, 3)
            else:
                with open(curr_intrinsics, "r") as f:
                    curr_intrinsics = np.array(json.load(f)).reshape(3, 3)
            curr_intrinsics = curr_intrinsics[[0, 1, 0, 1], [0, 1, 2, 2]].tolist()
        # 根据数据信息中的深度缩放因子（depth_scale）对深度图进行缩放
        if "depth_scale" in data_info and self.depth_scale != data_info["depth_scale"]:
            self.depth_scale = data_info["depth_scale"]
        elif "depth_scale" not in data_info and self.depth_scale is None:
            self.depth_scale = 1
        
        # 如果深度提示存在，同样对其进行缩放
        if curr_depth is not None:
            curr_depth = curr_depth / self.depth_scale
        # 加载深度掩码（Depth Mask），用于标记深度图中的有效区域
        if curr_depth is not None:
            curr_depth_mask = self.read_file(
                data_path["depth_mask_path"], image=curr_depth, const=1, dtype=int
            )
            if len(curr_depth_mask.shape) == 3:
                curr_depth_mask = curr_depth_mask[..., -1]
            # 如果深度图存在，更新深度掩码以移除无效深度值（NaN值），并设置无效区域的深度值为0
            curr_depth_mask = (
                curr_depth_mask
                * (~np.isnan(curr_depth)).astype(int)
                * (~np.isinf(curr_depth)).astype(int)
            )
            curr_depth[~curr_depth_mask.astype(bool)] = 0
        if data_path["normal_path"] is not None:
            curr_normal = self.load_normal(
                data_path["normal_path"], image=curr_rgb, const=0, dtype=np.float32
            )
        
        data_batch = dict(
            curr_intrinsics=curr_intrinsics,
            curr_rgb=curr_rgb,
            curr_depth=curr_depth,
            curr_depth_mask=curr_depth_mask,
            curr_normal=curr_normal,
        )
        return data_batch

    def __len__(self):
        return len(self.data_infos)

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)["files"]
        return data
    
    def save_json(self, data_info, json_path):
        with open(json_path, 'w') as f:
            json.dump({"files":data_info}, f, indent=2)

    def preprocess_items(self, item_idx, preprocess_funcs):
        info = self.data_infos[item_idx]
        data_batch = self.load_data(info)
        update_info = copy.deepcopy(info)
        for func in preprocess_funcs:
            update_info = func(data_batch, update_info, self.output_root)
        return update_info

def preprocess_normal_svd_json_only(data_batch, data_info, output_root):
    normal_prefix = 'NormalSVD_v2/'
    normal_path = os.path.join(normal_prefix, 'normal', data_info["rgb"].replace('.jpg', '.npy'))
    normal_mask_path = os.path.join(normal_prefix, 'normal_mask', data_info["rgb"].replace('.jpg', '.npy'))
    data_info["normal_svd"] = normal_path
    data_info["normal_mask_svd"] = normal_mask_path
    return data_info
    

def preprocess_normal_svd(data_batch, data_info, output_root):
    normal_prefix = 'NormalSVD_v2/'
    normal_path = os.path.join(normal_prefix, 'normal', data_info["rgb"].replace('.jpg', '.npy'))
    normal_mask_path = os.path.join(normal_prefix, 'normal_mask', data_info["rgb"].replace('.jpg', '.npy'))
    normal_vis_path = os.path.join(normal_prefix, 'normal_vis', data_info["rgb"])
    
    curr_depth = torch.as_tensor(data_batch["curr_depth"]).cuda()
    curr_depth_mask = torch.as_tensor(data_batch["curr_depth_mask"]).cuda()
    
    if "curr_pointmap" in data_batch:
        curr_pointmap = torch.as_tensor(data_batch["curr_pointmap"]).cuda()
    else:
        curr_intrinsics = data_batch["curr_intrinsics"]
        curr_intrinsics_mat = create_intrinsics_matrix(curr_intrinsics)
        curr_pointmap = load_pointmap(curr_depth, curr_intrinsics_mat).cuda()
        data_batch["curr_pointmap"] = curr_pointmap.cpu().numpy()
    
    normal_svd, normal_svd_mask = pointmap_to_normal_svd(curr_pointmap.permute(1,2,0), curr_depth_mask, patch_size=3)
    normal_svd_mask = (curr_depth_mask & normal_svd_mask).bool()
    
    data_info["normal_svd"] = normal_path
    data_info["normal_mask_svd"] = normal_mask_path
    
    normal_path = os.path.join(output_root, normal_path)
    normal_mask_path = os.path.join(output_root, normal_mask_path)
    normal_vis_path = os.path.join(output_root, normal_vis_path)
    
    os.makedirs(os.path.dirname(normal_path), exist_ok=True)
    np.save(normal_path, normal_svd.permute(1, 2, 0).cpu().numpy())
    
    os.makedirs(os.path.dirname(normal_mask_path), exist_ok=True)
    np.save(normal_mask_path, normal_svd_mask.cpu().numpy())
    
    normal_svd = torch.where(normal_svd_mask, normal_svd, 0).permute(1,2,0)
    normal_colored = normal_svd.cpu().numpy() * [0.5, -0.5, -0.5] + 0.5
    normal_colored = (normal_colored.clip(0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(normal_vis_path), exist_ok=True)
    cv2.imwrite(normal_vis_path, cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR))
    return data_info

def pointmap_to_normal_svd(pointmap: torch.Tensor, depth_mask: torch.Tensor, patch_size: int = 3):
    """
    Compute surface normals via local plane fitting using SVD.

    Args:
        pointmap: [B, H, W, 3] or [H, W, 3] — 3D points in camera coordinates
        depth_mask: [B, H, W] or [H, W] — boolean mask (True = valid)
        patch_size: int, must be odd (e.g., 3)

    Returns:
        normals: [B, 3, H, W] — normalized surface normals
        valid_mask: [B, H, W] — True where normal is valid
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    squeeze_batch = False
    # Standardize pointmap to [B, H, W, 3]
    if pointmap.dim() == 3:
        pointmap = pointmap.unsqueeze(0)  # [H, W, 3] -> [1, H, W, 3]
        squeeze_batch = True
    assert pointmap.dim() == 4 and pointmap.shape[-1] == 3

    # Standardize depth_mask to [B, H, W]
    if depth_mask.dim() == 2:
        depth_mask = depth_mask.unsqueeze(0)  # [H, W] -> [1, H, W]
    assert depth_mask.dim() == 3, f"depth_mask must be 2D or 3D, got {depth_mask.dim()}D"

    B, H, W, _ = pointmap.shape
    assert depth_mask.shape == (
        B,
        H,
        W,
    ), f"Shape mismatch: pointmap {pointmap.shape}, mask {depth_mask.shape}"

    device = pointmap.device

    half = patch_size // 2

    # Pad pointmap and mask
    # Use constant padding with zeros; we'll mask out invalid points later
    pointmap_pad = F.pad(
        pointmap, (0, 0, half, half, half, half), mode="constant", value=0.0
    )  # [B, H+p, W+p, 3]
    mask_pad = F.pad(
        depth_mask.float(), (half, half, half, half), mode="constant", value=0.0
    ).bool()  # [B, H+p, W+p]

    # Unfold to get patches: [B, H, W, patch_size, patch_size, 3]
    patches_xyz = pointmap_pad.unfold(1, patch_size, 1).unfold(
        2, patch_size, 1
    )  # [B, H, W, 3, ps, ps]
    patches_xyz = patches_xyz.permute(0, 1, 2, 4, 5, 3)  # [B, H, W, ps, ps, 3]

    patches_mask = mask_pad.unfold(1, patch_size, 1).unfold(2, patch_size, 1)  # [B, H, W, ps, ps]

    # Reshape to [B*H*W, ps*ps, 3] for batched SVD
    patches_xyz = patches_xyz.reshape(B * H * W, patch_size * patch_size, 3)
    patches_mask = patches_mask.reshape(B * H * W, patch_size * patch_size)

    # Count valid points per patch
    valid_counts = patches_mask.sum(dim=1)  # [B*H*W]

    # Minimum number of points to fit a plane (at least 3 non-collinear)
    min_valid = 3
    enough_points = valid_counts >= min_valid  # [B*H*W]

    # Initialize normals
    normals = torch.zeros(B * H * W, 3, device=device)

    if enough_points.any():
        # Select only patches with enough valid points
        valid_patches_xyz = patches_xyz[enough_points]  # [N, K, 3]
        valid_patches_mask = patches_mask[enough_points]  # [N, K]

        # Zero out invalid points (set to 0, but they won't affect centroid if masked)
        # Better: compute centroid only from valid points
        valid_counts_sel = valid_counts[enough_points].float()  # [N]

        # Compute centroid (mean of valid points)
        weighted_sum = (valid_patches_xyz * valid_patches_mask.unsqueeze(-1)).sum(dim=1)  # [N, 3]
        centroid = weighted_sum / valid_counts_sel.unsqueeze(-1)  # [N, 3]

        # Center the points
        centered = valid_patches_xyz - centroid.unsqueeze(1)  # [N, K, 3]

        # Zero out invalid points in centered (to avoid NaN in SVD)
        centered = centered * valid_patches_mask.unsqueeze(-1)

        # Perform SVD: [N, K, 3] -> compute covariance-like matrix
        # We can use torch.svd on each patch, but it's slow. Instead, compute SVD directly.
        # Note: SVD of centered (N, K, 3) gives U, S, V where V is [N, 3, 3]
        try:
            _, _, V = torch.svd(centered)
        except RuntimeError as e:
            # Fallback: use eig of covariance matrix
            cov = torch.bmm(centered.transpose(-1, -2), centered)  # [N, 3, 3]
            _, V = torch.linalg.eigh(cov)
            V = V.transpose(-1, -2)

        # Normal is the last column of V (smallest singular value)
        normal_est = V[:, :, -1]  # [N, 3]

        # Re-orient: normal should point towards camera (i.e., opposite to view direction)
        # View direction ≈ centroid (from origin to surface)
        dot = (normal_est * centroid).sum(dim=1, keepdim=True)  # [N, 1]
        normal_est = torch.where(dot > 0, -normal_est, normal_est)

        # Store back
        normals[enough_points] = normal_est

    # Reshape back to [B, H, W, 3]
    normals = normals.reshape(B, H, W, 3)
    valid_mask = enough_points.reshape(B, H, W)

    # Normalize (in case of numerical issues)
    norm = torch.norm(normals, dim=-1, keepdim=True)
    normals = normals / (norm + 1e-8)

    # Zero out invalid normals
    normals = normals * valid_mask.unsqueeze(-1)

    # Output format: [B, 3, H, W]
    normals = normals.permute(0, 3, 1, 2).contiguous()

    if squeeze_batch:
        normals = normals.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
    return normals, valid_mask

def load_pointmap(depth, intrinsics):
    """
    Generates a point cloud map from a depth map and intrinsics camera parameters.

    Args:
        depth (torch.Tensor or np.ndarray): The depth map as a tensor or NumPy array.
        intrinsics (torch.Tensor or np.ndarray): The intrinsics camera parameters as a tensor or NumPy array.

    Returns:
        torch.Tensor: The generated point cloud map as a tensor.
    """
    # Convert tensors to NumPy arrays if necessary.
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze(0).cpu().numpy()  # [ 1, h, w] -> [h, w]
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()

    height, width = depth.shape

    # Create a grid of pixel coordinates (u, v).
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5, indexing="xy")
    uv = np.stack([u, v], axis=-1)

    # Convert the pixel coordinates to homogeneous coordinates.
    uv_homogeneous = np.concatenate([uv, np.ones([height, width, 1])], axis=-1).reshape(-1, 3)

    # Invert the intrinsics matrix to convert from image coordinates to world coordinates.
    K_inv = np.linalg.inv(intrinsics)

    # Compute the direction vectors from the camera to each pixel.
    directions = uv_homogeneous @ K_inv.T

    # Multiply the direction vectors by the depth values to get the 3D points.
    points = depth.reshape(-1, 1) * directions

    # Center and normalize the point cloud based on the valid points.
    # points = self.center_and_normalize_point_cloud(points, valid_mask.reshape(-1))

    # Reshape the point cloud to match the original depth map dimensions and transpose axes.
    points = points.reshape(height, width, 3).transpose(2, 0, 1)

    # Clip the point cloud values to be within [-1.0, 1.0].
    # points = np.clip(points, a_min=-1.0, a_max=1.0)

    # Convert the point cloud back to a PyTorch tensor.
    points = torch.from_numpy(points).float()

    return points

def create_intrinsics_matrix(intrinsics):
    """Create an intrinsics matrix from a list of parameters."""
    intrinsics_mat = torch.zeros((3, 3)).float()
    intrinsics_mat[0, 0] = intrinsics[0]
    intrinsics_mat[1, 1] = intrinsics[1]
    intrinsics_mat[0, 2] = intrinsics[2]
    intrinsics_mat[1, 2] = intrinsics[3]
    intrinsics_mat[2, 2] = 1.0
    return intrinsics_mat

if __name__ == '__main__':
    data_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_root = '/mnt/netdata/Team/AI/datasets/TMD/'
    output_meta_root = '/mnt/netdata/Team/AI/datasets/TMA_preprocess'
    version = 'v1_251016'
    
    dataset_name = 'hypersim'
    json_paths = [
        '/mnt/netdata/Team/AI/datasets/TMD/Hypersim/data_split_marigold_v2/test.json',
        '/mnt/netdata/Team/AI/datasets/TMD/Hypersim/data_split_marigold_v2/train.json',
    ]
    debug = False
    
    preprocess_funcs = [
        preprocess_normal_svd
    ]
    for json_path in json_paths:
        json_name = os.path.basename(json_path).split(".")[0]
        preprocessor = HypersimPreprocess(
            json_path,
            data_root,
            output_root,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        def preprocess_single_item(idx):
            """线程安全的单条目预处理函数"""
            return idx, preprocessor.preprocess_items(idx, preprocess_funcs)
        
        # 初始化结果列表（保持原始顺序）
        n_items = len(preprocessor)
        if debug:
            n_items = 100
        processed_infos = [None] * n_items
        
        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(preprocess_single_item, i) for i in range(n_items)]
            
            # 使用 tqdm 包装 as_completed 以显示进度
            for future in tqdm(as_completed(futures), total=n_items, desc=f'Preprocessing dataset {dataset_name}, split: {json_name}'):
                idx, result = future.result()
                processed_infos[idx] = result
        print(preprocessor.depth_scale)
        # for i in tqdm(range(len(preprocessor)), desc='preprocessing dataset', total=len(preprocessor)):
        #     processed_infos[i] = preprocessor.preprocess_items(i, preprocess_funcs)
        if not debug:
            os.makedirs(f'{output_meta_root}/{version}', exist_ok=True)
            preprocessor.save_json(processed_infos, f'{output_meta_root}/{version}/{dataset_name}_{json_name}.json')
