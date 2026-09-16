import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def revalue(map, lower, upper, start, scale):
    mask = (map > lower) & (map <= upper)
    if np.sum(mask) >= 1.0:
        mn, mx = map[mask].min(), map[mask].max()
        map[mask] = ((map[mask] - mn) / (mx - mn + 1e-7)) * scale + start

    return map


def depth_err_to_colorbar(est, gt=None, with_bar=False, cmap="jet"):
    error_bar_height = 50
    if gt is None:
        gt = np.zeros_like(est)
        valid = est > 0
        max_depth = est.max()
    else:
        valid = gt > 0
        max_depth = gt.max()
    error_map = np.abs(est - gt) * valid
    h, w = error_map.shape

    maxvalue = error_map.max()
    if max_depth < 30:
        breakpoints = np.array([0, 0.1, 0.5, 1.25, 2, 4, max(10, maxvalue)])
    else:
        breakpoints = np.array([0, 0.1, 0.5, 1.25, 2, 4, max(90, maxvalue)])
    points = np.array([0, 0.25, 0.38, 0.66, 0.83, 0.95, 1])
    num_bins = np.array(
        [
            0,
            w // 8,
            w // 8,
            w // 4,
            w // 4,
            w // 8,
            w - (w // 4 + w // 4 + w // 8 + w // 8 + w // 8),
        ]
    )
    acc_num_bins = np.cumsum(num_bins)

    for i in range(1, len(breakpoints)):
        scale = points[i] - points[i - 1]
        start = points[i - 1]
        lower = breakpoints[i - 1]
        upper = breakpoints[i]
        error_map = revalue(error_map, lower, upper, start, scale)

    # [0, 1], [H, W, 3]
    error_map = plt.cm.get_cmap(cmap)(error_map)[:, :, :3]

    # mark invalid px as black
    error_map = error_map * valid[:, :, None]

    if not with_bar:
        return error_map

    error_bar = np.array([])
    for i in range(1, len(num_bins)):
        error_bar = np.concatenate((error_bar, np.linspace(points[i - 1], points[i], num_bins[i])))

    error_bar = (
        np.repeat(error_bar, error_bar_height).reshape(w, error_bar_height).transpose(1, 0)
    )  # [error_bar_height, w]
    error_bar_map = plt.cm.get_cmap(cmap)(error_bar)[:, :, :3]
    plt.xticks(ticks=acc_num_bins, labels=[str(f) for f in breakpoints])
    plt.axis("on")

    # [0, 1], [H, W, 3]
    error_map = np.concatenate((error_map, error_bar_map[..., :3]), axis=0)[..., :3]

    return error_map


def sparse_to_dense_depth(sparse_depth_map):
    """
    Converts a sparse depth map to a dense depth map using bilinear interpolation.
    Args:
    sparse_depth_map (np.ndarray): Sparse depth map where depth values are non-zero at specific locations.
    resolution (tuple): The resolution (height, width) of the desired dense depth map.
    Returns:
    np.ndarray: Dense depth map.
    """
    # Get non-zero indices (x, y positions of known depth values)
    sparse_depth_map = sparse_depth_map.squeeze().cpu().numpy()
    resolution = sparse_depth_map.shape
    y_indices, x_indices = np.nonzero(sparse_depth_map)
    # Get depth values at those positions
    known_depth_values = sparse_depth_map[y_indices, x_indices]
    # Create a grid of coordinates for the desired dense depth map
    x, y = np.meshgrid(np.arange(resolution[1]), np.arange(resolution[0]))
    # Perform bilinear interpolation
    dense_depth_map = griddata((x_indices, y_indices), known_depth_values, (x, y), method="linear")
    dense_depth_map = np.nan_to_num(dense_depth_map, nan=0.0)
    return dense_depth_map


def compute_depth_distribution(images, name="base"):
    """计算所有深度图的深度分布"""
    all_data = np.concatenate([img.flatten() for img in images])
    hist, bins = np.histogram(all_data, bins=256, range=(np.min(all_data), np.max(all_data)))
    plot_distribution(hist, bins, name)
    return hist, bins


def plot_distribution(hist, bins, name="base"):
    """绘制深度分布直方图"""
    plt.figure()
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), color="blue")
    plt.title("Depth Distribution")
    plt.xlabel("Depth Value")
    plt.ylabel("Frequency")
    plt.savefig(f"depth_distribution_{name}.png")
    plt.close()
    print("save:", f"depth_distribution_{name}.png")


# if __name__ == "__main__":
#     from hAlgorithm.datasets.base_dataset import BaseDataset
#     dataset = BaseDataset(
#         phase="train",
#         name="base",
#         seed=0,
#         data_root="/mnt/personal/TMD/datasets/",
#         data_path="/mnt/personal/ts/data/Marigold/data_split/hypersim/val80.json",
#         sampling_strategy="all",
#         train_transforms=[
#             dict(
#                 type="hAlgorithm.datasets.transforms.transforms.Resize",
#                 width=640,
#                 height=480,
#             ),
#             dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
#             dict(
#                 type="hAlgorithm.datasets.transforms.transforms.Normalize",
#                 mean=[127.5, 127.5, 127.5],
#                 std=[127.5, 127.5, 127.5],
#             ),
#             # x / 255 * 2 - 1
#         ],
#         test_transforms=[
#             dict(
#                 type="hAlgorithm.datasets.transforms.transforms.Resize",
#                 width=640,
#                 height=480,
#             ),
#             dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
#             dict(
#                 type="hAlgorithm.datasets.transforms.transforms.Normalize",
#                 mean=[127.5, 127.5, 127.5],
#                 std=[127.5, 127.5, 127.5],
#             ),
#         ],
#         depth_scale=1000.0,
#         min_depth=1e-5,
#         max_depth=100.0,
#         normalize_depth=dict(
#             type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
#             norm_min=-1.0,
#             norm_max=1.0,
#             min_max_quantile=0.02,
#             clip=True,
#         ),
#         with_pointmap=True,
#         sparse_depth_ratio=0.05,
#         sky_index=-1,
#         recalculate_normal=True,
#         normal_transform=dict(
#             type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"
#         ),
#         debug=True,
#     )

#     for index in range(50):
#         print(index)
#         data = dataset.__getitem__(index)
#         sparse_depth = data["sparse_depth"]
#         sparse_depth = sparse_to_dense_depth(sparse_depth)

#         sparse_depth = (sparse_depth - sparse_depth.min()) / (sparse_depth.max() - sparse_depth.min())

#         sparse_depth_colored = colorize_depth_maps(
#             sparse_depth,
#             0,
#             1,
#             cmap="turbo",
#             valid_mask=sparse_depth_mask,
#         )
#         save_path = os.path.join("debug/sparse_to_dense/", f"{idx:04d}.jpg")
#         sparse_depth_colored.save(save_path)


def compute_depth_distribution(images, path, name="base"):
    """计算所有深度图的深度分布"""
    all_data = images

    def save_plot(data, mode):
        try:
            hist, bins = np.histogram(data, bins=256, range=(np.min(data), min(np.max(data), 250)))
            plot_distribution(hist, bins, path, name, mode=mode)
        except:
            print("distribution out of range")

    save_plot(all_data[:, 0], "min")
    save_plot(all_data[:, -1], "max")
    save_plot(all_data[:, -2], "max0.95")
    save_plot(all_data[:, 10], "median")
    save_plot(all_data.reshape(-1), "all")
    return np.histogram(all_data, bins=256, range=(np.min(all_data), min(np.max(all_data), 250)))


def plot_distribution(hist, bins, path, name="base", mode="all"):
    """绘制深度分布直方图"""
    from matplotlib import pyplot as plt

    plt.figure()
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), color="blue")
    plt.title(f"{mode} Depth Distribution")
    plt.xlabel("Depth Value")
    plt.ylabel("Frequency")
    plt.savefig(f"{path}/depth_{mode}_distribution_{name}.png")
    plt.close()
    print("save:", f"{path}/depth_{mode}_distribution_{name}.png")


def compute_zero_regions(mask, thresh=1):
    """
    Compute the regions on the top, bottom, left, and right of a binary mask
    that are completely zero.

    Args:
        mask (np.ndarray): Binary mask of shape [H, W], where 0 indicates invalid regions.

    Returns:
        dict: A dictionary containing the number of rows/columns that are completely zero
              in the top, bottom, left, and right regions.
    """
    H, W = mask.shape

    # Initialize results
    top_zero = 0
    bottom_zero = 0
    left_zero = 0
    right_zero = 0

    # Compute top zero region
    for i in range(H):
        if mask[i, :].sum() < thresh:  # Check if the entire row is zero
            top_zero += 1
        else:
            break

    # Compute bottom zero region
    for i in range(H - 1, -1, -1):
        if mask[i, :].sum() < thresh:  # Check if the entire row is zero
            bottom_zero += 1
        else:
            break

    # Compute left zero region
    for j in range(W):
        if mask[:, j].sum() < thresh:  # Check if the entire column is zero
            left_zero += 1
        else:
            break

    # Compute right zero region
    for j in range(W - 1, -1, -1):
        if mask[:, j].sum() < thresh:  # Check if the entire column is zero
            right_zero += 1
        else:
            break

    return top_zero, bottom_zero, left_zero, right_zero
