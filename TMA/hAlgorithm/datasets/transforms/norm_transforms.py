import numpy as np


class DepthToNormal(object):
    def __init__(self):
        pass

    def __call__(self, intrinsics, depth):
        """
        Converts a depth map to a normal map using the camera's projection matrix.

        Args:
            intrinsics (tuple): A tuple of four floats (fx, fy, cx, cy) representing the camera's intrinsics parameters.
            depth (numpy.ndarray): A 2D array of shape [height, width] representing the depth map.

        Returns:
            numpy.ndarray: A 3D array of shape [height, width, 3] representing the normal map.
        """

        img_height, img_width = depth.shape

        # Pad the depth map to keep the output size the same
        depth_padded = np.pad(depth, pad_width=1, mode="edge")

        # Calculate the intrinsics camera parameters from the projection matrix
        fx, fy, cx, cy = intrinsics

        # Create a grid of pixel coordinates in the image plane for the padded depth map
        xx, yy = np.meshgrid(np.arange(img_width + 2), np.arange(img_height + 2))
        xx = (xx - cx) / fx  # Normalize X coordinates
        yy = (yy - cy) / fy  # Normalize Y coordinates

        # Combine with normalized coordinates to get 3D points for the padded depth map
        inv_proj_xyz = np.stack((xx * depth_padded, yy * depth_padded, depth_padded), axis=-1)

        # Compute depth differences between neighboring pixels for the padded depth map
        depth_center = depth_padded[1:-1, 1:-1]
        depth_diff_left = np.abs(depth_center - depth_padded[1:-1, :-2])[..., None]
        depth_diff_right = np.abs(depth_center - depth_padded[1:-1, 2:])[..., None]
        depth_diff_up = np.abs(depth_center - depth_padded[:-2, 1:-1])[..., None]
        depth_diff_down = np.abs(depth_center - depth_padded[2:, 1:-1])[..., None]

        # Extract the central region of the 3D points for normal calculation
        inv_proj_xyz_center = inv_proj_xyz[1:-1, 1:-1, :]

        # Calculate gradients (dx, dy) by subtracting adjacent 3D points
        dx_left = inv_proj_xyz_center - inv_proj_xyz[1:-1, :-2, :]
        dx_right = inv_proj_xyz[1:-1, 2:, :] - inv_proj_xyz_center
        dy_up = inv_proj_xyz[:-2, 1:-1, :] - inv_proj_xyz_center
        dy_down = inv_proj_xyz_center - inv_proj_xyz[2:, 1:-1, :]

        # Choose the gradient with the smaller depth difference to reduce noise
        dx = np.where(depth_diff_left < depth_diff_right, dx_left, dx_right)
        dy = np.where(depth_diff_up < depth_diff_down, dy_up, dy_down)

        # Compute the cross product of the gradients to get the normal vectors
        normal_from_depth = np.cross(dx, dy, axis=-1)

        # Adjust the direction of the normals to point away from the camera
        view_dir = inv_proj_xyz_center
        dot_product = np.sum(view_dir * normal_from_depth, axis=-1, keepdims=True)
        normal_from_depth = normal_from_depth * ((dot_product < 0).astype(float) * 2 - 1)

        # Normalize the normal vectors to unit length
        norm = np.linalg.norm(normal_from_depth, axis=-1, keepdims=True)
        normal_from_depth = normal_from_depth / (
            norm + 1e-10
        )  # Add epsilon to avoid division by zero

        normal_from_depth *= -1

        return normal_from_depth
