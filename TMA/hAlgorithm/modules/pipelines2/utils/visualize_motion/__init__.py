from .utils import MotionVisUtils
from .visualizer_2d import MotionVisualizer2D
from .gif_generator import GifGenerator
from .visualizer_3d import MotionVisualizer3D, SparseMotionVisualizer3D
from .sparse_visualizer import MotionVisualizer


def to_numpy(tensor):
    return MotionVisUtils.to_numpy(tensor)


def normalize_rgb(img):
    return MotionVisUtils.normalize_rgb(img)


def get_depth_from_output(output, h, w):
    return MotionVisUtils.get_depth_from_output(output, h, w)


def project_scene_flow_to_2d(scene_flow_3d, depth, intrinsics, src_extrinsics=None, tgt_extrinsics=None):
    return MotionVisUtils.project_scene_flow_to_2d(
        scene_flow_3d, depth, intrinsics, src_extrinsics=src_extrinsics, tgt_extrinsics=tgt_extrinsics
    )


def get_image_from_output(output, h, w):
    return MotionVisUtils.get_image_from_output(output, h, w)


def compute_motion_colors(vectors):
    return MotionVisUtils.compute_motion_colors(vectors)


def generate_flow_gif(flow_img_dir, output_path=None, fps=2.0, size=280):
    return GifGenerator.generate_flow_gif(flow_img_dir, output_path=output_path, fps=fps, size=size)


def vis_motion_head_results(cfg, mv_outputs, motion_out_dir, data_idx, meta_data=None):
    MotionVisualizer2D(cfg).run(mv_outputs, motion_out_dir, data_idx, meta_data)


def vis_motion_3d_rerun(cfg, mv_outputs, out_dir, data_idx, meta_data=None):
    MotionVisualizer3D(cfg).run(mv_outputs, out_dir, data_idx, meta_data)


def vis_sparse_motion_3d_rerun(cfg, mv_outputs, out_dir, data_idx, meta_data=None):
    SparseMotionVisualizer3D(cfg).run(mv_outputs, out_dir, data_idx, meta_data)


__all__ = [
    "MotionVisUtils",
    "MotionVisualizer2D",
    "GifGenerator",
    "MotionVisualizer3D",
    "SparseMotionVisualizer3D",
    "MotionVisualizer",
    "to_numpy",
    "normalize_rgb",
    "get_depth_from_output",
    "project_scene_flow_to_2d",
    "get_image_from_output",
    "compute_motion_colors",
    "generate_flow_gif",
    "vis_motion_head_results",
    "vis_motion_3d_rerun",
    "vis_sparse_motion_3d_rerun",
]
