"""Fisheye 4D tracking datasets.

This module provides dataset classes for fisheye camera data with proper
polynomial distortion handling. Unlike the standard datasets_4d which assume
pinhole cameras, these classes correctly handle:

- Fisheye polynomial projection/unprojection (Blender model)
- Circular valid region cropping
- Static trajectory generation with correct fisheye geometry
"""

from hAlgorithm.datasets_4d_fisheye.base_fisheye_track_dataset import BaseFisheyeTrackDataset
from hAlgorithm.datasets_4d_fisheye.hasim_fisheye_track_dataset import HaSimFisheyeTrackDataset

__all__ = [
    "BaseFisheyeTrackDataset",
    "HaSimFisheyeTrackDataset",
]
