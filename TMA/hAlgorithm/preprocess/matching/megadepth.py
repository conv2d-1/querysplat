import os
from PIL import Image
import h5py
import numpy as np
import torch
import math

class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        scene_name = None,
    ) -> None:
        self.data_root = data_root
        self.scene_name = os.path.splitext(scene_name)[0]
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]

    def organize_scene_infos(
        self,
        data_root, 
        min_overlap=0.0,
        max_overlap=1.0,
        max_num_pairs = None,
    ):
        total_frames = []
        img_info_frame_id = []
        cur_frame_id = 0
        for i, rgb in enumerate(self.image_paths):
            if rgb is not None:
                depth = self.depth_paths[i]
                intrinsics = self.intrinsics[i].tolist()
                intrinsics = [intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]]
                pose_w2c = self.poses[i].tolist()
                rgb = os.path.join(self.data_root, rgb).replace(data_root, '')
                depth = os.path.join(self.data_root, depth).replace(data_root, '')
                frame_entry = {
                    "frame_id": cur_frame_id,
                    "views": [
                        {
                            "view_id": 0,
                            "rgb": rgb,
                            "depth": depth,
                            "cam_in": intrinsics,
                            "extrinsics": pose_w2c,
                            "depth_scale": 1.0,
                        }
                    ]
                }
                total_frames.append(frame_entry)
                img_info_frame_id.append(cur_frame_id)
                cur_frame_id = cur_frame_id + 1
            else:
                img_info_frame_id.append(None)
        
        scene_entry = {
            self.scene_name: total_frames
        }
        
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        if max_num_pairs is not None and len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        pairs = []
        for pair in self.pairs:
            img0, img1 = pair[0], pair[1]
            frame0 = img_info_frame_id[img0]
            frame1 = img_info_frame_id[img1]
            assert total_frames[frame0]["views"][0]["rgb"].endswith(self.image_paths[img0])
            assert total_frames[frame1]["views"][0]["rgb"].endswith(self.image_paths[img1])
            pairs.append(
                [[frame0, 0], [frame1, 0]]
            )
        pair_entry = {
            self.scene_name: pairs
        }
        return scene_entry, pair_entry


class MegadepthBuilder:
    def __init__(self, data_root="data/megadepth", loftr_ignore=True, imc21_ignore = True) -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        # LoFTR did the D2-net preprocessing differently than we did and got more ignore scenes, can optionially ignore those
        self.loftr_ignore_scenes = set(['0121.npy', '0133.npy', '0168.npy', '0178.npy', '0229.npy', '0349.npy', '0412.npy', '0430.npy', '0443.npy', '1001.npy', '5014.npy', '5015.npy', '5016.npy'])
        self.imc21_scenes = set(['0008.npy', '0019.npy', '0021.npy', '0024.npy', '0025.npy', '0032.npy', '0063.npy', '1589.npy'])
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]
        self.loftr_ignore = loftr_ignore
        self.imc21_ignore = imc21_ignore

    def build_scenes(self, split="train", scene_names = None, **kwargs):
        if split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        elif split == "custom":
            scene_names = scene_names
        else:
            raise ValueError(f"Split {split} not available")
        scenes = []
        for scene_name in scene_names:
            if self.loftr_ignore and scene_name in self.loftr_ignore_scenes:
                continue
            if self.imc21_ignore and scene_name in self.imc21_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            scenes.append(
                MegadepthScene(
                    self.data_root, scene_info,scene_name = scene_name, **kwargs
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws


mega = MegadepthBuilder(data_root="/mnt/netdata/Team/AI/datasets/TMD/MegaDepth", loftr_ignore=True, imc21_ignore = True)
megadepth_train1 = mega.build_scenes(
    split="test_loftr",#train_loftr
)
data_root="/mnt/netdata/Team/AI/datasets/TMD/"
data_json = {
    "mf_files": {},
}
pair_json = {
    "image_pairs" : {}
}
for scene in megadepth_train1:
    scene, pair = scene.organize_scene_infos(data_root, min_overlap=0.01, max_num_pairs=30_000)
    data_json["mf_files"].update(scene)
    pair_json["image_pairs"].update(pair)

import json
# with open("./megadepth_mf_test.json", 'w') as f:
#     json.dump(data_json, f, indent=2, ensure_ascii=False)
with open("./test_pair_overlap_1.json", 'w') as f:
    json.dump(pair_json, f, indent=2, ensure_ascii=False)

