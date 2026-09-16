import os
import sys

sys.path.append(os.getcwd())

import json
import logging
import random
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_util

from hAlgorithm.datasets.transforms.transforms import Compose
from hAlgorithm.utils import instantiate_from_config


class BaseDatasetSeg(Dataset):
    """Base dataset class for segmentation tasks."""

    def __init__(
        self,
        phase: str = "train",
        name: str = "seg_dataset",
        data_root: str = None,
        data_path: str = None,
        seed: int = 0,
        sampling_strategy: str = "all",
        train_transforms: List[Dict] = None,
        test_transforms: List[Dict] = None,
        scene_name: str = None,
        debug: bool = False,
        return_masks: bool = True,
        return_boxes: bool = True,
        mask_format: str = "polygon",
        box_format: str = "xyxy",
        **kwargs,
    ):
        self.phase = phase
        self.name = name
        self.data_root = data_root
        self.data_path = data_path
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.scene_name = scene_name
        self.return_masks = return_masks
        self.return_boxes = return_boxes
        self.mask_format = mask_format
        self.box_format = box_format
        self.debug = debug

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        transform_config = train_transforms if phase == "train" else test_transforms
        self.data_transforms = Compose([instantiate_from_config(cfg) for cfg in (transform_config or [])])
        self.data_infos = self.get_data_infos()
        print(f"[{name}] Loaded {len(self.data_infos)} samples, GT format: box={self.box_format}, mask={self.mask_format}")

    def get_data_infos(self) -> List[Dict]:
        """Load dataset info from JSON file."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mf_files = data.get("mf_files", {})
        if self.scene_name and self.scene_name in mf_files:
            scene_data, selected_scene = mf_files[self.scene_name], self.scene_name
        else:
            selected_scene = list(mf_files.keys())[0]
            scene_data = mf_files[selected_scene]

        if self.data_root is None:
            raise ValueError("data_root must be provided")

        frames = scene_data.get("frames", [])
        for frame in frames:
            frame["scene"] = selected_scene

        frames = self._apply_sampling_strategy(frames)
        logging.info(f"Selected {len(frames)} frames from scene: {selected_scene}")
        return frames

    def _apply_sampling_strategy(self, data: List[Dict]) -> List[Dict]:
        if self.sampling_strategy == "all":
            return data
        if ":" in self.sampling_strategy:
            strategy, num = self.sampling_strategy.split(":")
            num = int(num)
            if strategy == "first":
                return data[:num]
            elif strategy == "end":
                return data[-num:]
            elif strategy == "index":
                return data[num : num + 1] if num < len(data) else []
            elif strategy == "random":
                indices = random.sample(range(len(data)), min(num, len(data)))
                return [data[i] for i in sorted(indices)]
        return data

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict:
        try:
            return self._get_item_impl(idx)
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            if self.phase == "train":
                return self.__getitem__(random.randint(0, len(self) - 1))
            raise

    def _get_item_impl(self, idx: int) -> Dict:
        data_info = self.data_infos[idx]

        rgb_path = data_info["rgb"]
        if not os.path.isabs(rgb_path):
            rgb_path = os.path.join(self.data_root, rgb_path)
        image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

        label_path = data_info["label"]
        if not os.path.isabs(label_path):
            label_path = os.path.join(self.data_root, label_path)
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        parsed_labels = self._parse_label_json(label_data, image.shape[:2])
        
        # Distinguish between: no label file vs. no objects in scene
        # - parsed_labels is None: label parsing failed (data error) -> skip
        # - parsed_labels exists but masks is empty: valid sample with no objects (all background)
        if self.phase == "train" and self.return_masks:
            if parsed_labels is None:
                raise ValueError(f"Sample {idx}: label parsing failed, skipping")

        transform_info = {"name": self.name}
        if self.data_transforms and len(self.data_transforms.transforms) > 0:
            image_transformed, _, _, _, _, _, transform_info = self.data_transforms(
                image=image,
                intrinsics=None,
                depth=None,
                depth_mask=None,
                normal=None,
                other_labels=[],
                transform_info=transform_info,
            )
        else:
            image_transformed = torch.from_numpy(image).permute(2, 0, 1).float()

        # Apply same resize to ground truth masks and boxes
        if parsed_labels and "resize_ratio_h" in transform_info:
            ratio_h = transform_info["resize_ratio_h"]
            ratio_w = transform_info["resize_ratio_w"]
            new_h, new_w = image_transformed.shape[1], image_transformed.shape[2]
            
            # Resize masks (list of numpy arrays) - batch process for speed
            if "masks" in parsed_labels and self.mask_format == "bitmap":
                masks = parsed_labels["masks"]
                if isinstance(masks, list) and len(masks) > 0:
                    # Stack all masks for batch resize
                    masks_array = np.stack(masks, axis=0)  # [N, H, W]
                    # Batch resize: reshape to [N*H, W] -> resize -> reshape back
                    n, orig_h, orig_w = masks_array.shape
                    masks_resized = cv2.resize(
                        masks_array.transpose(1, 2, 0),  # [H, W, N]
                        (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST
                    )  # [new_H, new_W, N]
                    if n == 1:
                        masks_resized = masks_resized[:, :, np.newaxis]
                    # Split back to list
                    resized_masks = [masks_resized[:, :, i] for i in range(n)]
                    parsed_labels["masks"] = resized_masks
            
            # Resize boxes (scale coordinates)
            if "boxes" in parsed_labels:
                boxes = parsed_labels["boxes"]
                if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                    if self.box_format == "xyxy":
                        boxes[:, [0, 2]] *= ratio_w  # x1, x2
                        boxes[:, [1, 3]] *= ratio_h  # y1, y2
                    # cxcywh normalized format doesn't need scaling
                    parsed_labels["boxes"] = boxes

        data_dict = {
            "image": image_transformed,
            "label_data": label_data,
            "meta_data": {
                "name": self.name,
                "data_idx": idx,
                "input_height": image_transformed.shape[1],
                "input_width": image_transformed.shape[2],
                "origin_height": image.shape[0],
                "origin_width": image.shape[1],
                "rgb_path": rgb_path,
                "label_path": label_path,
            },
        }
        if parsed_labels:
            data_dict.update(parsed_labels)
        if self.debug:
            self._debug_sample(idx, data_dict, image)
        return data_dict

    def _parse_label_json(self, label_data: Dict, image_shape: Tuple[int, int]) -> Optional[Dict]:
        """Parse label JSON (supports COCO and LabelMe formats)."""
        if "annotations" in label_data:
            return self._parse_coco_format(label_data, image_shape)
        elif "shapes" in label_data:
            return self._parse_labelme_format(label_data, image_shape)
        return None

    def _parse_coco_format(self, label_data: Dict, image_shape: Tuple[int, int]) -> Optional[Dict]:
        """Parse COCO format annotations."""
        annotations = label_data.get("annotations", [])
        if not annotations:
            return None

        h, w = image_shape
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for ann in annotations:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, bw, bh = bbox
            x1, y1 = max(0, min(x, w - 1)), max(0, min(y, h - 1))
            x2, y2 = max(0, min(x + bw, w - 1)), max(0, min(y + bh, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            if self.return_boxes:
                boxes.append([x1, y1, x2, y2])
            labels.append(self._label_name_to_idx(ann.get("category_id", "unknown")))
            areas.append(ann.get("area", bw * bh))
            iscrowd.append(ann.get("iscrowd", 0))

            if self.return_masks:
                seg = ann.get("segmentation")
                if seg:
                    masks.append(self._process_segmentation(seg, h, w))

        if not labels:
            return None

        result = {
            "labels": labels,  # Keep as list for variable batch collation
            "area": areas,
            "iscrowd": iscrowd,
            "num_instances": len(labels),
        }
        if self.return_boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            result["boxes"] = self._convert_boxes(boxes_tensor, h, w)
        if self.return_masks and masks:
            # Keep masks as list (not stacked tensor) for variable instance count per sample
            result["masks"] = masks
        return result

    def _process_segmentation(self, seg, h: int, w: int):
        """Process segmentation based on mask_format."""
        if isinstance(seg, dict) and "counts" in seg:
            return seg if self.mask_format == "rle" else self._rle_to_bitmap(seg, h, w)
        elif isinstance(seg, list) and seg:
            if self.mask_format == "polygon":
                return seg
            bitmap = self._coco_polygon_to_bitmap(seg, h, w)
            return bitmap if self.mask_format == "bitmap" else self._bitmap_to_rle(bitmap)
        return None

    def _convert_boxes(self, boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert boxes to target format. xyxy (pixel) or cxcywh (normalized 0-1)."""
        if self.box_format == "xyxy" or boxes.numel() == 0:
            return boxes
        # xyxy -> cxcywh normalized
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        return torch.stack([cx, cy, bw, bh], dim=1)

    def _parse_labelme_format(self, label_data: Dict, image_shape: Tuple[int, int]) -> Optional[Dict]:
        """Parse LabelMe format annotations."""
        shapes = label_data.get("shapes", [])
        if not shapes:
            return None

        h, w = image_shape
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for shape in shapes:
            points = shape.get("points", [])
            if len(points) < 3:
                continue

            pts = np.array(points, dtype=np.float32)
            x_min, x_max = np.clip([pts[:, 0].min(), pts[:, 0].max()], 0, w - 1)
            y_min, y_max = np.clip([pts[:, 1].min(), pts[:, 1].max()], 0, h - 1)
            if x_max - x_min <= 1 or y_max - y_min <= 1:
                continue

            if self.return_boxes:
                boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self._label_name_to_idx(shape.get("label", "unknown")))
            areas.append(self._polygon_area(pts))
            iscrowd.append(0)

            if self.return_masks:
                if self.mask_format == "polygon":
                    masks.append(pts.flatten().tolist())
                elif self.mask_format == "rle":
                    masks.append(self._polygon_to_rle(pts, h, w))
                else:
                    masks.append(self._polygon_to_bitmap(pts, h, w))

        if not labels:
            return None

        result = {
            "labels": labels,  # Keep as list for variable batch collation
            "area": areas,
            "iscrowd": iscrowd,
            "num_instances": len(labels),
        }
        if self.return_boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            result["boxes"] = self._convert_boxes(boxes_tensor, h, w)
        if self.return_masks and masks:
            # Keep masks as list (not stacked tensor) for variable instance count per sample
            result["masks"] = masks
        return result

    def _label_name_to_idx(self, label_name: str) -> int:
        """Convert label name to index. Override for custom mapping."""
        return hash(label_name) % 1000

    def _polygon_area(self, polygon: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        x, y = polygon[:, 0], polygon[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _polygon_to_bitmap(self, polygon: np.ndarray, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask

    def _polygon_to_rle(self, polygon: np.ndarray, h: int, w: int) -> Dict:
        return self._bitmap_to_rle(self._polygon_to_bitmap(polygon, h, w))

    def _coco_polygon_to_bitmap(self, polygons: List, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        for polygon in polygons:
            if len(polygon) >= 6:
                cv2.fillPoly(mask, [np.array(polygon).reshape(-1, 2).astype(np.int32)], 1)
        return mask

    def _rle_to_bitmap(self, rle: Dict, h: int, w: int) -> np.ndarray:
        try:
            rle_copy = rle.copy()
            if isinstance(rle_copy.get("counts"), str):
                rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
            return mask_util.decode(rle_copy)
        except:
            return np.zeros((h, w), dtype=np.uint8)

    def _bitmap_to_rle(self, mask: np.ndarray) -> Dict:
        rle = mask_util.encode(np.asfortranarray(mask))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def _debug_sample(self, idx: int, data_dict: Dict, original_image: np.ndarray):
        outdir = f"./debug/dataset_seg/{self.name}/{idx:04d}"
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(f"{outdir}/image.jpg", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        with open(f"{outdir}/label.json", "w", encoding="utf-8") as f:
            json.dump(data_dict["label_data"], f, indent=2, ensure_ascii=False)
        logging.info(f"Debug saved to: {outdir}")

    def visualize_sample(self, sample: Dict, save_path: str = None, show: bool = False) -> np.ndarray:
        """Visualize sample with boxes and masks."""
        if "meta_data" in sample and "rgb_path" in sample["meta_data"]:
            image = cv2.cvtColor(cv2.imread(sample["meta_data"]["rgb_path"]), cv2.COLOR_BGR2RGB)
        else:
            img = sample["image"].permute(1, 2, 0).numpy()
            image = np.clip((img + 1) / 2 * 255 if img.min() < 0 else img * 255 if img.max() <= 1 else img, 0, 255).astype(np.uint8)

        vis = image.copy()
        labels = sample.get("labels")
        n = len(labels) if labels is not None else 0

        if n > 0:
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(n, 3), dtype=np.uint8)

            # Draw masks
            masks = sample.get("masks")
            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    for i, m in enumerate(masks.numpy()[:n]):
                        colored = np.zeros_like(vis)
                        colored[m > 0] = colors[i].tolist()
                        vis = cv2.addWeighted(vis, 1, colored, 0.5, 0)
                elif isinstance(masks, list):
                    for i, m in enumerate(masks[:n]):
                        # Handle different mask formats
                        if isinstance(m, dict) and "counts" in m:
                            # RLE format
                            mask = self._rle_to_bitmap(m, vis.shape[0], vis.shape[1])
                        elif isinstance(m, np.ndarray):
                            # Bitmap format (numpy array)
                            mask = m
                            if mask.shape[:2] != vis.shape[:2]:
                                mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                        else:
                            continue
                        colored = np.zeros_like(vis)
                        colored[mask > 0] = colors[i].tolist()
                        vis = cv2.addWeighted(vis, 1, colored, 0.5, 0)

            # Draw boxes
            boxes = sample.get("boxes")
            if boxes is not None:
                boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
                labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
                for i, box in enumerate(boxes_np):
                    x1, y1, x2, y2 = box.astype(int)
                    c = tuple(colors[i].tolist())
                    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
                    if i < len(labels_np):
                        txt = self._get_class_name(int(labels_np[i]))
                        sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(vis, (x1, y1 - sz[1] - 10), (x1 + sz[0] + 5, y1), c, -1)
                        cv2.putText(vis, txt, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if show:
            cv2.imshow("Vis", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return vis

    def _get_class_name(self, label_idx: int) -> str:
        """Get class name from index. Override in subclass."""
        return f"class_{label_idx}"


if __name__ == "__main__":
    # Example usage
    dataset = BaseDatasetSeg(
        phase="train",
        name="seg_test",
        data_root="/mnt/nasTeam/Kosmo/processed_data/kosmo",  # Required for relative paths
        data_path="data_seg_创意园_right_sharp.json",
        seed=0,
        sampling_strategy="first:10",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=512,
                height=512,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        debug=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading first sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label data keys: {list(sample['label_data'].keys())}")
    print(f"Meta data: {sample['meta_data']}")
