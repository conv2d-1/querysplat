"""
Custom segmentation dataset with predefined class mapping.
Example of how to extend BaseDatasetSeg with your own class categories.
"""

from typing import Dict, List
from hAlgorithm.datasets_seg.base_dataset import BaseDatasetSeg


class CustomSegDataset(BaseDatasetSeg):
    """
    Custom segmentation dataset with predefined class names.
    
    Example usage:
        dataset = CustomSegDataset(
            phase="train",
            name="custom_seg",
            data_root="/mnt/nasTeam/Kosmo/processed_data/kosmo",
            data_path="data_seg.json",
            class_names=["background", "person", "car", "building"],
            ...
        )
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        **kwargs
    ):
        """
        Initialize custom segmentation dataset.
        
        Args:
            class_names: List of class names in order (index 0 = background)
            **kwargs: Arguments passed to BaseDatasetSeg
        """
        # Set default class names if not provided
        if class_names is None:
            class_names = [
                "background",
                "person",
                "car",
                "bicycle",
                "motorcycle",
                "truck",
                "bus",
                "building",
                "road",
                "sidewalk",
                "vegetation",
                "sky",
            ]
        
        self.class_names = class_names
        
        # Create bidirectional mapping
        self.class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class_name = {idx: name for idx, name in enumerate(class_names)}
        
        super().__init__(**kwargs)
    
    def _label_name_to_idx(self, label_name: str) -> int:
        """
        Convert label name to class index using predefined mapping.
        
        Args:
            label_name: Class name from annotation
            
        Returns:
            Class index (0 = background if not found)
        """
        # Try exact match first
        if label_name in self.class_name_to_idx:
            return self.class_name_to_idx[label_name]
        
        # Try case-insensitive match
        label_lower = label_name.lower()
        for name, idx in self.class_name_to_idx.items():
            if name.lower() == label_lower:
                return idx
        
        # Return background index (0) if not found
        return 0
    
    def _get_class_name(self, label_idx: int) -> str:
        """
        Override base class method to provide actual class names.
        Used internally by visualize_sample() and other methods.
        
        Args:
            label_idx: Class index
            
        Returns:
            Class name string
        """
        return self.idx_to_class_name.get(label_idx, "unknown")
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.class_names)


# Example with specific domain classes
class VehicleSegDataset(CustomSegDataset):
    """Vehicle-focused segmentation dataset."""
    
    def __init__(self, **kwargs):
        # Vehicle-specific classes
        class_names = [
            "background",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "person",
            "traffic_sign",
            "traffic_light",
            "road",
            "lane_marking",
        ]
        super().__init__(class_names=class_names, **kwargs)


class KosmoSegDataset(CustomSegDataset):
    """Urban scene segmentation dataset."""
    
    def __init__(self, **kwargs):
        # Urban scene classes with background
        class_names = [
            "background",  # index 0
            "person",      # index 1
            "window",      # index 2
            "sky"          # index 3
        ]
        super().__init__(class_names=class_names, **kwargs)


if __name__ == "__main__":
    # Example usage
    dataset = KosmoSegDataset(
        phase="train",
        name="custom_seg_test",
        data_root="/mnt/nasTeam/Kosmo/processed_data/kosmo",
        data_path="data_seg_创意园_right_sharp.json",
        seed=0,
        sampling_strategy="first:5",
        return_masks=True,
        return_boxes=True,
        mask_format="bitmap",
        debug=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {dataset.class_names}")
    
    # Test loading first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Number of objects: {len(sample.get('labels', []))}")
    
    if 'boxes' in sample:
        print(f"  Boxes shape: {sample['boxes'].shape}")
        print(f"  First 3 boxes: {sample['boxes'][:3]}")
    
    if 'labels' in sample:
        print(f"  Labels: {sample['labels'][:10]}")
        print(f"  Class names: {[dataset._get_class_name(int(idx)) for idx in sample['labels'][:10]]}")
    
    if 'masks' in sample:
        if isinstance(sample['masks'], list):
            print(f"  Number of masks: {len(sample['masks'])}")
        else:
            print(f"  Masks shape: {sample['masks'].shape}")
    
    # Visualize the sample
    print("\nGenerating visualization...")
    vis_image = dataset.visualize_sample(
        sample,
        save_path="./debug/dataset_seg/custom_seg_test/sample_0_visualization.jpg",
        show=False
    )
    print(f"Visualization shape: {vis_image.shape}")
    print("\n✓ Visualization saved successfully!")
