"""
Sequential Clip Sampler for Any4D-aligned evaluation.

Selects the first N consecutive frames from the video sequence.
Example: view_num=50 -> indices [0, 1, 2, ..., 49]
"""


class SequentialClipSampler:
    """
    Sequentially samples the first N consecutive frames from views.
    
    This is aligned with Any4D evaluation protocol which uses
    consecutive frames as input.
    
    Args:
        view_num: Number of consecutive frames to sample
        start_idx: Starting frame index (default: 0)
        debug: Whether to print debug information
        max_view: Maximum view number (for compatibility)
    
    Example:
        sampler = SequentialClipSampler(view_num=50)
        # For 200 frames: returns frames [0, 1, 2, ..., 49]
    """
    
    def __init__(self, view_num=None, start_idx=0, debug=False, max_view=None, **kwargs):
        self.view_num = view_num
        self.start_idx = start_idx
        self.debug = debug
        self.max_view = max_view

    def get_max_view_num(self):
        if self.max_view is not None:
            return self.max_view

        if self.view_num is not None:
            if isinstance(self.view_num, int):
                return self.view_num
            else:
                return max(self.view_num)

        return self.max_view

    def get_min_view_num(self):
        if self.view_num is not None:
            if isinstance(self.view_num, int):
                return self.view_num
            else:
                return min(self.view_num)
        else:
            return None

    def __call__(self, views):
        """
        Select first N consecutive frames starting from start_idx.
        
        Args:
            views: List of view dictionaries
            
        Returns:
            List of N consecutive views starting from start_idx
        """
        if self.view_num is None:
            return views
        
        # Calculate end index
        start = self.start_idx
        end = min(start + self.view_num, len(views))
        
        # Handle case where we don't have enough frames
        if end - start < self.view_num:
            # If not enough frames from start_idx, adjust start to get view_num frames
            if len(views) >= self.view_num:
                start = len(views) - self.view_num
                end = len(views)
            else:
                # Not enough frames total, use all available
                start = 0
                end = len(views)
        
        indices = list(range(start, end))
        
        if self.debug:
            print(f"SequentialClipSampler: start={start}, end={end}")
            print(f"SequentialClipSampler: indices={indices}")
            if views and "frame_id" in views[0]:
                print(f"SequentialClipSampler: frame_ids={[views[i]['frame_id'] for i in indices]}")
        
        return [views[i] for i in indices]