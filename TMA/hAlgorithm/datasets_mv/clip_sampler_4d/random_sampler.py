import random

import numpy as np


class Sampler4D:
    def __init__(
        self,
        frame_num,
        view_num,
        max_frame=None,
        max_view=None,
        frame_step=1,
        max_nums=None,
        keep_first_frame=False,
        seed=None,
        debug=False,
        **kwargs,
    ):

        self.frame_num = frame_num
        self.max_frame = max_frame
        self.frame_step = frame_step

        self.view_num = view_num
        self.max_view = max_view
        self.max_nums = max_nums

        self.keep_first_frame = keep_first_frame

        self.seed = seed
        self.debug = debug

    def get_max_frame_num(self):
        if self.max_frame is not None:
            return self.max_frame

        if self.frame_num is not None:
            if isinstance(self.frame_num, int):
                return (
                    self.frame_num
                    if self.frame_step is None
                    else 1 + (self.frame_num - 1) * self.frame_step
                )
            else:
                return (
                    max(self.frame_num)
                    if self.frame_step is None
                    else 1 + (max(self.frame_num) - 1) * self.frame_step
                )

        return self.max_frame

    def get_max_view_num(self):
        if self.max_view is not None:
            return self.max_view

        if self.view_num is not None:
            if isinstance(self.view_num, int):
                return self.view_num
            else:
                return max(self.view_num)

        return self.max_view

    def __call__(self, clip):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if isinstance(self.frame_num, (list, tuple)):
            frame_num = random.sample(range(min(self.frame_num), max(self.frame_num) + 1), 1)[0] 
        else:
            frame_num = self.frame_num
        
        if isinstance(self.view_num, (list, tuple)):
            if self.max_nums is not None:
                view_num_min = min(min(self.view_num), self.max_nums // frame_num)
                view_num_max = min(max(self.view_num), self.max_nums // frame_num)
                view_num = random.sample(range(view_num_min, view_num_max + 1), 1)[0] 
            else:
                view_num = random.sample(range(min(self.view_num), max(self.view_num) + 1), 1)[0] 
        else:
            if self.max_nums is not None:
                view_num = min(self.view_num, self.max_nums // frame_num)
            else:
                view_num = self.view_num

        # Frames sampler
        frame_ids = sorted(list(clip.keys()))

        if self.frame_step is not None:
            frame_step = self.frame_step
        else:
            max_step = int((len(frame_ids) - 1) / max(1, frame_num - 1))
            frame_step = random.sample(range(1, max_step + 1), 1)[0]

        last_indice = len(frame_ids) - (1 + frame_step * (frame_num - 1))

        if self.keep_first_frame:
            first_frame_indice = 0
        else:
            index = range(0, last_indice+1)
            first_frame_indice = random.sample(index, 1)[0]
        # first_frame_indice = last_indice
        frame_indices = [first_frame_indice + i * frame_step for i in range(frame_num)]
        clip_new = [clip[frame_ids[i]] for i in frame_indices]

        # Views sampler
        base_views = clip_new[0]
        index = range(len(base_views))
        view_indices = random.sample(index, view_num)

        clip_out = []
        for views in clip_new:
            sample = []
            for idx in view_indices:
                assert base_views[idx]["view_id"] == views[idx]["view_id"]
                sample.append(views[idx])
            clip_out.append(sample)

        if self.debug:
            print(f"Sampler4D")
            print(
                f"frame indices {frame_indices}, frame ids {[frame_ids[i] for i in frame_indices]}"
            )
            print(
                f"view indices {view_indices}, view ids {[base_views[i]['view_id'] for i in view_indices]}"
            )

        return clip_out
