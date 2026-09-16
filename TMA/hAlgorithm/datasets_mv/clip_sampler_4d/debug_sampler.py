class Sampler4D:
    def __init__(
        self,
        frame_num,
        view_num,
        max_frame=None,
        max_view=None,
        debug=False,
        frame_step=None,
        **kwargs,
    ):

        self.frame_num = frame_num
        self.max_frame = max_frame
        self.frame_step = frame_step

        self.view_num = view_num
        self.max_view = max_view

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
        # Frames sampler
        frame_ids = sorted(list(clip.keys()))
        if self.frame_step is not None:
            frame_indices = [round(i * self.frame_step) for i in range(1, self.frame_num)]
            frame_indices = [0] + frame_indices
        else:
            step = (len(frame_ids) - 1) / (self.frame_num - 1)
            frame_indices = [round(i * step) for i in range(1, self.frame_num - 1)]
            frame_indices = [0] + frame_indices + [len(frame_ids) - 1]

        clip_new = [clip[frame_ids[i]] for i in frame_indices]

        # Views sampler
        base_views = clip_new[0]
        step = (len(base_views) - 1) / (self.view_num - 1)
        view_indices = [round(i * step) for i in range(1, self.view_num - 1)]
        view_indices = [0] + view_indices + [len(base_views) - 1]

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
