import random

import numpy as np


class MixClipSampler:
    def __init__(
        self,
        view_num=1,
        frame_step=1,
        clip_shuffle=False,
        temporal_sampling=False,
        resampling=False,
        seed=None,
    ):
        self.view_num = view_num
        self.frame_step = frame_step
        self.clip_shuffle = clip_shuffle
        self.temporal_sampling = temporal_sampling
        self.resampling = resampling
        self.seed = seed

    def get_max_view_num(self):
        max_view_num = self.view_num if isinstance(self.view_num, int) else max(self.view_num)
        max_frame_step = (
            self.frame_step if isinstance(self.frame_step, int) else max(self.frame_step)
        )
        return max_view_num * max_frame_step

    def get_min_view_num(self):
        min_view_num = self.view_num if isinstance(self.view_num, int) else min(self.view_num)
        min_frame_step = (
            self.frame_step if isinstance(self.frame_step, int) else min(self.frame_step)
        )
        return min_view_num * min_frame_step

    def get_view_num(self):
        if isinstance(self.view_num, int):
            return self.view_num
        else:
            return random.randint(min(self.view_num), max(self.view_num))

    def get_frame_step(self):
        if isinstance(self.frame_step, int):
            return self.frame_step
        else:
            return random.randint(min(self.frame_step), max(self.frame_step))

    def __call__(self, views):

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        view_num = self.get_view_num()
        view_ids = [view["view_id"] for view in views]

        if all([view_ids[0] == view_id for view_id in view_ids[1:]]):
            # frame_ids = [view["frame_id"] for view in views]

            # 相同 view_id 代表时序数据，进行时序采样
            sorted_views = sorted(views, key=lambda x: x["frame_id"])  # 按帧号排序
            frame_step = self.get_frame_step()

            # 计算总长度
            total_size = len(sorted_views)
            window_size = min(frame_step * (view_num - 1) + 1, total_size)

            # 随机片段
            left_id = random.randint(0, total_size - window_size)
            right_id = left_id + window_size - 1
            assert (
                right_id < total_size
            ), f"left_id:{left_id}, right_id:{right_id}, total_size:{total_size}"
            frame_step = int((right_id - left_id) / (view_num - 1))

            if self.temporal_sampling:
                # 随机采样
                index = list(range(left_id, right_id + 1))
                if window_size >= view_num:
                    # 无重复采样
                    selected_index = random.sample(index, view_num)
                else:
                    # 重复采样
                    assert self.resampling, f"window_size{window_size} < view_num{view_num}"
                    selected_index = np.random.choice(index, view_num)

                if not self.clip_shuffle:
                    selected_index = sorted(selected_index)
            else:
                if window_size >= view_num:
                    selected_index = list(range(left_id, right_id + 1, frame_step))
                else:
                    assert self.resampling, f"window_size{window_size} < view_num{view_num}"
                    selected_index = np.random.choice(list(range(left_id, right_id + 1)), view_num)
                    selected_index = sorted(selected_index)

                if self.clip_shuffle:
                    random.shuffle(selected_index)

            clip = [sorted_views[index] for index in selected_index]

        else:
            # 不同 view_id 代表多视角数据，进行视角采样
            index = list(range(len(view_ids)))

            if len(index) >= view_num:
                # 无重复采样
                selected_index = random.sample(index, view_num)
            else:
                # 重复采样
                assert self.resampling, "window_size > total_size"
                selected_index = np.random.choice(index, view_num)

            clip = [views[index] for index in selected_index]

        assert len(clip) >= view_num, f"len(clip)={len(clip)}, view_num={view_num}"
        return clip[:view_num]
