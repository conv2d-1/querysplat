import random

import numpy as np
from scipy.stats import norm


class RandomClipSampler:
    def __init__(self, view_num=1, shuffle=True, seed=None, debug=False, norm_pdf=False, **kwargs):
        self.view_num = view_num
        self.shuffle = shuffle
        self.seed = seed
        self.debug = debug

        self.norm_pdf = norm_pdf
        self.views = None
        self.views_prob = None

    def get_max_view_num(self):
        if isinstance(self.view_num, int):
            return self.view_num
        else:
            return max(self.view_num)

    def get_min_view_num(self):
        if isinstance(self.view_num, int):
            return self.view_num
        else:
            return min(self.view_num)

    def get_view_num(self):
        if isinstance(self.view_num, int):
            return self.view_num
        elif self.norm_pdf:
            if self.views is None or self.views_prob is None:
                view_num_min = min(self.view_num)
                view_num_max = max(self.view_num)
                view_num_mean = (view_num_min + view_num_max) * 0.5
                self.views = list(range(view_num_min, view_num_max + 1))
                self.views_prob = norm.pdf([view - view_num_mean for view in self.views])
                self.views_prob = self.views_prob / np.sum(self.views_prob)
            return random.choices(self.views, weights=self.views_prob, k=1)[0]
        else:
            return random.randint(min(self.view_num), max(self.view_num))

    def __call__(self, views):

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        view_num = self.get_view_num()

        # 不同 view_id 代表多视角数据，进行视角采样
        index = list(range(len(views)))

        if len(index) >= view_num:
            # 无重复采样
            selected_index = random.sample(index, view_num)
        else:
            # 重复采样
            selected_index = np.random.choice(index, view_num)

        if not self.shuffle:
            selected_index = sorted(selected_index)

        if self.debug:
            print("shuffle", self.shuffle, "index", selected_index)

        clip = [views[index] for index in selected_index]

        assert len(clip) >= view_num, f"len(clip)={len(clip)}, view_num={view_num}"
        return clip[:view_num]


class KeepFirstRandomClipSampler(RandomClipSampler):
    """保留第一帧，其他帧随机抽取"""

    def __init__(self, **kwargs):
        super(KeepFirstRandomClipSampler, self).__init__(**kwargs)

    def __call__(self, views):

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        view_num = self.get_view_num()

        # 不同 view_id 代表多视角数据，进行视角采样
        index = list(range(len(views)))

        if len(index) >= view_num:
            # 无重复采样
            selected_index = random.sample(index[1:], view_num - 1)
        else:
            # 重复采样
            selected_index = np.random.choice(index[1:], view_num - 1)

        if not self.shuffle:
            selected_index = sorted(selected_index)

        if self.debug:
            print("shuffle", self.shuffle, "index", selected_index)

        clip = [views[0]] + [views[index] for index in selected_index]

        assert len(clip) >= view_num, f"len(clip)={len(clip)}, view_num={view_num}"
        return clip[:view_num]


class KeepFirstClipSampler(RandomClipSampler):
    """从第一帧按顺序取 clip"""

    def __init__(self, **kwargs):
        super(KeepFirstClipSampler, self).__init__(**kwargs)

    def __call__(self, views):

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        view_num = self.get_view_num()

        if self.shuffle:
            selected_index = list(range(1, view_num))
            random.shuffle(selected_index)
            selected_index = [0] + selected_index
        else:
            selected_index = list(range(view_num))

        if self.debug:
            print("clip index", selected_index)

        return [views[index] for index in selected_index]


class RandomIntervalClipSampler(RandomClipSampler):
    """在 interval 范围内随机选间隔（受视频长度约束），并从合法范围内随机选起始帧，最大程度避免补帧"""

    def __init__(self, interval=(1, 5), **kwargs):
        super(RandomIntervalClipSampler, self).__init__(**kwargs)
        self.interval_range = interval

    def _sample_interval(self, total, view_num):
        # 处理 view_num == 1 的边界情况
        if view_num <= 1:
            return 1

        # 用户指定的 interval 范围
        if isinstance(self.interval_range, (list, tuple)) and len(self.interval_range) == 2:
            low, high = self.interval_range
        else:
            low = high = int(self.interval_range)

        # 根据视频长度计算最大可行 interval（确保能采满 view_num 帧，从 start=0 开始）
        max_feasible_interval = (total - 1) // (view_num - 1)
        
        # 如果视频太短，max_feasible_interval 可能为 0 或负数 → 至少为 1
        max_feasible_interval = max(1, max_feasible_interval)

        # 实际采样上限：不能超过用户设定的 high，也不能超过可行上限
        actual_high = min(high, max_feasible_interval)

        # 如果 low > actual_high，说明即使最小间隔也采不够 → 仍返回 feasible 值（后续会补帧）
        if low > actual_high:
            interval = actual_high  # 尽力而为
        else:
            interval = random.randint(low, actual_high)

        return interval

    def __call__(self, views):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        view_num = self.get_view_num()
        total = len(views)

        if total == 0:
            raise ValueError("Input views is empty")
        if view_num <= 0:
            raise ValueError("view_num must be positive")

        # 动态采样一个与视频长度兼容的 interval
        interval = self._sample_interval(total, view_num)

        # 所需最小长度（用于判断是否能无补帧采样）
        min_required_length = (view_num - 1) * interval + 1

        if total >= min_required_length:
            # ✅ 足够长：在 [0, max_start] 内随机选起始帧
            max_start = total - min_required_length
            start = random.randint(0, max_start)
            indices = [start + i * interval for i in range(view_num)]
        else:
            # ⚠️ 仍不够（理论上不应发生，除非 view_num=1 或极端情况），兜底处理
            start = random.randint(0, total - 1)
            indices = []
            current = start
            while len(indices) < view_num:
                if current < total:
                    indices.append(current)
                else:
                    break
                current += interval
            # 补足
            if len(indices) < view_num:
                remaining = view_num - len(indices)
                extra = np.random.choice(total, size=remaining, replace=True)
                indices.extend(extra.tolist())
            indices = indices[:view_num]

        # 打乱（仅当 shuffle=True）
        if self.shuffle:
            random.shuffle(indices)
        
        if self.debug:
            print(f"shuffle={self.shuffle}, interval={interval}, total={total}, indices={indices}")

        clip = [views[i] for i in indices]
        assert len(clip) == view_num, f"Expected {view_num} frames, got {len(clip)}"
        return clip
