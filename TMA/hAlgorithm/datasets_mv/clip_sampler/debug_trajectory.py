import random

import numpy as np


class DebugTrajectory:
    def __init__(self, view_num=None, debug=False, max_view=None, **kwargs):
        self.view_num = view_num
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
        if self.view_num is None:
            return views
        else:
            if self.view_num > 1:
                step = (len(views) - 1) / (self.view_num - 1) 
                indices = [round(i * step) for i in range(1, self.view_num - 1)]
                indices = [0] + indices + [len(views) - 1]
            else:
                indices = [0]

            if self.debug:
                print("DebugTrajectory, indices", indices)
                print("DebugTrajectory, frame id", [views[i]["frame_id"] for i in indices])
                print("DebugTrajectory, view id", [views[i]["view_id"] for i in indices])
            return [views[i] for i in indices]
