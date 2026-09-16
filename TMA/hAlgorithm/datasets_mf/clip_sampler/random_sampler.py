import numpy as np


class RandomSamplerBounded:
    def __init__(
        self,
        frame_num=2,
        view_num=1,
        init_frame_range=[10, 20],
        frame_range=[40, 50],
        view_range=1,
        warmup_steps=10,
    ) -> None:
        self.frame_num = frame_num
        self.view_num = view_num

        if isinstance(init_frame_range, (list, tuple)):
            self.init_min_frame_range = init_frame_range[0]
            self.init_max_frame_range = init_frame_range[1]
        else:
            self.init_min_frame_range = init_frame_range - 1
            self.init_max_frame_range = init_frame_range

        if isinstance(frame_range, (list, tuple)):
            self.min_frame_range = frame_range[0]
            self.max_frame_range = frame_range[1]
        else:
            self.min_frame_range = frame_range - 1
            self.max_frame_range = frame_range

        if isinstance(view_range, (list, tuple)):
            self.min_view_range = view_range[0]
            self.max_view_range = view_range[1]
        else:
            self.min_view_range = 1
            self.max_view_range = view_range

        self.step = 0
        self.warmup_steps = warmup_steps

    def presample_clip(self, sequences, cach_f):
        return sequences, cach_f

    def local_step(self):
        step = self.step
        self.step += 1
        return step

    def range_schedule(self, min_range, max_range):
        fraction = self.local_step() / self.warmup_steps
        return min(min_range + int((max_range - min_range) * fraction), max_range)

    def __call__(self, sequences, sequence_f, **kwargs):
        sequence_v = int(len(sequences) / sequence_f)
        cur_min_frame_range = self.range_schedule(self.init_min_frame_range, self.min_frame_range)
        cur_max_frame_range = self.range_schedule(self.init_max_frame_range, self.max_frame_range)
        cur_view_range = self.range_schedule(self.min_view_range, self.max_view_range)

        frame_range = np.random.randint(cur_min_frame_range, cur_max_frame_range)
        f_sample_indices = np.random.choice(frame_range, size=self.frame_num, replace=False)
        f_sample_indices.sort()
        init_sample_indices = np.random.randint(0, sequence_f - frame_range + 1)
        f_sample_indices = f_sample_indices + init_sample_indices

        v_sample_indices = np.random.choice(cur_view_range, size=self.view_num, replace=False)
        v_sample_indices.sort()
        init_v_sample_indices = np.random.randint(
            0, sequence_v - cur_view_range + 1, size=self.view_num
        )
        v_sample_indices = v_sample_indices + init_v_sample_indices

        sample_sequence = []
        for v in v_sample_indices:
            for f in f_sample_indices:
                sample_sequence.append(sequences[sequence_f * v + f])

        return sample_sequence

    def get_sample_fv(self):
        return self.max_frame_range, self.max_view_range
