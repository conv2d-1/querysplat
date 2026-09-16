class StepSampler:
    def __init__(self, frame_num=2, view_num=1, frame_step=1, view_step=1) -> None:
        self.frame_num = frame_num
        self.view_num = view_num
        self.frame_step = frame_step
        self.view_step = view_step

    def presample_clip(self, sequences, cach_f):
        sample_clip = []
        for v_i in range(self.view_num):
            for f_i in range(self.frame_num):
                cach_id = v_i * self.view_step * cach_f + f_i * self.frame_step
                cur_item = sequences[cach_id]
                sample_clip.append(cur_item)
        return sample_clip, self.frame_num

    def __call__(self, sequences, sequnce_f, **kwargs):
        return sequences

    def get_sample_fv(self):
        return (self.frame_num - 1) * self.frame_step + 1, (self.view_num - 1) * self.view_step + 1


class StepSamplerCopy:
    def __init__(self, frame_num=2, view_num=1, frame_step=1, view_step=1) -> None:
        self.frame_num = frame_num
        self.view_num = view_num
        self.frame_step = frame_step
        self.view_step = view_step

    def presample_clip(self, sequences, cach_f):
        sample_clip = []
        for v_i in range(self.view_num):
            for f_i in range(self.frame_num):
                cach_id = v_i * self.view_step * cach_f + f_i * self.frame_step
                cur_item = sequences[cach_id]
                sample_clip.append(cur_item)
        return sample_clip, self.frame_num

    def __call__(self, sequences, sequnce_f, **kwargs):
        sample_sequences = [sequences[-1] for i in range(self.frame_num)]
        return sample_sequences

    def get_sample_fv(self):
        return (self.frame_num - 1) * self.frame_step + 1, (self.view_num - 1) * self.view_step + 1
