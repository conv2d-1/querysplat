import torch


class RandomSampler(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, datas, num_samples=None):
        num_samples = num_samples if num_samples is not None else self.num_samples
        if num_samples >= len(datas):
            return datas
        indices = torch.randperm(len(datas))[:num_samples]
        return datas[indices]
