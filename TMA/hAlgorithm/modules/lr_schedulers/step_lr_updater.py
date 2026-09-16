import numpy as np


class StepLrUpdater:
    def __init__(
        self,
        warmup_iters=0,
        warmup="linear",
        warmup_ratio=1e-6,
        steps=None,
        ratio=0.1,
    ):
        self.warmup_iters = warmup_iters
        self.warmup = warmup
        self.warmup_ratio = warmup_ratio
        self.steps = steps
        self.ratio = ratio

    def get_warmup_lr(self, cur_iters):
        if self.warmup == "constant":
            return self.warmup_ratio
        elif self.warmup == "linear":
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            return 1 - k
        elif self.warmup == "exp":
            k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
            return k
        return 1

    def get_lr(self, n_iter):
        coeff = 1.0
        for step in self.steps:
            if n_iter >= step:
                coeff *= self.ratio
        return coeff

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_iters:
            alpha = self.get_warmup_lr(n_iter)
        elif self.steps is None or len(self.steps) == 0:
            alpha = 1.0
        else:
            alpha = self.get_lr(n_iter)
        return alpha


if __name__ == "__main__":
    lr_scheduler = StepLrUpdater(
        warmup_iters=10000,
        warmup="linear",
        warmup_ratio=1e-6,
        steps=[100000, 200000, 300000],
    )

    x = np.arange(300000)
    alphas = [lr_scheduler(i) for i in x]
    import matplotlib.pyplot as plt

    plt.plot(alphas)
    plt.savefig("step_lr_scheduler.png")
