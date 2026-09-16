import numpy as np


class PolyLrUpdater:

    def __init__(
        self,
        base_lr,
        max_iters,
        warmup_iters=0,
        warmup="linear",
        warmup_ratio=1e-6,
        power=1.0,
        min_lr=0.0,
    ):
        self.base_lr = base_lr
        self.max_iters = max_iters

        self.warmup_iters = warmup_iters
        self.warmup = warmup
        self.warmup_ratio = warmup_ratio
        self.power = power
        self.min_lr = min_lr

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
        coeff = (1 - n_iter / self.max_iters) ** self.power
        return (1 - self.min_lr / self.base_lr) * coeff + self.min_lr / self.base_lr

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_iters:
            alpha = self.get_warmup_lr(n_iter)
        else:
            alpha = self.get_lr(n_iter)
        return alpha


if __name__ == "__main__":
    lr_scheduler = PolyLrUpdater(
        base_lr=5e-5,
        max_iters=200000,
        warmup_iters=200,
        warmup="linear",
        warmup_ratio=1e-6,
        power=0.9,
        min_lr=1e-8,
    )

    x = np.arange(200000)
    alphas = [lr_scheduler(i) for i in x]
    import matplotlib.pyplot as plt

    plt.plot(alphas)
    plt.savefig("poly_lr_scheduler.png")
