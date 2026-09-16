import matplotlib.pyplot as plt
import numpy as np


class CosineLrUpdater:
    def __init__(
        self,
        base_lr,
        max_iters,
        warmup_iters=0,
        warmup="linear",
        warmup_ratio=1e-6,
        min_lr=0.0,
    ):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup = warmup
        self.warmup_ratio = warmup_ratio
        self.min_lr = min_lr

    def get_warmup_lr(self, cur_iter):
        if self.warmup == "constant":
            return self.warmup_ratio
        elif self.warmup == "linear":
            # Linear increase from warmup_ratio to 1.0
            return self.warmup_ratio + (1 - self.warmup_ratio) * (cur_iter / self.warmup_iters)
        elif self.warmup == "exp":
            # Exponential increase
            return self.warmup_ratio ** ((1 - cur_iter / self.warmup_iters))
        elif self.warmup == "cosine":
            # Cosine warmup from warmup_ratio to 1.0
            alpha = (1 - np.cos(np.pi * cur_iter / self.warmup_iters)) / 2  # [0, 1]
            return self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        else:
            raise ValueError(f"Unsupported warmup method: {self.warmup}")

    def get_lr(self, cur_iter):
        # Cosine decay from 1.0 to min_lr ratio
        progress = (cur_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        cos_progress = np.cos(np.pi * progress)  # From 1 to -1
        lr_ratio = (1 + cos_progress) / 2  # Scale to [0, 1]
        # Scale and shift to [min_lr / base_lr, 1.0]
        scaled_lr_ratio = lr_ratio * (1 - self.min_lr / self.base_lr) + self.min_lr / self.base_lr
        return scaled_lr_ratio

    def __call__(self, cur_iter):
        if cur_iter < self.warmup_iters:
            alpha = self.get_warmup_lr(cur_iter)
        else:
            alpha = self.get_lr(cur_iter)
        return alpha


if __name__ == "__main__":
    lr_scheduler = CosineLrUpdater(
        base_lr=1e-5,
        max_iters=50000,
        warmup_iters=int(50000 * 0.4),
        warmup="cosine",
        warmup_ratio=1e-6,
        min_lr=1e-8,
    )

    x = np.arange(50000)
    alphas = [lr_scheduler(i) for i in x]
    print(alphas[0], alphas[-1])

    plt.plot(alphas)
    plt.title("Cosine LR Scheduler with Warmup")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate Ratio")
    plt.grid(True)
    plt.savefig("cosine_lr_scheduler.png")
    plt.show()
