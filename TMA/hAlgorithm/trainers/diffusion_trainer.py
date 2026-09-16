import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from hAlgorithm.utils import instantiate_from_config

from .base_trainer import BaseTrainer


class DiffusionTrainer(BaseTrainer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def register_schedule(self):
        """Registers the learning rate schedule or any other training schedules."""
        # Optimizer !should be defined after input layer is adapted
        if self.optimizer is None:
            self.optimizer = Adam(self.model.get_train_parameters(), lr=self.lr)
        else:
            optimizer_type = self.optimizer.pop("type")
            self.optimizer = getattr(torch.optim, optimizer_type)(
                self.model.get_train_parameters(), **self.optimizer
            )

        # LR scheduler
        lr_func = instantiate_from_config(self.lr_scheduler)
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)
