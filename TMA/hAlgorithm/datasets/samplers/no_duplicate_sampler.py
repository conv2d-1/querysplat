import logging

import numpy as np
import torch
from torch.utils.data import Sampler


class NoDuplicateDistributedSampler(Sampler):
    """A distributed sampler that avoids duplicate samples across ranks.

    When the dataset has fewer samples than the number of replicas (GPUs),
    this sampler pads with repeated indices so that every rank gets at least
    one sample.  This prevents hangs caused by some ranks executing zero
    forward passes while collective operations (barrier / all_gather) wait
    for all ranks to participate.
    """

    def __init__(self, dataset, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Automatically get num_replicas and rank from the distributed environment
        self.num_replicas = (
            torch.distributed.get_world_size()
        )  # Get total number of processes (GPUs)
        self.rank = torch.distributed.get_rank()  # Get the rank of the current process

        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))

        # If the dataset cannot be evenly divided by num_replicas, drop_last controls whether to discard the last incomplete batch
        if self.drop_last and self.num_samples % self.num_replicas != 0:
            self.num_samples -= self.num_samples % self.num_replicas

        # Number of samples assigned to each process
        self.num_samples_per_replica = self.num_samples // self.num_replicas

        # Handle the extra samples that cannot be evenly divided
        self.extra_samples = self.num_samples % self.num_replicas

        # When there are fewer samples than replicas, some ranks would get 0
        # samples.  This causes DDP forward count mismatch and collective-op
        # hangs.  Pad with repeated samples so every rank gets at least 1.
        self._padded = False
        if self.num_samples_per_replica == 0 and self.rank >= self.extra_samples:
            self._padded = True

    def __iter__(self):
        # Calculate the range of data for each process
        start = self.rank * self.num_samples_per_replica
        end = start + self.num_samples_per_replica
        indices = self.indices[start:end]

        # If there are extra samples, distribute them among the first few processes
        if self.rank < self.extra_samples:
            indices = indices + [self.num_samples_per_replica * self.num_replicas + self.rank]

        # Ensure every rank has at least 1 sample by assigning a padded index
        if len(indices) == 0:
            # Assign a wrapped index so all ranks execute at least one forward pass
            padded_idx = self.rank % self.num_samples
            indices = [padded_idx]
            if not hasattr(self, "_pad_warned"):
                logging.warning(
                    f"[NoDuplicateDistributedSampler] rank {self.rank} has 0 samples "
                    f"(dataset size {self.num_samples} < world_size {self.num_replicas}). "
                    f"Padding with index {padded_idx} to prevent distributed hang."
                )
                self._pad_warned = True

        # Shuffle the data if needed
        if self.shuffle:
            # Set a random seed so each process gets a different shuffle order
            seed = torch.initial_seed() % 2**32
            np.random.seed(seed)
            np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        # Return the number of samples for each process
        base = self.num_samples_per_replica + (1 if self.rank < self.extra_samples else 0)
        # Ensure at least 1 to match the padded iteration
        return max(base, 1) if self.num_samples > 0 else 0

    @property
    def is_padded(self):
        """Whether this rank received a padded (duplicate) sample."""
        return self._padded
