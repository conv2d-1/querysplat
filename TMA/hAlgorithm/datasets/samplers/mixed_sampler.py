import logging
import random
import numpy as np

import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class MixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probabilities.
    Compatible with datasets at different resolutions.

    Args:
        dataset_list (list): List of datasets to sample from.
        batch_size_list (list): List of batch sizes corresponding to each dataset.
        drop_last (bool): If True, drop the last incomplete batch for each dataset.
        shuffle (bool): If True, shuffle the data before sampling.
        prob_list (list or None): List of probabilities for selecting each dataset.
                                  If None, probabilities are determined by the number of batches in each dataset.
        generator (torch.Generator or None): Generator for reproducibility. Defaults to None.
    """

    def __init__(
        self,
        dataset_list: list,
        batch_size_list: list,
        drop_last: bool,
        shuffle: bool,
        prob_list: list = None,
        generator: torch.Generator = None,
        set_random_size: bool = False,
        aspect_ratio_range=None,
        max_size_list=None,
    ):
        # Validate input parameters
        if len(dataset_list) != len(batch_size_list):
            raise ValueError("Length of dataset_list and batch_size_list must be equal.")

        if prob_list is not None and len(prob_list) != len(dataset_list):
            raise ValueError("Length of prob_list must match the length of dataset_list.")

        self.dataset_list = dataset_list
        self.batch_size_list = batch_size_list
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator
        self.set_random_size = set_random_size

        self.aspect_ratio_range = aspect_ratio_range
        if self.aspect_ratio_range is not None:
            if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
                raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")

        self.max_size_list = max_size_list

        # Initialize dataset lengths and cumulative lengths
        self._initialize_dataset_lengths()

        # Create BatchSamplers for each dataset
        self._create_batch_samplers()

        # Initialize probabilities
        self._initialize_probabilities(prob_list)

        # Precompute batches for each dataset
        self._precompute_batches()

    def _initialize_dataset_lengths(self):
        """Initialize dataset lengths and cumulative lengths."""
        dataset_names = [ds.name for ds in self.dataset_list]
        self.dataset_length = [len(ds) for ds in self.dataset_list]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(len(self.dataset_list) + 1)
        ]  # cumulative dataset length
        datasets_info = [(name, num) for name, num in zip(dataset_names, self.dataset_length)]
        logging.info(f"MixedBatchSampler, datasets: {datasets_info}")

    def _create_batch_samplers(self):
        """Create BatchSamplers for each dataset based on shuffle flag."""
        if self.shuffle:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=RandomSampler(ds, replacement=False, generator=self.generator),
                    batch_size=bs,
                    drop_last=self.drop_last,
                )
                for ds, bs in zip(self.dataset_list, self.batch_size_list)
            ]
        else:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=SequentialSampler(ds),
                    batch_size=bs,
                    drop_last=self.drop_last,
                )
                for ds, bs in zip(self.dataset_list, self.batch_size_list)
            ]

    def _initialize_probabilities(self, prob_list):
        """Initialize probabilities for selecting each dataset."""
        if prob_list is None:
            # If no probabilities are provided, use the number of batches as weights
            n_batches = [len(bs) for bs in self.src_batch_samplers]
            self.prob_list = torch.tensor(n_batches, dtype=torch.float) / sum(n_batches)
        else:
            self.prob_list = torch.as_tensor(prob_list, dtype=torch.float)

        dataset_names = [dataset.name for dataset in self.dataset_list]
        prob_list = self.prob_list.numpy().tolist()
        logging.info(
            f"MixedBatchSampler, prob_list: {[info for info in zip(dataset_names, prob_list)]}"
        )

    def _precompute_batches(self):
        """Precompute all batches for each dataset."""
        self.raw_batches = [list(bs) for bs in self.src_batch_samplers]
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)

    def __iter__(self):
        """Iterate over batches, yielding indices corresponding to the concatenated dataset.

        Yields:
            list(int): A batch of indices, corresponding to ConcatDataset of dataset_list.
        """
        for bi in range(self.n_total_batch):
            # Select a dataset based on the probability distribution
            idx_ds = torch.multinomial(
                self.prob_list, 1, replacement=True, generator=self.generator
            ).item()

            # If the current dataset's batch list is empty, regenerate it
            if not self.raw_batches[idx_ds]:
                self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])

            # Get a batch from the selected dataset
            batch_raw = self.raw_batches[idx_ds].pop()

            # Shift the batch indices by the cumulative dataset length
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]

            if self.aspect_ratio_range is None and self.max_size_list is None:
                yield batch
            else:
                random_max_size = random_aspect_ratio = None
                if self.aspect_ratio_range is not None:
                    random_aspect_ratio = round(random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)
                if self.max_size_list is not None:
                    random_max_size = random.choice(self.max_size_list)
                batch = [[n, {"aspect_ratio": random_aspect_ratio, "max_size": random_max_size}] for n in batch]
                yield batch

    def __len__(self):
        """Return the total number of batches."""
        return self.n_total_batch


class MixedMaxIterBatchSampler(MixedBatchSampler):
    def __init__(
        self,
        max_iter: int,
        **kwargs,
    ):
        super(MixedMaxIterBatchSampler, self).__init__(**kwargs)
        self.n_total_batch = max_iter


if "__main__" == __name__:
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, start, len) -> None:
            super().__init__()
            self.start = start
            self.len = len

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            return self.start + index

    dataset_1 = SimpleDataset(0, 10)
    dataset_2 = SimpleDataset(200, 20)
    dataset_3 = SimpleDataset(1000, 50)

    concat_dataset = ConcatDataset([dataset_1, dataset_2, dataset_3])  # will directly concatenate

    mixed_sampler = MixedBatchSampler(
        dataset_list=[dataset_1, dataset_2, dataset_3],
        batch_size_list=[4, 4, 4],
        drop_last=True,
        shuffle=False,
        prob_list=[0.6, 0.3, 0.1],
        generator=torch.Generator().manual_seed(0),
    )

    loader = DataLoader(concat_dataset, batch_sampler=mixed_sampler)

    for d in loader:
        print(d)
