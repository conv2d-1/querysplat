import logging
import random
import numpy as np

import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class DebugBatchSampler(BatchSampler):
    def __init__(self, dataset_list: list):
        self.dataset_list = dataset_list

        # Initialize dataset lengths and cumulative lengths
        self._initialize_dataset_lengths()

        # Create BatchSamplers for each dataset
        self._create_batch_samplers()

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
        logging.info(f"DebugBatchSampler, datasets: {datasets_info}")

    def _create_batch_samplers(self):
        """Create BatchSamplers for each dataset based on shuffle flag."""
        self.src_batch_samplers = [
            BatchSampler(
                sampler=SequentialSampler(ds),
                batch_size=1,
                drop_last=False,
            )
            for ds in self.dataset_list
        ]

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
        idx_ds = 0

        for bi in range(self.n_total_batch):
            # If the current dataset's batch list is empty, regenerate it
            if not self.raw_batches[idx_ds]:
                idx_ds += 1
            
            if idx_ds >= len(self.raw_batches):
                break

            # Get a batch from the selected dataset
            batch_raw = self.raw_batches[idx_ds].pop()

            # Shift the batch indices by the cumulative dataset length
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]
            yield batch 

    def __len__(self):
        """Return the total number of batches."""
        return self.n_total_batch




if "__main__" == __name__:
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, start, len) -> None:
            super().__init__()
            self.start = start
            self.len = len
            self.name = ""

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            return self.start + index

    dataset_1 = SimpleDataset(0, 10)
    dataset_2 = SimpleDataset(0, 20)
    dataset_3 = SimpleDataset(0, 50)

    concat_dataset = ConcatDataset([dataset_1, dataset_2, dataset_3])  # will directly concatenate

    mixed_sampler = DebugBatchSampler(
        dataset_list=[dataset_1, dataset_2, dataset_3],
    )

    loader = DataLoader(concat_dataset, batch_sampler=mixed_sampler)

    for d in loader:
        print(d)
