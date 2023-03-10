import torch
from dataclasses import dataclass

@dataclass
class InterleavedSamplerConfig:
    dataset_key: object
    every_n_epochs: int
    every_n_updates: int
    every_n_samples: int
    drop_last: bool


class InterleavedSampler:
    def __init__(self, datasets, dataset_key, drop_last, configs, batch_size, every_n_updates):
        super().__init__()
        self.datasets = datasets
        self.dataset = self.datasets[dataset_key]
        self.drop_last = drop_last
        self.configs = configs
        self.batch_size = batch_size
        self.every_n_updates = every_n_updates
        self.sample_idx = 0

    def __iter__(self):
        if self.drop_last:
            batches_per_epoch = len(self.dataset) // self.batch_size
        else:
            batches_per_epoch = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        samples_per_epoch = min(len(self.dataset, batches_per_epoch * self.batch_size))
        while True:
            for i in range(len(self.dataset)):
                yield i
                self.sample_idx += 1
                sample_in_batch = self.sample_idx % self.batch_size
                update = self.sample_idx // self.batch_size
                if sample_in_batch == 0 and update % self.every_n_updates == 0:
                    for j in range(len(self.dataset2)):
                        yield len(self.dataset) + j
                # drop last
                if self.sample_idx % samples_per_epoch == 0:
                    break

