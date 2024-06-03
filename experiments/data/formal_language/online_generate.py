# Copyright (c) NXAI GmbH and its affiliates 2024
# Andreas Auer
import numpy as np
from torch.utils.data.dataset import Dataset

from . import generate


class OnlineTaskGenerate(Dataset):

    def __init__(self, seed, generate_kwargs):
        self._generate_kwargs = generate_kwargs
        self._count = generate_kwargs["count"]
        assert self._count < (1 << 32)
        self.base_seed = np.random.default_rng(seed=seed).integers((1 << 32) - self._count)
        self._generate_kwargs.pop("seed")
        self._generate_kwargs.pop("count")

    def __getitem__(self, index):
        seed = self.base_seed + index
        batch, batch_mask = generate.GEN_FUNCS[self._generate_kwargs["synth_lang_type"]](seed=seed, **self._generate_kwargs)
        assert batch.shape[0] == 1  # Currently one element per generation
        return batch[0], batch_mask[0]

    def __len__(self) -> int:
        return self._count


class _AccessWrapper:
    def __init__(self, access_func, count_func):
        self._access_func = access_func
        self._count_func = count_func

    def __getitem__(self, index):
        return self._access_func(index)

    def __len__(self) -> int:
        return self._count_func()


class OnlineTaskGenerateMaskedSeparate(Dataset):
    """
    Little wrapper to allow an interface where the data and mask are retrieved from "different" variables
    given the interface of the formal lang tasks
    """

    def __init__(self, seed, generate_kwargs):
        self._online_task_generate = OnlineTaskGenerate(seed, generate_kwargs)
        self._dataset_wrapper = _AccessWrapper(self._retrieve_batch_data, self._online_task_generate.__len__)
        self._dataset_mask_wrapper = _AccessWrapper(self._retrieve_batch_mask, self._online_task_generate.__len__)
        self._current_idx = None
        self._current_values = None

    def _retrieve(self, index):
        if self._current_idx != index:
            self._current_values = self._online_task_generate[index]
            self._current_idx = index

    def _retrieve_batch_data(self, index):
        self._retrieve(index)
        return self._current_values[0]

    def _retrieve_batch_mask(self, index):
        self._retrieve(index)
        return self._current_values[1]

    @property
    def dataset(self):
        return self._dataset_wrapper

    @property
    def dataset_mask(self):
        return self._dataset_mask_wrapper
