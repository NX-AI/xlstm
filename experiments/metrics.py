# Copyright (c) NXAI GmbH and its affiliates 2024
from typing import Optional

import torch
import torchmetrics
from torchmetrics import Metric


class SequenceAccuracy(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True

    def __init__(self, **kwargs):
        super().__init__()
        self._acc = torchmetrics.Accuracy(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.reshape((-1, preds.shape[-1]))
        target = target.flatten()
        return self._acc.update(preds, target)

    def compute(self):
        return self._acc.compute()

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._acc.reset()
