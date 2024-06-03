# Copyright (c) NXAI GmbH and its affiliates 2024
# Andreas Auer, Korbinian Pöppel
import inspect
import itertools
import random
from dataclasses import asdict, dataclass, field, make_dataclass
from typing import Mapping, List

import torch
import torchmetrics

from .generate import ALL_ARGS
from .online_generate import OnlineTaskGenerateMaskedSeparate
from ..utils import DataGen
from ...metrics import SequenceAccuracy


def create_dataclass(cls_name, params_list, additional=[]):
    fields = []
    for param in params_list:
        if param.default != inspect._empty:
            fields.append((param.name, param.annotation, field(default=param.default)))
        else:
            fields.append((param.name, param.annotation))

    fields += additional
    return make_dataclass(cls_name, fields)


FormLangDatasetConfigBase = create_dataclass(
    "SyntheticMaskedDatasetConfig",
    (arg for arg in ALL_ARGS.values() if arg.name != "kwargs"),
    [
        ("synth_lang_type", str, field(default="parity")),
        ("count", dict, field(default_factory=dict({"train": 1, "validation": 1}))),
        ("subpar", dict, field(default=None)),
    ],
)


@dataclass
class FormLangDatasetConfig(FormLangDatasetConfigBase):
    shift: int = 1
    enable_mask: bool = False
    additional_prefix_tokens: int = (
        0  # e.g. adding [CLS]... before for global attention memory
    )
    additional_suffix_tokens: int = (
        0  # e.g. adding [SEP]... after for global attention memory
    )
    additional_premask_tokens: int = (
        0  # e.g. adding [PREMASK]... before every mask to enable "causal working memory"
    )


class FormLangDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        dataset_mask,
        context_length,
        vocab_size,
        pad_idx=0,
        target_pad_idx=-1,
        shift: int = 1,
        enable_mask: bool = True,
        additional_prefix_tokens: int = 0,
        additional_suffix_tokens: int = 0,
        additional_premask_tokens: int = 0,
    ):
        self._context_length = context_length
        self._dataset = dataset
        self._dataset_mask = dataset_mask
        self.shift = shift
        self.enable_mask = enable_mask
        self._vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.target_pad_idx = target_pad_idx
        self.additional_prefix_tokens = additional_prefix_tokens
        self.additional_suffix_tokens = additional_suffix_tokens
        self.additional_premask_tokens = additional_premask_tokens

    def __getitem__(self, idx):
        prefix_token_offset = (
            self.vocab_size
            - self.additional_prefix_tokens
            - self.additional_suffix_tokens
            - self.additional_premask_tokens
        )
        suffix_token_offset = (
            self.vocab_size
            - self.additional_suffix_tokens
            - self.additional_premask_tokens
        )
        prefix_tokens = torch.arange(
            prefix_token_offset, prefix_token_offset + self.additional_prefix_tokens
        )
        suffix_tokens = torch.arange(
            suffix_token_offset, suffix_token_offset + self.additional_suffix_tokens
        )
        seq = torch.from_numpy(self._dataset[idx])
        seq = torch.concat((prefix_tokens, seq, suffix_tokens))
        mask = torch.from_numpy(self._dataset_mask[idx])
        mask = torch.concat(
            (
                torch.zeros([self.additional_prefix_tokens]).to(
                    device=mask.device, dtype=mask.dtype
                ),
                mask,
                torch.zeros([self.additional_suffix_tokens]).to(
                    device=mask.device, dtype=mask.dtype
                ),
            )
        )
        mask = mask.contiguous()
        if self.additional_premask_tokens:
            new_seq = []
            new_mask = []
            premask_token_offset = self.vocab_size - self.additional_premask_tokens
            for s, m in zip(seq, mask):
                if m == 1:
                    for t in range(
                        premask_token_offset,
                        premask_token_offset + self.additional_premask_tokens,
                    ):
                        new_seq.append(t)
                        new_mask.append(0)
                    new_seq.append(s)
                    new_mask.append(m)
                else:
                    new_seq.append(s)
                    new_mask.append(m)
            seq = torch.tensor(new_seq)
            mask = torch.tensor(new_mask)

        if self.enable_mask:
            # mask
            seqIn = torch.where(mask == 1, self.pad_idx, seq).to(dtype=torch.long)
            seqOut = torch.where(mask == 1, seq, self.target_pad_idx).to(
                dtype=torch.long
            )
        else:
            seqIn = seq.to(dtype=torch.long)
            seqOut = seq.to(dtype=torch.long)
        if self.shift > 0:
            res = seqIn[: -self.shift], seqOut[self.shift :]
        elif self.shift < 0:
            res = seqIn[-self.shift :], seqOut[: self.shift]
        else:
            res = seqIn, seqOut
        return res

    def __len__(self):
        return len(self._dataset)

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def context_length(self):
        return self._context_length


class FormLangDatasetGenerator(DataGen):
    config_class = FormLangDatasetConfig

    def __init__(self, cfg: FormLangDatasetConfig, check_existing: bool = False):
        self.config = cfg
        self.check_existing = check_existing
        self.subsets = ["train", "validation", "test"]
        self.seeds = self._generate_seeds()
        setattr(self.config, "seeds", self.seeds)
        self._generate_dataset()

    def _generate_dataset(self):
        self.datasets = {}
        for subset in self.subsets:
            for subset_part in self._resolve_subset_subparts(subset):
                kwargs = asdict(self.config)
                kwargs["vocab_size"] = (
                    kwargs["vocab_size"]
                    - (
                        kwargs["additional_prefix_tokens"]
                        if "additional_prefix_tokens" in kwargs
                        else 0
                    )
                    - (
                        kwargs["additional_suffix_tokens"]
                        if "additional_suffix_tokens" in kwargs
                        else 0
                    )
                    - (
                        kwargs["additional_premask_tokens"]
                        if "additional_premask_tokens" in kwargs
                        else 0
                    )
                )
                if (
                    "subpar" in kwargs
                    and self.config.subpar
                    and subset_part in self.config.subpar
                    and self.config.subpar[subset_part] is not None
                ):
                    subconfig = self.config.subpar[subset_part]
                    for param in subconfig:
                        kwargs[param] = subconfig[param]

                kwargs["count"] = kwargs["count"][subset]
                online_generator = OnlineTaskGenerateMaskedSeparate(
                    self.seeds[subset_part], kwargs
                )
                self.datasets[subset_part] = FormLangDataset(
                    online_generator.dataset,
                    online_generator.dataset_mask,
                    context_length=self.config.context_length,
                    vocab_size=self.config.vocab_size,
                    pad_idx=0,
                    target_pad_idx=-1,
                    shift=self.config.shift,
                    enable_mask=self.config.enable_mask,
                    additional_prefix_tokens=self.config.additional_prefix_tokens,
                    additional_premask_tokens=self.config.additional_premask_tokens,
                    additional_suffix_tokens=self.config.additional_suffix_tokens,
                )

    def _resolve_subset_subparts(self, subset) -> List[str]:
        subset_subparts = []
        if (
            self.config.subpar is not None
            and len(
                subparts := [
                    subpart
                    for subpart in self.config.subpar.keys()
                    if subpart.startswith(subset)
                    and self.config.subpar[subpart] is not None
                ]
            )
            > 0
        ):
            for subpart in subparts:
                subset_subparts.append(subpart)
        else:
            subset_subparts.append(subset)
        return subset_subparts

    def _generate_seeds(self):
        rng = random.Random(self.config.seed)
        seeds = {
            subpart: rng.randint(0, 1 << 32)
            for subpart in itertools.chain.from_iterable(
                [self._resolve_subset_subparts(subset) for subset in self.subsets]
            )
        }
        return seeds

    @property
    def train_split(self) -> FormLangDataset:
        return self.datasets["train"]

    @property
    def validation_split(self) -> Mapping[str, FormLangDataset]:
        return {k: v for k, v in self.datasets.items() if k.startswith("validation")}

    @property
    def test_split(self) -> Mapping[str, FormLangDataset]:
        return {k: v for k, v in self.datasets.items() if k.startswith("test")}

    @property
    def train_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection(
            SequenceAccuracy(
                task="multiclass", num_classes=self.vocab_size, ignore_index=-1
            )
        )

    @property
    def validation_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection(
            SequenceAccuracy(
                task="multiclass", num_classes=self.vocab_size, ignore_index=-1
            )
        )

    @property
    def test_metrics(self) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection(
            SequenceAccuracy(
                task="multiclass", num_classes=self.vocab_size, ignore_index=-1
            )
        )

    @property
    def vocab_size(self):
        return self.config.vocab_size

    @property
    def context_length(self):
        return self.config.context_length
