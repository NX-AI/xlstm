# Copyright (c) NXAI GmbH and its affiliates 2024
# Andreas Auer, Maximilian Beck
from abc import abstractmethod
from typing import Tuple, Any, Optional, Mapping

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torchmetrics
from torch.utils.data import TensorDataset


class DataGen:

    @property
    @abstractmethod
    def train_split(self) -> torch.utils.data.Dataset:
        pass

    @property
    @abstractmethod
    def validation_split(self) -> Mapping[str, torch.utils.data.Dataset]:
        pass

    @property
    @abstractmethod
    def train_metrics(self) -> torchmetrics.MetricCollection:
        pass

    @property
    @abstractmethod
    def validation_metrics(self) -> torchmetrics.MetricCollection:
        pass

class SequenceTensorDataset(TensorDataset):

    def __init__(self, tensors: Tuple[Any, Any], vocab_size: int, context_length: int) -> None:
        super().__init__(*tensors)
        self._vocab_size = vocab_size
        self._context_length = context_length

    @property
    def vocab_size(self) -> Optional[int]:
        return self._vocab_size

    @property
    def context_length(self) -> int:
        return self._context_length


class CacheMixin:

    @staticmethod
    def check_exist(config, directory: Path, check_existing: bool):
        if directory is not None and os.path.exists(str(directory / "config.json")):
            # check if
            if check_existing:
                ds_hash = CacheMixin.dataset_hash(directory)
                with open(str(directory / "config.json")) as fp:
                    read_conf = json.load(fp)
                conf_dict = asdict(config)
                conf_dict["hash"] = ds_hash
                # compare except for data dir
                del read_conf["data_dir"]
                del conf_dict["data_dir"]
                assert read_conf == conf_dict, (
                    f"Non-matching configuration: " f"Read: {read_conf} - Current: {conf_dict}"
                )
            return True
        else:
            return False

    @staticmethod
    def post_generate(config, directory):
        ds_hash = CacheMixin.dataset_hash(directory)
        conf_dict = asdict(config)
        conf_dict["hash"] = ds_hash
        with open(str(directory / "config.json"), "w") as fp:
            json.dump(conf_dict, fp)

    @staticmethod
    def dataset_hash(subdir):
        return calc_joint_md5sum(subdir, exclude=["config.json"])


def calc_joint_md5sum(dir_path, exclude=[]):
    md5 = hashlib.md5()
    file_names = sorted(os.listdir(dir_path))
    for file_name in file_names:
        if file_name in exclude:
            continue
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                file_data = f.read()
                md5.update(file_data)
    return md5.hexdigest()

