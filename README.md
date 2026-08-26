# xLSTM Plus

This is a fork of the original [NX-AI/xlstm](https://github.com/NX-AI/xlstm) repository.

It includes additonal features for more flexible training and support to Truncated Backpropagation Through Time (TBPTT).


## Installation
Create a virtual environment (e.g., with Conda from the file `environment_pt240cu124.yaml`) and install via pip:
```bash
pip install git+https://github.com/LeZeez/xlstm_plus.git

### Or

git clone https://github.com/LeZeez/xlstm_plus.git
cd xlstm_plus
pip install -e .
```

Triton Kernels (required for xLSTM Large 7B and provides faster training):
```bash
pip install mlstm_kernels
```



## Features

### Auto mode for xLSTMLarge
Uses triton kernels when sequences aligns with kernels constraints (e.g., seq_len divisible by chunk size, head dimension > 16) and the rest of sequences are passed to the native scan.

When kernels are not available, or you are on CPU, it falls back to native scan with no constraints.

```python
import torch
from xlstm_plus.xlstm_large import xLSTMLarge, xLSTMLargeConfig

config = xLSTMLargeConfig(
    embedding_dim=256,
    num_heads=4,
    num_blocks=4,
    vocab_size=1000,
    chunk_size=64,
    mode="auto",  # Uses Triton when S % 64 == 0; falls back to native scan otherwise
    return_last_states=True,
)
model = xLSTMLarge(config)
```

### Parameters `boundaries` and `return_detached_states` in forward pass
Artificially injects `-1000` in the forget gate at the first token of each sequence, given that a tensor of bools in size `(B, T)` is provided - with `True` corresponding to the first token - so you can pack documents without padding. Returned states would be already detached, ready to be passed to the next batch (TBPTT).

```python
# Pack multiple documents into a single sequence without cross-document attention leakage
B, S = 2, 64
x = torch.randint(0, 1000, (B, S))

# Mark position 32 as the start of Document 2
boundaries = torch.zeros(B, S, dtype=torch.bool)
boundaries[:, 32] = True

logits, state = model(x, boundaries=boundaries, return_detached_states=True)
```

### Checkpointing
Checkpointing decreases memory usage in exchange for slower training (~20-30% slower).

```python
from xlstm_plus.xlstm_large import xLSTMLarge, xLSTMLargeConfig

config = xLSTMLargeConfig(
    embedding_dim=256,
    num_heads=4,
    num_blocks=4,
    vocab_size=1000,
    chunk_size=64,
    use_checkpoint=True,  # Saves VRAM with non-reentrant activation checkpointing
    return_last_states=True,
    mode="auto",
)
model = xLSTMLarge(config)
```

### Functions `zero_rows()` and `detach_states()`

Utilities for asynchronous batching and continuous TBPTT memory management:
- `detach_states(state)`: Recursively detaches all nested tensors in the state dictionary across steps.
- `zero_rows(state, mask)`: In-place zeroes memory for sequences in the batch that reached EOS.

```python
import torch
from xlstm_plus import detach_states, zero_rows
from xlstm_plus.xlstm_large import xLSTMLarge, xLSTMLargeConfig

config = xLSTMLargeConfig(
    embedding_dim=256,
    num_heads=4,
    num_blocks=4,
    vocab_size=1000,
    return_last_states=True,
    mode="auto",
)
model = xLSTMLarge(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

state = None
for x_chunk, y_chunk, finished_mask in train_dataloader:
    optimizer.zero_grad()
    logits, state = model(x_chunk, state=state)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 1000), y_chunk.view(-1))
    loss.backward()
    optimizer.step()

    # Detach states before the next batch/chunk
    state = detach_states(state)

    # In-place reset memory rows for finished sequences in the active batch
    # finished_mask: 1D bool tensor of shape (batch_size,)
    if finished_mask.any():
        zero_rows(state, finished_mask)
```



# (&darr; Read note) Original repository README 

*Note: packages imports here are renamed to `xlstm_plus`*

<div align="center">

# xLSTM: Extended Long Short-Term Memory

[![Paper](https://img.shields.io/static/v1?label=Paper&message=2405.04517&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2405.04517)
[![PyPI](https://img.shields.io/pypi/v/xlstm?color=blue)](https://pypi.org/project/xlstm/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/xlstm?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/tirex-ts)
![GitHub Repo stars](https://img.shields.io/github/stars/NX-AI/xlstm)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

![xLSTM Figure](./res/desc_xlstm_overview.svg)

</div>

> **Paper:** https://arxiv.org/abs/2405.04517
>
> **Authors:** Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter

## About

xLSTM is a new Recurrent Neural Network architecture based on ideas of the original LSTM.
Through Exponential Gating with appropriate normalization and stabilization techniques and a new Matrix Memory it overcomes the limitations of the original LSTM 
and shows promising performance on Language Modeling when compared to Transformers or State Space Models.

:rotating_light: We trained a 7B parameter xLSTM Language Model on 2.3T tokens! :rotating_light:

We refer to the optimized architecture for our xLSTM 7B as xLSTM Large. 

## Minimal Installation

*Removed (see Installation section above)*

## Requirements

This package is based on PyTorch and was tested for versions `>=1.8`. For a well-tested environment, install the `environment_pt240cu124.yaml` as:
```bash
conda env create -n xlstm -f environment_pt240cu124.yaml
conda activate xlstm
``` 

For the xLSTM Large 7B model we require our [`mlstm_kernels`](https://github.com/NX-AI/mlstm_kernels) package, which provides fast kernels for the xLSTM.

<div align="center">

# xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference
[![Paper](https://img.shields.io/static/v1?label=Paper&message=2503.13427&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2503.13427)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-xLSTM_7B-yellow?logo=huggingface)](https://huggingface.co/NX-AI/xLSTM-7b)
[![License](https://img.shields.io/badge/license-nxai_community-green)](https://github.com/NX-AI/tirex-internal/blob/main/LICENSE)


> **Paper:** https://arxiv.org/abs/2503.13427
>
> **Authors:** Maximilian Beck, Korbinian Pöppel, Phillip Lippe, Richard Kurle, Patrick M. Blies, Günter Klambauer, Sebastian Böck, Sepp Hochreiter

![xLSTM Figure](./res/xlstm_7b_poster.svg)

</div>

We have optimized the xLSTM architecture in terms of training throughput and stability. 
The code for the updated architecture is located in `xlstm/xlstm_large`.

The model weights are available on Huggingface at https://huggingface.co/NX-AI/xLSTM-7b. 

## How to use the xLSTM Large 7B and its architecture

We provide a standalone single file implementation of the xLSTM Large architecture in [`xlstm/xlstm_large/model.py`](https://github.com/NX-AI/xlstm/blob/main/xlstm/xlstm_large/model.py).
This implementation requires our [`mlstm_kernels`](https://github.com/NX-AI/mlstm_kernels) package and other than that has no dependency on the NeurIPS xLSTM architecture implementation.

For a quick start, we provide a [`demo.ipynb`](https://github.com/NX-AI/xlstm/blob/main/notebooks/xlstm_large/demo.ipynb) notebook for the xLSTM Large architecture at `notebooks/xlstm_large/demo.ipynb`. 

In this notebook we import our config and model class, initialize a random model and perform a forward pass, like so:

```python
import torch
from xlstm_plus.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# configure the model with TFLA Triton kernels
xlstm_config = xLSTMLargeConfig(
    embedding_dim=512,
    num_heads=4,
    num_blocks=6,
    vocab_size=2048,
    return_last_states=True,
    mode="inference",
    chunkwise_kernel="chunkwise--triton_xl_chunk", # xl_chunk == TFLA kernels
    sequence_kernel="native_sequence__triton",
    step_kernel="triton",
)
# instantiate the model
xlstm = xLSTMLarge(xlstm_config)
xlstm = xlstm.to("cuda")
# create inputs
input = torch.randint(0, 2048, (3, 256)).to("cuda")
# run a forward pass
out = xlstm(input)
out.shape[1:] == (256, 2048)
```

## Recommendation for other hardware

We have tested our model mostly on NVIDIA GPUs, however our Triton kernels should also run on AMD GPUs. 
For other platforms, like Apple Metal, we recommend using the native PyTorch implementations for now:

```python 
xlstm_config = xLSTMLargeConfig(
    embedding_dim=512,
    num_heads=4,
    num_blocks=6,
    vocab_size=2048,
    return_last_states=True,
    mode="inference",
    chunkwise_kernel="chunkwise--native_autograd", # no Triton kernels
    sequence_kernel="native_sequence__native", # no Triton kernels
    step_kernel="native", # no Triton kernels
)
```

If you are working inside Apple's MLX ecosystem, check out the community-driven
[xLSTM-metal](https://github.com/MLXPorts/xLSTM-metal) port which provides an
MLX-native implementation of xLSTM targeting Apple Silicon.

# Models from the xLSTM NeurIPS Paper

This section explains how to use the models from the xLSTM paper.

## How to use the xLSTM architecture from our NeurIPS paper

For non language applications or for integrating in other architectures you can use the `xLSTMBlockStack` and for language modeling or other token-based applications you can use the `xLSTMLMModel`.

### Using the sLSTM CUDA kernels

For the CUDA version of sLSTM, you need Compute Capability >= 8.0, see [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). If you have problems with the compilation, please try (thanks to [@zia1138](https://github.com/zia1138) for pointing out):
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
```

For all kinds of custom setups with torch and CUDA, keep in mind that versions have to match. Also, to make sure the correct CUDA libraries are included you can use the "XLSTM_EXTRA_INCLUDE_PATHS" environment variable now to inject different include paths, e.g.:

```bash
export XLSTM_EXTRA_INCLUDE_PATHS='/usr/local/include/cuda/:/usr/include/cuda/'
```

or within python:

```python
import os
os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'
```

for standalone, even faster sLSTM kernels, feel free to use the [FlashRNN](https://github.com/NX-AI/flashrnn) library.

### xLSTM Block Stack

The `xLSTMBLockStack` is meant for use as alternative backbone in existing projects. It is similar to a stack of Transformer blocks, but uses xLSTM blocks:

```python
import torch

from xlstm_plus import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,
    num_blocks=7,
    embedding_dim=128,
    slstm_at=[1],

)

xlstm_stack = xLSTMBlockStack(cfg)

x = torch.randn(4, 256, 128).to("cuda")
xlstm_stack = xlstm_stack.to("cuda")
y = xlstm_stack(x)
y.shape == x.shape
```

If you are working with yaml strings / files for configuration you can also use dacite to create the config dataclasses. This is the same as the snippet above:

```python
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm_plus import xLSTMBlockStack, xLSTMBlockStackConfig

xlstm_cfg = """ 
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
slstm_block:
  slstm:
    backend: cuda
    num_heads: 4
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: 256
num_blocks: 7
embedding_dim: 128
slstm_at: [1]
"""
cfg = OmegaConf.create(xlstm_cfg)
cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
xlstm_stack = xLSTMBlockStack(cfg)

x = torch.randn(4, 256, 128).to("cuda")
xlstm_stack = xlstm_stack.to("cuda")
y = xlstm_stack(x)
y.shape == x.shape

```


### xLSTM Language Model

The `xLSTMLMModel` is a wrapper around the `xLSTMBlockStack` that adds the token embedding and lm head.

```python
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm_plus import xLSTMLMModel, xLSTMLMModelConfig

xlstm_cfg = """ 
vocab_size: 50304
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
slstm_block:
  slstm:
    backend: cuda
    num_heads: 4
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: 256
num_blocks: 7
embedding_dim: 128
slstm_at: [1]
"""
cfg = OmegaConf.create(xlstm_cfg)
cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
xlstm_stack = xLSTMLMModel(cfg)

x = torch.randint(0, 50304, size=(4, 256)).to("cuda")
xlstm_stack = xlstm_stack.to("cuda")
y = xlstm_stack(x)
y.shape[1:] == (256, 50304)
```


## Experiments

The synthetic experiments show-casing the benefits of sLSTM over mLSTM and vice versa best are the Parity task and the Multi-Query Associative Recall task. The Parity task can only be solved with state-tracking capabilities provided by the memory-mixing of sLSTM. The Multi-Query Associative Recall task measures memorization capabilities, where the matrix-memory and state expansion of mLSTM is very beneficial.
In combination they do well on both tasks.

To run each, run the `main.py` in the experiments folder like:
```
PYTHONPATH=. python experiments/main.py --config experiments/parity_xlstm01.yaml   # xLSTM[0:1], sLSTM only
PYTHONPATH=. python experiments/main.py --config experiments/parity_xlstm10.yaml   # xLSTM[1:0], mLSTM only
PYTHONPATH=. python experiments/main.py --config experiments/parity_xlstm11.yaml   # xLSTM[1:1], mLSTM and sLSTM
```

Note that the training loop does not contain early stopping or test evaluation.


## Citation

If you use this codebase, or otherwise find our work valuable, please cite the xLSTM paper:
```
@inproceedings{beck:24xlstm,
  title = {xLSTM: Extended Long Short-Term Memory}, 
  author = {Maximilian Beck and Korbinian Pöppel and Markus Spanring and Andreas Auer and Oleksandra Prudnikova and Michael Kopp and Günter Klambauer and Johannes Brandstetter and Sepp Hochreiter},
  booktitle = {Thirty-eighth Conference on Neural Information Processing Systems},
  year = {2024},
  url = {https://arxiv.org/abs/2405.04517}, 
}

@article{beck:25xlstm7b,
  title = {{xLSTM 7B}: A Recurrent LLM for Fast and Efficient Inference},
  author = {Maximilian Beck and Korbinian Pöppel and Phillip Lippe and Richard Kurle and Patrick M. Blies and Günter Klambauer and Sebastian Böck and Sepp Hochreiter},
  booktitle = {Forty-second International Conference on Machine Learning},
  year = {2025},
  url = {https://arxiv.org/abs/2503.13427}
}

```
