import torch
import pytest
from .template_chunkwise_vs_recurrent import template_test_recurrent_vs_chunkwise


def test_recurrent_vs_chunkwise_native(
    chunkwise_kernel_name: str = "chunkwise--native_autograd",
    sequence_kernel_name: str = "native_sequence__native",
    step_kernel_name: str = "native",
):
    template_test_recurrent_vs_chunkwise(
        chunkwise_kernel_name=chunkwise_kernel_name,
        sequence_kernel_name=sequence_kernel_name,
        step_kernel_name=step_kernel_name,
    )


@pytest.mark.parametrize(
    "chunkwise_kernel_name",
    [
        "chunkwise--native_autograd",
        "chunkwise--triton_xl_chunk",
        "chunkwise--triton_limit_chunk",
    ],
)
@pytest.mark.parametrize(
    "sequence_kernel_name, step_kernel_name",
    [
        ["native_sequence__native", "native"],
        ["native_sequence__triton", "triton"],
    ],
)
def test_recurrent_vs_chunkwise_triton(
    chunkwise_kernel_name: str,
    sequence_kernel_name: str,
    step_kernel_name: str,
):
    template_test_recurrent_vs_chunkwise(
        chunkwise_kernel_name=chunkwise_kernel_name,
        sequence_kernel_name=sequence_kernel_name,
        step_kernel_name=step_kernel_name,
    )
