# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbinian Poeppel

import os
from typing import Sequence, Union
import logging

import time
import random

import torch
from torch.utils.cpp_extension import load as _load

LOGGER = logging.getLogger(__name__)


def defines_to_cflags(defines=Union[dict[str, Union[int, str]], Sequence[tuple[str, Union[str, int]]]]):
    cflags = []
    print(defines)
    if isinstance(defines, dict):
        defines = defines.items()
    for key, val in defines:
        cflags.append(f"-D{key}={str(val)}")
    return cflags


curdir = os.path.dirname(__file__)

# ROCm builds of torch expose the CUDA API through HIP: torch.cuda works, .cu
# sources are hipified transparently, but the toolchain is hipcc and cuBLAS is
# hipBLAS, so nvcc-only flags and cublas linkage must be swapped below.
IS_HIP = torch.version.hip is not None

if torch.cuda.is_available():
    from packaging import version

    if IS_HIP:
        from torch.utils.cpp_extension import ROCM_HOME

        os.environ["CUDA_LIB"] = os.path.join(ROCM_HOME or "/opt/rocm", "lib")
    elif version.parse(torch.__version__) >= version.parse("2.6.0"):
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(device_type="cuda")[-1])[0], "lib"
        )
    else:
        os.environ["CUDA_LIB"] = os.path.join(
            os.path.split(torch.utils.cpp_extension.include_paths(cuda=True)[-1])[0], "lib"
        )


EXTRA_INCLUDE_PATHS = (
    tuple(os.environ["XLSTM_EXTRA_INCLUDE_PATHS"].split(":")) if "XLSTM_EXTRA_INCLUDE_PATHS" in os.environ else ()
)
if "CONDA_PREFIX" in os.environ:
    # This enforces adding the correct include directory from the CUDA installation via torch. If you use the system
    # installation, you might have to add the cflags yourself.
    from pathlib import Path
    from packaging import version
    import glob

    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        matching_dirs = glob.glob(f"{os.environ['CONDA_PREFIX']}/targets/**", recursive=True)
        EXTRA_INCLUDE_PATHS = (
            EXTRA_INCLUDE_PATHS
            + tuple(map(str, (Path(os.environ["CONDA_PREFIX"]) / "targets").glob("**/include/")))[:1]
        )


def _hipify_sources(sources):
    """Translate the sLSTM CUDA sources — and the headers they include — to HIP.

    torch's JIT builder only hipifies the files passed as ``sources``, not the
    headers they pull in (blas.h, inline_ops*.cuh, ...), so those would keep
    their cuBLAS / ``__nv_bfloat16`` spellings while the hipified .cu bodies use
    the HIP ones. Instead copy the whole src tree into a cache dir, hipify
    everything there, and compile from that copy. The repo sources are never
    touched.
    """
    import shutil
    from torch.utils.hipify import hipify_python

    src_root = os.path.abspath(curdir)
    out_root = os.environ.get(
        "XLSTM_HIP_SRC_DIR",
        os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "xlstm",
            "hip_src",
        ),
    )
    for sub in ("cuda", "util"):
        shutil.copytree(
            os.path.join(src_root, sub), os.path.join(out_root, sub), dirs_exist_ok=True
        )
    hipify_python.hipify(
        project_directory=out_root,
        output_directory=out_root,
        includes=[os.path.join(out_root, "*")],
        is_pytorch_extension=True,
        show_detailed=False,
    )
    # hipify writes renamed copies (cuda/foo.cu -> hip/foo.hip, blas.h ->
    # blas_hip.h) and leaves the untranslated originals; overwrite the originals
    # with the hipified text so stale relative includes still resolve to HIP.
    for dirpath, _, filenames in os.walk(out_root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, out_root)
            hip_rel = hipify_python.get_hip_file_path(rel, is_pytorch_extension=True)
            hip_path = os.path.join(out_root, hip_rel)
            if hip_path != path and os.path.exists(hip_path):
                shutil.copyfile(hip_path, path)

    # Residual fixups hipify's substitution map does not cover: CUDA-only
    # driver headers that have no HIP counterpart, and the fp16 gemm pointer
    # (hipify rewrites &cublasHgemm -> &hipblasHgemm, whose hipblasHalf
    # signature is incompatible with the __half-typed wrapper; point it at the
    # local cublasHgemm2 wrapper instead, matching the strided path).
    import re

    _dead_includes = re.compile(
        r'^\s*#\s*include\s*[<"](?:cuda|cuda_runtime_api|cuda_device_runtime_api)\.h[>"]\s*$',
        re.MULTILINE,
    )
    for dirpath, _, filenames in os.walk(out_root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            with open(path, "r") as fh:
                text = fh.read()
            new_text = _dead_includes.sub("// [hip] removed CUDA-only include", text)
            new_text = re.sub(r"&\s*hipblasHgemm\b", "&cublasHgemm2", new_text)
            # bf16 blas support is gated on CUDART_VERSION, which HIP lacks;
            # enable the same block on ROCm.
            new_text = new_text.replace(
                "CUDART_VERSION >= 11020",
                "(CUDART_VERSION >= 11020 || defined(__HIP_PLATFORM_AMD__))",
            )
            if new_text != text:
                with open(path, "w") as fh:
                    fh.write(new_text)

    new_sources = []
    for source in sources:
        rel = os.path.relpath(os.path.abspath(source), src_root)
        hip_rel = hipify_python.get_hip_file_path(rel, is_pytorch_extension=True)
        hip_path = os.path.join(out_root, hip_rel)
        if not os.path.exists(hip_path):
            hip_path = os.path.join(out_root, rel)
        # The pybind glue (.cc) references the HIP fp16/bf16 types for its
        # dtype dispatch. Those headers only compile under clang, so route the
        # glue through hipcc by giving it a .hip extension (torch selects the
        # compiler by suffix); a plain-C++ host compile would fail.
        if hip_path.endswith((".cc", ".cpp")):
            hip_source = os.path.splitext(hip_path)[0] + "_glue.hip"
            shutil.copyfile(hip_path, hip_source)
            hip_path = hip_source
        new_sources.append(hip_path)
    return new_sources


def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    if IS_HIP:
        sources = _hipify_sources(sources)
    suffix = ""
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {"float": "f", "__half": "h", "__nv_bfloat16": "b", "__hip_bfloat16": "b", "true": "1", "false": "0"}
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        # *(f"-I{path}" for path in EXTRA_INCLUDE_PATHS)
    ]
    for eip in EXTRA_INCLUDE_PATHS:
        extra_cflags.append("-isystem")
        extra_cflags.append(eip)

    if IS_HIP:
        # hipcc rejects nvcc-only flags (-Xptxas, -gencode, -res-usage, ...).
        # cuBLAS calls hipify to hipBLAS, so link hipblas; force-include the
        # compat shim for the enums/intrinsics hipify does not translate.
        # Force-include the compat shim ONLY on the device (hipcc) pass: it
        # pulls in hip_bf16.h/hip_fp16.h, which rely on clang builtins and do
        # not compile under the g++ host compiler used for the pybind glue.
        compat_header = os.path.join(curdir, "util", "hip_compat.h")
        myargs = {
            "verbose": True,
            "with_cuda": True,
            "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lhipblas"],
            "extra_cflags": [*extra_cflags],
            "extra_cuda_cflags": [
                "-O3",
                "-ffast-math",
                "-include",
                compat_header,
                *extra_cflags,
                *extra_cuda_cflags,
            ],
        }
    else:
        myargs = {
            "verbose": True,
            "with_cuda": True,
            "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lcublas"],
            "extra_cflags": [*extra_cflags],
            "extra_cuda_cflags": [
                # "-gencode",
                # "arch=compute_70,code=compute_70",
                # "-dbg=1",
                '-Xptxas="-v"',
                "-gencode",
                "arch=compute_80,code=compute_80",
                "-res-usage",
                "--use_fast_math",
                "-O3",
                "-Xptxas -O3",
                "--extra-device-vectorization",
                *extra_cflags,
                *extra_cuda_cflags,
            ],
        }
    print(myargs)
    myargs.update(**kwargs)
    # add random waiting time to minimize deadlocks because of badly managed multicompile of pytorch ext
    time.sleep(random.random() * 10)
    LOGGER.info(f"Before compilation and loading of {name}.")
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")
    return mod
