# Original license text from pybind/scikit_build_example

# Copyright (c) 2016 The Pybind Development Team, All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to the author of this software, without
# imposing a separate written license agreement for such Enhancements, then you
# hereby grant the following license: a non-exclusive, royalty-free perpetual
# license to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.

# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 4.0
cmake_minimum_required(VERSION 3.18...4.0)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

# Find pybind11 and Python
find_package(pybind11 CONFIG REQUIRED)

# Find CUDA and set up the CUDA language
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
  message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}, building with CUDA support")
  if (DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(XLSTM_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  else()
    set(XLSTM_CUDA_ARCHITECTURES 75-real 80-real 86-real 90)
  endif()
  enable_language(CUDA)

  # Add a library using FindPython's tooling (pybind11 also provides a helper like
  # this)
  python_add_library(
      _slstm MODULE

      xlstm/blocks/slstm/src/cuda/slstm.cc
      xlstm/blocks/slstm/src/cuda/slstm_forward.cu
      xlstm/blocks/slstm/src/cuda/slstm_backward.cu
      xlstm/blocks/slstm/src/cuda/slstm_backward_cut.cu
      xlstm/blocks/slstm/src/cuda/slstm_pointwise.cu
      xlstm/blocks/slstm/src/util/blas.cu
      xlstm/blocks/slstm/src/util/cuda_error.cu

      WITH_SOABI
  )
  set_property(TARGET _slstm PROPERTY CUDA_ARCHITECTURES ${XLSTM_CUDA_ARCHITECTURES})

  # Build for CUDA architectures 8.0, 8.6, and 9.0 by default
  if (NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(ENV{TORCH_CUDA_ARCH_LIST} "7.5;8.0;8.6;9.0+PTX")
  endif()

  # Get Torch's CMake package from the build environment used by scikit-build.
  execute_process(
      COMMAND "${PYTHON_EXECUTABLE}"
      -c "import torch;print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
  )
  list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
  find_package(Torch REQUIRED)
  target_include_directories(_slstm PRIVATE ${TORCH_INCLUDE_DIRS})

  find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
  target_link_libraries(
      _slstm
      PRIVATE
      pybind11::headers
      ${TORCH_LIBRARIES}
      ${TORCH_PYTHON_LIBRARY}
      CUDA::cublas
      CUDA::cudart
  )

  # This is passing in the version as a define just as an example
  target_compile_definitions(_slstm PRIVATE
    TORCH_EXTENSION_NAME=_slstm
    SLSTM_HIDDEN_SIZE=-1
    SLSTM_BATCH_SIZE=8
    SLSTM_NUM_HEADS=4
    SLSTM_NUM_STATES=4
    SLSTM_DTYPE_B=float
    SLSTM_DTYPE_R=__nv_bfloat16
    SLSTM_DTYPE_W=__nv_bfloat16
    SLSTM_DTYPE_G=__nv_bfloat16
    SLSTM_DTYPE_S=__nv_bfloat16
    SLSTM_DTYPE_A=float
    SLSTM_NUM_GATES=4
    SLSTM_SIMPLE_AGG=true
    SLSTM_GRADIENT_RECURRENT_CLIPVAL_VALID=false
    SLSTM_GRADIENT_RECURRENT_CLIPVAL=0.0
  )

  # Add extra c flags like -U__CUDA_NO_HALF_OPERATORS__
  target_compile_options(_slstm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-res-usage>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
  )

  # The install directory is the output (wheel) directory
  install(TARGETS _slstm DESTINATION ${SKBUILD_PROJECT_NAME})

else()
  message(WARNING "CUDAToolkit not found, CUDA support will be disabled")
endif()
