// Copyright (c) NXAI GmbH and its affiliates 2023
// Korbinian Poeppel

// Adapted from the haste library
//
// See:
// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include "../util/blas.h"
#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"

#include "slstm.h"
#include "slstm_pointwise.cuh"
#include <driver_types.h>
#include <stdio.h>
#include <vector_types.h>

#define MAX_THREADS_PER_BLOCK 512

namespace {

uint round_to_power2(uint x) {
  uint pow2 = 1;
  while (x > 1) {
    x >>= 1;
    pow2 <<= 1;
  }
  return pow2;
}

uint _min(uint a, uint b) { return (a > b) ? b : a; }
uint _max(uint a, uint b) { return (a < b) ? b : a; }

} // anonymous namespace

namespace slstm {

struct ForwardPass::private_data {
  bool training;
  int batch_size;
  int hidden_size;
  int num_heads;
  cublasHandle_t main_blas_handle;
  cublasHandle_t blas_handle_R;
  cudaStream_t stream_R;
  cudaEvent_t event_R;
  cudaStream_t stream;
};

ForwardPass::ForwardPass(const bool training, const int batch_size,
                         const int hidden_size, const int num_heads,
                         const cublasHandle_t &blas_handle,
                         const cudaStream_t &stream)
    : data_(new private_data) {
  data_->training = training;
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->num_heads = num_heads;
  data_->main_blas_handle = blas_handle;
  data_->stream = stream;

  cublasCreate(&data_->blas_handle_R);
  cudaStreamCreate(&data_->stream_R);
  cudaEventCreateWithFlags(&data_->event_R, cudaEventDisableTiming);
  cublasSetStream(data_->blas_handle_R, data_->stream_R);
}

void ForwardPass::Set(const bool training, const int batch_size,
                      const int hidden_size, const int num_heads,
                      const cublasHandle_t &blas_handle,
                      const cudaStream_t &stream) {
  data_->training = training;
  data_->batch_size = batch_size;
  data_->hidden_size = hidden_size;
  data_->main_blas_handle = blas_handle;
  data_->stream = stream;
}

ForwardPass::~ForwardPass() {

  cudaStreamSynchronize(data_->stream_R);
  cudaStreamDestroy(data_->stream_R);
  cublasDestroy(data_->blas_handle_R);
  cudaEventDestroy(data_->event_R);
  delete data_;
}

int ForwardPass::Iterate(
    const cudaStream_t &stream,
    const SLSTM_DTYPE_R *R,  // Weight matrix for recurrent state (Ry) [y,H*4]
    const SLSTM_DTYPE_B *b,  // Bias for gates (Wx + Ry + b) [H*4]
    const SLSTM_DTYPE_W *x,  // Input vector [N,C]
    const SLSTM_DTYPE_S *s,  // Recurrent state [N,H]
    SLSTM_DTYPE_S *s_out,    // Output recurrent state [N,H]
    SLSTM_DTYPE_G *g_r,      // Input vector and storage
    SLSTM_DTYPE_G *g_i,      // Input vector and storage
    SLSTM_DTYPE_G *tmp_Ry) { // Temporary storage for Ry vector [N,H*4]
  // Constants for GEMM
  static const SLSTM_DTYPE_G alpha = scalar_one<SLSTM_DTYPE_G>();
  static const SLSTM_DTYPE_G beta = scalar_zero<SLSTM_DTYPE_G>();

  const blas<void>::set_pointer_mode scoped1(data_->main_blas_handle);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->main_blas_handle;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  // Make sure inputs are ready before we use them.

  auto err = cudaPeekAtLastError();

  int res = IterateInternal(x, R, b, s, batch_size * hidden_size, s_out,
                            batch_size * hidden_size, g_r, g_i, tmp_Ry);

  // Make sure outputs have settled.
  if (stream) {
    cudaEventRecord(data_->event_R, data_->stream_R);

    cudaStreamWaitEvent(stream, data_->event_R, 0);
  }

  cublasSetStream(blas_handle, save_stream);

  if (err != cudaSuccess || res) {
    return 1;
  }
  return 0;
}

int ForwardPass::IterateInternal(
    const SLSTM_DTYPE_W *x, const SLSTM_DTYPE_R *R,
    // Weight matrix for recurrent state (Ry) [y,H*4]
    const SLSTM_DTYPE_B *b, // Bias for gates (Wx + Ry + b) [H*4]
    const SLSTM_DTYPE_S *s, // Cell carry max state [S,B,H]
    const uint s_stride,
    SLSTM_DTYPE_S *s_out, // Output recurrent state [S,B,H]
    const uint s_out_stride,
    SLSTM_DTYPE_G *g_r,      // Output vector (Wx + Ry + b) [B,H*G] ?
    SLSTM_DTYPE_G *g_i,      // Output vector (Wx + Ry + b) [B,H*G] ?
    SLSTM_DTYPE_G *tmp_Ry) { // Temporary storage for Ry vector [B,H*G]
  static const SLSTM_DTYPE_G alpha = scalar_one<SLSTM_DTYPE_G>();
  static const SLSTM_DTYPE_G beta = scalar_zero<SLSTM_DTYPE_G>();

  const bool training = data_->training;
  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const int num_heads = data_->num_heads;
  // const cublasHandle_t blas_handle = data_->main_blas_handle;
  const cublasHandle_t blas_handle_R = data_->blas_handle_R;
  const cudaStream_t stream_R = data_->stream_R;

  // const cudaEvent_t event_R = data_->event_R;
  const int head_dim = hidden_size / num_heads;

  cublasSetStream(blas_handle_R, stream_R);
  auto res = blas<SLSTM_DTYPE_R>::gemmsb(
      blas_handle_R, CUBLAS_OP_N, CUBLAS_OP_N, SLSTM_NUM_GATES * head_dim,
      batch_size, head_dim, &alpha, R, SLSTM_NUM_GATES * head_dim,
      SLSTM_NUM_GATES * head_dim * head_dim, s, hidden_size, head_dim, &beta,
      tmp_Ry, SLSTM_NUM_GATES * hidden_size, SLSTM_NUM_GATES * head_dim,
      num_heads);

  if (res != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS ERROR: %d", res);
  }

  // Compute launch configuration for pointwise operations kernel.
  uint gridDimHead =
      _min(_max(32, round_to_power2(head_dim)), MAX_THREADS_PER_BLOCK);
  uint gridDimBatch = _max(1, _min(MAX_THREADS_PER_BLOCK / gridDimHead,
                                   round_to_power2(batch_size)));

  const dim3 blockDim(gridDimHead, gridDimBatch, 1);

  if (training) {
    const dim3 gridDim((head_dim + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y, num_heads);
    SLSTMPointwiseForward<true><<<gridDim, blockDim, 0, stream_R>>>(
        batch_size, hidden_size, num_heads, x, tmp_Ry, b, s, s_stride, s_out,
        s_out_stride, g_r, g_i);
  } else {
    const dim3 gridDim((head_dim + blockDim.x - 1) / blockDim.x,
                       (batch_size + blockDim.y - 1) / blockDim.y, num_heads);
    SLSTMPointwiseForward<false><<<gridDim, blockDim, 0, stream_R>>>(
        batch_size, hidden_size, num_heads, x, tmp_Ry, b, s, s_stride, s_out,
        s_out_stride, nullptr, nullptr);
  }

  auto err = cudaPeekAtLastError();
  if ((err != cudaSuccess) || (res != CUBLAS_STATUS_SUCCESS)) {
    return 1;
  }
  return 0;
}

int ForwardPass::Run(
    const int steps,
    const SLSTM_DTYPE_R *R,  // Weight matrix for recurrent state (Ry) [y,H*4]
    const SLSTM_DTYPE_B *b,  // Bias for gates (Wx + Ry + b) [H*4]
    const SLSTM_DTYPE_W *x,  // Input vector [T,N,C]
    SLSTM_DTYPE_S *s,        // Recurrent state [T+1,N,H]
    SLSTM_DTYPE_G *g_r,      // Output vector (Wx + Ry + b) [T,N,H*4]
    SLSTM_DTYPE_G *g_i,      // Output vector (Wx + Ry + b) [T,N,H*4]
    SLSTM_DTYPE_G *tmp_Ry) { // Temporary storage for Ry vector [N,H*4]

  static const SLSTM_DTYPE_R alpha = scalar_one<SLSTM_DTYPE_R>();
  static const SLSTM_DTYPE_R beta = scalar_zero<SLSTM_DTYPE_R>();

  const blas<void>::set_pointer_mode scoped1(data_->main_blas_handle);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  // const int num_heads = data_->num_heads;
  const cublasHandle_t blas_handle = data_->main_blas_handle;

  const cudaEvent_t event_R = data_->event_R;
  const cudaStream_t stream_R = data_->stream_R;
  bool use_input_stream = false;
  cudaStream_t save_stream;
  int res = 0;
  if (cublasGetStream(blas_handle, &save_stream) == CUBLAS_STATUS_SUCCESS) {
    use_input_stream = true;
  } else {
    use_input_stream = false;
  }

  cudaEventRecord(event_R, data_->stream);
  cudaStreamWaitEvent(stream_R, event_R);

  for (int t = 0; t < steps; ++t) {
    const int BH = batch_size * hidden_size;
    // printf("Iterating \n");
    res |= IterateInternal(
        x + t * BH * SLSTM_NUM_GATES, R, b, s + t * BH,
        (steps + 1) * batch_size * hidden_size, s + (t + 1) * BH,
        (steps + 1) * batch_size * hidden_size, g_r + t * BH * SLSTM_NUM_GATES,
        g_i + t * BH * SLSTM_NUM_GATES, tmp_Ry);
  }
  cudaEventRecord(event_R, stream_R);
  if (use_input_stream) {
    cudaStreamWaitEvent(save_stream, event_R);
  }
  cudaStreamWaitEvent(data_->stream, event_R);
  // cublasSetStream(blas_handle, save_stream);
  return res;
}

} // namespace slstm
