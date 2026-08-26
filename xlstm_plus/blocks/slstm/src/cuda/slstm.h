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

#pragma once

#include "../util/util.h"

#ifndef SLSTM_NUM_GATES
// all needed definitions from external
#define SLSTM_NUM_HEADS 1
#define SLSTM_HIDDEN_SIZE 512
#define SLSTM_BATCH_SIZE 8
#define SLSTM_NUM_GATES 4
#define SLSTM_NUM_STATES 4
#define SLSTM_DTYPE_R float
#define SLSTM_DTYPE_B float
#define SLSTM_DTYPE_W float
#define SLSTM_DTYPE_G float
#define SLSTM_DTYPE_S float
#define SLSTM_DTYPE_A float
// defines whether g = Wx + Ry + b for every gate, enables half the cache for
// backward
#define SLSTM_SIMPLE_AGG true
#endif

namespace slstm {

class ForwardPass {
public:
  // training: `true` if the caller intends to perform a backward pass to
  // compute gradients. batch_size: the number of training/inference inputs
  // provided in each tensor. input_size: the dimension of each input vector.
  // hidden_size: the expected dimension of each output vector.
  // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
  ForwardPass(const bool training, const int batch_size, const int hidden_size,
              const int num_heads, const cublasHandle_t &blas_handle,
              const cudaStream_t &stream = 0);

  // Releases internal resources.
  // Blocks until all iterations have completed executing on the GPU.
  ~ForwardPass();

  // Set internal values for single forward / backward
  void Set(const bool training, const int batch_size, const int hidden_size,
           const int num_heads, const cublasHandle_t &blas_handle,
           const cudaStream_t &stream = 0);

  // Performs one forward iteration of the LSTM cell.
  //
  // W: [C,H*4] the input weight matrix.
  // R: [H,H*4] the recurrent weight matrix.
  // b: [H*4] the bias vector.
  // x: [N,C] the XLSTMsig input for this iteration (N vectors, each with
  // dimension C). y: [N,H] the t-1 iteration's `y_out` or the initial hidden
  // state if this is the
  //     t=0 iteration (typically zeros).
  // c: [N,H] the t-1 iteration's `c_out` or the initial cell state if this is
  // the
  //     t=0 iteration (typically zeros).
  // y_out: [N,H] the XLSTMsig's output, and the input to the next iteration's
  // `h`. This
  //     pointer may be the same as `h`. Each iteration may reuse the same
  //     memory region.
  // c_out: [N,H] the XLSTMsig's internal cell state after this iteration is
  // complete. This
  //     will become the input to the next iteration's `c`.
  // g: [N,H*4] if `training` is `false`, this is scratch space and should not
  // be used by
  //     the caller. If `training` is `true`, this vector will contain
  //     intermediate activations for this iteration which must be provided
  //     as-is to the corresponding backward iteration. In either case, a new
  //     memory region must be provided for each iteration.
  // tmp_Ry: [N,H*4] additional temporary work space required for this
  // iteration. The caller
  //     should not use the contents of this vector. The same memory region may
  //     be provided for each iteration.
  int Iterate(const cudaStream_t &stream, const SLSTM_DTYPE_R *R,
              const SLSTM_DTYPE_B *b, const SLSTM_DTYPE_W *x,
              const SLSTM_DTYPE_S *s, SLSTM_DTYPE_S *s_out, SLSTM_DTYPE_G *g_r,
              SLSTM_DTYPE_G *g_i, SLSTM_DTYPE_G *tmp_Ry);

  // Runs the LSTM over all time steps. This method is faster than using a
  // per-step `Iterate` but requires that the entire input sequence be available
  // upfront. In some situations, this constraint may not be satisfiable (e.g.
  // autoregressive models). Users should prefer calling `Run` over `Iterate`
  // whenever possible.
  //
  // steps: the number of iterations to run (i.e. T).
  // W: [H,H*4] the input weight matrix.
  // R: [H,H*4] the recurrent weight matrix.
  // b: [H*4] the bias vector.
  // x: [T,N,C] the LSTM input for this iteration (N vectors, each with
  // dimension C). y: [T+1,N,H] the hidden state vectors across all time steps.
  // The t=0'th vector should
  //      be set to the desired initial hidden state (typically zeros). The rest
  //      of the vectors will be set by this function. `h[1:,:,:]` forms the
  //      output of this LSTM layer.
  // c: [T+1,N,H] the cell state vectors across all time steps. The t=0'th
  // vector should be
  //      set to the desired initial cell state (typically zeros). The rest of
  //      the vectors will be set by this function.
  // g: [T,N,H*4] if `training` is `false`, this is scratch space and should not
  // be used by
  //     the caller. If `training` is `true`, this parameter will contain
  //     intermediate activations which must be provided as-is to
  //     `BackwardPass::Run` or manually urolled for `BackwardPass::Iterate`.
  // tmp_Ry: [N,H*4] additional temporary work space required for this
  // iteration. The caller
  //     should not use the contents of this vector. The same memory region may
  //     be provided for each iteration.
  int Run(const int steps, const SLSTM_DTYPE_R *R, const SLSTM_DTYPE_B *b,
          const SLSTM_DTYPE_W *x, SLSTM_DTYPE_S *s, SLSTM_DTYPE_G *g_r,
          SLSTM_DTYPE_G *g_i, SLSTM_DTYPE_G *tmp_Ry);

private:
  int IterateInternal(const SLSTM_DTYPE_W *x, const SLSTM_DTYPE_R *R,
                      const SLSTM_DTYPE_B *b, const SLSTM_DTYPE_S *s,
                      const uint s_stride, SLSTM_DTYPE_S *s_out,
                      const uint s_out_stride, SLSTM_DTYPE_G *g_r,
                      SLSTM_DTYPE_G *g_i, SLSTM_DTYPE_G *tmp_Ry);

  struct private_data;
  private_data *data_;
};

class BackwardPass {
public:
  // batch_size: the number of training inputs provided in each tensor.
  // input_size: the dimension of each input vector.
  // hidden_size: the expected dimension of each output vector.
  // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
  BackwardPass(const int batch_size, const int hidden_size, const int num_heads,
               const cublasHandle_t &blas_handle,
               const cudaStream_t &stream = 0);

  // Set internal values for single forward / backward
  void Set(const int batch_size, const int hidden_size, const int num_heads,
           const cublasHandle_t &blas_handle, const cudaStream_t &stream = 0);

  // Releases internal resources.
  // Blocks until all iterations have completed executing on the GPU.
  ~BackwardPass();

  // Performs one backward iteration of the LSTM cell.
  //
  // Note that BackwardPass must be iterated in the reverse order as
  // ForwardPass. If ForwardPass iterates from 0 to T-1, BackwardPass needs to
  // iterate from T-1 down to 0. When iteration numbers are described, they will
  // be based on the iteration index (i.e., the T-1'th iteration of the forward
  // pass is the last call to ForwardPass::Iterate, whereas it is the first call
  // to BackwardPass::Iterate).
  //
  // W_t: [H*4,C] the transpose of the input weight matrix.
  // R_t: [H*4,H] the transpose of the recurrent weight matrix.
  // b: [H*4] the bias vector.
  // x_t: [C,N] the transpose of the LSTM input for this iteration.
  // y: [N,H] the hidden state of the t'th iteration or the initial hidden state
  // if this is
  //     the t=0 iteration (typically zeros).
  // c: [N,H] the t-1'th forward iteration's `c_out` or the initial cell state
  // if this is
  //     the t=0 iteration (typically zeros).
  // c_new: [N,H] the t'th forward iteration's `c_out` vector.
  // dh_new: [N,H] the gradient of the loss with respect to `y_out` at this
  // iteration. dc_new: [N,H] the gradient of the loss with respect to `c_out`
  // at this iteration. dx: [N,C] the gradient of the loss with respect to the
  // input at this time step. dW: [C,H*4] the gradient of the loss with respect
  // to the input weight matrix. dR: [H,H*4] the gradient of the loss with
  // respect to the recurrent weight matrix. db: [H*4] the gradient of the loss
  // with respect to the bias vector. dy: [N,H] NOTE: this is an input and
  // output parameter. Should be initialized to zeros
  //     for the T-1'th iteration and the same pointer should be passed in for
  //     each iteration. After a complete backward pass, this vector will
  //     contain the gradient of the loss with respect to the initial hidden
  //     state.
  // dc: [N,H] NOTE: this is an input and output parameter. Should be
  // initialized to zeros
  //     for the T-1'th iteration and the same pointer should be passed in for
  //     each iteration. After a complete backward pass, this vector will
  //     contain the gradient of the loss with respect to the initial cell
  //     state.
  // g: [N,H*4] the same tensor that was passed to `ForwardPass::Iterate` on its
  // corresponding
  //     iteration.
  int Iterate(const cudaStream_t &stream, const SLSTM_DTYPE_R *R_t,
              const SLSTM_DTYPE_B *b, const SLSTM_DTYPE_S *s,
              const SLSTM_DTYPE_S *s_new, const SLSTM_DTYPE_S *ds_new,
              SLSTM_DTYPE_R *dR, SLSTM_DTYPE_B *db, SLSTM_DTYPE_S *ds,
              SLSTM_DTYPE_G *g_r, SLSTM_DTYPE_G *g_i, SLSTM_DTYPE_G *g_b);

  // Runs the LSTM backward pass over all time steps. This method is faster than
  // using a per-step `Iterate` but requires that the entire input sequence be
  // available upfront. In some situations, this constraint may not be
  // satisfiable (e.g. autoregressive models). Users should prefer calling `Run`
  // over `Iterate` whenever possible.
  //
  // steps: the number of iterations to run (i.e. T).
  // W_t: [H*4,C] the transpose of the input weight matrix.
  // R_t: [H*4,H] the transpose of the recurrent weight matrix.
  // b: [H*4] the bias vector.
  // x_t: [C,T,N] the transpose of the LSTM input for this iteration.
  // y: [T+1,N,H] the hidden state vectors after running `ForwardPass::Run`.
  // c: [T+1,N,H] the cell state vectors after running `ForwardPass::Run`.
  // dh_new: [T+1,N,H] the gradient of the loss with respect to `h`.
  // dc_new: [T+1,N,H] the gradient of the loss with respect to `c`.
  // dx: [T,N,C] the gradient of the loss with respect to the input.
  // dW: [C,H*4] the gradient of the loss with respect to the input weight
  // matrix. dR: [H,H*4] the gradient of the loss with respect to the recurrent
  // weight matrix. db: [H*4] the gradient of the loss with respect to the bias
  // vector. dy: [N,H] NOTE: this is an input and output parameter. Should be
  // initialized to zeros.
  //     When this function returns, `dh` will contain the gradient of the loss
  //     with respect to the initial hidden state.
  // dc: [N,H] NOTE: this is an input and output parameter. Should be
  // initialized to zeros.
  //     When this function returns, `dc` will contain the gradient of the loss
  //     with respect to the initial cell state.
  // g: [T,N,H*4] the same tensor that was passed to `ForwardPass::Run`.
  int Run(const int steps, const SLSTM_DTYPE_R *R_t, const SLSTM_DTYPE_B *b,
          const SLSTM_DTYPE_S *s, const SLSTM_DTYPE_S *ds_new,
          SLSTM_DTYPE_R *dR, SLSTM_DTYPE_B *db, SLSTM_DTYPE_S *ds,
          SLSTM_DTYPE_G *g_r, SLSTM_DTYPE_G *g_i, SLSTM_DTYPE_G *g_b);

private:
  int IterateInternal(const SLSTM_DTYPE_R *R_t, const SLSTM_DTYPE_B *b,
                      const SLSTM_DTYPE_S *s, const uint s_stride,
                      const SLSTM_DTYPE_S *s_new, const uint s_new_stride,
                      const SLSTM_DTYPE_S *ds_new, const uint ds_new_stride,
                      SLSTM_DTYPE_S *ds, const uint ds_stride,
                      SLSTM_DTYPE_G *g_r, SLSTM_DTYPE_G *g_i,
                      SLSTM_DTYPE_G *g_b);
  struct private_data;
  private_data *data_;
};

class BackwardPassCut {
public:
  // batch_size: the number of training inputs provided in each tensor.
  // input_size: the dimension of each input vector.
  // hidden_size: the expected dimension of each output vector.
  // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
  BackwardPassCut(const int batch_size, const int hidden_size,
                  const int num_heads, const cublasHandle_t &blas_handle,
                  const cudaStream_t &stream = 0);

  // Set internal values for single forward / backward
  void Set(const int batch_size, const int hidden_size, const int num_heads,
           const cublasHandle_t &blas_handle, const cudaStream_t &stream = 0);

  // Releases internal resources.
  // Blocks until all iterations have completed executing on the GPU.
  ~BackwardPassCut();

  // Performs one backward iteration of the LSTM cell.
  //
  // Note that BackwardPass must be iterated in the reverse order as
  // ForwardPass. If ForwardPass iterates from 0 to T-1, BackwardPass needs to
  // iterate from T-1 down to 0. When iteration numbers are described, they will
  // be based on the iteration index (i.e., the T-1'th iteration of the forward
  // pass is the last call to ForwardPass::Iterate, whereas it is the first call
  // to BackwardPass::Iterate).
  //
  // W_t: [H*4,C] the transpose of the input weight matrix.
  // R_t: [H*4,H] the transpose of the recurrent weight matrix.
  // b: [H*4] the bias vector.
  // x_t: [C,N] the transpose of the LSTM input for this iteration.
  // y: [N,H] the hidden state of the t'th iteration or the initial hidden state
  // if this is
  //     the t=0 iteration (typically zeros).
  // c: [N,H] the t-1'th forward iteration's `c_out` or the initial cell state
  // if this is
  //     the t=0 iteration (typically zeros).
  // c_new: [N,H] the t'th forward iteration's `c_out` vector.
  // dh_new: [N,H] the gradient of the loss with respect to `y_out` at this
  // iteration. dc_new: [N,H] the gradient of the loss with respect to `c_out`
  // at this iteration. dx: [N,C] the gradient of the loss with respect to the
  // input at this time step. dW: [C,H*4] the gradient of the loss with respect
  // to the input weight matrix. dR: [H,H*4] the gradient of the loss with
  // respect to the recurrent weight matrix. db: [H*4] the gradient of the loss
  // with respect to the bias vector. dy: [N,H] NOTE: this is an input and
  // output parameter. Should be initialized to zeros
  //     for the T-1'th iteration and the same pointer should be passed in for
  //     each iteration. After a complete backward pass, this vector will
  //     contain the gradient of the loss with respect to the initial hidden
  //     state.
  // dc: [N,H] NOTE: this is an input and output parameter. Should be
  // initialized to zeros
  //     for the T-1'th iteration and the same pointer should be passed in for
  //     each iteration. After a complete backward pass, this vector will
  //     contain the gradient of the loss with respect to the initial cell
  //     state.
  // g: [N,H*4] the same tensor that was passed to `ForwardPass::Iterate` on its
  // corresponding
  //     iteration.
  int Iterate(const cudaStream_t &stream, const SLSTM_DTYPE_R *R_t,
              const SLSTM_DTYPE_B *b, const SLSTM_DTYPE_S *s,
              const SLSTM_DTYPE_S *s_new, const SLSTM_DTYPE_S *ds_new,
              SLSTM_DTYPE_R *dR, SLSTM_DTYPE_B *db, SLSTM_DTYPE_S *ds,
              SLSTM_DTYPE_G *g_r, SLSTM_DTYPE_G *g_i, SLSTM_DTYPE_G *g_bias);

  // Runs the LSTM backward pass over all time steps. This method is faster than
  // using a per-step `Iterate` but requires that the entire input sequence be
  // available upfront. In some situations, this constraint may not be
  // satisfiable (e.g. autoregressive models). Users should prefer calling `Run`
  // over `Iterate` whenever possible.
  //
  // steps: the number of iterations to run (i.e. T).
  // W_t: [H*4,C] the transpose of the input weight matrix.
  // R_t: [H*4,H] the transpose of the recurrent weight matrix.
  // b: [H*4] the bias vector.
  // x_t: [C,T,N] the transpose of the LSTM input for this iteration.
  // y: [T+1,N,H] the hidden state vectors after running `ForwardPass::Run`.
  // c: [T+1,N,H] the cell state vectors after running `ForwardPass::Run`.
  // dh_new: [T+1,N,H] the gradient of the loss with respect to `h`.
  // dc_new: [T+1,N,H] the gradient of the loss with respect to `c`.
  // dx: [T,N,C] the gradient of the loss with respect to the input.
  // dW: [C,H*4] the gradient of the loss with respect to the input weight
  // matrix. dR: [H,H*4] the gradient of the loss with respect to the recurrent
  // weight matrix. db: [H*4] the gradient of the loss with respect to the bias
  // vector. dy: [N,H] NOTE: this is an input and output parameter. Should be
  // initialized to zeros.
  //     When this function returns, `dh` will contain the gradient of the loss
  //     with respect to the initial hidden state.
  // dc: [N,H] NOTE: this is an input and output parameter. Should be
  // initialized to zeros.
  //     When this function returns, `dc` will contain the gradient of the loss
  //     with respect to the initial cell state.
  // g: [T,N,H*4] the same tensor that was passed to `ForwardPass::Run`.
  int Run(const int steps, const SLSTM_DTYPE_R *R_t, const SLSTM_DTYPE_B *b,
          const SLSTM_DTYPE_S *s, const SLSTM_DTYPE_S *ds_new,
          SLSTM_DTYPE_R *dR, SLSTM_DTYPE_B *db, SLSTM_DTYPE_S *ds,
          SLSTM_DTYPE_G *g_r, SLSTM_DTYPE_G *g_i, SLSTM_DTYPE_G *g_bias);

private:
  int IterateInternal(const SLSTM_DTYPE_R *R_t, const SLSTM_DTYPE_B *b,
                      const SLSTM_DTYPE_S *s, const uint s_stride,
                      const SLSTM_DTYPE_S *s_new, const uint s_new_stride,
                      const SLSTM_DTYPE_S *ds_new, const uint ds_new_stride,
                      SLSTM_DTYPE_S *ds, const uint ds_stride,
                      SLSTM_DTYPE_G *g_r, SLSTM_DTYPE_G *g_i,
                      SLSTM_DTYPE_G *g_bias);
  struct private_data;
  private_data *data_;
};

} // namespace slstm
