// Copyright (c) NXAI GmbH and its affiliates 2023
// Korbinian Poeppel

#include "../util/cuda_error.h"
#include "../util/inline_ops.cuh"
#include <cublas_v2.h>

#include "slstm.h"
#include "slstm_pointwise.cuh"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#ifndef SLSTM_NUM_GATES
#define SLSTM_NUM_GATES 4
#define SLSTM_NUM_STATES 2
#define SLSTM_GRADIENT_RECURRENT_CLIPVAL 0.
#define SLSTM_GRADIENT_RECURRENT_CLIPVAL_VALID false
#endif

static_assert(SLSTM_NUM_GATES == 4, "Gates must be 4");

namespace slstm {

template <bool Training>
__global__ void SLSTMPointwiseForward(
    const int batch_dim, const int hidden_dim, const int num_heads,
    const SLSTM_DTYPE_G *Wx, // Precomputed (Wx) vector
    const SLSTM_DTYPE_G *Ry, // Precomputed (Ry) vector
    const SLSTM_DTYPE_B *b,  // Bias for gates
    const SLSTM_DTYPE_S *s,  // Input  state
    const uint s_stride,
    SLSTM_DTYPE_S *s_out, // Output recurrent state
    const uint s_out_stride,
    SLSTM_DTYPE_G *g_r_out, // Output vector v (Wx + Ry + b) (only
                            // used if autoraining==true)
    SLSTM_DTYPE_G *g_i_out) {

  // We're in column-major order here, so increase x => increase row.
  const int row = blockDim.x * blockIdx.x + threadIdx.x; // hidden
  const int col = blockDim.y * blockIdx.y + threadIdx.y; // batch
  const int head_dim = hidden_dim / num_heads;
  const int head_idx = (blockDim.z * blockIdx.z + threadIdx.z) * head_dim;

  if (row >= head_dim || col >= batch_dim)
    return;

  // Base index into the Wx and Ry matrices.
  const int weight_idx =
      col * (hidden_dim * SLSTM_NUM_GATES) + row + SLSTM_NUM_GATES * head_idx;

  // Base index into the output matrix. autohis is different from `weight_idx`
  // because the number of rows are different between the two sets of matrices.
  const int output_idx = col * hidden_dim + row + head_idx;

  const int i_idx = weight_idx + 0. * head_dim;
  const int f_idx = weight_idx + 1. * head_dim;
  const int z_idx = weight_idx + 2. * head_dim;
  const int o_idx = weight_idx + 3. * head_dim;

  // #ifdef DEBUG
  //   if (i_idx == 0) {
  //     printf("Ry: %f, Wx: %f, b: %f\n", bfloat162float(Ry[i_idx]),
  //            bfloat162float(Ry[i_idx]), bfloat162float(Ry[i_idx]));
  //   }
  // #endif

  const auto c_cur = type2float(s[output_idx + 1 * s_stride]);
  const auto iraw = add_g(
      type2float(Wx[i_idx]),
      add_g(type2float(Ry[i_idx]),
            type2float(b[row + SLSTM_NUM_GATES * head_idx + 0 * head_dim])));
  const auto fraw = add_g(
      type2float(Wx[f_idx]),
      add_g(type2float(Ry[f_idx]),
            type2float(b[row + SLSTM_NUM_GATES * head_idx + 1 * head_dim])));
  const auto zraw = add_g(
      type2float(Wx[z_idx]),
      add_g(type2float(Ry[z_idx]),
            type2float(b[row + SLSTM_NUM_GATES * head_idx + 2 * head_dim])));
  const auto zval = tanh_g(zraw);
  const float one = 1.;
  const auto ogate = sigmoid_g(add_g(
      type2float(Wx[o_idx]),
      add_g(type2float(Ry[o_idx]),
            type2float(b[row + SLSTM_NUM_GATES * head_idx + 3 * head_dim]))));

  const auto igate = sigmoid_g(iraw);
  const auto fgate = sigmoid_g(fraw);
  // Compile-time constant branch should be eliminated by compiler so we have
  // straight-through code.
  if (Training) {
    g_r_out[i_idx] = float2type<SLSTM_DTYPE_G>(igate);
    g_r_out[f_idx] = float2type<SLSTM_DTYPE_G>(fgate);
    g_r_out[z_idx] = float2type<SLSTM_DTYPE_G>(zval);
    g_r_out[o_idx] = float2type<SLSTM_DTYPE_G>(ogate);
  }

  const auto c_new = add_g(mul_g(fgate, c_cur), mul_g(igate, zval));
  auto y_new = mul_g(ogate, tanh_g(c_new));

#if SLSTM_FORWARD_CLIPVAL_VALID
  y_new = clip_val_g(y_new, neg_g((float)SLSTM_FORWARD_CLIPVAL),
                     (float)SLSTM_FORWARD_CLIPVAL);
#endif

  s_out[output_idx + 0 * s_out_stride] = float2type<SLSTM_DTYPE_S>(y_new);
  s_out[output_idx + 1 * s_out_stride] = float2type<SLSTM_DTYPE_S>(c_new);
}

__global__ void SLSTMPointwiseBackward(
    const int batch_dim, const int hidden_dim, const int num_heads,
    const SLSTM_DTYPE_S *s, const uint s_stride, const SLSTM_DTYPE_G *g_r,
    const SLSTM_DTYPE_G *g_i,
    const SLSTM_DTYPE_B *b, // Bias for gates
    const SLSTM_DTYPE_S *s_new, const uint s_new_stride,
    const SLSTM_DTYPE_S *ds_new, const uint ds_new_stride,
    SLSTM_DTYPE_S *ds_inout, const uint ds_inout_stride,
    SLSTM_DTYPE_G *dg_r_out, SLSTM_DTYPE_G *dg_i_out, SLSTM_DTYPE_G *dg_b_out) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x; // hidden
  const int col = blockDim.y * blockIdx.y + threadIdx.y; // batch
  const int head_dim = hidden_dim / num_heads;
  const int head_idx = (blockDim.z * blockIdx.z + threadIdx.z) * head_dim;

  if (row >= head_dim || col >= batch_dim)
    return;

  const int base_idx = col * hidden_dim + row + head_idx;
  auto dy_recurrent = type2float(ds_inout[base_idx + 0 * ds_inout_stride]);

#if (SLSTM_GRADIENT_RECURRENT_CLIPVAL_VALID)
  dy_recurrent =
      clip_val_g(dy_recurrent, neg_g((float)SLSTM_GRADIENT_RECURRENT_CLIPVAL),
                 (float)SLSTM_GRADIENT_RECURRENT_CLIPVAL);
#endif
  const auto dy_total =
      add_g(type2float(ds_new[base_idx + 0 * ds_new_stride]), dy_recurrent);
  auto dc_total = add_g(type2float(ds_new[base_idx + 1 * ds_new_stride]),
                        type2float(ds_inout[base_idx + 1 * ds_inout_stride]));

  const int stride4_base_idx =
      col * (hidden_dim * SLSTM_NUM_GATES) + row + SLSTM_NUM_GATES * head_idx;
  const int i_idx = stride4_base_idx + 0 * head_dim;
  const int f_idx = stride4_base_idx + 1 * head_dim;
  const int z_idx = stride4_base_idx + 2 * head_dim;
  const int o_idx = stride4_base_idx + 3 * head_dim;

  const auto igate = type2float(g_r[i_idx]);
  const auto fgate = type2float(g_r[f_idx]);
  const auto zval = type2float(g_r[z_idx]);
  const auto ogate = type2float(g_r[o_idx]);
  const auto c_cur = type2float(s[base_idx + 1 * s_stride]);
  const auto c_new = type2float(s_new[base_idx + 1 * s_stride]);
  const float zero = 0.;
  const float one = 1.;
  const auto y_new = type2float(s_new[base_idx + 0 * s_new_stride]);
  const auto c_new_tanh = tanh_g(c_new);

  const auto dc_tanh = mul_g(ogate, dy_total);
  dc_total = add_g(dc_total, mul_g(d_tanh_g(c_new_tanh), dc_tanh));

  const auto di = mul_g(zval, dc_total);
  const auto df = mul_g(c_cur, dc_total);
  const auto dz = mul_g(igate, dc_total);
  const auto do_ = mul_g(c_new_tanh, dy_total);
  const auto dc_i = mul_g(fgate, dc_total);

  const auto dg_i = mul_g(d_sigmoid_g(igate), di);
  const auto dg_f = mul_g(d_sigmoid_g(fgate), df);
  const auto dg_z = mul_g(d_tanh_g(zval), dz);
  const auto dg_o = mul_g(d_sigmoid_g(ogate), do_);

  ds_inout[base_idx + 0 * ds_inout_stride] = float2type<SLSTM_DTYPE_S>(zero);
  ds_inout[base_idx + 1 * ds_inout_stride] = float2type<SLSTM_DTYPE_S>(dc_i);

  dg_r_out[i_idx] = float2type<SLSTM_DTYPE_G>(dg_i);
  dg_r_out[f_idx] = float2type<SLSTM_DTYPE_G>(dg_f);
  dg_r_out[z_idx] = float2type<SLSTM_DTYPE_G>(dg_z);
  dg_r_out[o_idx] = float2type<SLSTM_DTYPE_G>(dg_o);
}

SLSTM_POST_DEFINITIONS

} // namespace slstm
