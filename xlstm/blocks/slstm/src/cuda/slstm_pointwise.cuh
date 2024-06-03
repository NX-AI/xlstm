// Copyright (c) NXAI GmbH and its affiliates 2023
// Korbinian Poeppel

#pragma once

#include "slstm.h"

namespace slstm {

template <bool>
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
    SLSTM_DTYPE_G *g_i_out);

__global__ void SLSTMPointwiseBackward(
    const int batch_dim, const int hidden_dim, const int num_heads,
    const SLSTM_DTYPE_S *s, const uint s_stride, const SLSTM_DTYPE_G *g_r,
    const SLSTM_DTYPE_G *g_i, const SLSTM_DTYPE_B *bias,
    const SLSTM_DTYPE_S *s_new, const uint s_new_stride,
    const SLSTM_DTYPE_S *ds_new, const uint ds_new_stride,
    SLSTM_DTYPE_S *ds_inout, const uint ds_inout_stride,
    SLSTM_DTYPE_G *dg_r_out, SLSTM_DTYPE_G *dg_i_out, SLSTM_DTYPE_G *dg_b_out);

} // namespace slstm

#define SLSTM_POST_DEFINITIONS                                                 \
  template __global__ void SLSTMPointwiseForward<true>(                        \
      const int batch_dim, const int hidden_dim, const int num_heads,          \
      const SLSTM_DTYPE_G *Wx, const SLSTM_DTYPE_G *Ry,                        \
      const SLSTM_DTYPE_B *b, const SLSTM_DTYPE_S *s, const uint s_stride,     \
      SLSTM_DTYPE_S *s_out, const uint s_out_stride, SLSTM_DTYPE_G *g_r_out,   \
      SLSTM_DTYPE_G *g_i_out);                                                 \
  template __global__ void SLSTMPointwiseForward<false>(                       \
      const int batch_dim, const int hidden_dim, const int num_heads,          \
      const SLSTM_DTYPE_G *Wx, const SLSTM_DTYPE_G *Ry,                        \
      const SLSTM_DTYPE_B *b, const SLSTM_DTYPE_S *s, const uint s_stride,     \
      SLSTM_DTYPE_S *s_out, const uint s_out_stride, SLSTM_DTYPE_G *g_r_out,   \
      SLSTM_DTYPE_G *g_i_out);                                                 \
  __global__ void SLSTMPointwiseBackward(                                      \
      const int batch_dim, const int hidden_dim, const int num_heads,          \
      const SLSTM_DTYPE_S *s, const uint s_stride, const SLSTM_DTYPE_G *g_r,   \
      const SLSTM_DTYPE_G *g_i, const SLSTM_DTYPE_B *b,                        \
      const SLSTM_DTYPE_S *s_new, const uint s_new_stride,                     \
      const SLSTM_DTYPE_S *ds_new, const uint ds_new_stride,                   \
      SLSTM_DTYPE_S *ds_inout, const uint ds_inout_stride,                     \
      SLSTM_DTYPE_G *dg_r_out, SLSTM_DTYPE_G *dg_i_out,                        \
      SLSTM_DTYPE_G *dg_b_out);
