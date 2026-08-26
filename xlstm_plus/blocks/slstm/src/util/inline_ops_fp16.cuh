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

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#if CUDART_VERSION >= 11000
#include <cuda_fp16.h>
// #pragma message(AT " CUDART_VERSION with FP16: " TOSTRING(                     \
//     CUDART_VERSION) ", CUDA_ARCH: " TOSTRING(__CUDA_ARCH__))
#else
// #pragma message(AT " CUDART_VERSION: " TOSTRING(CUDART_VERSION))
#endif

// CONSTANTS
template <> __device__ __forceinline__ __half dscalar_three() {
  return __float2half(3.0f);
}

template <> __device__ __forceinline__ __half dscalar_two() {
  return __float2half(2.0f);
}

template <> __device__ __forceinline__ __half dscalar_one() {
  return __float2half(1.0f);
}

template <> __device__ __forceinline__ __half dscalar_half() {
  return __float2half(0.5f);
}

template <> __device__ __forceinline__ __half dscalar_zero() {
  return __float2half(0.0f);
}

template <> __forceinline__ __half scalar_one() { return __float2half(1.0f); }

template <> __forceinline__ __half scalar_zero() { return __float2half(0.0f); }

template <> __device__ __forceinline__ __half dscalar(double x) {
  return __float2half((float)x);
}

// -- CONSTANTS

// ARITHMETIC FUNCTIONS
// ADD
template <>
__device__ __forceinline__ __half add_g(const __half a, const __half b) {
  return __hadd_rn(a, b);
}

// SUB
template <>
__device__ __forceinline__ __half sub_g(const __half a, const __half b) {
  return __hsub_rn(a, b);
}

// NEG
template <> __device__ __forceinline__ __half neg_g(const __half a) {
  return __hneg(a);
}

// MUL
template <>
__device__ __forceinline__ __half mul_g(const __half a, const __half b) {
  return __hmul_rn(a, b);
}

// DIV
template <>
__device__ __forceinline__ __half div_g(const __half a, const __half b) {
  return __hdiv(a, b);
}

// -- ARITHMETIC FUNCTIONS

// COMPARISON OPERATIONS
template <>
__device__ __forceinline__ bool gt_g(const __half a, const __half b) {
  return __hgt(a, b);
}

template <>
__device__ __forceinline__ bool lt_g(const __half a, const __half b) {
  return __hgt(b, a);
}

template <> __device__ __forceinline__ bool gt_zero_g(const __half a) {
  return __hgt(a, __float2half(0.0f));
}

template <> __device__ __forceinline__ bool eq_zero_g(const __half a) {
  return __heq(a, __float2half(0.0f));
}

template <> __device__ __forceinline__ bool lt_zero_g(const __half a) {
  return __hgt(__float2half(0.0f), a);
}

// -- COMPARISON OPERATIONS

// Other functions
template <> __device__ __forceinline__ __half exp_g(const __half x) {
  return hexp(x);
}

template <> __device__ __forceinline__ __half log_g(const __half x) {
  return hlog(x);
}

template <> __device__ __forceinline__ __half tanh_g(const __half x) {
  __half zero = dscalar_zero<__half>();
  __half one = dscalar_one<__half>();
  __half two = dscalar_two<__half>();
  __half e2x;
  __half negx = x;
  if (gt_g(x, zero)) {
    negx = __hneg(x);
  }
  e2x = hexp(__hmul(two, negx));
  e2x = __hdiv(__hsub(one, e2x), __hadd(one, e2x));
  if (gt_g(x, zero)) {
    return e2x;
  } else {
    return __hneg(e2x);
  }
}

template <> __device__ __forceinline__ __half sigmoid_g(const __half x) {
  __half one = dscalar_one<__half>();
  __half expx;
  __half negx = x;
  if (gt_zero_g(x)) {
    negx = __hneg(x);
  }
  expx = __hdiv(one, __hadd(one, hexp(negx)));
  if (gt_zero_g(x)) {
    return expx;
  } else {
    return sub_g(one, expx);
  }
}

template <> __device__ __forceinline__ __half logsigmoid_g(const __half x) {
  __half one = dscalar_one<__half>();
  __half negx = x;
  if (gt_zero_g(x)) {
    negx = __hneg(x);
  }
  __half logaddexpnx = hlog(__hadd(one, hexp(negx)));
  if (gt_zero_g(x)) {
    return __hneg(logaddexpnx);
  } else {
    return __hsub(x, logaddexpnx);
  }
}

template <>
__device__ __forceinline__ __half sigmoid_unstable_g(const __half x) {
  __half one = dscalar_one<__half>();
  return __hdiv(one, __hadd(one, hexp(__hneg(x))));
}

template <>
__device__ __forceinline__ __half d_sigmoid_g(const __half sigmoid_output) {
  return __hmul(sigmoid_output, __hsub(dscalar_one<__half>(), sigmoid_output));
}

template <>
__device__ __forceinline__ __half d_tanh_g(const __half tanh_output) {
  return __hsub(dscalar_one<__half>(), __hmul(tanh_output, tanh_output));
}

template <>
__device__ __forceinline__ __half max_g(const __half a, const __half b) {
  return __hmax(a, b);
}

template <>
__device__ __forceinline__ __half min_g(const __half a, const __half b) {
  return __hmin(a, b);
}

template <> __device__ __forceinline__ float type2float(const __half x) {
  return __half2float(x);
}

template <> __device__ __forceinline__ __half float2type(const float x) {
  return __float2half(x);
}

template <> __device__ __forceinline__ bool isnan_g(const __half x) {
  return __hisnan(x);
}

template <> __device__ __forceinline__ bool isinf_g(const __half x) {
  return __hisinf(x);
}