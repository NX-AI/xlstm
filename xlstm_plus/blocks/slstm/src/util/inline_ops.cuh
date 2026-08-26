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
#define ROUND_UP_TO_MULTIPLE(x, y) (((x) + (y)-1) / (y)) * (y)

#define AT __FILE__ ":" TOSTRING(__LINE__)

// CONSTANTS
template <typename T> __device__ __forceinline__ T dscalar_three() {
  return static_cast<T>(3.0);
}

template <typename T> __device__ __forceinline__ T dscalar_two() {
  return static_cast<T>(2.0);
}

template <typename T> __device__ __forceinline__ T dscalar_one() {
  return static_cast<T>(1.0);
}

template <typename T> __device__ __forceinline__ T dscalar_half() {
  return static_cast<T>(0.5);
}

template <typename T> __device__ __forceinline__ T dscalar_zero() {
  return static_cast<T>(0.0);
}

template <typename T> __forceinline__ T scalar_one() {
  return static_cast<T>(1.0);
}

template <typename T> __forceinline__ T scalar_zero() {
  return static_cast<T>(0.0);
}

template <typename T> __device__ __forceinline__ T dscalar(double x) {
  return (T)x;
}

// -- CONSTANTS

// ARITHMETIC FUNCTIONS
// ADD
template <typename T> __device__ __forceinline__ T add_g(const T a, const T b) {
  return a + b;
}

// SUB
template <typename T> __device__ __forceinline__ T sub_g(const T a, const T b) {
  return a - b;
}

// NEG
template <typename T> __device__ __forceinline__ T neg_g(const T a) {
  return -a;
}

// MUL
template <typename T> __device__ __forceinline__ T mul_g(const T a, const T b) {
  return a * b;
}

// DIV
template <typename T> __device__ __forceinline__ T div_g(const T a, const T b) {
  return a / b;
}

// -- ARITHMETIC FUNCTIONS

// COMPARISON OPERATIONS

template <typename T>
__device__ __forceinline__ bool gt_g(const T a, const T b) {
  return a > b;
}

template <typename T> __device__ __forceinline__ bool gt_zero_g(const T a) {
  return a > 0.0;
}

template <typename T> __device__ __forceinline__ bool eq_zero_g(const T a) {
  return a == 0.0;
}

template <typename T>
__device__ __forceinline__ bool lt_g(const T a, const T b) {
  return a < b;
}

template <typename T> __device__ __forceinline__ bool lt_zero_g(const T a) {
  return a < 0.0;
}

// -- COMPARISON OPERATIONS

// Other functions
template <typename T> __device__ __forceinline__ T exp_g(const T x) {
  return exp(x);
}

template <> __device__ __forceinline__ float exp_g(const float x) {
  return expf(x);
}

template <typename T> __device__ __forceinline__ T tanh_g(const T x) {
  T negx = x;
  T expnx;
  T one = dscalar_one<T>();
  if (gt_zero_g(x)) {
    negx = neg_g(x);
  }
  expnx = exp_g(mul_g(dscalar_two<T>(), negx));
  expnx = div_g(sub_g(one, expnx), add_g(one, expnx));
  if (gt_zero_g(x)) {
    return expnx;
  } else {
    return neg_g(expnx);
  }
}

template <typename T> __device__ __forceinline__ T log_g(const T x) {
  return log(x);
}

template <> __device__ __forceinline__ float log_g(const float x) {
  return logf(x);
}

template <typename T>
__device__ __forceinline__ T sigmoid_unstable_g(const T x) {
  return div_g(dscalar_one<T>(), add_g(dscalar_one<T>(), exp_g(neg_g(x))));
}

template <typename T> __device__ __forceinline__ T sigmoid_g(const T x) {
  T negx = x;
  T expnx;
  T one = dscalar_one<T>();
  if (gt_zero_g(x)) {
    negx = neg_g(x);
  }
  expnx = exp_g(negx);
  expnx = div_g(one, add_g(one, expnx));
  if (gt_zero_g(x)) {
    return expnx;
  } else {
    return sub_g(one, expnx);
  }
}

template <typename T> __device__ __forceinline__ T logsigmoid_g(const T x) {
  T one = dscalar_one<T>();
  T negx = x;
  if (gt_zero_g(x)) {
    negx = neg_g(x);
  }
  T logaddexpnx = log_g(add_g(one, exp_g(negx)));
  if (gt_zero_g(x)) {
    return neg_g(logaddexpnx);
  } else {
    return sub_g(x, logaddexpnx);
  }
}

template <typename T>
__device__ __forceinline__ T d_sigmoid_g(const T sigmoid_output) {
  return mul_g(sigmoid_output, sub_g(dscalar_one<T>(), sigmoid_output));
}

template <typename T>
__device__ __forceinline__ T d_tanh_g(const T tanh_output) {
  return sub_g(dscalar_one<T>(), mul_g(tanh_output, tanh_output));
}

template <typename T> __device__ __forceinline__ T max_g(const T a, const T b) {
  return max(a, b);
}

template <typename T> __device__ __forceinline__ T min_g(const T a, const T b) {
  return min(a, b);
}

template <typename T>
__device__ __forceinline__ T clip_val_g(const T x, const T lower,
                                        const T upper) {
  return max_g(min_g(x, upper), lower);
}

template <typename T>
__device__ __forceinline__ bool low_half_gt_zero_2h(const T x);

template <typename T>
__device__ __forceinline__ bool high_half_gt_zero_2h(const T x);

template <typename T, typename HT>
__device__ __forceinline__ T join_halves_2h(const HT x, const HT y);

template <typename T> __device__ __forceinline__ float type2float(const T x);

template <> __device__ __forceinline__ float type2float(const float x) {
  return x;
}

template <> __device__ __forceinline__ float type2float(const double x) {
  return (double)x;
}

template <typename T> __device__ __forceinline__ T float2type(const float x);

template <> __device__ __forceinline__ float float2type(const float x) {
  return x;
}
template <> __device__ __forceinline__ double float2type(const float x) {
  return (double)x;
}

template <typename T> __device__ __forceinline__ bool isnan_g(const T x) {
  return isnan(x);
}

template <typename T> __device__ __forceinline__ bool isinf_g(const T x) {
  return isinf(x);
}

// #if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 600
#include <cuda_fp16.h>

// #pragma message(AT " CUDART_VERSION with FP16: " TOSTRING(CUDART_VERSION))
// #else
// #pragma message(AT " CUDART_VERSION: " TOSTRING(CUDART_VERSION))
// #endif

// #if CUDART_VERSION >= 11020 && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>

template <typename T> struct DoubleHalfTypes;

template <> struct DoubleHalfTypes<__half> {
  typedef __half2 Type;
};

template <> struct DoubleHalfTypes<__nv_bfloat16> {
  typedef __nv_bfloat162 Type;
};

template <> struct DoubleHalfTypes<float> {
  typedef float Type;
};

template <typename T> struct HalfDoubleTypes;

template <> struct HalfDoubleTypes<__half2> {
  typedef __half Type;
};

template <> struct HalfDoubleTypes<__nv_bfloat162> {
  typedef __nv_bfloat16 Type;
};

template <> struct HalfDoubleTypes<float> {
  typedef float Type;
};

template <> struct DoubleHalfTypes<double> {
  typedef double Type;
};

template <typename T, typename HT>
__device__ __forceinline__ HT low_half_2h(const T x) {}

template <typename T, typename HT>
__device__ __forceinline__ HT high_half_2h(const T x);

#include "inline_ops_fp16.cuh"

#include "inline_ops_2fp16.cuh"

#include "inline_ops_bf16.cuh"

#include "inline_ops_2bf16.cuh"

// #pragma message(AT " CUDART_VERSION with BF16: " TOSTRING(                     \
//     CUDART_VERSION) ", arch: " TOSTRING(__CUDA_ARCH__))
// #else
//  #pragma message(AT " CUDART_VERSION: " TOSTRING(CUDART_VERSION) ", arch: "
//  TOSTRING(__CUDA_ARCH__)) #endif
