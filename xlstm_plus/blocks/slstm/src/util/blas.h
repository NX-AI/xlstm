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

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T> struct blas {
  struct set_pointer_mode {
    set_pointer_mode(cublasHandle_t handle) : handle_(handle) {
      cublasGetPointerMode(handle_, &old_mode_);
      cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    }
    ~set_pointer_mode() { cublasSetPointerMode(handle_, old_mode_); }

  private:
    cublasHandle_t handle_;
    cublasPointerMode_t old_mode_;
  };
  struct enable_tensor_cores {
    enable_tensor_cores(cublasHandle_t handle) : handle_(handle) {
      cublasGetMathMode(handle_, &old_mode_);
      cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    }
    ~enable_tensor_cores() { cublasSetMathMode(handle_, old_mode_); }

  private:
    cublasHandle_t handle_;
    cublasMath_t old_mode_;
  };
};

cublasStatus_t cublasHgemmsb(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const __half *alpha, const __half *A, int lda,
                             long long int strideA, const __half *B, int ldb,
                             long long int strideB, const __half *beta,
                             __half *C, int ldc, long long int strideC,
                             int batchCount);

cublasStatus_t cublasSgemmsb(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const float *alpha, const float *A, int lda,
                             long long int strideA, const float *B, int ldb,
                             long long int strideB, const float *beta, float *C,
                             int ldc, long long int strideC, int batchCount);

cublasStatus_t cublasDgemmsb(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const double *alpha, const double *A, int lda,
                             long long int strideA, const double *B, int ldb,
                             long long int strideB, const double *beta,
                             double *C, int ldc, long long int strideC,
                             int batchCount);

cublasStatus_t cublasHgemv2(cublasHandle_t handle, cublasOperation_t trans,
                            int m, int n, const __half *alpha, const __half *A,
                            int lda, const __half *x, int incx,
                            const __half *beta, __half *y, int incy);

cublasStatus_t cublasHgemv3(cublasHandle_t handle, cublasOperation_t trans,
                            int m, int n, const __half *alpha, const __half *A,
                            int lda, const __half *x, int incx,
                            const __half *beta, __half *y, int incy);

cublasStatus_t cublasHgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const __half *alpha, const __half *A,
                           int lda, const __half *x, int incx,
                           const __half *beta, __half *y, int incy);

cublasStatus_t cublasHgemm2(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const __half *alpha, /* host or device pointer */
                            const __half *A, int lda, const __half *B, int ldb,
                            const __half *beta, /* host or device pointer */
                            __half *C, int ldc);

cublasStatus_t cublasHgemm3(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const __half *alpha, /* host or device pointer */
                            const __half *A, int lda, const __half *B, int ldb,
                            const __half *beta, /* host or device pointer */
                            __half *C, int ldc);

void initVector_d(cudaStream_t stream, double *data, int size, double value);
void initVector_f(cudaStream_t stream, float *data, int size, float value);
void initVector_h(cudaStream_t stream, __half *data, int size, __half value);

template <> struct blas<__half> {
  static constexpr decltype(cublasHgemm2) *gemm = &cublasHgemm;
  static constexpr decltype(cublasHgemmsb) *gemmsb = &cublasHgemmsb;
  static constexpr decltype(cublasHgemv2) *gemv = &cublasHgemv;
  static constexpr decltype(initVector_h) *initVector = &initVector_h;
};

template <> struct blas<float> {
  static constexpr decltype(cublasSgemm) *gemm = &cublasSgemm;
  static constexpr decltype(cublasSgemmsb) *gemmsb = &cublasSgemmsb;
  static constexpr decltype(cublasSgemv) *gemv = &cublasSgemv;
  static constexpr decltype(initVector_f) *initVector = &initVector_f;
};

template <> struct blas<double> {
  static constexpr decltype(cublasDgemm) *gemm = &cublasDgemm;
  static constexpr decltype(cublasDgemmsb) *gemmsb = &cublasDgemmsb;
  static constexpr decltype(cublasDgemv) *gemv = &cublasDgemv;
  static constexpr decltype(initVector_d) *initVector = &initVector_d;
};

#if CUDART_VERSION >= 11020
#include <cuda_bf16.h>

cublasStatus_t cublasBgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const __nv_bfloat16 *alpha,
                           const __nv_bfloat16 *A, int lda,
                           const __nv_bfloat16 *x, int incx,
                           const __nv_bfloat16 *beta, __nv_bfloat16 *y,
                           int incy);

cublasStatus_t
cublasBgemm(cublasHandle_t handle, cublasOperation_t transa,
            cublasOperation_t transb, int m, int n, int k,
            const __nv_bfloat16 *alpha, /* host or device pointer */
            const __nv_bfloat16 *A, int lda, const __nv_bfloat16 *B, int ldb,
            const __nv_bfloat16 *beta, /* host or device pointer */
            __nv_bfloat16 *C, int ldc);

cublasStatus_t cublasBgemmsb(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const __nv_bfloat16 *alpha, const __nv_bfloat16 *A,
                             int lda, long long int strideA,
                             const __nv_bfloat16 *B, int ldb,
                             long long int strideB, const __nv_bfloat16 *beta,
                             __nv_bfloat16 *C, int ldc, long long int strideC,
                             int batchCount);

void initVector_b(cudaStream_t stream, __nv_bfloat16 *data, int size,
                  __nv_bfloat16 value);

template <> struct blas<__nv_bfloat16> {
  static constexpr decltype(cublasBgemm) *gemm = &cublasBgemm;
  static constexpr decltype(cublasBgemmsb) *gemmsb = &cublasBgemmsb;
  static constexpr decltype(cublasBgemv) *gemv = &cublasBgemv;
  static constexpr decltype(initVector_b) *initVector = &initVector_b;
};

#endif
