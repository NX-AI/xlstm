// Copyright (c) NXAI GmbH and its affiliates 2023
//
// HIP/ROCm compatibility shim for the sLSTM CUDA kernel.
//
// torch's hipify pass rewrites most CUDA symbols in the kernel sources
// (cublas* -> hipblas*, __nv_bfloat16 -> __hip_bfloat16, cuda_runtime ->
// hip_runtime, ...), but a handful of cuBLAS enums/types and a few math
// intrinsics have no entry in its substitution map. This header — force
// included on ROCm builds via `-include` — provides the missing pieces so the
// hipified sources compile unchanged. Modeled on llama.cpp's
// ggml-cuda/vendors/hip.h.
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hipblas/hipblas.h>

// --- cuBLAS math-mode API: hipify translates the calls but not these enums/
// types. hipBLAS keeps HIPBLAS_TENSOR_OP_MATH as a (deprecated) no-op alias.
#ifndef CUBLAS_TENSOR_OP_MATH
#define CUBLAS_TENSOR_OP_MATH HIPBLAS_TENSOR_OP_MATH
#endif
#ifndef CUBLAS_DEFAULT_MATH
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#endif
using cublasMath_t = hipblasMath_t;

// --- data-type / compute-type enums used by cublasGemmEx-style calls.
#ifndef CUDA_R_16F
#define CUDA_R_16F  HIPBLAS_R_16F
#endif
#ifndef CUDA_R_16BF
#define CUDA_R_16BF HIPBLAS_R_16B
#endif
#ifndef CUDA_R_32F
#define CUDA_R_32F  HIPBLAS_R_32F
#endif

// cublasGemmEx compute types and algo selector (hipify leaves these untouched).
#ifndef CUBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#endif
#ifndef CUBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#endif
#ifndef CUBLAS_COMPUTE_32F_FAST_16F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#endif
#ifndef CUBLAS_GEMM_DFALT_TENSOR_OP
#define CUBLAS_GEMM_DFALT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#endif
#ifndef CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#endif

// The SLSTM_DTYPE_* compiler defines inject the literal type name
// __nv_bfloat16 into slstm.h at compile time (so hipify, which only rewrites
// source text, never sees it). ROCm defines nv_bfloat16 but not the
// double-underscore CUDA spelling, so alias it here.
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

// --- bf16 round-to-nearest scalar->bf16 conversion used by the pointwise
// kernels. CUDA spells it __floatbfloat_rn; HIP only provides __float2bfloat16.
#ifndef __floatbfloat_rn
static __device__ __forceinline__ __hip_bfloat16 __floatbfloat_rn(float x) {
    return __float2bfloat16(x);
}
#endif

// --- packed bf16 scalar-broadcast conversion. CUDA's __float2bfloat162_rn(float)
// broadcasts a scalar into both lanes; HIP only defines the float2 (vector)
// overload, so add the scalar form. It must be __host__ __device__ because the
// scalar_*() helpers that call it are host-callable. (__hmax2/__hmin2 for
// __hip_bfloat162 DO exist natively on ROCm, so they are not redefined here.)
static __host__ __device__ __forceinline__ __hip_bfloat162 __float2bfloat162_rn(float x) {
    const __hip_bfloat16 h = __float2bfloat16(x);
    return __halves2bfloat162(h, h);
}

// --- packed fp16 max/min. ROCm provides no __hmax2/__hmin2 for __half2, and
// __half2 implicitly converts to __hip_bfloat162, so an unshimmed call silently
// binds the bf16 overload and returns the wrong type. Provide explicit __half2
// versions so overload resolution prefers them.
static __host__ __device__ __forceinline__ __half2 __hmax2(const __half2 a, const __half2 b) {
    return __halves2half2(__hmax(__low2half(a), __low2half(b)),
                          __hmax(__high2half(a), __high2half(b)));
}
static __host__ __device__ __forceinline__ __half2 __hmin2(const __half2 a, const __half2 b) {
    return __halves2half2(__hmin(__low2half(a), __low2half(b)),
                          __hmin(__high2half(a), __high2half(b)));
}

// --- hipblasHgemm takes hipblasHalf (uint16_t) pointers, but the fp16 gemv
// wrapper in blas.cu passes __half pointers (bit-identical). Add a thin __half
// overload that reinterprets to the hipBLAS type. The differing pointer type
// makes this a genuine overload (no recursion).
static inline hipblasStatus_t hipblasHgemm(
    hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k, const __half *alpha, const __half *A, int lda,
    const __half *B, int ldb, const __half *beta, __half *C, int ldc) {
    return hipblasHgemm(handle, transa, transb, m, n, k,
                        reinterpret_cast<const hipblasHalf *>(alpha),
                        reinterpret_cast<const hipblasHalf *>(A), lda,
                        reinterpret_cast<const hipblasHalf *>(B), ldb,
                        reinterpret_cast<const hipblasHalf *>(beta),
                        reinterpret_cast<hipblasHalf *>(C), ldc);
}

#endif // __HIP_PLATFORM_AMD__ || USE_ROCM
