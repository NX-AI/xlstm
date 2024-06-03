// Copyright (c) NXAI GmbH and its affiliates 2023
// Korbinian Poeppel

#include <iostream>

template <typename T>
__device__ __forceinline__ void print_val(const char *pref, const int col,
                                          const int row, const T val) {
  printf("<%d, %d> - %s: (f) %f\n", col, row, pref, val);
}

template <>
__device__ __forceinline__ void print_val(const char *pref, const int col,
                                          const int row, const int val) {
  printf("<%d, %d> - %s: (f) %d\n", col, row, pref, val);
}

template <>
__device__ __forceinline__ void print_val(const char *pref, const int col,
                                          const int row, const double val) {
  printf("<%d, %d> - %s: (d) %lf\n", col, row, pref, val);
}

template <>
__device__ __forceinline__ void print_val(const char *pref, const int col,
                                          const int row, const __half val) {
  printf("<%d, %d> - %s: (h) %f\n", col, row, pref, __half2float(val));
}

template <>
__device__ __forceinline__ void print_val(const char *pref, const int col,
                                          const int row,
                                          const __nv_bfloat16 val) {
  printf("<%d, %d> - %s: (b) %f\n", col, row, pref, __bfloat162float(val));
}

template <>
__device__ __forceinline__ void print_val(const char *pref, const int col,
                                          const int row, const __half2 val) {
  printf("<%d, %d> - %s: (h2) %f, %f\n", col, row, pref,
         __half2float(__low2half(val)), __half2float(__high2half(val)));
}

template <>
__device__ __forceinline__ void print_val(const char *pref, const int col,
                                          const int row,
                                          const __nv_bfloat162 val) {
  printf("<%d, %d> - %s: (b2) %f, %f\n", col, row, pref,
         __bfloat162float(__low2bfloat16(val)),
         __bfloat162float(__high2bfloat16(val)));
}
