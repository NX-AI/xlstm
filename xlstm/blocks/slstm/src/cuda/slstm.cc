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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <type_traits>
#include <vector>

#include "../util/support.h"
#include "slstm.h"

namespace {

using slstm::BackwardPass;
using slstm::BackwardPassCut;
using slstm::ForwardPass;

using torch::Tensor;

class sLSTMFunc {
private:
  ForwardPass fw;
  BackwardPass bw;
  BackwardPassCut bwc;

public:
  sLSTMFunc(const bool training, const int batch_size, const int hidden_size,
            const int num_heads)
      : fw(training, batch_size, hidden_size, num_heads, 0, 0),
        bw(batch_size, hidden_size, num_heads, 0, 0),
        bwc(batch_size, hidden_size, num_heads, 0, 0) {}

  std::vector<Tensor> forward(bool training, Tensor x, Tensor s0,
                              Tensor recurrent_kernel, Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto num_heads = recurrent_kernel.size(0);

    const auto hidden_size = recurrent_kernel.size(1) * num_heads;

    CHECK_INPUT(x);
    CHECK_INPUT(s0);
    CHECK_INPUT(recurrent_kernel);
    CHECK_INPUT(bias);

    TORCH_CHECK(x.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_W>(),
                "Bad input type");
    TORCH_CHECK(s0.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_S>(),
                "Bad input type");
    TORCH_CHECK(recurrent_kernel.scalar_type() ==
                    typeToTorchDtype<SLSTM_DTYPE_R>(),
                "Bad input type");
    TORCH_CHECK(bias.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_B>(),
                "Bad input type");

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());
    int res = 1;
    Tensor states = torch::empty(
        {SLSTM_NUM_STATES, time_steps + 1, batch_size, hidden_size},
        options.dtype(typeToTorchDtype<SLSTM_DTYPE_S>()));

    Tensor gate_cache_r;
    Tensor gate_cache_i;
    // if (training) {
    gate_cache_r =
        torch::empty({time_steps, batch_size, hidden_size * SLSTM_NUM_GATES},
                     options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
#if SLSTM_SIMPLE_AGG
    gate_cache_i =
        torch::empty({}, options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
#else
    // this is for both recurrent and input base caches, without additional
    // bias-only gates
    gate_cache_i =
        torch::empty({time_steps, batch_size, hidden_size * SLSTM_NUM_GATES},
                     options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
#endif
    // }
    // else {
    //   gate_cache_r =
    //       torch::empty({}, options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
    //   gate_cache_i =
    //       torch::empty({}, options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
    // }
    Tensor tmp_Ry =
        torch::empty({batch_size, hidden_size * SLSTM_NUM_GATES},
                     options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));

    for (uint i = 0; i < SLSTM_NUM_STATES; i++) {
      states[i][0] = s0[i];
    }
    torch::cuda::synchronize();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
        x.scalar_type(), "sLSTMFunc.forward", ([&] {
          fw.Set(training, batch_size, hidden_size, num_heads,
                 at::cuda::getCurrentCUDABlasHandle(),
                 at::cuda::getCurrentCUDAStream());
          res = fw.Run(
              time_steps,
              reinterpret_cast<SLSTM_DTYPE_R *>(recurrent_kernel.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_B *>(bias.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_W *>(x.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_S *>(states.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_r.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_i.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(tmp_Ry.data_ptr()));
        }));
    torch::cuda::synchronize();
    // change for hin version
    if (res != 0) {
      TORCH_CHECK(0, "Errors during CUDA kernel calls forward.");
    }
    return {states, gate_cache_r, gate_cache_i};
  }

  std::vector<Tensor> backward(Tensor recurrent_kernel_t, Tensor bias, Tensor s,
                               Tensor gate_cache_r, Tensor gate_cache_i,
                               Tensor ds_new) {
    const auto time_steps = gate_cache_r.size(0);
    const auto batch_size = gate_cache_r.size(1);
    const auto num_heads = recurrent_kernel_t.size(0);
    const auto hidden_size = recurrent_kernel_t.size(2) * num_heads;
    int res = 1;

    CHECK_INPUT(recurrent_kernel_t);
    CHECK_INPUT(bias);
    CHECK_INPUT(s);
    CHECK_INPUT(gate_cache_r);
    CHECK_INPUT(gate_cache_i);
    CHECK_INPUT(ds_new);

    TORCH_CHECK(ds_new.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_S>(),
                "Bad ds_new type");
    TORCH_CHECK(s.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_S>(),
                "Bad state type");
    TORCH_CHECK(gate_cache_i.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_G>(),
                "Bad gate_i type");
    TORCH_CHECK(gate_cache_r.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_G>(),
                "Bad gate_r type");
    TORCH_CHECK(recurrent_kernel_t.scalar_type() ==
                    typeToTorchDtype<SLSTM_DTYPE_R>(),
                "Bad input type");
    TORCH_CHECK(bias.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_B>(),
                "Bad input type");

    // AT_PRIVATE_CHECK_SELECTIVE_BUILD(x.scalar_type());
    // AT_PRIVATE_CHECK_SELECTIVE_BUILD(gate_cache_r.scalar_type());
    // AT_PRIVATE_CHECK_SELECTIVE_BUILD(recurrent_kernel_t.scalar_type());
    // AT_PRIVATE_CHECK_SELECTIVE_BUILD(bias.scalar_type());

    const auto options = recurrent_kernel_t.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dR = torch::zeros({SLSTM_NUM_HEADS, hidden_size / num_heads,
                              (hidden_size * SLSTM_NUM_GATES) / num_heads},
                             options);
    Tensor db = torch::zeros_like(bias);
    Tensor ds = torch::zeros({SLSTM_NUM_STATES, batch_size, hidden_size},
                             options.dtype(typeToTorchDtype<SLSTM_DTYPE_S>()));
#if SLSTM_SIMPLE_AGG
    Tensor gate_cache_bias =
        torch::ones({}, options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
#else
    // this is for both recurrent and input base caches, without additional
    // bias-only gates
    Tensor gate_cache_bias =
        torch::ones({time_steps, batch_size,
                     hidden_size * (SLSTM_NUM_GATES - SLSTM_NUM_GATES)},
                    options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
#endif
    torch::cuda::synchronize();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
        recurrent_kernel_t.scalar_type(), "SLSTMfunc.backward", ([&] {
          bw.Set(batch_size, hidden_size, num_heads,
                 at::cuda::getCurrentCUDABlasHandle(),
                 at::cuda::getCurrentCUDAStream());
          res = bw.Run(
              time_steps,
              reinterpret_cast<SLSTM_DTYPE_R *>(recurrent_kernel_t.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_B *>(bias.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_S *>(s.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_S *>(ds_new.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_R *>(dR.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_B *>(db.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_S *>(ds.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_r.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_i.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_bias.data_ptr()));
        }));
    if (res != 0) {
      TORCH_CHECK(0, "Errors during CUDA kernel calls backward.");
    }
    torch::cuda::synchronize();
#if SLSTM_SIMPLE_AGG
    return {gate_cache_r, ds, dR, db};
#else
    return {gate_cache_i, ds, dR, db};
#endif
  }

  std::vector<Tensor> backward_cut(Tensor recurrent_kernel_t, Tensor bias,
                                   Tensor s, Tensor gate_cache_r,
                                   Tensor gate_cache_i, Tensor ds_new) {
    const auto time_steps = gate_cache_r.size(0);
    const auto batch_size = gate_cache_r.size(1);
    const auto num_heads = recurrent_kernel_t.size(0);
    const auto hidden_size = recurrent_kernel_t.size(2) * num_heads;
    int res = 1;

    CHECK_INPUT(recurrent_kernel_t);
    CHECK_INPUT(bias);
    CHECK_INPUT(s);
    CHECK_INPUT(gate_cache_r);
    CHECK_INPUT(ds_new);

    TORCH_CHECK(ds_new.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_S>(),
                "Bad ds_new type");
    TORCH_CHECK(s.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_S>(),
                "Bad state type");
    TORCH_CHECK(gate_cache_i.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_G>(),
                "Bad gate_i type");
    TORCH_CHECK(gate_cache_r.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_G>(),
                "Bad gate_r type");
    TORCH_CHECK(recurrent_kernel_t.scalar_type() ==
                    typeToTorchDtype<SLSTM_DTYPE_R>(),
                "Bad input type");
    TORCH_CHECK(bias.scalar_type() == typeToTorchDtype<SLSTM_DTYPE_B>(),
                "Bad input type");

    const auto options = recurrent_kernel_t.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dR = torch::zeros({num_heads, hidden_size / num_heads,
                              (hidden_size * SLSTM_NUM_GATES) / num_heads},
                             options);
    Tensor db = torch::zeros_like(bias);
    Tensor ds = torch::zeros({SLSTM_NUM_STATES, batch_size, hidden_size},
                             options.dtype(typeToTorchDtype<SLSTM_DTYPE_S>()));

#if SLSTM_SIMPLE_AGG
    Tensor gate_cache_bias =
        torch::ones({}, options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
#else
    // this is for both recurrent and input base caches, without additional
    // bias-only gates
    Tensor gate_cache_bias =
        torch::ones({time_steps, batch_size,
                     hidden_size * (SLSTM_NUM_GATES - SLSTM_NUM_GATES)},
                    options.dtype(typeToTorchDtype<SLSTM_DTYPE_G>()));
#endif
    torch::cuda::synchronize();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF2(
        recurrent_kernel_t.scalar_type(), "SLSTMfunc.backward_cut", ([&] {
          bwc.Set(batch_size, hidden_size, num_heads,
                  at::cuda::getCurrentCUDABlasHandle(),
                  at::cuda::getCurrentCUDAStream());
          res = bwc.Run(
              time_steps,
              reinterpret_cast<SLSTM_DTYPE_R *>(recurrent_kernel_t.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_B *>(bias.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_S *>(s.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_S *>(ds_new.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_R *>(dR.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_B *>(db.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_S *>(ds.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_r.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_i.data_ptr()),
              reinterpret_cast<SLSTM_DTYPE_G *>(gate_cache_bias.data_ptr()));
        }));
    torch::cuda::synchronize();
    if (res != 0) {
      TORCH_CHECK(0, "Errors during CUDA kernel calls backward cut.");
    }
#if SLSTM_SIMPLE_AGG
    return {gate_cache_r, ds, dR, db};
#else
    return {gate_cache_i, ds, dR, db};
#endif
  }
};
} // anonymous namespace

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { PARAMETRIC_DEFINITIONS }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<sLSTMFunc>(m, "sLSTMFunc")
      .def(pybind11::init<const bool, const int, const int, const int>())
      .def("forward", &sLSTMFunc::forward)
      .def("backward", &sLSTMFunc::backward)
      .def("backward_cut", &sLSTMFunc::backward_cut);
}