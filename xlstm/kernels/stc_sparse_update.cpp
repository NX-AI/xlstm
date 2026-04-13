#include <torch/extension.h>

torch::Tensor stc_sparse_update_cuda(
    torch::Tensor c_state,
    torch::Tensor k_q,
    torch::Tensor v_q,
    torch::Tensor fg_act,
    torch::Tensor ig_act
);

torch::Tensor stc_sparse_update(
    torch::Tensor c_state,
    torch::Tensor k_q,
    torch::Tensor v_q,
    torch::Tensor fg_act,
    torch::Tensor ig_act
) {
    if (c_state.is_cuda()) {
        return stc_sparse_update_cuda(c_state, k_q, v_q, fg_act, ig_act);
    } else {
        // Fallback to PyTorch's native operations for CPU
        auto outer = torch::matmul(k_q, v_q.transpose(-1, -2));
        return fg_act * c_state + ig_act * outer;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stc_sparse_update, "STC Sparse Update forward");
}
