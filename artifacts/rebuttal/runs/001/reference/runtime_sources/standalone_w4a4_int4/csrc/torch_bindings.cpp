#include <torch/extension.h>

#include <optional>

std::tuple<torch::Tensor, torch::Tensor> pack_activation(torch::Tensor input);
std::tuple<torch::Tensor, torch::Tensor> pack_weight(torch::Tensor weight);
torch::Tensor w4a4_int4_gemm(
    torch::Tensor packed_activation,
    torch::Tensor activation_scale,
    torch::Tensor packed_weight,
    torch::Tensor weight_scale,
    std::optional<torch::Tensor> bias,
    int64_t m,
    int64_t n,
    int64_t k);
bool can_implement_w4a4_int4(int64_t m, int64_t n, int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_activation", &pack_activation, "Pack W4A4 INT4 activation");
  m.def("pack_weight", &pack_weight, "Pack W4A4 INT4 weight");
  m.def("w4a4_int4_gemm", &w4a4_int4_gemm, "W4A4 INT4 GEMM");
  m.def("can_implement_w4a4_int4", &can_implement_w4a4_int4, "Check W4A4 INT4 shape support");
}
