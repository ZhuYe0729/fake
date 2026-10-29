#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace w4a4_int4 {

constexpr int kAlignment = 32;
constexpr int kThreads = 256;
constexpr float kQMax = 7.0f;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be torch.uint8")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be torch.float32")

inline void check_shape(int64_t value, char const* name) {
  TORCH_CHECK(value > 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
              name, " must fit positive int32, got ", value);
}

enum DTypeCode {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
};

int dtype_code(at::ScalarType dtype) {
  if (dtype == at::kFloat) {
    return kFloat32;
  }
  if (dtype == at::kHalf) {
    return kFloat16;
  }
  TORCH_CHECK(dtype == at::kBFloat16, "unsupported dtype");
  return kBFloat16;
}

__device__ float read_float_value(void const* ptr, int dtype, int64_t idx) {
  if (dtype == kFloat32) {
    return static_cast<float const*>(ptr)[idx];
  }
  if (dtype == kFloat16) {
    return __half2float(static_cast<__half const*>(ptr)[idx]);
  }
  return __bfloat162float(static_cast<__nv_bfloat16 const*>(ptr)[idx]);
}

__device__ int8_t quantize_s4(float value, float scale) {
  float q = nearbyintf(value / scale);
  q = fminf(fmaxf(q, -8.0f), 7.0f);
  return static_cast<int8_t>(q);
}

__device__ uint8_t pack_two_s4(int8_t lo, int8_t hi) {
  return static_cast<uint8_t>((lo & 0x0f) | ((hi & 0x0f) << 4));
}

__global__ void compute_row_scales_kernel(
    void const* input,
    float* scales,
    int dtype,
    int rows,
    int cols) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float thread_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = read_float_value(input, dtype, static_cast<int64_t>(row) * cols + col);
    value = isfinite(value) ? value : 0.0f;
    thread_max = fmaxf(thread_max, fabsf(value));
  }

  __shared__ float smem[kThreads];
  smem[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scales[row] = fmaxf(smem[0] / kQMax, 1.0e-8f);
  }
}

__global__ void compute_weight_scales_kernel(
    void const* weight,
    float* scales,
    int dtype,
    int rows,
    int cols) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float thread_max = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = read_float_value(weight, dtype, static_cast<int64_t>(row) * cols + col);
    value = isfinite(value) ? value : 0.0f;
    thread_max = fmaxf(thread_max, fabsf(value));
  }

  __shared__ float smem[kThreads];
  smem[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scales[row] = fmaxf(smem[0] / kQMax, 1.0e-8f);
  }
}

__global__ void pack_activation_kernel(
    void const* input,
    float const* scales,
    uint8_t* packed,
    int dtype,
    int rows,
    int cols) {
  int64_t byte_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_bytes = static_cast<int64_t>(rows) * (cols / 2);
  if (byte_idx >= total_bytes) {
    return;
  }

  int row = static_cast<int>(byte_idx / (cols / 2));
  int col_pair = static_cast<int>(byte_idx - static_cast<int64_t>(row) * (cols / 2));
  int col0 = col_pair * 2;
  int64_t base = static_cast<int64_t>(row) * cols;
  float scale = scales[row];

  float v0 = read_float_value(input, dtype, base + col0);
  float v1 = read_float_value(input, dtype, base + col0 + 1);
  v0 = isfinite(v0) ? v0 : 0.0f;
  v1 = isfinite(v1) ? v1 : 0.0f;

  packed[byte_idx] = pack_two_s4(quantize_s4(v0, scale), quantize_s4(v1, scale));
}

__global__ void pack_weight_kernel(
    void const* weight,
    float const* scales,
    uint8_t* packed,
    int dtype,
    int rows,
    int cols) {
  int64_t byte_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_bytes = static_cast<int64_t>(rows) * (cols / 2);
  if (byte_idx >= total_bytes) {
    return;
  }

  int row = static_cast<int>(byte_idx / (cols / 2));
  int col_pair = static_cast<int>(byte_idx - static_cast<int64_t>(row) * (cols / 2));
  int col0 = col_pair * 2;
  int64_t base = static_cast<int64_t>(row) * cols;
  float scale = scales[row];

  float v0 = read_float_value(weight, dtype, base + col0);
  float v1 = read_float_value(weight, dtype, base + col0 + 1);
  v0 = isfinite(v0) ? v0 : 0.0f;
  v1 = isfinite(v1) ? v1 : 0.0f;

  packed[byte_idx] = pack_two_s4(quantize_s4(v0, scale), quantize_s4(v1, scale));
}

__global__ void scale_int32_to_bf16_kernel(
    int32_t const* acc,
    float const* activation_scale,
    float const* weight_scale,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* output,
    int m,
    int n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(m) * n;
  if (idx >= total) {
    return;
  }

  int row = static_cast<int>(idx / n);
  int col = static_cast<int>(idx - static_cast<int64_t>(row) * n);
  float value = static_cast<float>(acc[idx]) * activation_scale[row] * weight_scale[col];
  if (bias != nullptr) {
    value += __bfloat162float(bias[col]);
  }
  output[idx] = __float2bfloat16(value);
}

using ElementInput = cutlass::int4b_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInput,
    cutlass::layout::RowMajor,
    ElementInput,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombinationClamp<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;

bool supported_shape(int64_t m, int64_t n, int64_t k) {
  return m > 0 && n > 0 && k > 0 && (k % kAlignment == 0) && (n % kAlignment == 0);
}

void check_float_input(torch::Tensor input, char const* name) {
  CHECK_CUDA(input);
  CHECK_CONTIGUOUS(input);
  TORCH_CHECK(input.dim() == 2, name, " must be 2D");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
          input.scalar_type() == at::kBFloat16,
      name, " must be float32, float16, or bfloat16");
}

}  // namespace w4a4_int4

std::tuple<torch::Tensor, torch::Tensor> pack_activation(torch::Tensor input) {
  using namespace w4a4_int4;
  check_float_input(input, "input");
  c10::cuda::CUDAGuard device_guard(input.device());

  int64_t rows64 = input.size(0);
  int64_t cols64 = input.size(1);
  check_shape(rows64, "rows");
  check_shape(cols64, "cols");
  TORCH_CHECK(cols64 % kAlignment == 0, "activation K must be divisible by 32");

  int rows = static_cast<int>(rows64);
  int cols = static_cast<int>(cols64);
  auto scale = torch::empty({rows}, input.options().dtype(at::kFloat));
  auto packed = torch::empty({rows, cols / 2}, input.options().dtype(at::kByte));

  auto stream = at::cuda::getCurrentCUDAStream();
  int dtype = dtype_code(input.scalar_type());
  compute_row_scales_kernel<<<rows, kThreads, 0, stream>>>(
      input.const_data_ptr(), scale.data_ptr<float>(), dtype, rows, cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t total_bytes = static_cast<int64_t>(rows) * (cols / 2);
  int blocks = static_cast<int>((total_bytes + kThreads - 1) / kThreads);
  pack_activation_kernel<<<blocks, kThreads, 0, stream>>>(
      input.const_data_ptr(), scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
      dtype, rows, cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scale};
}

std::tuple<torch::Tensor, torch::Tensor> pack_weight(torch::Tensor weight) {
  using namespace w4a4_int4;
  check_float_input(weight, "weight");
  c10::cuda::CUDAGuard device_guard(weight.device());

  int64_t rows64 = weight.size(0);
  int64_t cols64 = weight.size(1);
  check_shape(rows64, "rows");
  check_shape(cols64, "cols");
  TORCH_CHECK(rows64 % kAlignment == 0, "weight N/out_features must be divisible by 32");
  TORCH_CHECK(cols64 % kAlignment == 0, "weight K/in_features must be divisible by 32");

  int rows = static_cast<int>(rows64);
  int cols = static_cast<int>(cols64);
  auto scale = torch::empty({rows}, weight.options().dtype(at::kFloat));
  auto packed = torch::empty({rows, cols / 2}, weight.options().dtype(at::kByte));

  auto stream = at::cuda::getCurrentCUDAStream();
  int dtype = dtype_code(weight.scalar_type());
  compute_weight_scales_kernel<<<rows, kThreads, 0, stream>>>(
      weight.const_data_ptr(), scale.data_ptr<float>(), dtype, rows, cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t total_bytes = static_cast<int64_t>(rows) * (cols / 2);
  int blocks = static_cast<int>((total_bytes + kThreads - 1) / kThreads);
  pack_weight_kernel<<<blocks, kThreads, 0, stream>>>(
      weight.const_data_ptr(), scale.data_ptr<float>(), packed.data_ptr<uint8_t>(),
      dtype, rows, cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scale};
}

bool can_implement_w4a4_int4(int64_t m, int64_t n, int64_t k) {
  using namespace w4a4_int4;
  if (!supported_shape(m, n, k)) {
    return false;
  }
  Gemm gemm;
  auto status = gemm.can_implement({
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
      {static_cast<ElementInput const*>(nullptr), static_cast<int>(k)},
      {static_cast<ElementInput const*>(nullptr), static_cast<int>(k)},
      {static_cast<ElementOutput const*>(nullptr), static_cast<int>(n)},
      {static_cast<ElementOutput*>(nullptr), static_cast<int>(n)},
      {1, 0}
  });
  return status == cutlass::Status::kSuccess;
}

torch::Tensor w4a4_int4_gemm(
    torch::Tensor packed_activation,
    torch::Tensor activation_scale,
    torch::Tensor packed_weight,
    torch::Tensor weight_scale,
    std::optional<torch::Tensor> bias,
    int64_t m64,
    int64_t n64,
    int64_t k64) {
  using namespace w4a4_int4;
  CHECK_CUDA(packed_activation);
  CHECK_CONTIGUOUS(packed_activation);
  CHECK_UINT8(packed_activation);
  CHECK_CUDA(packed_weight);
  CHECK_CONTIGUOUS(packed_weight);
  CHECK_UINT8(packed_weight);
  CHECK_CUDA(activation_scale);
  CHECK_CONTIGUOUS(activation_scale);
  CHECK_FLOAT(activation_scale);
  CHECK_CUDA(weight_scale);
  CHECK_CONTIGUOUS(weight_scale);
  CHECK_FLOAT(weight_scale);
  TORCH_CHECK(supported_shape(m64, n64, k64), "unsupported W4A4 INT4 GEMM shape");

  check_shape(m64, "M");
  check_shape(n64, "N");
  check_shape(k64, "K");
  int m = static_cast<int>(m64);
  int n = static_cast<int>(n64);
  int k = static_cast<int>(k64);

  TORCH_CHECK(packed_activation.dim() == 2, "packed_activation must be 2D");
  TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must be 2D");
  TORCH_CHECK(packed_activation.size(0) == m && packed_activation.size(1) == k / 2,
              "packed_activation shape mismatch");
  TORCH_CHECK(packed_weight.size(0) == n && packed_weight.size(1) == k / 2,
              "packed_weight shape mismatch");
  TORCH_CHECK(activation_scale.dim() == 1 && activation_scale.numel() == m,
              "activation_scale shape mismatch");
  TORCH_CHECK(weight_scale.dim() == 1 && weight_scale.numel() == n,
              "weight_scale shape mismatch");
  TORCH_CHECK(packed_activation.device() == packed_weight.device(), "packed tensors must share a device");
  TORCH_CHECK(activation_scale.device() == packed_activation.device(), "activation_scale device mismatch");
  TORCH_CHECK(weight_scale.device() == packed_activation.device(), "weight_scale device mismatch");

  torch::Tensor bias_tensor;
  __nv_bfloat16 const* bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    bias_tensor = bias.value();
    CHECK_CUDA(bias_tensor);
    CHECK_CONTIGUOUS(bias_tensor);
    TORCH_CHECK(bias_tensor.scalar_type() == at::kBFloat16, "bias must be torch.bfloat16");
    TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.numel() == n, "bias shape mismatch");
    TORCH_CHECK(bias_tensor.device() == packed_activation.device(), "bias device mismatch");
    bias_ptr = reinterpret_cast<__nv_bfloat16 const*>(bias_tensor.const_data_ptr<at::BFloat16>());
  }

  c10::cuda::CUDAGuard device_guard(packed_activation.device());
  auto int_options = packed_activation.options().dtype(at::kInt);
  auto out_options = packed_activation.options().dtype(at::kBFloat16);
  auto acc = torch::empty({m, n}, int_options);
  auto output = torch::empty({m, n}, out_options);

  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<ElementInput const*>(packed_activation.data_ptr<uint8_t>()), k},
      {reinterpret_cast<ElementInput const*>(packed_weight.data_ptr<uint8_t>()), k},
      {acc.data_ptr<int32_t>(), n},
      {acc.data_ptr<int32_t>(), n},
      {1, 0});

  auto status = gemm.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS cannot implement W4A4 INT4 GEMM");

  auto stream = at::cuda::getCurrentCUDAStream();
  status = gemm(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS W4A4 INT4 GEMM failed");
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t total = static_cast<int64_t>(m) * n;
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  scale_int32_to_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      acc.data_ptr<int32_t>(),
      activation_scale.data_ptr<float>(),
      weight_scale.data_ptr<float>(),
      bias_ptr,
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      m,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
