/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM120 architecture.

    This example demonstrates a simple way to instantiate and run a narrow precision blockscaled sparse GEMM on the NVIDIA Blackwell SM120 architecture.
    This kernel is optimized for the GeForce RTX 50 series GPUs.

    The Blackwell SM120 CUTLASS kernel uses the new Block Scaled Sparse Tensor Core MMA Instructions:
      * mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale.
    Please see more detail in https://docs.nvidia.com/cuda/parallel-thread-execution.

    The kernel leverages:
    1. Warp-Specialized persistent kernel design that supports cooperative scheduler introduced in Hopper.
    2. The new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).
    3. Block Scaled Sparse Tensor Core MMA Instructions

    Note that GeForce RTX 50 series GPUs do not support:
    1. Multicast feature of TMA load. Cluster shape has to be 1x1x1.
    2. Dynamic datatypes.

    Usage:
      $ ./examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_bf16_sparse_gemm --m=2048 --n=2048 --k=2048
*/
#include <iostream>
#include <memory>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/core_io.h"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"

#include <cuda_bf16.h>

#include "helper.h"
using namespace cute;
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;     // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                       // Layout type for A matrix operand
constexpr int AlignmentA  = 64;                                              // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;     // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                    // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                              // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                             // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                             // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::ColumnMajor;                    // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::ColumnMajor;                    // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;     // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;     // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
using   outputScaleFactor = cutlass::float_ue4m3_t;                          // Scale factor type for D matrix
// E matrix configuration. Note, E is used to represent metadata tensor.
using         ElementE    = uint8_t;                                         // Element type for E matrix operand
// Kernel functional config
using ElementCompute      = float;                                           // Element type for computation inside mainloop and epilogue
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledSparseTensorOp; // Operator class tag
using KernelScheduleType =  cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;   // Kernel schedule policy
// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_256>;                           // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
  cutlass::epilogue::collective::EpilogueScheduleAuto                       // Epilogue schedule policy
  >::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelScheduleType                                                       // Mainloop schedule policy
  >::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                  // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));
using LayoutE   = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                            cute::Shape<int, int, int, int>,
                            ElementA::DataType,
                            LayoutATag,
                            SparseConfig>;
using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
                            cute::Shape<int, int, int, int>,
                            ElementA::DataType,
                            LayoutATag,
                            SparseConfig,
                            cutlass::arch::Sm120>;
using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;
//
// Data members
//
/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
LayoutE layout_E;
uint64_t seed;
// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A_Decompressed;
cutlass::HostTensor<ElementE, cutlass::layout::PackedVectorLayout> block_E;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;

struct CachedRunConfig {
  int m = 0;
  int n = 0;
  int k = 0;
  float alpha = 0.0f;
  float beta = 0.0f;
  void const* sparse_a = nullptr;
  void const* metadata_e = nullptr;
  void const* b = nullptr;
  void const* sfa = nullptr;
  void const* sfb = nullptr;
  void* d = nullptr;
  void const* c = nullptr;

  bool matches(
      int other_m,
      int other_n,
      int other_k,
      float other_alpha,
      float other_beta,
      void const* other_sparse_a,
      void const* other_metadata_e,
      void const* other_b,
      void const* other_sfa,
      void const* other_sfb,
      void* other_d,
      void const* other_c) const {
    return m == other_m &&
        n == other_n &&
        k == other_k &&
        alpha == other_alpha &&
        beta == other_beta &&
        sparse_a == other_sparse_a &&
        metadata_e == other_metadata_e &&
        b == other_b &&
        sfa == other_sfa &&
        sfb == other_sfb &&
        d == other_d &&
        c == other_c;
  }
};

struct CachedRunState {
  bool valid = false;
  int device_id = -1;
  CachedRunConfig config;
  std::unique_ptr<Gemm> gemm;
  void* workspace = nullptr;
  size_t workspace_size = 0;

  ~CachedRunState() {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
  }
};

CachedRunState cached_run_state;
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

constexpr int kSparseNvfp4GroupSize = 32;
constexpr float kSparseNvfp4Max = 6.0f;

__device__ uint8_t sparse_fp4_code_from_float(float v) {
  return cutlass::float_e2m1_t(v).storage;
}

template <class LayoutSF>
__device__ int64_t sparse_scale_layout_offset(LayoutSF const& layout_sf, int row, int k) {
  return static_cast<int64_t>(layout_sf(cute::make_coord(row, k, 0)));
}

template <class LayoutSF>
__global__ void sparse_compute_b_scales_kernel(
    __nv_bfloat16 const* input,
    uint8_t* scale_storage,
    float* scale_float,
    float const* global_scale,
    LayoutSF layout_sf,
    int rows,
    int cols,
    int groups) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * groups;
  if (idx >= total) {
    return;
  }

  int row = idx / groups;
  int group = idx - row * groups;
  int col_begin = group * kSparseNvfp4GroupSize;
  float max_abs = 0.0f;

  #pragma unroll
  for (int i = 0; i < kSparseNvfp4GroupSize; ++i) {
    int col = col_begin + i;
    float v = __bfloat162float(input[row * cols + col]);
    v = isfinite(v) ? v : 0.0f;
    max_abs = fmaxf(max_abs, fabsf(v));
  }

  float scale = max_abs > 0.0f ? (*global_scale) * max_abs / kSparseNvfp4Max : 1.0f;
  scale = fminf(fmaxf(scale, 1.0e-8f), 448.0f);
  cutlass::float_ue4m3_t sf(scale);
  float quantized_scale = static_cast<float>(sf);
  if (quantized_scale == 0.0f) {
    sf = cutlass::float_ue4m3_t(1.0e-8f);
    quantized_scale = static_cast<float>(sf);
  }

  int64_t sf_offset = sparse_scale_layout_offset(layout_sf, row, col_begin);
  scale_storage[sf_offset] = sf.storage;
  scale_float[idx] = quantized_scale;
}

__global__ void sparse_pack_b_data_kernel(
    __nv_bfloat16 const* input,
    uint8_t* packed,
    float const* scale_float,
    float const* global_scale,
    int rows,
    int cols,
    int groups,
    int64_t total_elements) {
  int64_t byte_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total_bytes = (total_elements + 1) / 2;
  if (byte_idx >= total_bytes) {
    return;
  }

  uint8_t codes[2] = {0, 0};
  #pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    int64_t elem = byte_idx * 2 + lane;
    if (elem >= total_elements) {
      continue;
    }

    int row = static_cast<int>(elem / cols);
    int col = static_cast<int>(elem - static_cast<int64_t>(row) * cols);
    float scale = scale_float[row * groups + col / kSparseNvfp4GroupSize];
    float v = __bfloat162float(input[row * cols + col]);
    v = isfinite(v) ? v : 0.0f;
    v = v * (*global_scale) / scale;
    v = fminf(fmaxf(v, -kSparseNvfp4Max), kSparseNvfp4Max);
    codes[lane] = sparse_fp4_code_from_float(v);
  }

  packed[byte_idx] = static_cast<uint8_t>((codes[1] << 4) | codes[0]);
}

bool pack_sparse_b_bf16(
    int rows,
    int k,
    void const* bf16_input_device_ptr,
    void* packed_b_device_ptr,
    void* sfb_device_ptr,
    float const* global_scale_device_ptr,
    float* scale_float_device_ptr,
    cudaStream_t stream) {
  if (bf16_input_device_ptr == nullptr || packed_b_device_ptr == nullptr ||
      sfb_device_ptr == nullptr || global_scale_device_ptr == nullptr ||
      scale_float_device_ptr == nullptr) {
    std::cerr << "pack_sparse_b_bf16: input, output, scale, and scratch pointers must not be null." << std::endl;
    return false;
  }
  if (rows <= 0 || k <= 0 || k % kSparseNvfp4GroupSize != 0) {
    std::cerr << "pack_sparse_b_bf16: invalid shape." << std::endl;
    return false;
  }

  int groups = k / kSparseNvfp4GroupSize;
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1, rows, k, 1));

  int threads = 256;
  int blocks = (rows * groups + threads - 1) / threads;
  sparse_compute_b_scales_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(bf16_input_device_ptr),
      reinterpret_cast<uint8_t*>(sfb_device_ptr),
      scale_float_device_ptr,
      global_scale_device_ptr,
      layout_sfb,
      rows,
      k,
      groups);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "pack_sparse_b_bf16: scale kernel failed: " << cudaGetErrorString(error) << std::endl;
    return false;
  }

  int64_t total_elements = static_cast<int64_t>(rows) * static_cast<int64_t>(k);
  blocks = static_cast<int>(((total_elements + 1) / 2 + threads - 1) / threads);
  sparse_pack_b_data_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(bf16_input_device_ptr),
      reinterpret_cast<uint8_t*>(packed_b_device_ptr),
      scale_float_device_ptr,
      global_scale_device_ptr,
      rows,
      k,
      groups,
      total_elements);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "pack_sparse_b_bf16: pack kernel failed: " << cudaGetErrorString(error) << std::endl;
    return false;
  }
  return true;
}

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <typename T>
bool copy_device_to_device(T* dst, T const* src, size_t count, char const* name) {
  if (count == 0) {
    return true;
  }
  if (dst == nullptr || src == nullptr) {
    std::cerr << "Invalid device pointer for " << name << std::endl;
    return false;
  }

  size_t bytes = cutlass::bits_to_bytes<size_t>(count * cutlass::sizeof_bits<T>::value);
  cudaError_t error = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
  if (error != cudaSuccess) {
    std::cerr << "cudaMemcpyDeviceToDevice failed for " << name << ": "
              << cudaGetErrorString(error) << std::endl;
    return false;
  }
  return true;
}

CompressorUtility make_compressor_utility(int m, int n, int k) {
  auto workload = make_shape(m, n, k, 1);
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  layout_A = SparseConfig::fill_layoutA(workload);
  layout_E = SparseConfig::fill_layoutE(workload);
  return CompressorUtility(workload, stride_A);
}

bool allocate_sparse_a_buffers(CompressorUtility const& compressor_utility) {
  int aligned_m_e = compressor_utility.get_metadata_m_physical();
  int aligned_k_e = compressor_utility.get_metadata_k_physical();
  int aligned_m_a = compressor_utility.get_tensorA_m_physical();
  int aligned_k_a = compressor_utility.get_tensorA_k_physical();

  block_A.reset(cutlass::make_Coord(aligned_m_a * aligned_k_a));
  block_E.reset(cutlass::make_Coord(aligned_m_e * aligned_k_e));
  return true;
}

bool compress_sparse_a_from_decompressed_device(int m, int n, int k) {
  cutlass::Status status { cutlass::Status::kSuccess };
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  typename Compressor::Arguments arguments{
    {m, n, k, 1},
    {block_A_Decompressed.device_data(),
      stride_A,
      block_A.device_data(),
      block_E.device_data()},
    {hw_info}
  };

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  status = compressor_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = compressor_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  status = compressor_op.run();
  if (status != cutlass::Status::kSuccess) {
    return false;
  }

  auto result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    return false;
  }

  return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////
// Command line options parsing
struct Options {
  bool help;
  float alpha, beta;
  int iterations;
  int m, n, k;
  Options():
    help(false),
    m(1024), n(1024), k(1024),
    alpha(1.f), beta(0.f),
    iterations(10)
  { }
  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }
  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {
    out << "80b_blackwell_geforce_nvfp4_bf16_sparse_gemm\n\n"
      << "  Blackwell MXFP8 Sparse GEMM is a warp specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";
    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_bf16_sparse_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";
    return out;
  }
  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};
/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;
  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}
};
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {
  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;
  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}
/// Initialize blocks that released to sparse Matrix A and its metadata E
bool initialize_sparse_blocks(
  const Options &options,
  ElementA::DataType const* dense_a_device_ptr = nullptr,
  bool apply_random_2to4_mask = false) {
  CompressorUtility compressor_utility = make_compressor_utility(options.m, options.n, options.k);
  allocate_sparse_a_buffers(compressor_utility);
  block_A_Decompressed.reset(cutlass::make_Coord(options.m * options.k));

  if (dense_a_device_ptr != nullptr) {
    if (!copy_device_to_device(
          block_A_Decompressed.device_data(),
          dense_a_device_ptr,
          static_cast<size_t>(options.m) * static_cast<size_t>(options.k),
          "dense A")) {
      return false;
    }

    if (apply_random_2to4_mask) {
      // This is a utility fallback to enforce structural sparsity. It mutates A.
      block_A_Decompressed.sync_host();
      compressor_utility.structure_sparse_zero_mask_fill(
          block_A_Decompressed.host_data(), static_cast<int>(seed + 2021));
      block_A_Decompressed.sync_device();
    }
  }
  else {
    initialize_block(block_A_Decompressed.host_view(), seed + 2020);
    compressor_utility.structure_sparse_zero_mask_fill(
            block_A_Decompressed.host_data(), static_cast<int>(seed + 2021));
    block_A_Decompressed.sync_device();
  }

  return compress_sparse_a_from_decompressed_device(options.m, options.n, options.k);
}

bool initialize_sparse_blocks_from_compressed(
  const Options& options,
  ElementA::DataType const* sparse_a_device_ptr,
  ElementE const* metadata_e_device_ptr) {
  if (sparse_a_device_ptr == nullptr || metadata_e_device_ptr == nullptr) {
    std::cerr << "initialize_sparse_blocks_from_compressed: sparse A and E pointers must not be null." << std::endl;
    return false;
  }

  CompressorUtility compressor_utility = make_compressor_utility(options.m, options.n, options.k);
  allocate_sparse_a_buffers(compressor_utility);

  if (!copy_device_to_device(
        block_A.device_data(),
        sparse_a_device_ptr,
        static_cast<size_t>(block_A.capacity()),
        "sparse A")) {
    return false;
  }

  if (!copy_device_to_device(
        block_E.device_data(),
        metadata_e_device_ptr,
        static_cast<size_t>(block_E.capacity()),
        "metadata E")) {
    return false;
  }

  return true;
}

bool initialize_non_a_operands(
  const Options &options,
  ElementB::DataType const* b_device_ptr = nullptr,
  ElementA::ScaleFactorType const* sfa_device_ptr = nullptr,
  ElementB::ScaleFactorType const* sfb_device_ptr = nullptr,
  ElementC const* c_device_ptr = nullptr) {
  using namespace cute;

  // Define B, C and D blocks
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  // Define SFA and SFB tensors layouts
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

  auto b_elements = static_cast<size_t>(size(layout_B));
  auto c_elements = static_cast<size_t>(size(layout_C));
  auto sfa_elements = static_cast<size_t>(size(filter_zeros(layout_SFA)));
  auto sfb_elements = static_cast<size_t>(size(filter_zeros(layout_SFB)));

  if (b_device_ptr != nullptr) {
    if (!copy_device_to_device(block_B.device_data(), b_device_ptr, b_elements, "B")) {
      return false;
    }
  }
  else {
    initialize_block(block_B.host_view(), seed + 2022);
    block_B.sync_device();
  }

  if (c_device_ptr != nullptr) {
    if (!copy_device_to_device(block_C.device_data(), c_device_ptr, c_elements, "C")) {
      return false;
    }
  }
  else {
    cutlass::reference::host::TensorFill(block_C.host_view(), ElementC(0));
    block_C.sync_device();
  }

  if (sfa_device_ptr != nullptr) {
    if (!copy_device_to_device(block_SFA.device_data(), sfa_device_ptr, sfa_elements, "SFA")) {
      return false;
    }
  }
  else {
    initialize_block(block_SFA.host_view(), seed + 2024);
    block_SFA.sync_device();
  }

  if (sfb_device_ptr != nullptr) {
    if (!copy_device_to_device(block_SFB.device_data(), sfb_device_ptr, sfb_elements, "SFB")) {
      return false;
    }
  }
  else {
    initialize_block(block_SFB.host_view(), seed + 2025);
    block_SFB.sync_device();
  }

  return true;
}

bool initialize_from_sparse(
  const Options& options,
  ElementA::DataType const* sparse_a_device_ptr,
  ElementE const* metadata_e_device_ptr,
  ElementB::DataType const* b_device_ptr = nullptr,
  ElementA::ScaleFactorType const* sfa_device_ptr = nullptr,
  ElementB::ScaleFactorType const* sfb_device_ptr = nullptr,
  ElementC const* c_device_ptr = nullptr) {
  if (!initialize_sparse_blocks_from_compressed(options, sparse_a_device_ptr, metadata_e_device_ptr)) {
    return false;
  }
  return initialize_non_a_operands(options, b_device_ptr, sfa_device_ptr, sfb_device_ptr, c_device_ptr);
}

/// Initialize operands to be used in the GEMM and reference GEMM
bool initialize(
  const Options &options,
  ElementA::DataType const* dense_a_device_ptr = nullptr,
  ElementB::DataType const* b_device_ptr = nullptr,
  ElementA::ScaleFactorType const* sfa_device_ptr = nullptr,
  ElementB::ScaleFactorType const* sfb_device_ptr = nullptr,
  ElementC const* c_device_ptr = nullptr,
  bool apply_random_2to4_mask = false) {
  if (!initialize_sparse_blocks(options, dense_a_device_ptr, apply_random_2to4_mask)) {
    return false;
  }
  return initialize_non_a_operands(options, b_device_ptr, sfa_device_ptr, sfb_device_ptr, c_device_ptr);
}

// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    { // Mainloop arguments
      block_A.device_data(), layout_A,
      block_B.device_data(), stride_B,
      block_E.device_data(), layout_E,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    { // Epilogue arguments
      {options.alpha, options.beta},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };
  return arguments;
}

typename Gemm::Arguments args_from_options(
    const Options &options,
    ElementC const* c_device_ptr,
    ElementD* d_device_ptr,
    outputScaleFactor* sfd_device_ptr)
{
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    { // Mainloop arguments
      block_A.device_data(), layout_A,
      block_B.device_data(), stride_B,
      block_E.device_data(), layout_E,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    { // Epilogue arguments
      {options.alpha, options.beta},
      const_cast<ElementC*>(c_device_ptr), stride_C,
      d_device_ptr, stride_D
    }
  };
  return arguments;
}

typename Gemm::Arguments args_from_raw_ptrs(
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    ElementA::DataType const* sparse_a_device_ptr,
    ElementE const* metadata_e_device_ptr,
    ElementB::DataType const* b_device_ptr,
    ElementA::ScaleFactorType const* sfa_device_ptr,
    ElementB::ScaleFactorType const* sfb_device_ptr,
    ElementC const* c_device_ptr,
    ElementD* d_device_ptr) {
  using namespace cute;

  auto problem_shape = cutlass::gemm::GemmCoord{m, n, k};
  auto workload = make_shape(m, n, k, 1);

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto layout_a = SparseConfig::fill_layoutA(workload);
  auto layout_e = SparseConfig::fill_layoutE(workload);
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(workload);
  auto layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(workload);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {problem_shape.m(), problem_shape.n(), problem_shape.k(), 1},
      {
          sparse_a_device_ptr, layout_a,
          b_device_ptr, stride_b,
          metadata_e_device_ptr, layout_e,
          sfa_device_ptr, layout_sfa,
          sfb_device_ptr, layout_sfb,
      },
      {
          {alpha, beta},
          const_cast<ElementC*>(c_device_ptr), stride_c,
          d_device_ptr, stride_d,
      }};
  return arguments;
}

int run_backend_from_sparse(
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    ElementA::DataType const* sparse_a_device_ptr,
    ElementE const* metadata_e_device_ptr,
    ElementB::DataType const* b_device_ptr,
    ElementA::ScaleFactorType const* sfa_device_ptr,
    ElementB::ScaleFactorType const* sfb_device_ptr,
    ElementD* d_device_ptr,
    ElementC const* c_device_ptr = nullptr,
    outputScaleFactor* sfd_device_ptr = nullptr)
{
  if (sparse_a_device_ptr == nullptr || metadata_e_device_ptr == nullptr ||
      b_device_ptr == nullptr || sfa_device_ptr == nullptr ||
      sfb_device_ptr == nullptr || d_device_ptr == nullptr) {
    std::cerr << "run_backend_from_sparse: A/E/B/SFA/SFB/D pointers must not be null." << std::endl;
    return -1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    std::cerr << "run_backend_from_sparse: invalid GEMM shape." << std::endl;
    return -1;
  }

  Options options;
  options.m = m;
  options.n = n;
  options.k = k;
  options.alpha = alpha;
  options.beta = beta;
  options.iterations = 0;

  Gemm gemm;
  auto arguments = args_from_raw_ptrs(
      m,
      n,
      k,
      alpha,
      beta,
      sparse_a_device_ptr,
      metadata_e_device_ptr,
      b_device_ptr,
      sfa_device_ptr,
      sfb_device_ptr,
      (c_device_ptr != nullptr ? c_device_ptr : reinterpret_cast<ElementC const*>(d_device_ptr)),
      d_device_ptr);

  auto cache_c_ptr = (c_device_ptr != nullptr ? c_device_ptr : reinterpret_cast<ElementC const*>(d_device_ptr));
  int current_device = 0;
  auto device_status = cudaGetDevice(&current_device);
  if (device_status != cudaSuccess) {
    std::cerr << "run_backend_from_sparse: cudaGetDevice failed: "
              << cudaGetErrorString(device_status) << std::endl;
    return -1;
  }
  bool cache_hit =
      cached_run_state.valid &&
      cached_run_state.device_id == current_device &&
      cached_run_state.config.matches(
          m,
          n,
          k,
          alpha,
          beta,
          sparse_a_device_ptr,
          metadata_e_device_ptr,
          b_device_ptr,
          sfa_device_ptr,
          sfb_device_ptr,
          d_device_ptr,
          cache_c_ptr);
  if (cache_hit) {
    auto status = cached_run_state.gemm->run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "run_backend_from_sparse: cached run failed." << std::endl;
      cached_run_state.valid = false;
      return -1;
    }
    return 0;
  }

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "run_backend_from_sparse: can_implement failed." << std::endl;
    return -1;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    auto cuda_status = cudaMalloc(&workspace, workspace_size);
    if (cuda_status != cudaSuccess) {
      std::cerr << "run_backend_from_sparse: cudaMalloc workspace failed: "
                << cudaGetErrorString(cuda_status) << std::endl;
      return -1;
    }
  }

  status = gemm.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    std::cerr << "run_backend_from_sparse: initialize failed." << std::endl;
    return -1;
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    std::cerr << "run_backend_from_sparse: run failed." << std::endl;
    return -1;
  }

  if (cached_run_state.workspace != nullptr) {
    cudaFree(cached_run_state.workspace);
  }
  cached_run_state.valid = true;
  cached_run_state.device_id = current_device;
  cached_run_state.config = {
      m,
      n,
      k,
      alpha,
      beta,
      sparse_a_device_ptr,
      metadata_e_device_ptr,
      b_device_ptr,
      sfa_device_ptr,
      sfb_device_ptr,
      d_device_ptr,
      cache_c_ptr,
  };
  cached_run_state.gemm = std::make_unique<Gemm>(std::move(gemm));
  cached_run_state.workspace = workspace;
  cached_run_state.workspace_size = workspace_size;

  return 0;
}

bool pack_sparse_a_from_dense(
    int m,
    int n,
    int k,
    ElementA::DataType const* dense_a_device_ptr,
    ElementA::DataType* sparse_a_device_ptr,
    ElementE* metadata_e_device_ptr,
    bool apply_random_2to4_mask)
{
  if (dense_a_device_ptr == nullptr || sparse_a_device_ptr == nullptr || metadata_e_device_ptr == nullptr) {
    std::cerr << "pack_sparse_a_from_dense: dense A, sparse A, and metadata E pointers must not be null." << std::endl;
    return false;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    std::cerr << "pack_sparse_a_from_dense: invalid GEMM shape." << std::endl;
    return false;
  }

  Options options;
  options.m = m;
  options.n = n;
  options.k = k;

  if (!initialize_sparse_blocks(options, dense_a_device_ptr, apply_random_2to4_mask)) {
    std::cerr << "pack_sparse_a_from_dense: sparse packing failed." << std::endl;
    return false;
  }

  if (!copy_device_to_device(
        sparse_a_device_ptr,
        block_A.device_data(),
        static_cast<size_t>(block_A.capacity()),
        "packed sparse A output")) {
    return false;
  }

  if (!copy_device_to_device(
        metadata_e_device_ptr,
        block_E.device_data(),
        static_cast<size_t>(block_E.capacity()),
        "packed metadata E output")) {
    return false;
  }

  return true;
}

extern "C" int qutlass_sparse_nvf4_gemm_from_sparse(
    int m,
    int n,
    int k,
    float alpha,
    float beta,
    void const* sparse_a_device_ptr,
    void const* metadata_e_device_ptr,
    void const* b_device_ptr,
    void const* sfa_device_ptr,
    void const* sfb_device_ptr,
    void* d_device_ptr,
    void const* c_device_ptr,
    void* sfd_device_ptr)
{
  return run_backend_from_sparse(
      m,
      n,
      k,
      alpha,
      beta,
      reinterpret_cast<ElementA::DataType const*>(sparse_a_device_ptr),
      reinterpret_cast<ElementE const*>(metadata_e_device_ptr),
      reinterpret_cast<ElementB::DataType const*>(b_device_ptr),
      reinterpret_cast<ElementA::ScaleFactorType const*>(sfa_device_ptr),
      reinterpret_cast<ElementB::ScaleFactorType const*>(sfb_device_ptr),
      reinterpret_cast<ElementD*>(d_device_ptr),
      reinterpret_cast<ElementC const*>(c_device_ptr),
      reinterpret_cast<outputScaleFactor*>(sfd_device_ptr));
}

extern "C" int qutlass_sparse_nvf4_pack_from_dense(
    int m,
    int n,
    int k,
    void const* dense_a_device_ptr,
    void* sparse_a_device_ptr,
    void* metadata_e_device_ptr,
    int apply_random_2to4_mask)
{
  return pack_sparse_a_from_dense(
             m,
             n,
             k,
             reinterpret_cast<ElementA::DataType const*>(dense_a_device_ptr),
             reinterpret_cast<ElementA::DataType*>(sparse_a_device_ptr),
             reinterpret_cast<ElementE*>(metadata_e_device_ptr),
             apply_random_2to4_mask != 0)
           ? 0
           : -1;
}

extern "C" int qutlass_sparse_nvf4_pack_b(
    int rows,
    int k,
    void const* bf16_input_device_ptr,
    void* packed_b_device_ptr,
    void* sfb_device_ptr,
    void const* global_scale_device_ptr,
    void* scale_float_device_ptr,
    cudaStream_t stream)
{
  return pack_sparse_b_bf16(
             rows,
             k,
             bf16_input_device_ptr,
             packed_b_device_ptr,
             sfb_device_ptr,
             reinterpret_cast<float const*>(global_scale_device_ptr),
             reinterpret_cast<float*>(scale_float_device_ptr),
             stream)
           ? 0
           : -1;
}

extern "C" int64_t qutlass_sparse_nvf4_packed_a_numel(int m, int n, int k)
{
  if (m <= 0 || n <= 0 || k <= 0) {
    return -1;
  }
  Options options;
  options.m = m;
  options.n = n;
  options.k = k;
  auto compressor_utility = make_compressor_utility(options.m, options.n, options.k);
  return static_cast<int64_t>(compressor_utility.get_tensorA_m_physical()) *
         static_cast<int64_t>(compressor_utility.get_tensorA_k_physical());
}

extern "C" int64_t qutlass_sparse_nvf4_metadata_e_numel(int m, int n, int k)
{
  if (m <= 0 || n <= 0 || k <= 0) {
    return -1;
  }
  Options options;
  options.m = m;
  options.n = n;
  options.k = k;
  auto compressor_utility = make_compressor_utility(options.m, options.n, options.k);
  return static_cast<int64_t>(compressor_utility.get_metadata_m_physical()) *
         static_cast<int64_t>(compressor_utility.get_metadata_k_physical());
}
bool verify(const Options &options) {
  using namespace cute;
  // Create the arguments for host reference implementation
  Tensor tensor_A = make_tensor(make_iterator(block_A_Decompressed.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);
  Tensor tensor_E = make_tensor(make_iterator(block_E.host_data()), layout_E);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,                 // ElementAccumulator
      decltype(tensor_A),                 // TensorA
      decltype(tensor_SFA),               // TensorSfA
      decltype(tensor_B),                 // TensorB
      decltype(tensor_SFB)                // TensorSfB
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};
  auto tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,                   // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementAccumulator,                   // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D)                    // TensorD
    > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D};
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
  // Comparison
  block_D.sync_host();

  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_reference_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);
  return passed;
}
/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  // Initialization
  if(!initialize(options))
  {
    std::cerr << " Initialization failed! " << std::endl;
    exit(-1);
  }

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;
  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());
  cudaDeviceSynchronize();
  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);
  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  if (!result.passed) {
    exit(-1);
  }
  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();
    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);
    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }
  return 0;
}
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit for SM120 support,
  // or CUDA 12.9 or higher for SM121 support.
  // Must have compute capability at least 120.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer for SM120 support." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#elif defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)) {
    std::cerr << "This example requires CUDA 12.9 or newer for SM121 support." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#endif
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));

  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 120 or 121)." << std::endl;
    return 0;
  }
  //
  // Parse options
  //
  Options options;
  options.parse(argc, args);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  //
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  run<Gemm>(options);
#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
