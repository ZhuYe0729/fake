#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <spinfer_api.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace {

void check_status(cusparseStatus_t status, char const* what) {
  TORCH_CHECK(status == CUSPARSE_STATUS_SUCCESS, what, " failed with cuSPARSE status ", static_cast<int>(status));
}

void check_cuda(cudaError_t status, char const* what) {
  TORCH_CHECK(status == cudaSuccess, what, " failed: ", cudaGetErrorString(status));
}

void check_shape(int64_t value, char const* name) {
  TORCH_CHECK(value > 0 && value <= static_cast<int64_t>(INT32_MAX), name, " must fit positive int32, got ", value);
}

void check_supported_shape(int64_t m, int64_t n, int64_t k) {
  check_shape(m, "M");
  check_shape(n, "N");
  check_shape(k, "K");
  TORCH_CHECK(m % 8 == 0, "M/out_features must be divisible by 8 for cuSPARSELt sparse BF16, got ", m);
  TORCH_CHECK(n % 8 == 0, "N/tokens must be divisible by 8 for cuSPARSELt sparse BF16, got ", n);
  TORCH_CHECK(k % 64 == 0, "K/in_features must be divisible by 64 for cuSPARSELt sparse BF16, got ", k);
}

void check_spinfer_supported_shape(int64_t m, int64_t n, int64_t k) {
  check_shape(m, "M");
  check_shape(n, "N");
  check_shape(k, "K");
  TORCH_CHECK(m % 64 == 0, "M/out_features must be divisible by 64 for SpInfer sparse BF16, got ", m);
  TORCH_CHECK(k % 64 == 0, "K/in_features must be divisible by 64 for SpInfer sparse BF16, got ", k);
  TORCH_CHECK(
      n == 8 || n == 16 || n == 32 || n == 64 || n == 128 || n % 128 == 0,
      "N/tokens must be one of 8,16,32,64,128 or divisible by 128 for SpInfer sparse BF16, got ",
      n);
}

struct LtHandle {
  cusparseLtHandle_t handle{};

  LtHandle() {
    check_status(cusparseLtInit(&handle), "cusparseLtInit");
  }

  ~LtHandle() {
    cusparseLtDestroy(&handle);
  }

  LtHandle(LtHandle const&) = delete;
  LtHandle& operator=(LtHandle const&) = delete;
};

struct MatDescriptor {
  cusparseLtMatDescriptor_t desc{};
  bool initialized = false;

  ~MatDescriptor() {
    if (initialized) {
      cusparseLtMatDescriptorDestroy(&desc);
    }
  }

  MatDescriptor() = default;
  MatDescriptor(MatDescriptor const&) = delete;
  MatDescriptor& operator=(MatDescriptor const&) = delete;

  MatDescriptor(MatDescriptor&& other) noexcept : desc(other.desc), initialized(other.initialized) {
    other.initialized = false;
  }

  MatDescriptor& operator=(MatDescriptor&& other) noexcept {
    if (this != &other) {
      if (initialized) {
        cusparseLtMatDescriptorDestroy(&desc);
      }
      desc = other.desc;
      initialized = other.initialized;
      other.initialized = false;
    }
    return *this;
  }
};

struct MatmulPlan {
  cusparseLtMatmulPlan_t plan{};
  bool initialized = false;

  ~MatmulPlan() {
    if (initialized) {
      cusparseLtMatmulPlanDestroy(&plan);
    }
  }

  MatmulPlan() = default;
  MatmulPlan(MatmulPlan const&) = delete;
  MatmulPlan& operator=(MatmulPlan const&) = delete;

  MatmulPlan(MatmulPlan&& other) noexcept : plan(other.plan), initialized(other.initialized) {
    other.initialized = false;
  }

  MatmulPlan& operator=(MatmulPlan&& other) noexcept {
    if (this != &other) {
      if (initialized) {
        cusparseLtMatmulPlanDestroy(&plan);
      }
      plan = other.plan;
      initialized = other.initialized;
      other.initialized = false;
    }
    return *this;
  }
};

MatDescriptor make_sparse_a_desc(LtHandle const& lt, int64_t m, int64_t k) {
  MatDescriptor a;
  check_status(
      cusparseLtStructuredDescriptorInit(
          &lt.handle,
          &a.desc,
          m,
          k,
          k,
          16,
          CUDA_R_16BF,
          CUSPARSE_ORDER_ROW,
          CUSPARSELT_SPARSITY_50_PERCENT),
      "cusparseLtStructuredDescriptorInit(A)");
  a.initialized = true;
  return a;
}

MatDescriptor make_dense_desc(
    LtHandle const& lt,
    int64_t rows,
    int64_t cols,
    int64_t ld,
    cusparseOrder_t order,
    char const* name) {
  MatDescriptor desc;
  std::ostringstream what;
  what << "cusparseLtDenseDescriptorInit(" << name << ")";
  check_status(
      cusparseLtDenseDescriptorInit(&lt.handle, &desc.desc, rows, cols, ld, 16, CUDA_R_16BF, order),
      what.str().c_str());
  desc.initialized = true;
  return desc;
}

MatmulPlan make_matmul_plan(
    LtHandle const& lt,
    MatDescriptor const& a,
    MatDescriptor const& b,
    MatDescriptor const& c,
    MatDescriptor const& d,
    cusparseLtMatmulDescriptor_t& matmul,
    cusparseLtMatmulAlgSelection_t& alg) {
  check_status(
      cusparseLtMatmulDescriptorInit(
          &lt.handle,
          &matmul,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &a.desc,
          &b.desc,
          &c.desc,
          &d.desc,
          CUSPARSE_COMPUTE_32F),
      "cusparseLtMatmulDescriptorInit");

  check_status(
      cusparseLtMatmulAlgSelectionInit(&lt.handle, &alg, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT),
      "cusparseLtMatmulAlgSelectionInit");

  MatmulPlan plan;
  check_status(cusparseLtMatmulPlanInit(&lt.handle, &plan.plan, &matmul, &alg), "cusparseLtMatmulPlanInit");
  plan.initialized = true;
  return plan;
}

struct CachedMatmulState {
  bool valid = false;
  int device_id = -1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t compressed_a_size = 0;
  size_t expected_compressed_size = 0;
  std::unique_ptr<LtHandle> lt;
  MatDescriptor a;
  MatDescriptor b;
  MatDescriptor c;
  MatDescriptor d;
  cusparseLtMatmulDescriptor_t matmul{};
  cusparseLtMatmulAlgSelection_t alg{};
  MatmulPlan plan;
  void* workspace = nullptr;
  size_t workspace_size = 0;

  ~CachedMatmulState() {
    reset();
  }

  void reset() {
    if (workspace != nullptr) {
      cudaFree(workspace);
      workspace = nullptr;
    }
    workspace_size = 0;
    plan = MatmulPlan{};
    d = MatDescriptor{};
    c = MatDescriptor{};
    b = MatDescriptor{};
    a = MatDescriptor{};
    lt.reset();
    valid = false;
    device_id = -1;
    m = n = k = 0;
    compressed_a_size = 0;
    expected_compressed_size = 0;
  }
};

struct MatmulCacheKey {
  int device_id = -1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t compressed_a_size = 0;

  bool operator==(MatmulCacheKey const& other) const {
    return device_id == other.device_id && m == other.m && n == other.n && k == other.k &&
        compressed_a_size == other.compressed_a_size;
  }
};

struct MatmulCacheKeyHash {
  size_t operator()(MatmulCacheKey const& key) const {
    size_t seed = std::hash<int>{}(key.device_id);
    auto mix = [&seed](int64_t value) {
      seed ^= std::hash<int64_t>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    mix(key.m);
    mix(key.n);
    mix(key.k);
    mix(key.compressed_a_size);
    return seed;
  }
};

constexpr size_t kMaxMatmulCacheEntries = 512;
using MatmulLruList = std::list<MatmulCacheKey>;
using MatmulCacheValue = std::pair<std::unique_ptr<CachedMatmulState>, MatmulLruList::iterator>;

MatmulLruList matmul_cache_lru;
std::unordered_map<MatmulCacheKey, MatmulCacheValue, MatmulCacheKeyHash> matmul_cache;

void clear_matmul_cache() {
  matmul_cache.clear();
  matmul_cache_lru.clear();
}

CachedMatmulState& get_cached_matmul(int64_t m, int64_t n, int64_t k, int64_t compressed_a_size) {
  int current_device = 0;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
  MatmulCacheKey key{current_device, m, n, k, compressed_a_size};

  auto found = matmul_cache.find(key);
  if (found != matmul_cache.end()) {
    matmul_cache_lru.splice(matmul_cache_lru.begin(), matmul_cache_lru, found->second.second);
    found->second.second = matmul_cache_lru.begin();
    return *found->second.first;
  }

  auto entry = std::make_unique<CachedMatmulState>();
  entry->device_id = current_device;
  entry->m = m;
  entry->n = n;
  entry->k = k;
  entry->compressed_a_size = compressed_a_size;
  entry->lt = std::make_unique<LtHandle>();
  entry->a = make_sparse_a_desc(*entry->lt, m, k);
  entry->b = make_dense_desc(*entry->lt, k, n, k, CUSPARSE_ORDER_COL, "B");
  entry->c = make_dense_desc(*entry->lt, m, n, m, CUSPARSE_ORDER_COL, "C");
  entry->d = make_dense_desc(*entry->lt, m, n, m, CUSPARSE_ORDER_COL, "D");

  size_t ignored_buffer_size = 0;
  check_status(
      cusparseLtSpMMACompressedSize2(&entry->lt->handle, &entry->a.desc, &entry->expected_compressed_size, &ignored_buffer_size),
      "cusparseLtSpMMACompressedSize2(matmul)");

  entry->plan = make_matmul_plan(*entry->lt, entry->a, entry->b, entry->c, entry->d, entry->matmul, entry->alg);
  check_status(
      cusparseLtMatmulGetWorkspace(&entry->lt->handle, &entry->plan.plan, &entry->workspace_size),
      "cusparseLtMatmulGetWorkspace");
  if (entry->workspace_size > 0) {
    check_cuda(cudaMalloc(&entry->workspace, entry->workspace_size), "cudaMalloc(matmul workspace)");
  }
  entry->valid = true;

  matmul_cache_lru.push_front(key);
  auto inserted = matmul_cache.emplace(key, MatmulCacheValue{std::move(entry), matmul_cache_lru.begin()});
  while (matmul_cache.size() > kMaxMatmulCacheEntries) {
    auto evict_key = matmul_cache_lru.back();
    matmul_cache.erase(evict_key);
    matmul_cache_lru.pop_back();
  }
  return *inserted.first->second.first;
}

void check_pruned_weight(LtHandle const& lt, MatDescriptor const& a, torch::Tensor const& dense_a) {
  auto valid_device = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(dense_a.device()));
  auto stream = at::cuda::getCurrentCUDAStream(dense_a.device().index()).stream();
  check_status(
      cusparseLtSpMMAPruneCheck2(
          &lt.handle,
          &a.desc,
          1,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          dense_a.data_ptr(),
          valid_device.data_ptr<int>(),
          stream),
      "cusparseLtSpMMAPruneCheck2");
  int valid_host = 0;
  check_cuda(
      cudaMemcpyAsync(
          &valid_host,
          valid_device.data_ptr<int>(),
          sizeof(int),
          cudaMemcpyDeviceToHost,
          stream),
      "cudaMemcpyAsync(prune valid)");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(prune valid)");
  TORCH_CHECK(valid_host != 0, "weight failed cuSPARSELt 2:4 prune check; valid flag=", valid_host);
}

int cusparselt_version() {
  LtHandle lt;
  int version = 0;
  check_status(cusparseLtGetVersion(&lt.handle, &version), "cusparseLtGetVersion");
  return version;
}

struct SpinferHostBuffers {
  half* compressed = nullptr;
  int* tile_offsets = nullptr;
  int* tile_offsets_median = nullptr;
  int* tile_offsets_global = nullptr;
  uint64_t* bitmap = nullptr;

  ~SpinferHostBuffers() {
    std::free(compressed);
    std::free(tile_offsets);
    std::free(tile_offsets_median);
    std::free(tile_offsets_global);
    std::free(bitmap);
  }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> spinfer_sparse_bf16_compress_weight(
    torch::Tensor const& dense_a) {
  torch::checkAllContiguous("spinfer_sparse_bf16_compress_weight", {{dense_a, "dense_a", 0}});
  torch::checkDeviceType("spinfer_sparse_bf16_compress_weight", {dense_a}, at::DeviceType::CUDA);
  TORCH_CHECK(dense_a.scalar_type() == at::kBFloat16, "dense_a must be bfloat16");
  TORCH_CHECK(dense_a.dim() == 2, "dense_a must be 2D [M, K]");

  int64_t m = dense_a.size(0);
  int64_t k = dense_a.size(1);
  check_spinfer_supported_shape(m, 8, k);

  const at::cuda::OptionalCUDAGuard device_guard(dense_a.device());
  auto dense_half_cpu = dense_a.to(torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU)).contiguous();

  SpinferHostBuffers host;
  int max_nnz_intile = 0;
  int num_global_tiles = spinfer_init_sparse_matrix(
      reinterpret_cast<half*>(dense_half_cpu.data_ptr<at::Half>()),
      static_cast<int>(m),
      static_cast<int>(k),
      8,
      16,
      64,
      8,
      64,
      64,
      &host.compressed,
      &host.tile_offsets,
      &host.tile_offsets_median,
      &host.tile_offsets_global,
      &host.bitmap,
      max_nnz_intile);
  TORCH_CHECK(num_global_tiles >= 0, "spinfer_init_sparse_matrix failed");

  int64_t num_global_tiles_i64 = static_cast<int64_t>(num_global_tiles);
  int64_t num_local_tiles = num_global_tiles_i64 * 8 * 8;
  int64_t num_median_tiles = num_global_tiles_i64 * 4;
  int64_t value_count = static_cast<int64_t>(host.tile_offsets_global[num_global_tiles]);
  TORCH_CHECK(value_count > 0, "SpInfer compressed weight has zero values");

  auto stream = at::cuda::getCurrentCUDAStream(dense_a.device().index()).stream();
  auto compressed = torch::empty({value_count}, torch::TensorOptions().dtype(torch::kFloat16).device(dense_a.device()));
  auto tile_offsets_global =
      torch::empty({num_global_tiles_i64 + 1}, torch::TensorOptions().dtype(torch::kInt32).device(dense_a.device()));
  auto tile_offsets_median =
      torch::empty({num_median_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(dense_a.device()));
  auto bitmap = torch::empty({num_local_tiles}, torch::TensorOptions().dtype(torch::kInt64).device(dense_a.device()));

  check_cuda(
      cudaMemcpyAsync(
          compressed.data_ptr<at::Half>(),
          host.compressed,
          static_cast<size_t>(value_count) * sizeof(half),
          cudaMemcpyHostToDevice,
          stream),
      "cudaMemcpyAsync(spinfer compressed)");
  check_cuda(
      cudaMemcpyAsync(
          tile_offsets_global.data_ptr<int>(),
          host.tile_offsets_global,
          static_cast<size_t>(num_global_tiles_i64 + 1) * sizeof(int),
          cudaMemcpyHostToDevice,
          stream),
      "cudaMemcpyAsync(spinfer tile_offsets_global)");
  check_cuda(
      cudaMemcpyAsync(
          tile_offsets_median.data_ptr<int>(),
          host.tile_offsets_median,
          static_cast<size_t>(num_median_tiles) * sizeof(int),
          cudaMemcpyHostToDevice,
          stream),
      "cudaMemcpyAsync(spinfer tile_offsets_median)");
  check_cuda(
      cudaMemcpyAsync(
          bitmap.data_ptr<int64_t>(),
          host.bitmap,
          static_cast<size_t>(num_local_tiles) * sizeof(uint64_t),
          cudaMemcpyHostToDevice,
          stream),
      "cudaMemcpyAsync(spinfer bitmap)");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(spinfer weight upload)");

  return std::make_tuple(compressed, tile_offsets_global, tile_offsets_median, bitmap, max_nnz_intile);
}

std::tuple<torch::Tensor, torch::Tensor> sparse_bf16_compress_weight(torch::Tensor const& dense_a) {
  torch::checkAllContiguous("sparse_bf16_compress_weight", {{dense_a, "dense_a", 0}});
  torch::checkDeviceType("sparse_bf16_compress_weight", {dense_a}, at::DeviceType::CUDA);
  TORCH_CHECK(dense_a.scalar_type() == at::kBFloat16, "dense_a must be bfloat16");
  TORCH_CHECK(dense_a.dim() == 2, "dense_a must be 2D [M, K]");

  clear_matmul_cache();

  int64_t m = dense_a.size(0);
  int64_t k = dense_a.size(1);
  check_supported_shape(m, 8, k);

  const at::cuda::OptionalCUDAGuard device_guard(dense_a.device());
  LtHandle lt;
  auto a = make_sparse_a_desc(lt, m, k);
  check_pruned_weight(lt, a, dense_a);

  size_t compressed_size = 0;
  size_t compressed_buffer_size = 0;
  check_status(
      cusparseLtSpMMACompressedSize2(&lt.handle, &a.desc, &compressed_size, &compressed_buffer_size),
      "cusparseLtSpMMACompressedSize2");
  TORCH_CHECK(compressed_size > 0, "cuSPARSELt returned zero compressed size");

  auto compressed = torch::empty(
      {static_cast<int64_t>(compressed_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(dense_a.device()));
  auto metadata = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(dense_a.device()));
  auto compress_buffer = torch::empty(
      {static_cast<int64_t>(compressed_buffer_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(dense_a.device()));

  auto stream = at::cuda::getCurrentCUDAStream(dense_a.device().index()).stream();
  check_status(
      cusparseLtSpMMACompress2(
          &lt.handle,
          &a.desc,
          1,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          dense_a.data_ptr(),
          compressed.data_ptr(),
          compress_buffer.numel() ? compress_buffer.data_ptr() : nullptr,
          stream),
      "cusparseLtSpMMACompress2");
  return std::make_tuple(compressed, metadata);
}

torch::Tensor sparse_bf16_matmul(
    int64_t m,
    int64_t n,
    int64_t k,
    torch::Tensor const& compressed_a,
    torch::Tensor const& b_bf16) {
  torch::checkAllContiguous(
      "sparse_bf16_matmul",
      {{compressed_a, "compressed_a", 0}, {b_bf16, "b_bf16", 1}});
  torch::checkDeviceType("sparse_bf16_matmul", {compressed_a, b_bf16}, at::DeviceType::CUDA);
  torch::checkAllSameGPU(
      "sparse_bf16_matmul",
      {{compressed_a, "compressed_a", 0}, {b_bf16, "b_bf16", 1}});

  TORCH_CHECK(compressed_a.scalar_type() == at::kByte, "compressed_a must be uint8");
  TORCH_CHECK(b_bf16.scalar_type() == at::kBFloat16, "b_bf16 must be bfloat16");
  TORCH_CHECK(b_bf16.dim() == 2, "b_bf16 must be 2D [N, K]");
  TORCH_CHECK(b_bf16.size(0) == n && b_bf16.size(1) == k, "b_bf16 shape does not match N/K");
  check_supported_shape(m, n, k);

  const at::cuda::OptionalCUDAGuard device_guard(compressed_a.device());
  auto& cached = get_cached_matmul(m, n, k, compressed_a.numel());
  TORCH_CHECK(
      compressed_a.numel() == static_cast<int64_t>(cached.expected_compressed_size),
      "compressed_a has unexpected size for M/K, got ",
      compressed_a.numel(),
      " expected ",
      cached.expected_compressed_size);

  auto out = torch::empty({n, m}, torch::TensorOptions().dtype(torch::kBFloat16).device(compressed_a.device()));

  float alpha_host = 1.0f;
  float beta_host = 0.0f;
  auto stream = at::cuda::getCurrentCUDAStream(compressed_a.device().index()).stream();
  check_status(
      cusparseLtMatmul(
          &cached.lt->handle,
          &cached.plan.plan,
          &alpha_host,
          compressed_a.data_ptr(),
          b_bf16.data_ptr(),
          &beta_host,
          out.data_ptr(),
          out.data_ptr(),
          cached.workspace,
          &stream,
          1),
      "cusparseLtMatmul");
  return out;
}

torch::Tensor spinfer_sparse_bf16_matmul(
    int64_t m,
    int64_t n,
    int64_t k,
    torch::Tensor const& compressed_a,
    torch::Tensor const& tile_offsets_global,
    torch::Tensor const& tile_offsets_median,
    torch::Tensor const& bitmap,
    int64_t max_nnz_intile,
    torch::Tensor const& b_bf16) {
  torch::checkAllContiguous(
      "spinfer_sparse_bf16_matmul",
      {{compressed_a, "compressed_a", 0},
       {tile_offsets_global, "tile_offsets_global", 1},
       {tile_offsets_median, "tile_offsets_median", 2},
       {bitmap, "bitmap", 3},
       {b_bf16, "b_bf16", 4}});
  torch::checkDeviceType(
      "spinfer_sparse_bf16_matmul",
      {compressed_a, tile_offsets_global, tile_offsets_median, bitmap, b_bf16},
      at::DeviceType::CUDA);
  torch::checkAllSameGPU(
      "spinfer_sparse_bf16_matmul",
      {{compressed_a, "compressed_a", 0},
       {tile_offsets_global, "tile_offsets_global", 1},
       {tile_offsets_median, "tile_offsets_median", 2},
       {bitmap, "bitmap", 3},
       {b_bf16, "b_bf16", 4}});

  TORCH_CHECK(compressed_a.scalar_type() == at::kHalf, "compressed_a must be float16 for SpInfer");
  TORCH_CHECK(tile_offsets_global.scalar_type() == at::kInt, "tile_offsets_global must be int32");
  TORCH_CHECK(tile_offsets_median.scalar_type() == at::kInt, "tile_offsets_median must be int32");
  TORCH_CHECK(bitmap.scalar_type() == at::kLong, "bitmap must be int64");
  TORCH_CHECK(b_bf16.scalar_type() == at::kBFloat16, "b_bf16 must be bfloat16");
  TORCH_CHECK(b_bf16.dim() == 2, "b_bf16 must be 2D [N, K]");
  TORCH_CHECK(b_bf16.size(0) == n && b_bf16.size(1) == k, "b_bf16 shape does not match N/K");
  TORCH_CHECK(max_nnz_intile > 0 && max_nnz_intile <= INT32_MAX, "max_nnz_intile must fit positive int32");
  check_spinfer_supported_shape(m, n, k);

  int64_t num_global_tiles = (m / 64) * (k / 64);
  TORCH_CHECK(tile_offsets_global.numel() == num_global_tiles + 1, "tile_offsets_global has unexpected size");
  TORCH_CHECK(tile_offsets_median.numel() == num_global_tiles * 4, "tile_offsets_median has unexpected size");
  TORCH_CHECK(bitmap.numel() == num_global_tiles * 64, "bitmap has unexpected size");

  const at::cuda::OptionalCUDAGuard device_guard(compressed_a.device());
  auto b_half = b_bf16.to(torch::kFloat16).contiguous();
  auto out_half = torch::empty({n, m}, torch::TensorOptions().dtype(torch::kFloat16).device(compressed_a.device()));

  int split_k = spinfer_pick_splitk(static_cast<int>(m), static_cast<int>(k));
  torch::Tensor reduction_workspace;
  half* reduction_ptr = nullptr;
  if (split_k > 1) {
    reduction_workspace =
        torch::empty({split_k * m * n}, torch::TensorOptions().dtype(torch::kFloat16).device(compressed_a.device()));
    reduction_ptr = reinterpret_cast<half*>(reduction_workspace.data_ptr<at::Half>());
  }

  auto stream = at::cuda::getCurrentCUDAStream(compressed_a.device().index()).stream();
  check_cuda(
      cudaMemsetAsync(out_half.data_ptr<at::Half>(), 0, static_cast<size_t>(out_half.numel()) * sizeof(half), stream),
      "cudaMemsetAsync(spinfer output)");
  check_cuda(
      spinfer_spmm(
          stream,
          nullptr,
          reinterpret_cast<half const*>(compressed_a.data_ptr<at::Half>()),
          tile_offsets_global.data_ptr<int>(),
          tile_offsets_median.data_ptr<int>(),
          reinterpret_cast<uint64_t const*>(bitmap.data_ptr<int64_t>()),
          static_cast<int>(max_nnz_intile),
          reinterpret_cast<half const*>(b_half.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out_half.data_ptr<at::Half>()),
          static_cast<int>(m),
          static_cast<int>(n),
          static_cast<int>(k),
          reduction_ptr,
          split_k),
      "spinfer_spmm");
  return out_half.to(torch::kBFloat16);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cusparselt_version", &cusparselt_version, "Return the linked cuSPARSELt version.");
  m.def(
      "sparse_bf16_compress_weight",
      &sparse_bf16_compress_weight,
      "Compress a 2:4 structured sparse BF16 weight with cuSPARSELt.");
  m.def(
      "sparse_bf16_matmul",
      &sparse_bf16_matmul,
      "Run cuSPARSELt sparse BF16 matmul using a pre-compressed sparse A weight.");
  m.def(
      "spinfer_sparse_bf16_compress_weight",
      &spinfer_sparse_bf16_compress_weight,
      "Compress a sparse BF16 weight into SpInfer bitmap-v3 format.");
  m.def(
      "spinfer_sparse_bf16_matmul",
      &spinfer_sparse_bf16_matmul,
      "Run SpInfer sparse matmul for the sparse BF16 wrapper.");
}
