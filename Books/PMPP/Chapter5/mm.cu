#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kBlockSize = 16;

__global__ void matmul_kernel(const float* a,
                              const float* b,
                              float* c,
                              int m,
                              int n,
                              int k) {
  __shared__ float tile_a[kBlockSize][kBlockSize];
  __shared__ float tile_b[kBlockSize][kBlockSize];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  unsigned num_phases = (k + kBlockSize - 1) / kBlockSize;
  for (unsigned ph = 0; ph < num_phases; ++ph) {
    // 1. Load tile
    tile_a[ty][tx] = a[row * k + ph * kBlockSize + tx];
    tile_b[ty][tx] = b[n * (ph * kBlockSize + ty) + col];
    __syncthreads();

    // 2. Compute on tile
    for (unsigned i = 0; i < kBlockSize; ++i) {
      acc += tile_a[ty][i] * tile_b[i][tx];
    }
    __syncthreads();
  }

  c[row * n + col] = acc;
}

/*
  * This version of the kernel adds boundary checks when loading tiles. This is
  * necessary when m, n, or k is not a multiple of kBlockSize.
*/
__global__ void matmul_kernel_checked(const float* a,
                              const float* b,
                              float* c,
                              int m,
                              int n,
                              int k) {
  __shared__ float tile_a[kBlockSize][kBlockSize];
  __shared__ float tile_b[kBlockSize][kBlockSize];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  unsigned num_phases = (k + kBlockSize - 1) / kBlockSize;
  for (unsigned ph = 0; ph < num_phases; ++ph) {
    // 1. Load tile
    if (row < m && ph * kBlockSize + tx < k) {
      tile_a[ty][tx] = a[row * k + ph * kBlockSize + tx];
    } else {
      tile_a[ty][tx] = 0.0f;
    }
    if (ph * kBlockSize + ty < k && col < n) {
      tile_b[ty][tx] = b[n * (ph * kBlockSize + ty) + col];
    } else {
      tile_b[ty][tx] = 0.0f;
    }
    __syncthreads();

    // 2. Compute on tile
    // ! We don't need checks here because out-of-bound threads will contribute 0 to the final result.
    for (unsigned i = 0; i < kBlockSize; ++i) {
      acc += tile_a[ty][i] * tile_b[i][tx];
    }
    __syncthreads();
  }

  c[row * n + col] = acc;
}

void check_cuda(cudaError_t error, const char* what) {
  if (error == cudaSuccess) {
    return;
  }

  std::cerr << what << ": " << cudaGetErrorString(error) << '\n';
  std::exit(EXIT_FAILURE);
}

}  // namespace

int main() {
  constexpr int m = 64;
  constexpr int n = 64;
  constexpr int k = 64;

  std::cout << "m=" << m << ", n=" << n << ", k=" << k << '\n';
  std::cout << "Initializing host data...\n";

  std::vector<float> h_a(m * k);
  std::vector<float> h_b(k * n);
  std::vector<float> h_c(m * n, 0.0f);
  std::vector<float> h_ref(m * n, 0.0f);

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < k; ++col) {
      h_a[row * k + col] = static_cast<float>((row + col) % 7 - 3);
    }
  }

  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      h_b[row * n + col] = static_cast<float>((row * 3 + col) % 5 - 2);
    }
  }

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int i = 0; i < k; ++i) {
        acc += h_a[row * k + i] * h_b[i * n + col];
      }
      h_ref[row * n + col] = acc;
    }
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  std::cout << "Allocating device memory...\n";
  check_cuda(cudaMalloc(&d_a, h_a.size() * sizeof(float)), "cudaMalloc(d_a)");
  check_cuda(cudaMalloc(&d_b, h_b.size() * sizeof(float)), "cudaMalloc(d_b)");
  check_cuda(cudaMalloc(&d_c, h_c.size() * sizeof(float)), "cudaMalloc(d_c)");

  std::cout << "Copying data to device...\n";
  check_cuda(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy H2D A");
  check_cuda(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy H2D B");

  std::cout << "Launching kernel...\n";
  dim3 block(kBlockSize, kBlockSize);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);

  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  check_cuda(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H C");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < h_c.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(h_c[i] - h_ref[i]));
  }

  std::cout << "max_abs_diff=" << max_abs_diff << '\n';
  if (max_abs_diff > 1e-4f) {
    std::cerr << "verification failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "verification passed\n";
  return EXIT_SUCCESS;
}
