#include <iostream>
#include <cassert>
#include <random>

__global__ void add_vector(float *d_a, float *d_b, float *d_c, float p, float N)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    d_c[idx] = p * d_a[idx] + d_b[idx];
  }
}

int main()
{
  const int N = 10;
  int id = cudaGetDevice(&id);

  float *a, *b, *c;
  cudaMallocManaged(&a, N * sizeof(float));
  cudaMallocManaged(&b, N * sizeof(float));
  cudaMallocManaged(&c, N * sizeof(float));

  for (std::size_t i = 0; i < N; i++)
  {
    a[i] = (float)rand() / RAND_MAX;
    b[i] = (float)rand() / RAND_MAX;
  }

  // Uncomment cudaMemPrefetchAsync to prefetch from host to device
  cudaMemPrefetchAsync(a, N * sizeof(float), id);
  cudaMemPrefetchAsync(b, N * sizeof(float), id);

  const int block_size = 64;
  const int grid_size = (N + block_size - 1) / block_size;
  std::cout << "Device ID: " << id << ". Grid size: " << grid_size << ". Block size: " << block_size << "." << std::endl;
  add_vector<<<grid_size, block_size>>>(a, b, c, 1.9f, N);

  // Wait for all previous operations before using values
  // We need this because we don't get the implicit synchronization of
  // cudaMemcpy like in the original example
  cudaDeviceSynchronize();

  // Uncomment cudaMemPrefetchAsync to prefetch from host to device
  cudaMemPrefetchAsync(c, N * sizeof(float), cudaCpuDeviceId);

  for (std::size_t i = 0; i < N; i++)
  {
    assert(c[i] == 1.9f * a[i] + b[i]);
  }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return 0;
}