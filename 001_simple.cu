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

  float *h_a = new float[N];
  float *h_b = new float[N];
  float *h_c = new float[N];
  for (std::size_t i = 0; i < N; i++)
  {
    h_a[i] = (float)rand() / RAND_MAX;
    h_b[i] = (float)rand() / RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);

  const int block_size = 64;
  const int grid_size = (N + block_size - 1) / block_size;
  std::cout << "Grid size: " << grid_size << ". Block size: " << block_size << "." << std::endl;
  add_vector<<<grid_size, block_size>>>(d_a, d_b, d_c, 1.9f, N);
  cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (std::size_t i = 0; i < N; i++)
  {
    assert(h_c[i] == 1.9f * h_a[i] + h_b[i]);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}