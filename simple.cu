#include <iostream>
#include <cuda_runtime.h>
#include <random>

float *random_floats(std::size_t N)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0, 1);

  float *result = new float[N];
  for (std::size_t i = 0; i < N; i++)
  {
    result[i] = dist(gen);
  }
  return result;
}

void print_array(float *arr, int N)
{
  std::cout << "[";
  for (std::size_t i = 0; i < N; i++)
  {
    std::cout << arr[i];
    if (i == N - 1)
    {
      std::cout << "]" << std::endl;
    }
    else
    {
      std::cout << ", ";
    }
  }
}

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
  const int block_size = 4;

  float *h_a = random_floats(N);
  float *h_b = random_floats(N);
  float *h_c = new float[N];
  print_array(h_a, N);
  print_array(h_b, N);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);

  int grid_size = (N + block_size - 1) / block_size;
  std::cout << "Grid size: " << grid_size << ". Block size: " << block_size << std::endl;
  add_vector<<<grid_size, block_size>>>(d_a, d_b, d_c, 1.9f, N);

  cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  print_array(h_c, N);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}