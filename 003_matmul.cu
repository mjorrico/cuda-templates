#include <iostream>
#include <cassert>

__global__ void matmul(int *a, int *b, int *c, int n)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ((row < n) && (col < n))
  {
    for (std::size_t k = 0; k < n; k++)
    {
      c[row * n + col] += a[row * n + k] * b[k * n + col];
    }
  }
}

void validate(int *a, int *b, int *c, int n)
{
  int local_value;
  for (std::size_t i = 0; i < n; i++)
  {
    for (std::size_t j = 0; j < n; j++)
    {
      local_value = 0;
      for (std::size_t k = 0; k < n; k++)
      {
        local_value += a[i * n + k] * b[k * n + j];
      }
      assert(local_value == c[i * n + j]);
    }
  }
  std::cout << "Validation completed." << std::endl;
}

int main()
{
  // Matrix size of 1024 x 1024
  int n = 1 << 10;
  std::size_t bytes = n * n * sizeof(int);

  // Host pointers
  int *h_a, *h_b, *h_c;

  // Allocate host memory
  h_a = new int[n * n]{};
  h_b = new int[n * n]{};
  h_c = new int[n * n]{};

  // Initialize matrices
  srand(time(NULL));
  for (std::size_t i = 0; i < n * n; i++)
  {
    h_a[i] = rand() % 5;
    h_b[i] = rand() % 5;
  }

  // Device pointers
  int *d_a, *d_b, *d_c;

  // Allocate device memory
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);

  // CUDA threads config
  int BLOCKSIZE = 16;
  int GRIDSIZE = (n + BLOCKSIZE - 1) / BLOCKSIZE;
  dim3 grid(GRIDSIZE, GRIDSIZE);
  dim3 threads(BLOCKSIZE, BLOCKSIZE);

  // Kernel launch
  matmul<<<grid, threads>>>(d_a, d_b, d_c, n);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  validate(h_a, h_b, h_c, n);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
}