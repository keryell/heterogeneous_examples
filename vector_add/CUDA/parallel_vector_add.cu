/*
  @author Ken O'Brien <kennetho@xilinx.com>

  Parallel vector addition for CUDA devices.
*/

#include <iostream>
#include <stdexcept>

constexpr size_t N = 3;
using Vector = float[N];


__global__ void vector_add(const float *a, const float *b, float *c) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx < N)
    c[idx] = a[idx] + b[idx];
}

void checkError(cudaError_t err) {
  if(err != cudaSuccess) {
    throw std::domain_error("CUDA ERROR: "
     + std::string { cudaGetErrorString(err) });
  }
}


int main() {
  Vector a = {1, 2, 3};
  Vector b = {5, 6, 8};
  Vector c;

  float *d_a, *d_b, *d_c;

  checkError(cudaMalloc((void**) &d_a, N*sizeof(*d_a)));
  checkError(cudaMalloc((void**) &d_b, N*sizeof(*d_b)));
  checkError(cudaMalloc((void**) &d_c, N*sizeof(*d_c)));

  checkError(cudaMemcpy(d_a, a, N*sizeof(*a), cudaMemcpyHostToDevice));
  checkError(cudaMemcpy(d_b, b, N*sizeof(*b), cudaMemcpyHostToDevice));

  vector_add<<<1, N>>>(d_a, d_b, d_c);

  checkError(cudaMemcpy(c, d_c, N*sizeof(*c), cudaMemcpyDeviceToHost));

  std::cout << std::endl << "Result: " << std::endl;
  for(auto e: c)
    std::cout << e << " ";
  std::cout << std::endl;

  checkError(cudaFree(d_a));
  checkError(cudaFree(d_b));
  checkError(cudaFree(d_c));

  return 0;
}
