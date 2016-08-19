#include <iostream>
#include <iterator>

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

constexpr size_t N = 3;
using Vector = float[N];

int main() {
  Vector a = { 1, 2, 3 };
  Vector b = { 5, 6, 8 };
  Vector c;

  // The input read-only buffers for OpenCL on default context
  cl::Buffer buffer_a { std::begin(a), std::end(a), true};
  cl::Buffer buffer_b { std::begin(b), std::end(b), true};

  // The output buffer for OpenCL on default context
  cl::Buffer buffer_c { CL_MEM_WRITE_ONLY, sizeof(c) };

  // Construct an OpenCL program from the source file
  const char kernel_source[] = R"(
__kernel void vector_add(const __global float *a,
                         const __global float *b,
                         __global float *c) {
  c[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];
}
)";

  // Compile and build the program
  cl::Program p { kernel_source, true };
  // Create the kernel functor taking 3 buffers as parameter
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> k { p, "vector_add" };

  // Call the kernel with N work-items on default command queue
  k(cl::EnqueueArgs(cl::NDRange(N)), buffer_a, buffer_b, buffer_c);

  // Get the output data from the accelerator
  cl::copy(buffer_c, std::begin(c), std::end(c));

  std::cout << std::endl << "Result:" << std::endl;
  for(auto e : c)
    std::cout << e << " ";
  std::cout << std::endl;
}
