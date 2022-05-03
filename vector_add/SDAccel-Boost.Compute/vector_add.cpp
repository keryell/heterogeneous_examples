/* Simple OpenCL vector addition using Boost.Compute C++ host API and
   precompiled kernel */

#include <boost/compute.hpp>
#include <iostream>
#include <iterator>

constexpr size_t N = 3;
using Vector = float[N];

int main() {
  Vector a = { 1, 2, 3 };
  Vector b = { 5, 6, 8 };
  Vector c;

  /* In the following, you can use the following environment to select
     the device to be chosen at runtime
     BOOST_COMPUTE_DEFAULT_DEVICE
     BOOST_COMPUTE_DEFAULT_DEVICE_TYPE
     BOOST_COMPUTE_DEFAULT_PLATFORM
     BOOST_COMPUTE_DEFAULT_VENDOR

     for example doing in bash
     export BOOST_COMPUTE_DEFAULT_VENDOR=Xilinx
     will probably select for execution a Xilinx FPGA on the machine
  */

  // Create the OpenCL context to attach resources on the device
  auto context = boost::compute::system::default_context();
  // Create the OpenCL command queue to control the device
  auto command_queue = boost::compute::system::default_queue();

  // The input buffers for OpenCL
  boost::compute::buffer buffer_a { context, sizeof(a), CL_MEM_READ_ONLY };
  boost::compute::buffer buffer_b { context, sizeof(b), CL_MEM_READ_ONLY };

  // The output buffer for OpenCL
  boost::compute::buffer buffer_c { context, sizeof(c), CL_MEM_WRITE_ONLY };

  // Construct an OpenCL program from the precompiled kernel file
  auto program =
    boost::compute::program::create_with_binary_file("vector_add.xclbin",
                                                     context);
  program.build();

  auto kernel = boost::compute::kernel { program, "vector_add" };

  // Send the input data to the accelerator
  command_queue.enqueue_write_buffer(buffer_a, 0 /* Offset */,
                                      sizeof(a), &a[0]);
  command_queue.enqueue_write_buffer(buffer_b, 0 /* Offset */,
                                      sizeof(b), &b[0]);

  kernel.set_args(buffer_a, buffer_b, buffer_c);

  boost::compute::extents<1> offset { 0 };
  boost::compute::extents<1> global { N };
  // Use only 1 CU
  boost::compute::extents<1> local { N };
  // Launch the kernel
  command_queue.enqueue_nd_range_kernel(kernel, offset, global, local);

  // Get the output data from the accelerator
  command_queue.enqueue_read_buffer(buffer_c, 0 /* Offset */,
                                    sizeof(c), &c[0]);

  std::cout << std::endl << "Result:" << std::endl;
  for(auto e : c)
    std::cout << e << " ";
  std::cout << std::endl;
}
