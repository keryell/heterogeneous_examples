/** Simple streaming example

 */

#include <boost/compute.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

// 1 Mi elements
#define N (2<<20)
#define TYPE int
// To inline constants in kernel source code string
#define K_TYPE BOOST_PP_STRINGIZE(TYPE)
#define K_N BOOST_PP_STRINGIZE(N)

int main() {
  std::vector<TYPE> input(N);
  std::vector<TYPE> output(N);

  // Create the OpenCL context to attach resources on the device
  auto context = boost::compute::system::default_context();
  // Create the OpenCL command queue to control the device
  auto command_queue = boost::compute::system::default_queue();

  // The input buffer for OpenCL
  boost::compute::buffer ib { context, N*sizeof(TYPE), CL_MEM_READ_ONLY };

  // The output buffer for OpenCL
  boost::compute::buffer ob { context, N*sizeof(TYPE), CL_MEM_WRITE_ONLY };

  // Construct an OpenCL program from the source string
  auto program = boost::compute::program::create_with_source(R"(
    __kernel void
    simple_stream(const __global )" K_TYPE R"( *ib,
                  __global )" K_TYPE R"( *ob) {
          for (int i = 0; i != )" K_N R"(; ++i)
            ob[i] = ib[i] + 1;
    }
      )", boost::compute::system::default_context());

  program.build();

  auto kernel = boost::compute::kernel { program, "simple_stream" };

  // Initalize host data with increasing numbers starting at 0
  std::iota(input.begin(), input.end(), 0);

  // Send the input data to the accelerator
  command_queue.enqueue_write_buffer(ib, 0 /* Offset */,
                                     N*sizeof(TYPE), input.data());

  kernel.set_args(ib, ob);

  boost::compute::extents<1> offset { 0 };
  // Use 1 work-group with 1 work-item
  boost::compute::extents<1> global { 1 };
  boost::compute::extents<1> local { 1 };
  // Launch the kernel
  command_queue.enqueue_nd_range_kernel(kernel, offset, global, local);

  // Get the output data from the accelerator
  command_queue.enqueue_read_buffer(ob, 0 /* Offset */,
                                    N*sizeof(TYPE), output.data());

  for (std::size_t i = 0; i != N; ++i)
    if (output[i] != input[i] + 1)
      throw std::runtime_error { "Wrong result" };
}
