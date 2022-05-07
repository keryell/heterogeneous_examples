// Use plain OpenCL code in SYCL

#include <cassert>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include <CL/opencl.h>

constexpr int size = 4;

auto check_error(auto&& function) {
  cl_int err;
  auto ret = function(&err);
  if (err != CL_SUCCESS)
    std::exit(err);
  return ret;
};

int main() {
  sycl::buffer<int> a { size };
  sycl::buffer<int> b { size };
  sycl::buffer<int> c { size };

  {
    sycl::host_accessor a_a { a };
    sycl::host_accessor a_b { b };
    for (int i = 0; i < size; ++i) {
      a_a[i] = i;
      a_b[i] = i + 42;
    }
  }

  sycl::queue q;
  std::array kernel_source{R"(
      __kernel void vector_add(const __global float *a,
                               const __global float *b,
                               __global float *c) {
        c[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];
      }
      )"};
  cl_context oc = sycl::get_native<sycl::backend::opencl>(q.get_context());
  auto program = check_error([&](auto err) {
    return clCreateProgramWithSource(oc, kernel_source.size(),
                                     kernel_source.data(), nullptr, err);
  });
  check_error([&](auto err) {
    return (*err =
                clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr));
  });
  sycl::kernel k = sycl::make_kernel<sycl::backend::opencl>(
      check_error(
          [&](auto err) { return clCreateKernel(program, "vector_add", err); }),
      q.get_context());

  q.submit([&](sycl::handler& cgh) {
    cgh.set_args(sycl::accessor { a, cgh, sycl::read_only },
                 sycl::accessor { b, cgh, sycl::read_only },
                 sycl::accessor { c, cgh, sycl::write_only, sycl::no_init });
    cgh.parallel_for(size, k);
  });

  {
    sycl::host_accessor a_a { a };
    sycl::host_accessor a_b { b };
    sycl::host_accessor a_c { c };
    for (int i = 0; i < size; ++i)
      assert(a_c[i] == a_a[i] + a_b[i]);
  }
}
