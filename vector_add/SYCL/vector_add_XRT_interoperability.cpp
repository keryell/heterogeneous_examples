// ./build-Release/bin/clang++ -fsycl -fsycl-targets=fpga64_hls_hw_emu -std=c++20 -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lOpenCL -luuid -lxrt_coreutil -g3 test_interop.cpp -o a.out

#include <sycl/ext/xilinx/xrt.hpp>
#include <sycl/sycl.hpp>
#include <xrt/xrt_kernel.h>
#include <xrt.h>

class Kernel;

constexpr auto curr_be = sycl::backend::xrt;

void kernel_bundle_test() {
  sycl::queue q;
  sycl::context ctx = q.get_context();
  sycl::kernel_bundle b = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  sycl::kernel k = b.get_kernel(sycl::get_kernel_id<Kernel>());

  sycl::buffer<int, 1> buf(4);
  q.submit([&](sycl::handler &h) {
    sycl::accessor acc(buf, h, sycl::write_only);
    h.set_arg(0, acc);
    h.single_task(k);
  });
  exit(1);
}

void sycl_interop_test() {
  xrt::device xdev(0);
  sycl::device sdev(sycl::make_device<curr_be>(xdev));
  xrt::device xdev2(sycl::get_native<curr_be>(sdev));
  sycl::device dev(sycl::make_device<curr_be>(xdev));

  sycl::queue q(dev);
  sycl::context ctx = q.get_context();

  sycl::kernel_bundle b =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  sycl::kernel korig = b.get_kernel(sycl::get_kernel_id<Kernel>());

  auto xkern = sycl::get_native<curr_be>(korig);
  auto k = sycl::make_kernel<curr_be>(xkern, ctx);

  sycl::buffer<int, 1> buf(4);
  q.submit([&](sycl::handler &h) {
    sycl::accessor acc(buf, h, sycl::write_only);
    h.set_arg(0, acc);
    h.single_task(k);
  });

  {
    sycl::host_accessor a{buf};
    assert(a[0] == 1);
  }

  // if (0)
  //   q.submit([&](sycl::handler &cgh) {
  //     sycl::accessor acc(buf, cgh, sycl::write_only);
  //     cgh.single_task<Kernel>([=]() { acc[0] = 1; });
  //   });
}

int main() {
  sycl::queue q;
  xrt::device xdev = sycl::get_native<sycl::backend::xrt>(q.get_device());
  xrt::kernel xk(xdev, xdev.load_xclbin("vadd.hw_emu.xclbin"), "vadd");
  sycl::kernel k(sycl::make_kernel<sycl::backend::xrt>(xk, q.get_context()));
  int size = 4;

  sycl::buffer<int, 1> a(size);
  sycl::buffer<int, 1> b(size);
  sycl::buffer<int, 1> c(size);

  {
    sycl::host_accessor a_a(a);
    sycl::host_accessor a_b(b);
    for (int i = 0; i < size; i++) {
      a_a[i] = i;
      a_b[i] = i + 1;
    }
  }

  q.submit([&](sycl::handler &cgh) {
    cgh.set_args(sycl::accessor{a, cgh, sycl::read_only},
                 sycl::accessor{b, cgh, sycl::read_only},
                 sycl::accessor{c, cgh, sycl::write_only}, size);
    cgh.single_task(k);
  });
  {
    sycl::host_accessor a_a(a);
    sycl::host_accessor a_b(b);
    sycl::host_accessor a_c(c);
    for (int i = 0; i < size; i++) {
      int res = a_a[i] + a_b[i];
      int val = a_c[i];
      assert(val == res);
    }
  }
}
