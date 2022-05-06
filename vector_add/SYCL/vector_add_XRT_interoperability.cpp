// Why the following #include order?
#include <sycl/ext/xilinx/xrt.hpp>
#include <sycl/sycl.hpp>
#include <xrt/xrt_kernel.h>
#include <xrt.h>

constexpr int size = 4;

int main() {
  sycl::queue q;
  xrt::device xdev = sycl::get_native<sycl::backend::xrt>(q.get_device());
  xrt::kernel xk { xdev, xdev.load_xclbin("vadd.hw_emu.xclbin"), "vadd" };
  sycl::kernel k { sycl::make_kernel<sycl::backend::xrt>(xk, q.get_context()) };
  sycl::buffer<int> a {size};
  sycl::buffer<int> b {size};
  sycl::buffer<int> c {size};

  {
    sycl::host_accessor a_a { a };
    sycl::host_accessor a_b { b };
    for (int i = 0; i < size; ++i) {
      a_a[i] = i;
      a_b[i] = i + 1;
    }
  }

  q.submit([&](sycl::handler& cgh) {
    cgh.set_args(sycl::accessor { a, cgh, sycl::read_only },
                 sycl::accessor { b, cgh, sycl::read_only },
                 sycl::accessor { c, cgh, sycl::write_only }, size);
    cgh.single_task(k);
  });
  {
    sycl::host_accessor a_a { a };
    sycl::host_accessor a_b { b };
    sycl::host_accessor a_c { c };
    for (int i = 0; i < size; ++i) {
      int res = a_a[i] + a_b[i];
      int val = a_c[i];
      assert(val == res);
    }
  }
}
