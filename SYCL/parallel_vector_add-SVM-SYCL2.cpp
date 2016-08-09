/* RUN: %{execute}%s | %{filecheck} %s
   CHECK: Result:
   CHECK-NEXT: 6 8 11
*/
#include <CL/sycl.hpp>
#include <iostream>

constexpr size_t N = 3;
using Vector = float[N];

int main() {
  Vector a = { 1, 2, 3 };
  Vector b = { 5, 6, 8 };
  Vector c;

  // Enqueue a synchronous parallel kernel
  parallel_for<>(N, [=] (auto index) { c[index] = a[index] + b[index]; });

  std::cout << std::endl << "Result:" << std::endl;
  for(auto e : c)
    std::cout << e << " ";
  std::cout << std::endl;
}
