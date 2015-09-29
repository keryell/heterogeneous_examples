#include <iostream>

constexpr size_t N = 3;
using Vector = float[N];

int main() {
  Vector a = { 1, 2, 3 };
  Vector b = { 5, 6, 8 };
  Vector c;

#pragma omp target parallel for map(to: a[:], b[:]) \
                                map(from: c[:])
  for (auto i = 0; i < N; ++i)
    c[i] = a[i] + b[i];

  std::cout << std::endl << "Result:" << std::endl;
  for(auto e : c)
    std::cout << e << " ";
  std::cout << std::endl;
}
