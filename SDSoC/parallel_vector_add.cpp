#include <iostream>

constexpr size_t N = 3;

#pragma SDS data access_pattern(a:SEQUENTIAL, b:SEQUENTIAL, c:SEQUENTIAL)
void vector_add(const float a[N],
                const float b[N],
                float c[N]);

int main() {
  float a[N] = { 1, 2, 3 };
  float b[N] = { 5, 6, 8 };
  float c[N];

  vector_add(a, b, c);

  std::cout << std::endl << "Result:" << std::endl;
  for(auto e : c)
    std::cout << e << " ";
  std::cout << std::endl;
}
