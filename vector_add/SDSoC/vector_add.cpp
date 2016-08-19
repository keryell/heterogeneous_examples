#include <cstdlib>

constexpr size_t N = 3;

void vector_add(const float a[N],
                const float b[N],
                float c[N]) {
#pragma HLS PIPELINE II=1
  for (size_t i = 0; i < N; ++i)
    c[i] = a[i] + b[i];
}
