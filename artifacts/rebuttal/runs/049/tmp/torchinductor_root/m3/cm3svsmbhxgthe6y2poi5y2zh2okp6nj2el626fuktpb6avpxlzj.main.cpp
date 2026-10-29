
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <c10/util/generic_math.h>

#if defined(_MSC_VER)
#  define EXTERN_DLL_EXPORT extern "C" __declspec(dllexport)
#else
#  define EXTERN_DLL_EXPORT extern "C"
#endif

EXTERN_DLL_EXPORT int8_t guard(int64_t *int_values, double *float_values) {
  int64_t L_x_stride_0_ = int_values[0], L_x_size_1_ = int_values[1], L_values_size_0_ = int_values[2], L_x_size_0_ = int_values[3];

  return (L_x_stride_0_ == L_x_size_1_) && (L_values_size_0_ == L_x_size_0_) && (L_x_size_1_*L_x_size_0_ <= 2147483647L) && ((2L <= L_x_size_0_) & (L_x_size_0_ <= 2147483647L)) && (2L <= L_x_size_1_);
}
