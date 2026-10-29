
#include <cstdint>
#include <immintrin.h>

extern "C" __m512bh __avx512_bf16_chk_kernel(__m512 a, __m512 b) {
    return _mm512_cvtne2ps_pbh(a, b);
}
