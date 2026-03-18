/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_max_star_horizontal_16i
 *
 * \b Deprecation
 *
 * This kernel is deprecated, no replacement has been identified.
 *
 * \b Overview
 *
 * Computes the maximum of each consecutive pair of 16-bit integers in the input
 * vector. For every two adjacent elements (src0[2k], src0[2k+1]), the larger value is
 * written to the output, producing num_points/2 results. This is a simplified max*
 * (max-star) operation used in log-MAP decoding for communications applications.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_max_star_horizontal_16i(short* target, const short* src0, unsigned int
 * num_points); \endcode
 *
 * \b Inputs
 * \li src0: The input vector of 16-bit integers. Must contain num_points elements, where
 * num_points is even.
 * \li num_points: The number of 16-bit integer data points in the input vector (must be
 * even).
 *
 * \b Outputs
 * \li target: The output vector containing num_points/2 maximum values, one per
 * consecutive input pair.
 *
 * \b Example
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 *
 * int main() {
 *     unsigned int num_points = 8;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     short* src0 = (short*)volk_malloc(sizeof(short) * num_points, alignment);
 *     short* target = (short*)volk_malloc(sizeof(short) * (num_points / 2), alignment);
 *
 *     // Initialize with pairs of values
 *     src0[0] = 10;  src0[1] = 5;
 *     src0[2] = -3;  src0[3] = 7;
 *     src0[4] = 15;  src0[5] = 15;
 *     src0[6] = -8;  src0[7] = -2;
 *
 *     volk_16i_max_star_horizontal_16i(target, src0, num_points);
 *
 *     // Expected: max of each pair
 *     // target[0] = max(10, 5)   = 10
 *     // target[1] = max(-3, 7)   = 7
 *     // target[2] = max(15, 15)  = 15
 *     // target[3] = max(-8, -2)  = -2
 *     for (unsigned int i = 0; i < num_points / 2; i++) {
 *         printf("target[%u] = %d\n", i, target[i]);
 *     }
 *
 *     volk_free(src0);
 *     volk_free(target);
 *     return 0;
 * }
 * \endcode
 */

#ifndef INCLUDED_volk_16i_max_star_horizontal_16i_u_H
#define INCLUDED_volk_16i_max_star_horizontal_16i_u_H

#include <volk/volk_common.h>

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_16i_max_star_horizontal_16i_generic(int16_t* target,
                                                            const int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    for (i = 0; i < bound; i += 2) {
        target[i >> 1] = (src0[i] > src0[i + 1]) ? src0[i] : src0[i + 1];
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>

static inline void volk_16i_max_star_horizontal_16i_u_sse2(int16_t* target,
                                                            const int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 16;
    unsigned int i;

    for (i = 0; i < sse_iters; i++) {
        __m128i v0 = _mm_loadu_si128((const __m128i*)src0);
        __m128i v1 = _mm_loadu_si128((const __m128i*)(src0 + 8));

        /* Sign-extend even/odd int16 elements to int32 via shift */
        __m128i evens0 = _mm_srai_epi32(_mm_slli_epi32(v0, 16), 16);
        __m128i odds0 = _mm_srai_epi32(v0, 16);
        __m128i evens1 = _mm_srai_epi32(_mm_slli_epi32(v1, 16), 16);
        __m128i odds1 = _mm_srai_epi32(v1, 16);

        /* 32-bit max via compare + select (SSE2 lacks _mm_max_epi32) */
        __m128i gt0 = _mm_cmpgt_epi32(evens0, odds0);
        __m128i max0 =
            _mm_or_si128(_mm_and_si128(gt0, evens0), _mm_andnot_si128(gt0, odds0));
        __m128i gt1 = _mm_cmpgt_epi32(evens1, odds1);
        __m128i max1 =
            _mm_or_si128(_mm_and_si128(gt1, evens1), _mm_andnot_si128(gt1, odds1));

        /* Pack 32-bit results to 16-bit (values already in int16 range) */
        _mm_storeu_si128((__m128i*)target, _mm_packs_epi32(max0, max1));
        src0 += 16;
        target += 8;
    }

    volk_16i_max_star_horizontal_16i_generic(
        target, src0, num_points - sse_iters * 16);
}

#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_max_star_horizontal_16i_u_ssse3(int16_t* target,
                                                             const int16_t* src0,
                                                             unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    static const uint8_t shuf_even_lo[16] = {
        0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };
    static const uint8_t shuf_even_hi[16] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d
    };
    static const uint8_t shuf_odd_lo[16] = {
        0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };
    static const uint8_t shuf_odd_hi[16] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f
    };

    __m128i xmm_even_lo = _mm_load_si128((const __m128i*)shuf_even_lo);
    __m128i xmm_even_hi = _mm_load_si128((const __m128i*)shuf_even_hi);
    __m128i xmm_odd_lo = _mm_load_si128((const __m128i*)shuf_odd_lo);
    __m128i xmm_odd_hi = _mm_load_si128((const __m128i*)shuf_odd_hi);

    __m128i* p_target = (__m128i*)target;
    const __m128i* p_src0 = (const __m128i*)src0;

    int bound = num_bytes >> 5;
    int intermediate = (num_bytes >> 4) & 1;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    for (i = 0; i < bound; ++i) {
        __m128i xmm0 = _mm_loadu_si128(&p_src0[0]);
        __m128i xmm1 = _mm_loadu_si128(&p_src0[1]);
        p_src0 += 2;

        __m128i evens = _mm_or_si128(
            _mm_shuffle_epi8(xmm0, xmm_even_lo),
            _mm_shuffle_epi8(xmm1, xmm_even_hi));
        __m128i odds = _mm_or_si128(
            _mm_shuffle_epi8(xmm0, xmm_odd_lo),
            _mm_shuffle_epi8(xmm1, xmm_odd_hi));

        _mm_storeu_si128(p_target, _mm_max_epi16(evens, odds));
        p_target += 1;
    }

    if (intermediate) {
        __m128i xmm0 = _mm_loadu_si128(p_src0);
        p_src0 += 1;

        __m128i evens = _mm_shuffle_epi8(xmm0, xmm_even_lo);
        __m128i odds = _mm_shuffle_epi8(xmm0, xmm_odd_lo);
        __m128i result = _mm_max_epi16(evens, odds);

        _mm_storel_pd((double*)p_target, bit128_p(&result)->double_vec);
        p_target = (__m128i*)((int8_t*)p_target + 8);
    }

    int processed = (bound << 4) + (intermediate << 3);
    volk_16i_max_star_horizontal_16i_generic(
        target + (processed >> 1), src0 + processed, leftovers);
}

#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2

#include <immintrin.h>

static inline void volk_16i_max_star_horizontal_16i_u_avx2(int16_t* target,
                                                            const int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 32;
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)src0);
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(src0 + 16));

        __m256i evens0 = _mm256_srai_epi32(_mm256_slli_epi32(v0, 16), 16);
        __m256i odds0 = _mm256_srai_epi32(v0, 16);
        __m256i max0 = _mm256_max_epi32(evens0, odds0);

        __m256i evens1 = _mm256_srai_epi32(_mm256_slli_epi32(v1, 16), 16);
        __m256i odds1 = _mm256_srai_epi32(v1, 16);
        __m256i max1 = _mm256_max_epi32(evens1, odds1);

        /* Pack and fix cross-lane artifact from _mm256_packs_epi32 */
        __m256i packed = _mm256_packs_epi32(max0, max1);
        packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));

        _mm256_storeu_si256((__m256i*)target, packed);
        src0 += 32;
        target += 16;
    }

    volk_16i_max_star_horizontal_16i_generic(
        target, src0, num_points - avx_iters * 32);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW

#include <immintrin.h>

static inline void volk_16i_max_star_horizontal_16i_u_avx512bw(int16_t* target,
                                                                const int16_t* src0,
                                                                unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 64;
    unsigned int i;

    const __m512i fixup = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    for (i = 0; i < avx512_iters; i++) {
        __m512i v0 = _mm512_loadu_si512((const __m512i*)src0);
        __m512i v1 = _mm512_loadu_si512((const __m512i*)(src0 + 32));

        __m512i evens0 = _mm512_srai_epi32(_mm512_slli_epi32(v0, 16), 16);
        __m512i odds0 = _mm512_srai_epi32(v0, 16);
        __m512i max0 = _mm512_max_epi32(evens0, odds0);

        __m512i evens1 = _mm512_srai_epi32(_mm512_slli_epi32(v1, 16), 16);
        __m512i odds1 = _mm512_srai_epi32(v1, 16);
        __m512i max1 = _mm512_max_epi32(evens1, odds1);

        /* Pack and fix cross-lane artifact from _mm512_packs_epi32 */
        __m512i packed = _mm512_packs_epi32(max0, max1);
        packed = _mm512_permutexvar_epi64(fixup, packed);

        _mm512_storeu_si512((__m512i*)target, packed);
        src0 += 64;
        target += 32;
    }

    volk_16i_max_star_horizontal_16i_generic(
        target, src0, num_points - avx512_iters * 64);
}

#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_NEON

#include <arm_neon.h>
static inline void volk_16i_max_star_horizontal_16i_neon(int16_t* target,
                                                         const int16_t* src0,
                                                         unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 16;
    unsigned number;
    int16x8x2_t input_vec;
    int16x8_t max_vec;
    for (number = 0; number < eighth_points; ++number) {
        input_vec = vld2q_s16(src0);
        max_vec = vmaxq_s16(input_vec.val[0], input_vec.val[1]);
        vst1q_s16(target, max_vec);
        src0 += 16;
        target += 8;
    }
    for (number = 0; number < num_points % 16; number += 2) {
        target[number >> 1] = (src0[number] > src0[number + 1])
                                  ? src0[number]
                                  : src0[number + 1];
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV7
extern void volk_16i_max_star_horizontal_16i_neonasm(int16_t* target,
                                                     const int16_t* src0,
                                                     unsigned int num_points);
#endif /* LV_HAVE_NEONV7 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_max_star_horizontal_16i_rvv(int16_t* target,
                                                         const int16_t* src0,
                                                         unsigned int num_points)
{
    /* Process consecutive pairs: target[i] = max(src0[2i], src0[2i+1]) */
    size_t n = (size_t)num_points / 2;
    for (size_t vl; n > 0; n -= vl, src0 += vl * 2, target += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        /* Load interleaved pairs with stride-2 segment load */
        vint16m4_t v0 = __riscv_vlse16_v_i16m4(src0, 2 * sizeof(int16_t), vl);
        vint16m4_t v1 = __riscv_vlse16_v_i16m4(src0 + 1, 2 * sizeof(int16_t), vl);
        vint16m4_t result = __riscv_vmax(v0, v1, vl);
        __riscv_vse16(target, result, vl);
    }
}
#endif /* LV_HAVE_RVV */

#endif /* INCLUDED_volk_16i_max_star_horizontal_16i_u_H */

#ifndef INCLUDED_volk_16i_max_star_horizontal_16i_a_H
#define INCLUDED_volk_16i_max_star_horizontal_16i_a_H

#include <volk/volk_common.h>

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>

static inline void volk_16i_max_star_horizontal_16i_a_sse2(int16_t* target,
                                                            const int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 16;
    unsigned int i;

    for (i = 0; i < sse_iters; i++) {
        __m128i v0 = _mm_load_si128((const __m128i*)src0);
        __m128i v1 = _mm_load_si128((const __m128i*)(src0 + 8));

        /* Sign-extend even/odd int16 elements to int32 via shift */
        __m128i evens0 = _mm_srai_epi32(_mm_slli_epi32(v0, 16), 16);
        __m128i odds0 = _mm_srai_epi32(v0, 16);
        __m128i evens1 = _mm_srai_epi32(_mm_slli_epi32(v1, 16), 16);
        __m128i odds1 = _mm_srai_epi32(v1, 16);

        /* 32-bit max via compare + select (SSE2 lacks _mm_max_epi32) */
        __m128i gt0 = _mm_cmpgt_epi32(evens0, odds0);
        __m128i max0 =
            _mm_or_si128(_mm_and_si128(gt0, evens0), _mm_andnot_si128(gt0, odds0));
        __m128i gt1 = _mm_cmpgt_epi32(evens1, odds1);
        __m128i max1 =
            _mm_or_si128(_mm_and_si128(gt1, evens1), _mm_andnot_si128(gt1, odds1));

        /* Pack 32-bit results to 16-bit (values already in int16 range) */
        _mm_store_si128((__m128i*)target, _mm_packs_epi32(max0, max1));
        src0 += 16;
        target += 8;
    }

    volk_16i_max_star_horizontal_16i_generic(
        target, src0, num_points - sse_iters * 16);
}

#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_max_star_horizontal_16i_a_ssse3(int16_t* target,
                                                            const int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    /* Shuffle masks to deinterleave even/odd 16-bit elements */
    static const uint8_t shuf_even_lo[16] = {
        0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };
    static const uint8_t shuf_even_hi[16] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x0c, 0x0d
    };
    static const uint8_t shuf_odd_lo[16] = {
        0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    };
    static const uint8_t shuf_odd_hi[16] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f
    };

    __m128i xmm_even_lo = _mm_load_si128((const __m128i*)shuf_even_lo);
    __m128i xmm_even_hi = _mm_load_si128((const __m128i*)shuf_even_hi);
    __m128i xmm_odd_lo = _mm_load_si128((const __m128i*)shuf_odd_lo);
    __m128i xmm_odd_hi = _mm_load_si128((const __m128i*)shuf_odd_hi);

    __m128i* p_target = (__m128i*)target;
    const __m128i* p_src0 = (const __m128i*)src0;

    int bound = num_bytes >> 5;
    int intermediate = (num_bytes >> 4) & 1;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    for (i = 0; i < bound; ++i) {
        __m128i xmm0 = _mm_load_si128(&p_src0[0]);
        __m128i xmm1 = _mm_load_si128(&p_src0[1]);
        p_src0 += 2;

        __m128i evens = _mm_or_si128(
            _mm_shuffle_epi8(xmm0, xmm_even_lo),
            _mm_shuffle_epi8(xmm1, xmm_even_hi));
        __m128i odds = _mm_or_si128(
            _mm_shuffle_epi8(xmm0, xmm_odd_lo),
            _mm_shuffle_epi8(xmm1, xmm_odd_hi));

        _mm_store_si128(p_target, _mm_max_epi16(evens, odds));
        p_target += 1;
    }

    if (intermediate) {
        __m128i xmm0 = _mm_load_si128(p_src0);
        p_src0 += 1;

        __m128i evens = _mm_shuffle_epi8(xmm0, xmm_even_lo);
        __m128i odds = _mm_shuffle_epi8(xmm0, xmm_odd_lo);
        __m128i result = _mm_max_epi16(evens, odds);

        _mm_storel_pd((double*)p_target, bit128_p(&result)->double_vec);
        p_target = (__m128i*)((int8_t*)p_target + 8);
    }

    for (i = (bound << 4) + (intermediate << 3);
         i < (bound << 4) + (intermediate << 3) + leftovers;
         i += 2) {
        target[i >> 1] = (src0[i] > src0[i + 1]) ? src0[i] : src0[i + 1];
    }
}

#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2

#include <immintrin.h>

static inline void volk_16i_max_star_horizontal_16i_a_avx2(int16_t* target,
                                                            const int16_t* src0,
                                                            unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 32;
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        __m256i v0 = _mm256_load_si256((const __m256i*)src0);
        __m256i v1 = _mm256_load_si256((const __m256i*)(src0 + 16));

        __m256i evens0 = _mm256_srai_epi32(_mm256_slli_epi32(v0, 16), 16);
        __m256i odds0 = _mm256_srai_epi32(v0, 16);
        __m256i max0 = _mm256_max_epi32(evens0, odds0);

        __m256i evens1 = _mm256_srai_epi32(_mm256_slli_epi32(v1, 16), 16);
        __m256i odds1 = _mm256_srai_epi32(v1, 16);
        __m256i max1 = _mm256_max_epi32(evens1, odds1);

        /* Pack and fix cross-lane artifact from _mm256_packs_epi32 */
        __m256i packed = _mm256_packs_epi32(max0, max1);
        packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));

        _mm256_store_si256((__m256i*)target, packed);
        src0 += 32;
        target += 16;
    }

    volk_16i_max_star_horizontal_16i_generic(
        target, src0, num_points - avx_iters * 32);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW

#include <immintrin.h>

static inline void volk_16i_max_star_horizontal_16i_a_avx512bw(int16_t* target,
                                                                const int16_t* src0,
                                                                unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 64;
    unsigned int i;

    const __m512i fixup = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    for (i = 0; i < avx512_iters; i++) {
        __m512i v0 = _mm512_load_si512((const __m512i*)src0);
        __m512i v1 = _mm512_load_si512((const __m512i*)(src0 + 32));

        __m512i evens0 = _mm512_srai_epi32(_mm512_slli_epi32(v0, 16), 16);
        __m512i odds0 = _mm512_srai_epi32(v0, 16);
        __m512i max0 = _mm512_max_epi32(evens0, odds0);

        __m512i evens1 = _mm512_srai_epi32(_mm512_slli_epi32(v1, 16), 16);
        __m512i odds1 = _mm512_srai_epi32(v1, 16);
        __m512i max1 = _mm512_max_epi32(evens1, odds1);

        /* Pack and fix cross-lane artifact from _mm512_packs_epi32 */
        __m512i packed = _mm512_packs_epi32(max0, max1);
        packed = _mm512_permutexvar_epi64(fixup, packed);

        _mm512_store_si512((__m512i*)target, packed);
        src0 += 64;
        target += 32;
    }

    volk_16i_max_star_horizontal_16i_generic(
        target, src0, num_points - avx512_iters * 64);
}

#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_16i_max_star_horizontal_16i_a_H */
