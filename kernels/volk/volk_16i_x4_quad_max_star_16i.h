/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_x4_quad_max_star_16i
 *
 * \b Deprecation
 *
 * This kernel is deprecated, no replacement has been identified.
 *
 * \b Overview
 *
 * Computes the element-wise maximum across four 16-bit integer input vectors using a
 * tournament-style comparison. For each element index, the kernel first selects the
 * larger value from each pair (src0 vs src1, and src2 vs src3), then selects the larger
 * of those two results. This implements the max-log-MAP approximation of the max-star
 * operation used in turbo and Viterbi decoding.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_x4_quad_max_star_16i(short* target, const short* src0, const short* src1,
 * const short* src2, const short* src3, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: First input vector of 16-bit integers (short).
 * \li src1: Second input vector of 16-bit integers (short).
 * \li src2: Third input vector of 16-bit integers (short).
 * \li src3: Fourth input vector of 16-bit integers (short).
 * \li num_points: The number of elements in each input/output vector.
 *
 * \b Outputs
 * \li target: Output vector of 16-bit integers (short) containing
 * max(max(src0[i], src1[i]), max(src2[i], src3[i])) for each element.
 *
 * \b Example
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 *
 * int main() {
 *     unsigned int N = 16;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     short* src0 = (short*)volk_malloc(N * sizeof(short), alignment);
 *     short* src1 = (short*)volk_malloc(N * sizeof(short), alignment);
 *     short* src2 = (short*)volk_malloc(N * sizeof(short), alignment);
 *     short* src3 = (short*)volk_malloc(N * sizeof(short), alignment);
 *     short* target = (short*)volk_malloc(N * sizeof(short), alignment);
 *
 *     // Initialize with sample trellis metric values
 *     for (unsigned int i = 0; i < N; i++) {
 *         src0[i] = (short)(100 - 10 * (int)i);
 *         src1[i] = (short)(5 * (int)i - 30);
 *         src2[i] = (short)(50 + 3 * (int)i);
 *         src3[i] = (short)(80 - 7 * (int)i);
 *     }
 *
 *     volk_16i_x4_quad_max_star_16i(target, src0, src1, src2, src3, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *         printf("target[%u] = %d\n", i, target[i]);
 *     }
 *
 *     volk_free(src0);
 *     volk_free(src1);
 *     volk_free(src2);
 *     volk_free(src3);
 *     volk_free(target);
 *     return 0;
 * }
 * \endcode
 */

#ifndef INCLUDED_volk_16i_x4_quad_max_star_16i_u_H
#define INCLUDED_volk_16i_x4_quad_max_star_16i_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_16i_x4_quad_max_star_16i_generic(short* target,
                                                         const short* src0,
                                                         const short* src1,
                                                         const short* src2,
                                                         const short* src3,
                                                         unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    short temp0 = 0;
    short temp1 = 0;
    for (i = 0; i < bound; ++i) {
        temp0 = (src0[i] > src1[i]) ? src0[i] : src1[i];
        temp1 = (src2[i] > src3[i]) ? src2[i] : src3[i];
        target[i] = (temp0 > temp1) ? temp0 : temp1;
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_x4_quad_max_star_16i_u_sse2(short* target,
                                                         const short* src0,
                                                         const short* src1,
                                                         const short* src2,
                                                         const short* src3,
                                                         unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    int leftovers = num_points - eighthPoints * 8;

    const __m128i* p_src0 = (const __m128i*)src0;
    const __m128i* p_src1 = (const __m128i*)src1;
    const __m128i* p_src2 = (const __m128i*)src2;
    const __m128i* p_src3 = (const __m128i*)src3;
    __m128i* p_target = (__m128i*)target;

    for (unsigned int i = 0; i < eighthPoints; ++i) {
        __m128i xmm1 = _mm_loadu_si128(p_src0);
        __m128i xmm2 = _mm_loadu_si128(p_src1);
        __m128i xmm3 = _mm_loadu_si128(p_src2);
        __m128i xmm4 = _mm_loadu_si128(p_src3);

        xmm1 = _mm_max_epi16(xmm1, xmm2);
        xmm3 = _mm_max_epi16(xmm3, xmm4);
        xmm1 = _mm_max_epi16(xmm1, xmm3);

        _mm_storeu_si128(p_target, xmm1);

        p_src0 += 1;
        p_src1 += 1;
        p_src2 += 1;
        p_src3 += 1;
        p_target += 1;
    }

    volk_16i_x4_quad_max_star_16i_generic(
        target + eighthPoints * 8, src0 + eighthPoints * 8, src1 + eighthPoints * 8,
        src2 + eighthPoints * 8, src3 + eighthPoints * 8, leftovers);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_x4_quad_max_star_16i_u_avx2(short* target,
                                                          const short* src0,
                                                          const short* src1,
                                                          const short* src2,
                                                          const short* src3,
                                                          unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    int leftovers = num_points - sixteenthPoints * 16;

    const __m256i* p_src0 = (const __m256i*)src0;
    const __m256i* p_src1 = (const __m256i*)src1;
    const __m256i* p_src2 = (const __m256i*)src2;
    const __m256i* p_src3 = (const __m256i*)src3;
    __m256i* p_target = (__m256i*)target;

    for (unsigned int i = 0; i < sixteenthPoints; ++i) {
        __m256i xmm1 = _mm256_loadu_si256(p_src0);
        __m256i xmm2 = _mm256_loadu_si256(p_src1);
        __m256i xmm3 = _mm256_loadu_si256(p_src2);
        __m256i xmm4 = _mm256_loadu_si256(p_src3);

        xmm1 = _mm256_max_epi16(xmm1, xmm2);
        xmm3 = _mm256_max_epi16(xmm3, xmm4);
        xmm1 = _mm256_max_epi16(xmm1, xmm3);

        _mm256_storeu_si256(p_target, xmm1);

        p_src0 += 1;
        p_src1 += 1;
        p_src2 += 1;
        p_src3 += 1;
        p_target += 1;
    }

    volk_16i_x4_quad_max_star_16i_generic(
        target + sixteenthPoints * 16, src0 + sixteenthPoints * 16,
        src1 + sixteenthPoints * 16, src2 + sixteenthPoints * 16,
        src3 + sixteenthPoints * 16, leftovers);
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_x4_quad_max_star_16i_u_avx512bw(short* target,
                                                              const short* src0,
                                                              const short* src1,
                                                              const short* src2,
                                                              const short* src3,
                                                              unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    int leftovers = num_points - thirtysecondPoints * 32;

    const __m512i* p_src0 = (const __m512i*)src0;
    const __m512i* p_src1 = (const __m512i*)src1;
    const __m512i* p_src2 = (const __m512i*)src2;
    const __m512i* p_src3 = (const __m512i*)src3;
    __m512i* p_target = (__m512i*)target;

    for (unsigned int i = 0; i < thirtysecondPoints; ++i) {
        __m512i xmm1 = _mm512_loadu_si512(p_src0);
        __m512i xmm2 = _mm512_loadu_si512(p_src1);
        __m512i xmm3 = _mm512_loadu_si512(p_src2);
        __m512i xmm4 = _mm512_loadu_si512(p_src3);

        xmm1 = _mm512_max_epi16(xmm1, xmm2);
        xmm3 = _mm512_max_epi16(xmm3, xmm4);
        xmm1 = _mm512_max_epi16(xmm1, xmm3);

        _mm512_storeu_si512(p_target, xmm1);

        p_src0 += 1;
        p_src1 += 1;
        p_src2 += 1;
        p_src3 += 1;
        p_target += 1;
    }

    volk_16i_x4_quad_max_star_16i_generic(
        target + thirtysecondPoints * 32, src0 + thirtysecondPoints * 32,
        src1 + thirtysecondPoints * 32, src2 + thirtysecondPoints * 32,
        src3 + thirtysecondPoints * 32, leftovers);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_NEON

#include <arm_neon.h>

static inline void volk_16i_x4_quad_max_star_16i_neon(short* target,
                                                      const short* src0,
                                                      const short* src1,
                                                      const short* src2,
                                                      const short* src3,
                                                      unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    unsigned i;

    int16x8_t src0_vec, src1_vec, src2_vec, src3_vec;
    int16x8_t result1_vec, result2_vec;
    for (i = 0; i < eighth_points; ++i) {
        src0_vec = vld1q_s16(src0);
        src1_vec = vld1q_s16(src1);
        src2_vec = vld1q_s16(src2);
        src3_vec = vld1q_s16(src3);

        result1_vec = vmaxq_s16(src0_vec, src1_vec);
        result2_vec = vmaxq_s16(src2_vec, src3_vec);
        result1_vec = vmaxq_s16(result1_vec, result2_vec);

        vst1q_s16(target, result1_vec);
        src0 += 8;
        src1 += 8;
        src2 += 8;
        src3 += 8;
        target += 8;
    }

    short temp0 = 0;
    short temp1 = 0;
    for (i = eighth_points * 8; i < num_points; ++i) {
        temp0 = (*src0 > *src1) ? *src0 : *src1;
        temp1 = (*src2 > *src3) ? *src2 : *src3;
        *target++ = (temp0 > temp1) ? temp0 : temp1;
        src0++;
        src1++;
        src2++;
        src3++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_x4_quad_max_star_16i_rvv(short* target,
                                                      const short* src0,
                                                      const short* src1,
                                                      const short* src2,
                                                      const short* src3,
                                                      unsigned int num_points)
{
    size_t n = (size_t)num_points;
    for (size_t vl; n > 0;
         n -= vl, src0 += vl, src1 += vl, src2 += vl, src3 += vl, target += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4_t v0 = __riscv_vle16_v_i16m4(src0, vl);
        vint16m4_t v1 = __riscv_vle16_v_i16m4(src1, vl);
        vint16m4_t v2 = __riscv_vle16_v_i16m4(src2, vl);
        vint16m4_t v3 = __riscv_vle16_v_i16m4(src3, vl);
        vint16m4_t m01 = __riscv_vmax(v0, v1, vl);
        vint16m4_t m23 = __riscv_vmax(v2, v3, vl);
        __riscv_vse16(target, __riscv_vmax(m01, m23, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#endif /* INCLUDED_volk_16i_x4_quad_max_star_16i_u_H */

#ifndef INCLUDED_volk_16i_x4_quad_max_star_16i_a_H
#define INCLUDED_volk_16i_x4_quad_max_star_16i_a_H

#include <inttypes.h>

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>

static inline void volk_16i_x4_quad_max_star_16i_a_sse2(short* target,
                                                        const short* src0,
                                                        const short* src1,
                                                        const short* src2,
                                                        const short* src3,
                                                        unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = (num_bytes >> 4);
    int bound_copy = bound;
    int leftovers = (num_bytes >> 1) & 7;

    __m128i* p_target;
    const __m128i *p_src0, *p_src1, *p_src2, *p_src3;
    p_target = (__m128i*)target;
    p_src0 = (const __m128i*)src0;
    p_src1 = (const __m128i*)src1;
    p_src2 = (const __m128i*)src2;
    p_src3 = (const __m128i*)src3;

    __m128i xmm1, xmm2, xmm3, xmm4;

    while (bound_copy > 0) {
        xmm1 = _mm_load_si128(p_src0);
        xmm2 = _mm_load_si128(p_src1);
        xmm3 = _mm_load_si128(p_src2);
        xmm4 = _mm_load_si128(p_src3);

        xmm1 = _mm_max_epi16(xmm1, xmm2);
        xmm3 = _mm_max_epi16(xmm3, xmm4);
        xmm1 = _mm_max_epi16(xmm1, xmm3);

        _mm_store_si128(p_target, xmm1);

        p_src0 += 1;
        p_src1 += 1;
        p_src2 += 1;
        p_src3 += 1;
        p_target += 1;
        bound_copy -= 1;
    }

    short temp0 = 0;
    short temp1 = 0;
    for (i = bound * 8; i < (bound * 8) + leftovers; ++i) {
        temp0 = (src0[i] > src1[i]) ? src0[i] : src1[i];
        temp1 = (src2[i] > src3[i]) ? src2[i] : src3[i];
        target[i] = (temp0 > temp1) ? temp0 : temp1;
    }
    return;
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_x4_quad_max_star_16i_a_avx2(short* target,
                                                          const short* src0,
                                                          const short* src1,
                                                          const short* src2,
                                                          const short* src3,
                                                          unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    int leftovers = num_points - sixteenthPoints * 16;

    const __m256i* p_src0 = (const __m256i*)src0;
    const __m256i* p_src1 = (const __m256i*)src1;
    const __m256i* p_src2 = (const __m256i*)src2;
    const __m256i* p_src3 = (const __m256i*)src3;
    __m256i* p_target = (__m256i*)target;

    for (unsigned int i = 0; i < sixteenthPoints; ++i) {
        __m256i xmm1 = _mm256_load_si256(p_src0);
        __m256i xmm2 = _mm256_load_si256(p_src1);
        __m256i xmm3 = _mm256_load_si256(p_src2);
        __m256i xmm4 = _mm256_load_si256(p_src3);

        xmm1 = _mm256_max_epi16(xmm1, xmm2);
        xmm3 = _mm256_max_epi16(xmm3, xmm4);
        xmm1 = _mm256_max_epi16(xmm1, xmm3);

        _mm256_store_si256(p_target, xmm1);

        p_src0 += 1;
        p_src1 += 1;
        p_src2 += 1;
        p_src3 += 1;
        p_target += 1;
    }

    volk_16i_x4_quad_max_star_16i_generic(
        target + sixteenthPoints * 16, src0 + sixteenthPoints * 16,
        src1 + sixteenthPoints * 16, src2 + sixteenthPoints * 16,
        src3 + sixteenthPoints * 16, leftovers);
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_x4_quad_max_star_16i_a_avx512bw(short* target,
                                                              const short* src0,
                                                              const short* src1,
                                                              const short* src2,
                                                              const short* src3,
                                                              unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    int leftovers = num_points - thirtysecondPoints * 32;

    const __m512i* p_src0 = (const __m512i*)src0;
    const __m512i* p_src1 = (const __m512i*)src1;
    const __m512i* p_src2 = (const __m512i*)src2;
    const __m512i* p_src3 = (const __m512i*)src3;
    __m512i* p_target = (__m512i*)target;

    for (unsigned int i = 0; i < thirtysecondPoints; ++i) {
        __m512i xmm1 = _mm512_load_si512(p_src0);
        __m512i xmm2 = _mm512_load_si512(p_src1);
        __m512i xmm3 = _mm512_load_si512(p_src2);
        __m512i xmm4 = _mm512_load_si512(p_src3);

        xmm1 = _mm512_max_epi16(xmm1, xmm2);
        xmm3 = _mm512_max_epi16(xmm3, xmm4);
        xmm1 = _mm512_max_epi16(xmm1, xmm3);

        _mm512_store_si512(p_target, xmm1);

        p_src0 += 1;
        p_src1 += 1;
        p_src2 += 1;
        p_src3 += 1;
        p_target += 1;
    }

    volk_16i_x4_quad_max_star_16i_generic(
        target + thirtysecondPoints * 32, src0 + thirtysecondPoints * 32,
        src1 + thirtysecondPoints * 32, src2 + thirtysecondPoints * 32,
        src3 + thirtysecondPoints * 32, leftovers);
}
#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_16i_x4_quad_max_star_16i_a_H */
