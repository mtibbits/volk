/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_max_star_16i
 *
 * \b Deprecation
 *
 * This kernel is deprecated. No replacement has been identified.
 *
 * \b Overview
 *
 * Computes the max* (max-star) operation over a vector of 16-bit signed integers,
 * returning the maximum value. In log-MAP decoding the max* operation selects the
 * dominant path metric; this implementation performs the max selection without the
 * log-domain correction term.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_max_star_16i(short* target, const short* src0, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li src0: The input vector of 16-bit signed integers (short).
 * \li num_points: The number of data points in the input vector.
 *
 * \b Outputs
 * \li target: The maximum value found in the input vector (single short value).
 *
 * \b Example
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main() {
 *     unsigned int N = 10;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     short* src0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 *     short* target = (short*)volk_malloc(sizeof(short), alignment);
 *
 *     // Initialize with sample path metrics
 *     for (unsigned int i = 0; i < N; ++i) {
 *       src0[i] = (short)(100 - (int)(i * 20));
 *     }
 *     // src0 = {100, 80, 60, 40, 20, 0, -20, -40, -60, -80}
 *
 *     volk_16i_max_star_16i(target, src0, N);
 *
 *     printf("max* = %d\n", target[0]);
 *     // Expected output: max* = 100
 *
 *     volk_free(src0);
 *     volk_free(target);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_16i_max_star_16i_u_H
#define INCLUDED_volk_16i_max_star_16i_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_16i_max_star_16i_generic(short* target, const short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    short candidate = src0[0];
    for (i = 1; i < bound; ++i) {
        candidate = (candidate > src0[i]) ? candidate : src0[i];
    }
    target[0] = candidate;
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_16i_max_star_16i_u_sse2(short* target, const short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    short candidate = src0[0];
    short cands[8];
    __m128i xmm0, xmm1;

    const __m128i* p_src0;

    p_src0 = (const __m128i*)src0;

    int bound = num_bytes >> 4;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    xmm0 = _mm_set1_epi16(candidate);

    for (i = 0; i < bound; ++i) {
        xmm1 = _mm_loadu_si128(p_src0);
        p_src0 += 1;

        xmm0 = _mm_max_epi16(xmm0, xmm1);
    }

    _mm_storeu_si128((__m128i*)cands, xmm0);

    for (i = 0; i < 8; ++i) {
        candidate = (candidate > cands[i]) ? candidate : cands[i];
    }

    if (leftovers > 0) {
        short tail_result;
        volk_16i_max_star_16i_generic(&tail_result, src0 + (bound << 3), leftovers);
        candidate = (candidate > tail_result) ? candidate : tail_result;
    }

    target[0] = candidate;
}

#endif /*LV_HAVE_SSE2*/


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_16i_max_star_16i_u_avx2(short* target, const short* src0, unsigned int num_points)
{
    short candidate = src0[0];

    const unsigned int sixteenthPoints = num_points / 16;
    int leftovers = num_points - sixteenthPoints * 16;

    __m256i xmm0 = _mm256_set1_epi16(candidate);
    const __m256i* p_src0 = (const __m256i*)src0;

    for (unsigned int i = 0; i < sixteenthPoints; ++i) {
        __m256i xmm1 = _mm256_loadu_si256(p_src0);
        p_src0 += 1;
        xmm0 = _mm256_max_epi16(xmm0, xmm1);
    }

    // Reduce 256-bit to 128-bit
    __m128i lo = _mm256_castsi256_si128(xmm0);
    __m128i hi = _mm256_extracti128_si256(xmm0, 1);
    __m128i reduced = _mm_max_epi16(lo, hi);

    __VOLK_ATTR_ALIGNED(16) short cands[8];
    _mm_storeu_si128((__m128i*)cands, reduced);

    for (int i = 0; i < 8; ++i) {
        candidate = (candidate > cands[i]) ? candidate : cands[i];
    }

    if (leftovers > 0) {
        short tail_result;
        volk_16i_max_star_16i_generic(
            &tail_result, src0 + sixteenthPoints * 16, leftovers);
        candidate = (candidate > tail_result) ? candidate : tail_result;
    }

    target[0] = candidate;
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void
volk_16i_max_star_16i_u_avx512bw(short* target, const short* src0, unsigned int num_points)
{
    short candidate = src0[0];

    const unsigned int thirtysecondPoints = num_points / 32;
    int leftovers = num_points - thirtysecondPoints * 32;

    __m512i xmm0 = _mm512_set1_epi16(candidate);
    const __m512i* p_src0 = (const __m512i*)src0;

    for (unsigned int i = 0; i < thirtysecondPoints; ++i) {
        __m512i xmm1 = _mm512_loadu_si512(p_src0);
        p_src0 += 1;
        xmm0 = _mm512_max_epi16(xmm0, xmm1);
    }

    // Reduce 512-bit to 256-bit to 128-bit
    __m256i lo256 = _mm512_castsi512_si256(xmm0);
    __m256i hi256 = _mm512_extracti64x4_epi64(xmm0, 1);
    __m256i reduced256 = _mm256_max_epi16(lo256, hi256);

    __m128i lo = _mm256_castsi256_si128(reduced256);
    __m128i hi = _mm256_extracti128_si256(reduced256, 1);
    __m128i reduced = _mm_max_epi16(lo, hi);

    __VOLK_ATTR_ALIGNED(16) short cands[8];
    _mm_storeu_si128((__m128i*)cands, reduced);

    for (int i = 0; i < 8; ++i) {
        candidate = (candidate > cands[i]) ? candidate : cands[i];
    }

    if (leftovers > 0) {
        short tail_result;
        volk_16i_max_star_16i_generic(
            &tail_result, src0 + thirtysecondPoints * 32, leftovers);
        candidate = (candidate > tail_result) ? candidate : tail_result;
    }

    target[0] = candidate;
}
#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_16i_max_star_16i_u_H */

#ifndef INCLUDED_volk_16i_max_star_16i_a_H
#define INCLUDED_volk_16i_max_star_16i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>

static inline void
volk_16i_max_star_16i_a_sse2(short* target, const short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    short candidate = src0[0];
    __VOLK_ATTR_ALIGNED(16) short cands[8];
    __m128i xmm0, xmm1;

    const __m128i* p_src0;

    p_src0 = (const __m128i*)src0;

    int bound = num_bytes >> 4;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    xmm0 = _mm_set1_epi16(candidate);

    for (i = 0; i < bound; ++i) {
        xmm1 = _mm_load_si128(p_src0);
        p_src0 += 1;

        xmm0 = _mm_max_epi16(xmm0, xmm1);
    }

    _mm_store_si128((__m128i*)cands, xmm0);

    for (i = 0; i < 8; ++i) {
        candidate = (candidate > cands[i]) ? candidate : cands[i];
    }

    for (i = 0; i < leftovers; ++i) {
        candidate = (candidate > src0[(bound << 3) + i])
                        ? candidate
                        : src0[(bound << 3) + i];
    }

    target[0] = candidate;
}

#endif /*LV_HAVE_SSE2*/


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_16i_max_star_16i_a_avx2(short* target, const short* src0, unsigned int num_points)
{
    short candidate = src0[0];

    const unsigned int sixteenthPoints = num_points / 16;
    int leftovers = num_points - sixteenthPoints * 16;

    __m256i xmm0 = _mm256_set1_epi16(candidate);
    const __m256i* p_src0 = (const __m256i*)src0;

    for (unsigned int i = 0; i < sixteenthPoints; ++i) {
        __m256i xmm1 = _mm256_load_si256(p_src0);
        p_src0 += 1;
        xmm0 = _mm256_max_epi16(xmm0, xmm1);
    }

    // Reduce 256-bit to 128-bit
    __m128i lo = _mm256_castsi256_si128(xmm0);
    __m128i hi = _mm256_extracti128_si256(xmm0, 1);
    __m128i reduced = _mm_max_epi16(lo, hi);

    __VOLK_ATTR_ALIGNED(16) short cands[8];
    _mm_store_si128((__m128i*)cands, reduced);

    for (int i = 0; i < 8; ++i) {
        candidate = (candidate > cands[i]) ? candidate : cands[i];
    }

    if (leftovers > 0) {
        short tail_result;
        volk_16i_max_star_16i_generic(
            &tail_result, src0 + sixteenthPoints * 16, leftovers);
        candidate = (candidate > tail_result) ? candidate : tail_result;
    }

    target[0] = candidate;
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void
volk_16i_max_star_16i_a_avx512bw(short* target, const short* src0, unsigned int num_points)
{
    short candidate = src0[0];

    const unsigned int thirtysecondPoints = num_points / 32;
    int leftovers = num_points - thirtysecondPoints * 32;

    __m512i xmm0 = _mm512_set1_epi16(candidate);
    const __m512i* p_src0 = (const __m512i*)src0;

    for (unsigned int i = 0; i < thirtysecondPoints; ++i) {
        __m512i xmm1 = _mm512_load_si512(p_src0);
        p_src0 += 1;
        xmm0 = _mm512_max_epi16(xmm0, xmm1);
    }

    // Reduce 512-bit to 256-bit to 128-bit
    __m256i lo256 = _mm512_castsi512_si256(xmm0);
    __m256i hi256 = _mm512_extracti64x4_epi64(xmm0, 1);
    __m256i reduced256 = _mm256_max_epi16(lo256, hi256);

    __m128i lo = _mm256_castsi256_si128(reduced256);
    __m128i hi = _mm256_extracti128_si256(reduced256, 1);
    __m128i reduced = _mm_max_epi16(lo, hi);

    __VOLK_ATTR_ALIGNED(16) short cands[8];
    _mm_store_si128((__m128i*)cands, reduced);

    for (int i = 0; i < 8; ++i) {
        candidate = (candidate > cands[i]) ? candidate : cands[i];
    }

    if (leftovers > 0) {
        short tail_result;
        volk_16i_max_star_16i_generic(
            &tail_result, src0 + thirtysecondPoints * 32, leftovers);
        candidate = (candidate > tail_result) ? candidate : tail_result;
    }

    target[0] = candidate;
}
#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_16i_max_star_16i_a_H */
