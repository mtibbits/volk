/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_permute_and_scalar_add
 *
 * \b Deprecation
 *
 * This kernel is deprecated.
 *
 * \b Overview
 *
 * Gathers elements from a source vector according to a permutation index
 * array, then adds masked scalar values controlled by four control vectors.
 * For each output element: target[i] = src0[permute_indexes[i]] +
 * (cntl0[i] & scalars[0]) + (cntl1[i] & scalars[1]) +
 * (cntl2[i] & scalars[2]) + (cntl3[i] & scalars[3]).
 *
 * This pattern of indexed gather with conditional scalar accumulation is
 * characteristic of trellis-based decoding, where survivor path states are
 * reordered (permuted) and branch metrics (scalars) are selectively added
 * according to transition masks. It can also serve other DSP operations
 * that combine sample reordering with masked offset correction.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_permute_and_scalar_add(short* target, short* src0, short*
 * permute_indexes, short* cntl0, short* cntl1, short* cntl2, short* cntl3, short*
 * scalars, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The source vector of 16-bit integers (short).
 * \li permute_indexes: Index array specifying which element of src0 to gather for each output position (short).
 * \li cntl0: Control mask vector for scalars[0] (short).
 * \li cntl1: Control mask vector for scalars[1] (short).
 * \li cntl2: Control mask vector for scalars[2] (short).
 * \li cntl3: Control mask vector for scalars[3] (short).
 * \li scalars: Array of four 16-bit scalar values (short).
 * \li num_points: The number of 16-bit elements to process.
 *
 * \b Outputs
 * \li target: The output vector of 16-bit integers (short).
 *
 * \b Example
 * Reverse a 4-element vector and add a constant offset via all-ones control masks.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * short* target = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* src0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* permute_indexes = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl1 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl2 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* cntl3 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* scalars = (short*)volk_malloc(sizeof(short) * 4, alignment);
 *
 * src0[0] = 10; src0[1] = 20; src0[2] = 30; src0[3] = 40;
 * permute_indexes[0] = 3; permute_indexes[1] = 2;
 * permute_indexes[2] = 1; permute_indexes[3] = 0;
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     cntl0[i] = -1; cntl1[i] = -1; cntl2[i] = -1; cntl3[i] = -1;
 * }
 * scalars[0] = 1; scalars[1] = 2; scalars[2] = 3; scalars[3] = 4;
 *
 * // Expected: target[i] = src0[reverse[i]] + (1 + 2 + 3 + 4) = src0[reverse[i]] + 10
 * // target = {50, 40, 30, 20}
 *
 * volk_16i_permute_and_scalar_add(target, src0, permute_indexes,
 *     cntl0, cntl1, cntl2, cntl3, scalars, N);
 *
 * printf("Expected: %d %d %d %d\n", 50, 40, 30, 20);
 * printf("Result:   %d %d %d %d\n", target[0], target[1], target[2], target[3]);
 *
 * volk_free(target);
 * volk_free(src0);
 * volk_free(permute_indexes);
 * volk_free(cntl0);
 * volk_free(cntl1);
 * volk_free(cntl2);
 * volk_free(cntl3);
 * volk_free(scalars);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_permute_and_scalar_add_u_H
#define INCLUDED_volk_16i_permute_and_scalar_add_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_16i_permute_and_scalar_add_generic(short* target,
                                                           const short* src0,
                                                           const short* permute_indexes,
                                                           const short* cntl0,
                                                           const short* cntl1,
                                                           const short* cntl2,
                                                           const short* cntl3,
                                                           const short* scalars,
                                                           unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    for (i = 0; i < bound; ++i) {
        target[i] = src0[permute_indexes[i]] + (cntl0[i] & scalars[0]) +
                    (cntl1[i] & scalars[1]) + (cntl2[i] & scalars[2]) +
                    (cntl3[i] & scalars[3]);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_permute_and_scalar_add_u_sse2(short* target,
                                                           const short* src0,
                                                           const short* permute_indexes,
                                                           const short* cntl0,
                                                           const short* cntl1,
                                                           const short* cntl2,
                                                           const short* cntl3,
                                                           const short* scalars,
                                                           unsigned int num_points)
{

    const unsigned int num_bytes = num_points * 2;

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    __m128i* p_target;
    const __m128i *p_cntl0, *p_cntl1, *p_cntl2, *p_cntl3, *p_scalars;

    const short* p_permute_indexes = permute_indexes;

    p_target = (__m128i*)target;
    p_cntl0 = (const __m128i*)cntl0;
    p_cntl1 = (const __m128i*)cntl1;
    p_cntl2 = (const __m128i*)cntl2;
    p_cntl3 = (const __m128i*)cntl3;
    p_scalars = (const __m128i*)scalars;

    int i = 0;

    int bound = (num_bytes >> 4);
    int leftovers = (num_bytes >> 1) & 7;

    xmm0 = _mm_loadu_si128(p_scalars);

    xmm1 = _mm_shufflelo_epi16(xmm0, 0);
    xmm2 = _mm_shufflelo_epi16(xmm0, 0x55);
    xmm3 = _mm_shufflelo_epi16(xmm0, 0xaa);
    xmm4 = _mm_shufflelo_epi16(xmm0, 0xff);

    xmm1 = _mm_shuffle_epi32(xmm1, 0x00);
    xmm2 = _mm_shuffle_epi32(xmm2, 0x00);
    xmm3 = _mm_shuffle_epi32(xmm3, 0x00);
    xmm4 = _mm_shuffle_epi32(xmm4, 0x00);


    for (; i < bound; ++i) {
        xmm0 = _mm_setzero_si128();
        xmm5 = _mm_setzero_si128();
        xmm6 = _mm_setzero_si128();
        xmm7 = _mm_setzero_si128();

        xmm0 = _mm_insert_epi16(xmm0, src0[p_permute_indexes[0]], 0);
        xmm5 = _mm_insert_epi16(xmm5, src0[p_permute_indexes[1]], 1);
        xmm6 = _mm_insert_epi16(xmm6, src0[p_permute_indexes[2]], 2);
        xmm7 = _mm_insert_epi16(xmm7, src0[p_permute_indexes[3]], 3);
        xmm0 = _mm_insert_epi16(xmm0, src0[p_permute_indexes[4]], 4);
        xmm5 = _mm_insert_epi16(xmm5, src0[p_permute_indexes[5]], 5);
        xmm6 = _mm_insert_epi16(xmm6, src0[p_permute_indexes[6]], 6);
        xmm7 = _mm_insert_epi16(xmm7, src0[p_permute_indexes[7]], 7);

        xmm0 = _mm_add_epi16(xmm0, xmm5);
        xmm6 = _mm_add_epi16(xmm6, xmm7);

        p_permute_indexes += 8;

        xmm0 = _mm_add_epi16(xmm0, xmm6);

        xmm5 = _mm_loadu_si128(p_cntl0);
        xmm6 = _mm_loadu_si128(p_cntl1);
        xmm7 = _mm_loadu_si128(p_cntl2);

        xmm5 = _mm_and_si128(xmm5, xmm1);
        xmm6 = _mm_and_si128(xmm6, xmm2);
        xmm7 = _mm_and_si128(xmm7, xmm3);

        xmm0 = _mm_add_epi16(xmm0, xmm5);

        xmm5 = _mm_loadu_si128(p_cntl3);

        xmm6 = _mm_add_epi16(xmm6, xmm7);

        p_cntl0 += 1;

        xmm5 = _mm_and_si128(xmm5, xmm4);

        xmm0 = _mm_add_epi16(xmm0, xmm6);

        p_cntl1 += 1;
        p_cntl2 += 1;

        xmm0 = _mm_add_epi16(xmm0, xmm5);

        p_cntl3 += 1;

        _mm_storeu_si128(p_target, xmm0);

        p_target += 1;
    }

    volk_16i_permute_and_scalar_add_generic(target + bound * 8,
                                            src0,
                                            permute_indexes + bound * 8,
                                            cntl0 + bound * 8,
                                            cntl1 + bound * 8,
                                            cntl2 + bound * 8,
                                            cntl3 + bound * 8,
                                            scalars,
                                            leftovers);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX2

#include <immintrin.h>

static inline void volk_16i_permute_and_scalar_add_u_avx2(short* target,
                                                           const short* src0,
                                                           const short* permute_indexes,
                                                           const short* cntl0,
                                                           const short* cntl1,
                                                           const short* cntl2,
                                                           const short* cntl3,
                                                           const short* scalars,
                                                           unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;
    const short* p_permute_indexes = permute_indexes;

    __m256i* p_target = (__m256i*)target;
    const __m256i* p_cntl0 = (const __m256i*)cntl0;
    const __m256i* p_cntl1 = (const __m256i*)cntl1;
    const __m256i* p_cntl2 = (const __m256i*)cntl2;
    const __m256i* p_cntl3 = (const __m256i*)cntl3;

    int i = 0;
    int bound = (num_bytes >> 5);
    int leftovers = (num_bytes >> 1) & 15;

    __m256i s0 = _mm256_set1_epi16(scalars[0]);
    __m256i s1 = _mm256_set1_epi16(scalars[1]);
    __m256i s2 = _mm256_set1_epi16(scalars[2]);
    __m256i s3 = _mm256_set1_epi16(scalars[3]);

    for (; i < bound; ++i) {
        /* Scalar gather: build two 128-bit halves */
        __m128i lo0 = _mm_setzero_si128();
        __m128i lo1 = _mm_setzero_si128();
        __m128i lo2 = _mm_setzero_si128();
        __m128i lo3 = _mm_setzero_si128();

        lo0 = _mm_insert_epi16(lo0, src0[p_permute_indexes[0]], 0);
        lo1 = _mm_insert_epi16(lo1, src0[p_permute_indexes[1]], 1);
        lo2 = _mm_insert_epi16(lo2, src0[p_permute_indexes[2]], 2);
        lo3 = _mm_insert_epi16(lo3, src0[p_permute_indexes[3]], 3);
        lo0 = _mm_insert_epi16(lo0, src0[p_permute_indexes[4]], 4);
        lo1 = _mm_insert_epi16(lo1, src0[p_permute_indexes[5]], 5);
        lo2 = _mm_insert_epi16(lo2, src0[p_permute_indexes[6]], 6);
        lo3 = _mm_insert_epi16(lo3, src0[p_permute_indexes[7]], 7);

        lo0 = _mm_add_epi16(lo0, lo1);
        lo2 = _mm_add_epi16(lo2, lo3);
        __m128i lo = _mm_add_epi16(lo0, lo2);

        __m128i hi0 = _mm_setzero_si128();
        __m128i hi1 = _mm_setzero_si128();
        __m128i hi2 = _mm_setzero_si128();
        __m128i hi3 = _mm_setzero_si128();

        hi0 = _mm_insert_epi16(hi0, src0[p_permute_indexes[8]], 0);
        hi1 = _mm_insert_epi16(hi1, src0[p_permute_indexes[9]], 1);
        hi2 = _mm_insert_epi16(hi2, src0[p_permute_indexes[10]], 2);
        hi3 = _mm_insert_epi16(hi3, src0[p_permute_indexes[11]], 3);
        hi0 = _mm_insert_epi16(hi0, src0[p_permute_indexes[12]], 4);
        hi1 = _mm_insert_epi16(hi1, src0[p_permute_indexes[13]], 5);
        hi2 = _mm_insert_epi16(hi2, src0[p_permute_indexes[14]], 6);
        hi3 = _mm_insert_epi16(hi3, src0[p_permute_indexes[15]], 7);

        hi0 = _mm_add_epi16(hi0, hi1);
        hi2 = _mm_add_epi16(hi2, hi3);
        __m128i hi = _mm_add_epi16(hi0, hi2);

        __m256i gathered =
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

        p_permute_indexes += 16;

        __m256i c0 = _mm256_loadu_si256(p_cntl0);
        __m256i c1 = _mm256_loadu_si256(p_cntl1);
        __m256i c2 = _mm256_loadu_si256(p_cntl2);
        __m256i c3 = _mm256_loadu_si256(p_cntl3);

        c0 = _mm256_and_si256(c0, s0);
        c1 = _mm256_and_si256(c1, s1);
        c2 = _mm256_and_si256(c2, s2);
        c3 = _mm256_and_si256(c3, s3);

        gathered = _mm256_add_epi16(gathered, c0);
        c1 = _mm256_add_epi16(c1, c2);
        gathered = _mm256_add_epi16(gathered, c1);
        gathered = _mm256_add_epi16(gathered, c3);

        _mm256_storeu_si256(p_target, gathered);

        p_target += 1;
        p_cntl0 += 1;
        p_cntl1 += 1;
        p_cntl2 += 1;
        p_cntl3 += 1;
    }

    volk_16i_permute_and_scalar_add_generic(target + bound * 16,
                                            src0,
                                            permute_indexes + bound * 16,
                                            cntl0 + bound * 16,
                                            cntl1 + bound * 16,
                                            cntl2 + bound * 16,
                                            cntl3 + bound * 16,
                                            scalars,
                                            leftovers);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW

#include <immintrin.h>

static inline void volk_16i_permute_and_scalar_add_u_avx512bw(
    short* target,
    const short* src0,
    const short* permute_indexes,
    const short* cntl0,
    const short* cntl1,
    const short* cntl2,
    const short* cntl3,
    const short* scalars,
    unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;
    const short* p_permute_indexes = permute_indexes;

    __m512i* p_target = (__m512i*)target;
    const __m512i* p_cntl0 = (const __m512i*)cntl0;
    const __m512i* p_cntl1 = (const __m512i*)cntl1;
    const __m512i* p_cntl2 = (const __m512i*)cntl2;
    const __m512i* p_cntl3 = (const __m512i*)cntl3;

    int i = 0;
    int bound = (num_bytes >> 6);
    int leftovers = (num_bytes >> 1) & 31;

    __m512i s0 = _mm512_set1_epi16(scalars[0]);
    __m512i s1 = _mm512_set1_epi16(scalars[1]);
    __m512i s2 = _mm512_set1_epi16(scalars[2]);
    __m512i s3 = _mm512_set1_epi16(scalars[3]);

    for (; i < bound; ++i) {
        /* Scalar gather: build four 128-bit parts, combine into 512-bit */
        __m128i p0a = _mm_setzero_si128();
        __m128i p0b = _mm_setzero_si128();
        __m128i p0c = _mm_setzero_si128();
        __m128i p0d = _mm_setzero_si128();

        p0a = _mm_insert_epi16(p0a, src0[p_permute_indexes[0]], 0);
        p0b = _mm_insert_epi16(p0b, src0[p_permute_indexes[1]], 1);
        p0c = _mm_insert_epi16(p0c, src0[p_permute_indexes[2]], 2);
        p0d = _mm_insert_epi16(p0d, src0[p_permute_indexes[3]], 3);
        p0a = _mm_insert_epi16(p0a, src0[p_permute_indexes[4]], 4);
        p0b = _mm_insert_epi16(p0b, src0[p_permute_indexes[5]], 5);
        p0c = _mm_insert_epi16(p0c, src0[p_permute_indexes[6]], 6);
        p0d = _mm_insert_epi16(p0d, src0[p_permute_indexes[7]], 7);

        p0a = _mm_add_epi16(p0a, p0b);
        p0c = _mm_add_epi16(p0c, p0d);
        __m128i part0 = _mm_add_epi16(p0a, p0c);

        __m128i p1a = _mm_setzero_si128();
        __m128i p1b = _mm_setzero_si128();
        __m128i p1c = _mm_setzero_si128();
        __m128i p1d = _mm_setzero_si128();

        p1a = _mm_insert_epi16(p1a, src0[p_permute_indexes[8]], 0);
        p1b = _mm_insert_epi16(p1b, src0[p_permute_indexes[9]], 1);
        p1c = _mm_insert_epi16(p1c, src0[p_permute_indexes[10]], 2);
        p1d = _mm_insert_epi16(p1d, src0[p_permute_indexes[11]], 3);
        p1a = _mm_insert_epi16(p1a, src0[p_permute_indexes[12]], 4);
        p1b = _mm_insert_epi16(p1b, src0[p_permute_indexes[13]], 5);
        p1c = _mm_insert_epi16(p1c, src0[p_permute_indexes[14]], 6);
        p1d = _mm_insert_epi16(p1d, src0[p_permute_indexes[15]], 7);

        p1a = _mm_add_epi16(p1a, p1b);
        p1c = _mm_add_epi16(p1c, p1d);
        __m128i part1 = _mm_add_epi16(p1a, p1c);

        __m128i p2a = _mm_setzero_si128();
        __m128i p2b = _mm_setzero_si128();
        __m128i p2c = _mm_setzero_si128();
        __m128i p2d = _mm_setzero_si128();

        p2a = _mm_insert_epi16(p2a, src0[p_permute_indexes[16]], 0);
        p2b = _mm_insert_epi16(p2b, src0[p_permute_indexes[17]], 1);
        p2c = _mm_insert_epi16(p2c, src0[p_permute_indexes[18]], 2);
        p2d = _mm_insert_epi16(p2d, src0[p_permute_indexes[19]], 3);
        p2a = _mm_insert_epi16(p2a, src0[p_permute_indexes[20]], 4);
        p2b = _mm_insert_epi16(p2b, src0[p_permute_indexes[21]], 5);
        p2c = _mm_insert_epi16(p2c, src0[p_permute_indexes[22]], 6);
        p2d = _mm_insert_epi16(p2d, src0[p_permute_indexes[23]], 7);

        p2a = _mm_add_epi16(p2a, p2b);
        p2c = _mm_add_epi16(p2c, p2d);
        __m128i part2 = _mm_add_epi16(p2a, p2c);

        __m128i p3a = _mm_setzero_si128();
        __m128i p3b = _mm_setzero_si128();
        __m128i p3c = _mm_setzero_si128();
        __m128i p3d = _mm_setzero_si128();

        p3a = _mm_insert_epi16(p3a, src0[p_permute_indexes[24]], 0);
        p3b = _mm_insert_epi16(p3b, src0[p_permute_indexes[25]], 1);
        p3c = _mm_insert_epi16(p3c, src0[p_permute_indexes[26]], 2);
        p3d = _mm_insert_epi16(p3d, src0[p_permute_indexes[27]], 3);
        p3a = _mm_insert_epi16(p3a, src0[p_permute_indexes[28]], 4);
        p3b = _mm_insert_epi16(p3b, src0[p_permute_indexes[29]], 5);
        p3c = _mm_insert_epi16(p3c, src0[p_permute_indexes[30]], 6);
        p3d = _mm_insert_epi16(p3d, src0[p_permute_indexes[31]], 7);

        p3a = _mm_add_epi16(p3a, p3b);
        p3c = _mm_add_epi16(p3c, p3d);
        __m128i part3 = _mm_add_epi16(p3a, p3c);

        __m256i lo256 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(part0), part1, 1);
        __m256i hi256 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(part2), part3, 1);
        __m512i gathered =
            _mm512_inserti64x4(_mm512_castsi256_si512(lo256), hi256, 1);

        p_permute_indexes += 32;

        __m512i c0 = _mm512_loadu_si512(p_cntl0);
        __m512i c1 = _mm512_loadu_si512(p_cntl1);
        __m512i c2 = _mm512_loadu_si512(p_cntl2);
        __m512i c3 = _mm512_loadu_si512(p_cntl3);

        c0 = _mm512_and_si512(c0, s0);
        c1 = _mm512_and_si512(c1, s1);
        c2 = _mm512_and_si512(c2, s2);
        c3 = _mm512_and_si512(c3, s3);

        gathered = _mm512_add_epi16(gathered, c0);
        c1 = _mm512_add_epi16(c1, c2);
        gathered = _mm512_add_epi16(gathered, c1);
        gathered = _mm512_add_epi16(gathered, c3);

        _mm512_storeu_si512(p_target, gathered);

        p_target += 1;
        p_cntl0 += 1;
        p_cntl1 += 1;
        p_cntl2 += 1;
        p_cntl3 += 1;
    }

    volk_16i_permute_and_scalar_add_generic(target + bound * 32,
                                            src0,
                                            permute_indexes + bound * 32,
                                            cntl0 + bound * 32,
                                            cntl1 + bound * 32,
                                            cntl2 + bound * 32,
                                            cntl3 + bound * 32,
                                            scalars,
                                            leftovers);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_permute_and_scalar_add_neon(short* target,
                                                         const short* src0,
                                                         const short* permute_indexes,
                                                         const short* cntl0,
                                                         const short* cntl1,
                                                         const short* cntl2,
                                                         const short* cntl3,
                                                         const short* scalars,
                                                         unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    const unsigned int leftovers = num_points - eighth_points * 8;

    int16x8_t s0 = vdupq_n_s16(scalars[0]);
    int16x8_t s1 = vdupq_n_s16(scalars[1]);
    int16x8_t s2 = vdupq_n_s16(scalars[2]);
    int16x8_t s3 = vdupq_n_s16(scalars[3]);

    for (unsigned int i = 0; i < eighth_points; ++i) {
        /* Scalar gather — no NEON gather for int16 */
        int16_t gathered[8];
        gathered[0] = src0[permute_indexes[0]];
        gathered[1] = src0[permute_indexes[1]];
        gathered[2] = src0[permute_indexes[2]];
        gathered[3] = src0[permute_indexes[3]];
        gathered[4] = src0[permute_indexes[4]];
        gathered[5] = src0[permute_indexes[5]];
        gathered[6] = src0[permute_indexes[6]];
        gathered[7] = src0[permute_indexes[7]];
        permute_indexes += 8;

        int16x8_t g = vld1q_s16(gathered);

        int16x8_t c0 = vandq_s16(vld1q_s16(cntl0), s0);
        int16x8_t c1 = vandq_s16(vld1q_s16(cntl1), s1);
        int16x8_t c2 = vandq_s16(vld1q_s16(cntl2), s2);
        int16x8_t c3 = vandq_s16(vld1q_s16(cntl3), s3);

        g = vaddq_s16(g, c0);
        c1 = vaddq_s16(c1, c2);
        g = vaddq_s16(g, c1);
        g = vaddq_s16(g, c3);

        vst1q_s16(target, g);

        target += 8;
        cntl0 += 8;
        cntl1 += 8;
        cntl2 += 8;
        cntl3 += 8;
    }

    volk_16i_permute_and_scalar_add_generic(target,
                                            src0,
                                            permute_indexes,
                                            cntl0,
                                            cntl1,
                                            cntl2,
                                            cntl3,
                                            scalars,
                                            leftovers);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_permute_and_scalar_add_rvv(short* target,
                                                        const short* src0,
                                                        const short* permute_indexes,
                                                        const short* cntl0,
                                                        const short* cntl1,
                                                        const short* cntl2,
                                                        const short* cntl3,
                                                        const short* scalars,
                                                        unsigned int num_points)
{
    const short s0 = scalars[0];
    const short s1 = scalars[1];
    const short s2 = scalars[2];
    const short s3 = scalars[3];

    size_t n = (size_t)num_points;
    for (size_t vl; n > 0;
         n -= vl, permute_indexes += vl, cntl0 += vl, cntl1 += vl, cntl2 += vl,
                  cntl3 += vl, target += vl) {
        vl = __riscv_vsetvl_e16m4(n);

        /* Gather from src0 using permute_indexes */
        vint16m4_t idx = __riscv_vle16_v_i16m4(permute_indexes, vl);
        vuint16m4_t uidx = __riscv_vreinterpret_u16m4(idx);
        vint16m4_t gathered = __riscv_vluxei16_v_i16m4(src0, __riscv_vsll(uidx, 1, vl), vl);

        /* Load control vectors and AND with scalars, then accumulate */
        vint16m4_t c0 = __riscv_vle16_v_i16m4(cntl0, vl);
        vint16m4_t c1 = __riscv_vle16_v_i16m4(cntl1, vl);
        vint16m4_t c2 = __riscv_vle16_v_i16m4(cntl2, vl);
        vint16m4_t c3 = __riscv_vle16_v_i16m4(cntl3, vl);

        vint16m4_t sum = gathered;
        sum = __riscv_vadd(sum, __riscv_vand(c0, s0, vl), vl);
        sum = __riscv_vadd(sum, __riscv_vand(c1, s1, vl), vl);
        sum = __riscv_vadd(sum, __riscv_vand(c2, s2, vl), vl);
        sum = __riscv_vadd(sum, __riscv_vand(c3, s3, vl), vl);

        __riscv_vse16(target, sum, vl);
    }
}
#endif /* LV_HAVE_RVV */

#endif /* INCLUDED_volk_16i_permute_and_scalar_add_u_H */

#ifndef INCLUDED_volk_16i_permute_and_scalar_add_a_H
#define INCLUDED_volk_16i_permute_and_scalar_add_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_permute_and_scalar_add_a_sse2(short* target,
                                                          const short* src0,
                                                          const short* permute_indexes,
                                                          const short* cntl0,
                                                          const short* cntl1,
                                                          const short* cntl2,
                                                          const short* cntl3,
                                                          const short* scalars,
                                                          unsigned int num_points)
{

    const unsigned int num_bytes = num_points * 2;

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    __m128i *p_target;
    const __m128i *p_cntl0, *p_cntl1, *p_cntl2, *p_cntl3, *p_scalars;

    const short* p_permute_indexes = permute_indexes;

    p_target = (__m128i*)target;
    p_cntl0 = (const __m128i*)cntl0;
    p_cntl1 = (const __m128i*)cntl1;
    p_cntl2 = (const __m128i*)cntl2;
    p_cntl3 = (const __m128i*)cntl3;
    p_scalars = (const __m128i*)scalars;

    int i = 0;

    int bound = (num_bytes >> 4);
    int leftovers = (num_bytes >> 1) & 7;

    xmm0 = _mm_load_si128(p_scalars);

    xmm1 = _mm_shufflelo_epi16(xmm0, 0);
    xmm2 = _mm_shufflelo_epi16(xmm0, 0x55);
    xmm3 = _mm_shufflelo_epi16(xmm0, 0xaa);
    xmm4 = _mm_shufflelo_epi16(xmm0, 0xff);

    xmm1 = _mm_shuffle_epi32(xmm1, 0x00);
    xmm2 = _mm_shuffle_epi32(xmm2, 0x00);
    xmm3 = _mm_shuffle_epi32(xmm3, 0x00);
    xmm4 = _mm_shuffle_epi32(xmm4, 0x00);


    for (; i < bound; ++i) {
        xmm0 = _mm_setzero_si128();
        xmm5 = _mm_setzero_si128();
        xmm6 = _mm_setzero_si128();
        xmm7 = _mm_setzero_si128();

        xmm0 = _mm_insert_epi16(xmm0, src0[p_permute_indexes[0]], 0);
        xmm5 = _mm_insert_epi16(xmm5, src0[p_permute_indexes[1]], 1);
        xmm6 = _mm_insert_epi16(xmm6, src0[p_permute_indexes[2]], 2);
        xmm7 = _mm_insert_epi16(xmm7, src0[p_permute_indexes[3]], 3);
        xmm0 = _mm_insert_epi16(xmm0, src0[p_permute_indexes[4]], 4);
        xmm5 = _mm_insert_epi16(xmm5, src0[p_permute_indexes[5]], 5);
        xmm6 = _mm_insert_epi16(xmm6, src0[p_permute_indexes[6]], 6);
        xmm7 = _mm_insert_epi16(xmm7, src0[p_permute_indexes[7]], 7);

        xmm0 = _mm_add_epi16(xmm0, xmm5);
        xmm6 = _mm_add_epi16(xmm6, xmm7);

        p_permute_indexes += 8;

        xmm0 = _mm_add_epi16(xmm0, xmm6);

        xmm5 = _mm_load_si128(p_cntl0);
        xmm6 = _mm_load_si128(p_cntl1);
        xmm7 = _mm_load_si128(p_cntl2);

        xmm5 = _mm_and_si128(xmm5, xmm1);
        xmm6 = _mm_and_si128(xmm6, xmm2);
        xmm7 = _mm_and_si128(xmm7, xmm3);

        xmm0 = _mm_add_epi16(xmm0, xmm5);

        xmm5 = _mm_load_si128(p_cntl3);

        xmm6 = _mm_add_epi16(xmm6, xmm7);

        p_cntl0 += 1;

        xmm5 = _mm_and_si128(xmm5, xmm4);

        xmm0 = _mm_add_epi16(xmm0, xmm6);

        p_cntl1 += 1;
        p_cntl2 += 1;

        xmm0 = _mm_add_epi16(xmm0, xmm5);

        p_cntl3 += 1;

        _mm_store_si128(p_target, xmm0);

        p_target += 1;
    }

    for (i = bound * 8; i < (bound * 8) + leftovers; ++i) {
        target[i] = src0[permute_indexes[i]] + (cntl0[i] & scalars[0]) +
                    (cntl1[i] & scalars[1]) + (cntl2[i] & scalars[2]) +
                    (cntl3[i] & scalars[3]);
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX2

#include <immintrin.h>

static inline void volk_16i_permute_and_scalar_add_a_avx2(short* target,
                                                           const short* src0,
                                                           const short* permute_indexes,
                                                           const short* cntl0,
                                                           const short* cntl1,
                                                           const short* cntl2,
                                                           const short* cntl3,
                                                           const short* scalars,
                                                           unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;
    const short* p_permute_indexes = permute_indexes;

    __m256i* p_target = (__m256i*)target;
    const __m256i* p_cntl0 = (const __m256i*)cntl0;
    const __m256i* p_cntl1 = (const __m256i*)cntl1;
    const __m256i* p_cntl2 = (const __m256i*)cntl2;
    const __m256i* p_cntl3 = (const __m256i*)cntl3;

    int i = 0;
    int bound = (num_bytes >> 5);
    int leftovers = (num_bytes >> 1) & 15;

    __m256i s0 = _mm256_set1_epi16(scalars[0]);
    __m256i s1 = _mm256_set1_epi16(scalars[1]);
    __m256i s2 = _mm256_set1_epi16(scalars[2]);
    __m256i s3 = _mm256_set1_epi16(scalars[3]);

    for (; i < bound; ++i) {
        /* Scalar gather: build two 128-bit halves */
        __m128i lo0 = _mm_setzero_si128();
        __m128i lo1 = _mm_setzero_si128();
        __m128i lo2 = _mm_setzero_si128();
        __m128i lo3 = _mm_setzero_si128();

        lo0 = _mm_insert_epi16(lo0, src0[p_permute_indexes[0]], 0);
        lo1 = _mm_insert_epi16(lo1, src0[p_permute_indexes[1]], 1);
        lo2 = _mm_insert_epi16(lo2, src0[p_permute_indexes[2]], 2);
        lo3 = _mm_insert_epi16(lo3, src0[p_permute_indexes[3]], 3);
        lo0 = _mm_insert_epi16(lo0, src0[p_permute_indexes[4]], 4);
        lo1 = _mm_insert_epi16(lo1, src0[p_permute_indexes[5]], 5);
        lo2 = _mm_insert_epi16(lo2, src0[p_permute_indexes[6]], 6);
        lo3 = _mm_insert_epi16(lo3, src0[p_permute_indexes[7]], 7);

        lo0 = _mm_add_epi16(lo0, lo1);
        lo2 = _mm_add_epi16(lo2, lo3);
        __m128i lo = _mm_add_epi16(lo0, lo2);

        __m128i hi0 = _mm_setzero_si128();
        __m128i hi1 = _mm_setzero_si128();
        __m128i hi2 = _mm_setzero_si128();
        __m128i hi3 = _mm_setzero_si128();

        hi0 = _mm_insert_epi16(hi0, src0[p_permute_indexes[8]], 0);
        hi1 = _mm_insert_epi16(hi1, src0[p_permute_indexes[9]], 1);
        hi2 = _mm_insert_epi16(hi2, src0[p_permute_indexes[10]], 2);
        hi3 = _mm_insert_epi16(hi3, src0[p_permute_indexes[11]], 3);
        hi0 = _mm_insert_epi16(hi0, src0[p_permute_indexes[12]], 4);
        hi1 = _mm_insert_epi16(hi1, src0[p_permute_indexes[13]], 5);
        hi2 = _mm_insert_epi16(hi2, src0[p_permute_indexes[14]], 6);
        hi3 = _mm_insert_epi16(hi3, src0[p_permute_indexes[15]], 7);

        hi0 = _mm_add_epi16(hi0, hi1);
        hi2 = _mm_add_epi16(hi2, hi3);
        __m128i hi = _mm_add_epi16(hi0, hi2);

        __m256i gathered =
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);

        p_permute_indexes += 16;

        __m256i c0 = _mm256_load_si256(p_cntl0);
        __m256i c1 = _mm256_load_si256(p_cntl1);
        __m256i c2 = _mm256_load_si256(p_cntl2);
        __m256i c3 = _mm256_load_si256(p_cntl3);

        c0 = _mm256_and_si256(c0, s0);
        c1 = _mm256_and_si256(c1, s1);
        c2 = _mm256_and_si256(c2, s2);
        c3 = _mm256_and_si256(c3, s3);

        gathered = _mm256_add_epi16(gathered, c0);
        c1 = _mm256_add_epi16(c1, c2);
        gathered = _mm256_add_epi16(gathered, c1);
        gathered = _mm256_add_epi16(gathered, c3);

        _mm256_store_si256(p_target, gathered);

        p_target += 1;
        p_cntl0 += 1;
        p_cntl1 += 1;
        p_cntl2 += 1;
        p_cntl3 += 1;
    }

    volk_16i_permute_and_scalar_add_generic(target + bound * 16,
                                            src0,
                                            permute_indexes + bound * 16,
                                            cntl0 + bound * 16,
                                            cntl1 + bound * 16,
                                            cntl2 + bound * 16,
                                            cntl3 + bound * 16,
                                            scalars,
                                            leftovers);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW

#include <immintrin.h>

static inline void volk_16i_permute_and_scalar_add_a_avx512bw(
    short* target,
    const short* src0,
    const short* permute_indexes,
    const short* cntl0,
    const short* cntl1,
    const short* cntl2,
    const short* cntl3,
    const short* scalars,
    unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;
    const short* p_permute_indexes = permute_indexes;

    __m512i* p_target = (__m512i*)target;
    const __m512i* p_cntl0 = (const __m512i*)cntl0;
    const __m512i* p_cntl1 = (const __m512i*)cntl1;
    const __m512i* p_cntl2 = (const __m512i*)cntl2;
    const __m512i* p_cntl3 = (const __m512i*)cntl3;

    int i = 0;
    int bound = (num_bytes >> 6);
    int leftovers = (num_bytes >> 1) & 31;

    __m512i s0 = _mm512_set1_epi16(scalars[0]);
    __m512i s1 = _mm512_set1_epi16(scalars[1]);
    __m512i s2 = _mm512_set1_epi16(scalars[2]);
    __m512i s3 = _mm512_set1_epi16(scalars[3]);

    for (; i < bound; ++i) {
        /* Scalar gather: build four 128-bit parts, combine into 512-bit */
        __m128i p0a = _mm_setzero_si128();
        __m128i p0b = _mm_setzero_si128();
        __m128i p0c = _mm_setzero_si128();
        __m128i p0d = _mm_setzero_si128();

        p0a = _mm_insert_epi16(p0a, src0[p_permute_indexes[0]], 0);
        p0b = _mm_insert_epi16(p0b, src0[p_permute_indexes[1]], 1);
        p0c = _mm_insert_epi16(p0c, src0[p_permute_indexes[2]], 2);
        p0d = _mm_insert_epi16(p0d, src0[p_permute_indexes[3]], 3);
        p0a = _mm_insert_epi16(p0a, src0[p_permute_indexes[4]], 4);
        p0b = _mm_insert_epi16(p0b, src0[p_permute_indexes[5]], 5);
        p0c = _mm_insert_epi16(p0c, src0[p_permute_indexes[6]], 6);
        p0d = _mm_insert_epi16(p0d, src0[p_permute_indexes[7]], 7);

        p0a = _mm_add_epi16(p0a, p0b);
        p0c = _mm_add_epi16(p0c, p0d);
        __m128i part0 = _mm_add_epi16(p0a, p0c);

        __m128i p1a = _mm_setzero_si128();
        __m128i p1b = _mm_setzero_si128();
        __m128i p1c = _mm_setzero_si128();
        __m128i p1d = _mm_setzero_si128();

        p1a = _mm_insert_epi16(p1a, src0[p_permute_indexes[8]], 0);
        p1b = _mm_insert_epi16(p1b, src0[p_permute_indexes[9]], 1);
        p1c = _mm_insert_epi16(p1c, src0[p_permute_indexes[10]], 2);
        p1d = _mm_insert_epi16(p1d, src0[p_permute_indexes[11]], 3);
        p1a = _mm_insert_epi16(p1a, src0[p_permute_indexes[12]], 4);
        p1b = _mm_insert_epi16(p1b, src0[p_permute_indexes[13]], 5);
        p1c = _mm_insert_epi16(p1c, src0[p_permute_indexes[14]], 6);
        p1d = _mm_insert_epi16(p1d, src0[p_permute_indexes[15]], 7);

        p1a = _mm_add_epi16(p1a, p1b);
        p1c = _mm_add_epi16(p1c, p1d);
        __m128i part1 = _mm_add_epi16(p1a, p1c);

        __m128i p2a = _mm_setzero_si128();
        __m128i p2b = _mm_setzero_si128();
        __m128i p2c = _mm_setzero_si128();
        __m128i p2d = _mm_setzero_si128();

        p2a = _mm_insert_epi16(p2a, src0[p_permute_indexes[16]], 0);
        p2b = _mm_insert_epi16(p2b, src0[p_permute_indexes[17]], 1);
        p2c = _mm_insert_epi16(p2c, src0[p_permute_indexes[18]], 2);
        p2d = _mm_insert_epi16(p2d, src0[p_permute_indexes[19]], 3);
        p2a = _mm_insert_epi16(p2a, src0[p_permute_indexes[20]], 4);
        p2b = _mm_insert_epi16(p2b, src0[p_permute_indexes[21]], 5);
        p2c = _mm_insert_epi16(p2c, src0[p_permute_indexes[22]], 6);
        p2d = _mm_insert_epi16(p2d, src0[p_permute_indexes[23]], 7);

        p2a = _mm_add_epi16(p2a, p2b);
        p2c = _mm_add_epi16(p2c, p2d);
        __m128i part2 = _mm_add_epi16(p2a, p2c);

        __m128i p3a = _mm_setzero_si128();
        __m128i p3b = _mm_setzero_si128();
        __m128i p3c = _mm_setzero_si128();
        __m128i p3d = _mm_setzero_si128();

        p3a = _mm_insert_epi16(p3a, src0[p_permute_indexes[24]], 0);
        p3b = _mm_insert_epi16(p3b, src0[p_permute_indexes[25]], 1);
        p3c = _mm_insert_epi16(p3c, src0[p_permute_indexes[26]], 2);
        p3d = _mm_insert_epi16(p3d, src0[p_permute_indexes[27]], 3);
        p3a = _mm_insert_epi16(p3a, src0[p_permute_indexes[28]], 4);
        p3b = _mm_insert_epi16(p3b, src0[p_permute_indexes[29]], 5);
        p3c = _mm_insert_epi16(p3c, src0[p_permute_indexes[30]], 6);
        p3d = _mm_insert_epi16(p3d, src0[p_permute_indexes[31]], 7);

        p3a = _mm_add_epi16(p3a, p3b);
        p3c = _mm_add_epi16(p3c, p3d);
        __m128i part3 = _mm_add_epi16(p3a, p3c);

        __m256i lo256 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(part0), part1, 1);
        __m256i hi256 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(part2), part3, 1);
        __m512i gathered =
            _mm512_inserti64x4(_mm512_castsi256_si512(lo256), hi256, 1);

        p_permute_indexes += 32;

        __m512i c0 = _mm512_load_si512(p_cntl0);
        __m512i c1 = _mm512_load_si512(p_cntl1);
        __m512i c2 = _mm512_load_si512(p_cntl2);
        __m512i c3 = _mm512_load_si512(p_cntl3);

        c0 = _mm512_and_si512(c0, s0);
        c1 = _mm512_and_si512(c1, s1);
        c2 = _mm512_and_si512(c2, s2);
        c3 = _mm512_and_si512(c3, s3);

        gathered = _mm512_add_epi16(gathered, c0);
        c1 = _mm512_add_epi16(c1, c2);
        gathered = _mm512_add_epi16(gathered, c1);
        gathered = _mm512_add_epi16(gathered, c3);

        _mm512_store_si512(p_target, gathered);

        p_target += 1;
        p_cntl0 += 1;
        p_cntl1 += 1;
        p_cntl2 += 1;
        p_cntl3 += 1;
    }

    volk_16i_permute_and_scalar_add_generic(target + bound * 32,
                                            src0,
                                            permute_indexes + bound * 32,
                                            cntl0 + bound * 32,
                                            cntl1 + bound * 32,
                                            cntl2 + bound * 32,
                                            cntl3 + bound * 32,
                                            scalars,
                                            leftovers);
}
#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_16i_permute_and_scalar_add_a_H */
