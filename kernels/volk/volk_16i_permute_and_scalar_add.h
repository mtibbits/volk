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

#ifndef INCLUDED_volk_16i_permute_and_scalar_add_a_H
#define INCLUDED_volk_16i_permute_and_scalar_add_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2

#include <emmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_permute_and_scalar_add_a_sse2(short* target,
                                                          short* src0,
                                                          short* permute_indexes,
                                                          short* cntl0,
                                                          short* cntl1,
                                                          short* cntl2,
                                                          short* cntl3,
                                                          short* scalars,
                                                          unsigned int num_points)
{

    const unsigned int num_bytes = num_points * 2;

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    __m128i *p_target, *p_cntl0, *p_cntl1, *p_cntl2, *p_cntl3, *p_scalars;

    short* p_permute_indexes = permute_indexes;

    p_target = (__m128i*)target;
    p_cntl0 = (__m128i*)cntl0;
    p_cntl1 = (__m128i*)cntl1;
    p_cntl2 = (__m128i*)cntl2;
    p_cntl3 = (__m128i*)cntl3;
    p_scalars = (__m128i*)scalars;

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
#endif /*LV_HAVE_SSE*/


#ifdef LV_HAVE_GENERIC
static inline void volk_16i_permute_and_scalar_add_generic(short* target,
                                                           short* src0,
                                                           short* permute_indexes,
                                                           short* cntl0,
                                                           short* cntl1,
                                                           short* cntl2,
                                                           short* cntl3,
                                                           short* scalars,
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

#endif /*LV_HAVE_GENERIC*/

#endif /*INCLUDED_volk_16i_permute_and_scalar_add_a_H*/
