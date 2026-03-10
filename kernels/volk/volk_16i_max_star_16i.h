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
 * void volk_16i_max_star_16i(short* target, short* src0, unsigned int num_points);
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
volk_16i_max_star_16i_generic(short* target, short* src0, unsigned int num_points)
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

#endif /* INCLUDED_volk_16i_max_star_16i_u_H */

#ifndef INCLUDED_volk_16i_max_star_16i_a_H
#define INCLUDED_volk_16i_max_star_16i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void
volk_16i_max_star_16i_a_ssse3(short* target, short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    short candidate = src0[0];
    short cands[8];
    __m128i xmm0, xmm1, xmm3, xmm4, xmm5, xmm6;

    const __m128i* p_src0;

    p_src0 = (const __m128i*)src0;

    int bound = num_bytes >> 4;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    xmm0 = _mm_set1_epi16(candidate);

    for (i = 0; i < bound; ++i) {
        xmm1 = _mm_load_si128(p_src0);
        p_src0 += 1;
        // xmm2 = _mm_sub_epi16(xmm1, xmm0);

        xmm3 = _mm_cmpgt_epi16(xmm0, xmm1);
        xmm4 = _mm_cmpeq_epi16(xmm0, xmm1);
        xmm5 = _mm_cmpgt_epi16(xmm1, xmm0);

        xmm6 = _mm_xor_si128(xmm4, xmm5);

        xmm3 = _mm_and_si128(xmm3, xmm0);
        xmm4 = _mm_and_si128(xmm6, xmm1);

        xmm0 = _mm_add_epi16(xmm3, xmm4);
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

#endif /*LV_HAVE_SSSE3*/

#endif /* INCLUDED_volk_16i_max_star_16i_a_H */
