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
 * This kernel is deprecated.
 *
 * \b Overview
 *
 * Computes the maximum value in a vector of 16-bit signed integers using the
 * max* (max-star) operation and stores the result in the output. The kernel
 * iterates through the input samples, keeping a running maximum via
 * element-wise comparison.
 *
 * The max* operation appears in log-domain decoding algorithms such as
 * log-MAP and SOVA used in turbo and convolutional decoders, where branch
 * metrics represented as fixed-point values must be reduced to a single
 * maximum.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_max_star_16i(short* target, short* src0, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li src0: The input vector of samples (short).
 * \li num_points: The number of data points in the input vector.
 *
 * \b Outputs
 * \li target: The maximum value found in the input vector (short).
 *
 * \b Example
 * Find the maximum value in a small vector of known values.
 * \code
 * unsigned int N = 6;
 * unsigned int alignment = volk_get_alignment();
 *
 * short* src0 = (short*)volk_malloc(sizeof(short) * N, alignment);
 * short* target = (short*)volk_malloc(sizeof(short), alignment);
 *
 * src0[0] = 3; src0[1] = -2; src0[2] = 7;
 * src0[3] = 1; src0[4] = 7; src0[5] = -5;
 *
 * short expected = 7;
 *
 * volk_16i_max_star_16i(target, src0, N);
 *
 * printf("Expected: %d\n", expected);
 * printf("Result:   %d\n", target[0]);
 *
 * volk_free(src0);
 * volk_free(target);
 * \endcode
 */

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

    __m128i* p_src0;

    p_src0 = (__m128i*)src0;

    int bound = num_bytes >> 4;
    int leftovers = (num_bytes >> 1) & 7;

    int i = 0;

    xmm1 = _mm_setzero_si128();
    xmm0 = _mm_setzero_si128();
    //_mm_insert_epi16(xmm0, candidate, 0);

    xmm0 = _mm_shuffle_epi8(xmm0, xmm1);

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
        candidate = ((short)(candidate - cands[i]) > 0) ? candidate : cands[i];
    }

    for (i = 0; i < leftovers; ++i) {
        candidate = ((short)(candidate - src0[(bound << 3) + i]) > 0)
                        ? candidate
                        : src0[(bound << 3) + i];
    }

    target[0] = candidate;
}

#endif /*LV_HAVE_SSSE3*/

#ifdef LV_HAVE_GENERIC

static inline void
volk_16i_max_star_16i_generic(short* target, short* src0, unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 2;

    int i = 0;

    int bound = num_bytes >> 1;

    short candidate = src0[0];
    for (i = 1; i < bound; ++i) {
        candidate = ((short)(candidate - src0[i]) > 0) ? candidate : src0[i];
    }
    target[0] = candidate;
}

#endif /*LV_HAVE_GENERIC*/


#endif /*INCLUDED_volk_16i_max_star_16i_a_H*/
