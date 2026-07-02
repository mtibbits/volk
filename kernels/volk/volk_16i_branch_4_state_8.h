/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_branch_4_state_8
 *
 * \b Deprecation
 *
 * This kernel is deprecated.
 *
 * \b Overview
 *
 * Computes 32 branch metrics (4 groups of 8) for a 4-state trellis by
 * permuting source state metrics, adding group-dependent scalar offsets,
 * and combining with masked control words. Each output element is:
 * target[i*8+j] = src0[perm] + scalar_offset + (cntl2[k] & scalars[2]) +
 * (cntl3[k] & scalars[3]).
 *
 * This kernel supports the add-compare-select (ACS) stage of a Viterbi
 * decoder. In trellis-based decoding, branch metrics quantify how well
 * each hypothesized state transition matches the received signal. The
 * permuters select which predecessor state metrics to use, while the
 * scalars and control words encode the trellis branch structure.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_branch_4_state_8(short* target, short* src0, char** permuters,
 * short* cntl2, short* cntl3, short* scalars)
 * \endcode
 *
 * \b Inputs
 * \li src0: Source state metrics, 8 elements (short).
 * \li permuters: Four 16-byte permutation tables that index into src0 (char**).
 * \li cntl2: Control word array, 32 elements, bitwise-ANDed with scalars[2] (short).
 * \li cntl3: Control word array, 32 elements, bitwise-ANDed with scalars[3] (short).
 * \li scalars: Four scalar values for group-dependent offsets and masking (short).
 *
 * \b Outputs
 * \li target: The 32 computed branch metrics, 4 groups of 8 (short).
 *
 * \b Example
 * Compute branch metrics with identity permutation and zero control masks.
 * \code
 * unsigned int alignment = volk_get_alignment();
 *
 * short* target = (short*)volk_malloc(sizeof(short) * 32, alignment);
 * short* src0 = (short*)volk_malloc(sizeof(short) * 8, alignment);
 * short* cntl2 = (short*)volk_malloc(sizeof(short) * 32, alignment);
 * short* cntl3 = (short*)volk_malloc(sizeof(short) * 32, alignment);
 * short* scalars = (short*)volk_malloc(sizeof(short) * 4, alignment);
 *
 * // 8 source state metrics
 * for (unsigned int i = 0; i < 8; ++i) {
 *   src0[i] = (short)(i + 1);
 * }
 *
 * // Identity permutation tables (byte offsets into src0)
 * char* perm0 = (char*)volk_malloc(16, alignment);
 * char* perm1 = (char*)volk_malloc(16, alignment);
 * char* perm2 = (char*)volk_malloc(16, alignment);
 * char* perm3 = (char*)volk_malloc(16, alignment);
 * for (int j = 0; j < 8; ++j) {
 *   perm0[j * 2] = (char)(j * 2); perm0[j * 2 + 1] = (char)(j * 2 + 1);
 *   perm1[j * 2] = (char)(j * 2); perm1[j * 2 + 1] = (char)(j * 2 + 1);
 *   perm2[j * 2] = (char)(j * 2); perm2[j * 2 + 1] = (char)(j * 2 + 1);
 *   perm3[j * 2] = (char)(j * 2); perm3[j * 2 + 1] = (char)(j * 2 + 1);
 * }
 * char* permuters[4] = { perm0, perm1, perm2, perm3 };
 *
 * // Zero controls so masked terms vanish
 * for (unsigned int i = 0; i < 32; ++i) {
 *   cntl2[i] = 0;
 *   cntl3[i] = 0;
 * }
 *
 * // scalars[0]=10, scalars[1]=20, scalars[2..3]=0
 * scalars[0] = 10; scalars[1] = 20; scalars[2] = 0; scalars[3] = 0;
 *
 * volk_16i_branch_4_state_8(target, src0, permuters, cntl2, cntl3, scalars);
 *
 * // Group 0 (i=0): offset = ((0+1)%2)*10 + (((0>>1)^1))*20 = 10+20 = 30
 * // target[0] = src0[0] + 30 = 1 + 30 = 31
 * printf("Expected target[0]: 31\n");
 * printf("Result   target[0]: %d\n", target[0]);
 *
 * volk_free(target);
 * volk_free(src0);
 * volk_free(cntl2);
 * volk_free(cntl3);
 * volk_free(scalars);
 * volk_free(perm0);
 * volk_free(perm1);
 * volk_free(perm2);
 * volk_free(perm3);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_branch_4_state_8_a_H
#define INCLUDED_volk_16i_branch_4_state_8_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_branch_4_state_8_a_ssse3(short* target,
                                                     short* src0,
                                                     char** permuters,
                                                     short* cntl2,
                                                     short* cntl3,
                                                     short* scalars)
{
    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11;
    __m128i *p_target, *p_src0, *p_cntl2, *p_cntl3, *p_scalars;

    p_target = (__m128i*)target;
    p_src0 = (__m128i*)src0;
    p_cntl2 = (__m128i*)cntl2;
    p_cntl3 = (__m128i*)cntl3;
    p_scalars = (__m128i*)scalars;

    xmm0 = _mm_load_si128(p_scalars);

    xmm1 = _mm_shufflelo_epi16(xmm0, 0);
    xmm2 = _mm_shufflelo_epi16(xmm0, 0x55);
    xmm3 = _mm_shufflelo_epi16(xmm0, 0xaa);
    xmm4 = _mm_shufflelo_epi16(xmm0, 0xff);

    xmm1 = _mm_shuffle_epi32(xmm1, 0x00);
    xmm2 = _mm_shuffle_epi32(xmm2, 0x00);
    xmm3 = _mm_shuffle_epi32(xmm3, 0x00);
    xmm4 = _mm_shuffle_epi32(xmm4, 0x00);

    xmm0 = _mm_load_si128((__m128i*)permuters[0]);
    xmm6 = _mm_load_si128((__m128i*)permuters[1]);
    xmm8 = _mm_load_si128((__m128i*)permuters[2]);
    xmm10 = _mm_load_si128((__m128i*)permuters[3]);

    xmm5 = _mm_load_si128(p_src0);
    xmm0 = _mm_shuffle_epi8(xmm5, xmm0);
    xmm6 = _mm_shuffle_epi8(xmm5, xmm6);
    xmm8 = _mm_shuffle_epi8(xmm5, xmm8);
    xmm10 = _mm_shuffle_epi8(xmm5, xmm10);

    xmm5 = _mm_add_epi16(xmm1, xmm2);

    xmm6 = _mm_add_epi16(xmm2, xmm6);
    xmm8 = _mm_add_epi16(xmm1, xmm8);

    xmm7 = _mm_load_si128(p_cntl2);
    xmm9 = _mm_load_si128(p_cntl3);

    xmm0 = _mm_add_epi16(xmm5, xmm0);

    xmm7 = _mm_and_si128(xmm7, xmm3);
    xmm9 = _mm_and_si128(xmm9, xmm4);

    xmm5 = _mm_load_si128(&p_cntl2[1]);
    xmm11 = _mm_load_si128(&p_cntl3[1]);

    xmm7 = _mm_add_epi16(xmm7, xmm9);

    xmm5 = _mm_and_si128(xmm5, xmm3);
    xmm11 = _mm_and_si128(xmm11, xmm4);

    xmm0 = _mm_add_epi16(xmm0, xmm7);


    xmm7 = _mm_load_si128(&p_cntl2[2]);
    xmm9 = _mm_load_si128(&p_cntl3[2]);

    xmm5 = _mm_add_epi16(xmm5, xmm11);

    xmm7 = _mm_and_si128(xmm7, xmm3);
    xmm9 = _mm_and_si128(xmm9, xmm4);

    xmm6 = _mm_add_epi16(xmm6, xmm5);


    xmm5 = _mm_load_si128(&p_cntl2[3]);
    xmm11 = _mm_load_si128(&p_cntl3[3]);

    xmm7 = _mm_add_epi16(xmm7, xmm9);

    xmm5 = _mm_and_si128(xmm5, xmm3);
    xmm11 = _mm_and_si128(xmm11, xmm4);

    xmm8 = _mm_add_epi16(xmm8, xmm7);

    xmm5 = _mm_add_epi16(xmm5, xmm11);

    _mm_store_si128(p_target, xmm0);
    _mm_store_si128(&p_target[1], xmm6);

    xmm10 = _mm_add_epi16(xmm5, xmm10);

    _mm_store_si128(&p_target[2], xmm8);

    _mm_store_si128(&p_target[3], xmm10);
}


#endif /*LV_HAVE_SSEs*/

#ifdef LV_HAVE_GENERIC
static inline void volk_16i_branch_4_state_8_generic(short* target,
                                                     short* src0,
                                                     char** permuters,
                                                     short* cntl2,
                                                     short* cntl3,
                                                     short* scalars)
{
    int i = 0;

    int bound = 4;

    for (; i < bound; ++i) {
        target[i * 8] = src0[((char)permuters[i][0]) / 2] + ((i + 1) % 2 * scalars[0]) +
                        (((i >> 1) ^ 1) * scalars[1]) + (cntl2[i * 8] & scalars[2]) +
                        (cntl3[i * 8] & scalars[3]);
        target[i * 8 + 1] = src0[((char)permuters[i][1 * 2]) / 2] +
                            ((i + 1) % 2 * scalars[0]) + (((i >> 1) ^ 1) * scalars[1]) +
                            (cntl2[i * 8 + 1] & scalars[2]) +
                            (cntl3[i * 8 + 1] & scalars[3]);
        target[i * 8 + 2] = src0[((char)permuters[i][2 * 2]) / 2] +
                            ((i + 1) % 2 * scalars[0]) + (((i >> 1) ^ 1) * scalars[1]) +
                            (cntl2[i * 8 + 2] & scalars[2]) +
                            (cntl3[i * 8 + 2] & scalars[3]);
        target[i * 8 + 3] = src0[((char)permuters[i][3 * 2]) / 2] +
                            ((i + 1) % 2 * scalars[0]) + (((i >> 1) ^ 1) * scalars[1]) +
                            (cntl2[i * 8 + 3] & scalars[2]) +
                            (cntl3[i * 8 + 3] & scalars[3]);
        target[i * 8 + 4] = src0[((char)permuters[i][4 * 2]) / 2] +
                            ((i + 1) % 2 * scalars[0]) + (((i >> 1) ^ 1) * scalars[1]) +
                            (cntl2[i * 8 + 4] & scalars[2]) +
                            (cntl3[i * 8 + 4] & scalars[3]);
        target[i * 8 + 5] = src0[((char)permuters[i][5 * 2]) / 2] +
                            ((i + 1) % 2 * scalars[0]) + (((i >> 1) ^ 1) * scalars[1]) +
                            (cntl2[i * 8 + 5] & scalars[2]) +
                            (cntl3[i * 8 + 5] & scalars[3]);
        target[i * 8 + 6] = src0[((char)permuters[i][6 * 2]) / 2] +
                            ((i + 1) % 2 * scalars[0]) + (((i >> 1) ^ 1) * scalars[1]) +
                            (cntl2[i * 8 + 6] & scalars[2]) +
                            (cntl3[i * 8 + 6] & scalars[3]);
        target[i * 8 + 7] = src0[((char)permuters[i][7 * 2]) / 2] +
                            ((i + 1) % 2 * scalars[0]) + (((i >> 1) ^ 1) * scalars[1]) +
                            (cntl2[i * 8 + 7] & scalars[2]) +
                            (cntl3[i * 8 + 7] & scalars[3]);
    }
}

#endif /*LV_HAVE_GENERIC*/


#endif /*INCLUDED_volk_16i_branch_4_state_8_a_H*/
