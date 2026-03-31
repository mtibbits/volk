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

#ifndef INCLUDED_volk_16i_branch_4_state_8_u_H
#define INCLUDED_volk_16i_branch_4_state_8_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_16i_branch_4_state_8_generic(short* target,
                                                     const short* src0,
                                                     char** permuters,
                                                     const short* cntl2,
                                                     const short* cntl3,
                                                     const short* scalars)
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

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_branch_4_state_8_u_ssse3(short* target,
                                                     const short* src0,
                                                     char** permuters,
                                                     const short* cntl2,
                                                     const short* cntl3,
                                                     const short* scalars)
{
    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11;
    __m128i *p_target;
    const __m128i *p_src0, *p_cntl2, *p_cntl3, *p_scalars;

    p_target = (__m128i*)target;
    p_src0 = (const __m128i*)src0;
    p_cntl2 = (const __m128i*)cntl2;
    p_cntl3 = (const __m128i*)cntl3;
    p_scalars = (const __m128i*)scalars;

    xmm0 = _mm_loadu_si128(p_scalars);

    xmm1 = _mm_shufflelo_epi16(xmm0, 0);
    xmm2 = _mm_shufflelo_epi16(xmm0, 0x55);
    xmm3 = _mm_shufflelo_epi16(xmm0, 0xaa);
    xmm4 = _mm_shufflelo_epi16(xmm0, 0xff);

    xmm1 = _mm_shuffle_epi32(xmm1, 0x00);
    xmm2 = _mm_shuffle_epi32(xmm2, 0x00);
    xmm3 = _mm_shuffle_epi32(xmm3, 0x00);
    xmm4 = _mm_shuffle_epi32(xmm4, 0x00);

    xmm0 = _mm_loadu_si128((const __m128i*)permuters[0]);
    xmm6 = _mm_loadu_si128((const __m128i*)permuters[1]);
    xmm8 = _mm_loadu_si128((const __m128i*)permuters[2]);
    xmm10 = _mm_loadu_si128((const __m128i*)permuters[3]);

    xmm5 = _mm_loadu_si128(p_src0);
    xmm0 = _mm_shuffle_epi8(xmm5, xmm0);
    xmm6 = _mm_shuffle_epi8(xmm5, xmm6);
    xmm8 = _mm_shuffle_epi8(xmm5, xmm8);
    xmm10 = _mm_shuffle_epi8(xmm5, xmm10);

    xmm5 = _mm_add_epi16(xmm1, xmm2);

    xmm6 = _mm_add_epi16(xmm2, xmm6);
    xmm8 = _mm_add_epi16(xmm1, xmm8);

    xmm7 = _mm_loadu_si128(p_cntl2);
    xmm9 = _mm_loadu_si128(p_cntl3);

    xmm0 = _mm_add_epi16(xmm5, xmm0);

    xmm7 = _mm_and_si128(xmm7, xmm3);
    xmm9 = _mm_and_si128(xmm9, xmm4);

    xmm5 = _mm_loadu_si128(&p_cntl2[1]);
    xmm11 = _mm_loadu_si128(&p_cntl3[1]);

    xmm7 = _mm_add_epi16(xmm7, xmm9);

    xmm5 = _mm_and_si128(xmm5, xmm3);
    xmm11 = _mm_and_si128(xmm11, xmm4);

    xmm0 = _mm_add_epi16(xmm0, xmm7);


    xmm7 = _mm_loadu_si128(&p_cntl2[2]);
    xmm9 = _mm_loadu_si128(&p_cntl3[2]);

    xmm5 = _mm_add_epi16(xmm5, xmm11);

    xmm7 = _mm_and_si128(xmm7, xmm3);
    xmm9 = _mm_and_si128(xmm9, xmm4);

    xmm6 = _mm_add_epi16(xmm6, xmm5);


    xmm5 = _mm_loadu_si128(&p_cntl2[3]);
    xmm11 = _mm_loadu_si128(&p_cntl3[3]);

    xmm7 = _mm_add_epi16(xmm7, xmm9);

    xmm5 = _mm_and_si128(xmm5, xmm3);
    xmm11 = _mm_and_si128(xmm11, xmm4);

    xmm8 = _mm_add_epi16(xmm8, xmm7);

    xmm5 = _mm_add_epi16(xmm5, xmm11);

    _mm_storeu_si128(p_target, xmm0);
    _mm_storeu_si128(&p_target[1], xmm6);

    xmm10 = _mm_add_epi16(xmm5, xmm10);

    _mm_storeu_si128(&p_target[2], xmm8);

    _mm_storeu_si128(&p_target[3], xmm10);
}


#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2

#include <immintrin.h>

static inline void volk_16i_branch_4_state_8_u_avx2(short* target,
                                                     const short* src0,
                                                     char** permuters,
                                                     const short* cntl2,
                                                     const short* cntl3,
                                                     const short* scalars)
{
    __m128i src128 = _mm_loadu_si128((const __m128i*)src0);
    __m256i src256 = _mm256_broadcastsi128_si256(src128);

    __m256i s2 = _mm256_set1_epi16(scalars[2]);
    __m256i s3 = _mm256_set1_epi16(scalars[3]);

    /* Groups 0 & 1 */
    __m256i perm01 = _mm256_set_m128i(
        _mm_loadu_si128((const __m128i*)permuters[1]),
        _mm_loadu_si128((const __m128i*)permuters[0]));
    __m256i shuffled01 = _mm256_shuffle_epi8(src256, perm01);

    __m256i offset01 = _mm256_set_m128i(_mm_set1_epi16(scalars[1]),
                                        _mm_set1_epi16(scalars[0] + scalars[1]));

    __m256i result01 = _mm256_add_epi16(shuffled01, offset01);
    __m256i c2_01 = _mm256_loadu_si256((const __m256i*)cntl2);
    __m256i c3_01 = _mm256_loadu_si256((const __m256i*)cntl3);
    result01 = _mm256_add_epi16(result01, _mm256_and_si256(c2_01, s2));
    result01 = _mm256_add_epi16(result01, _mm256_and_si256(c3_01, s3));
    _mm256_storeu_si256((__m256i*)target, result01);

    /* Groups 2 & 3 */
    __m256i perm23 = _mm256_set_m128i(
        _mm_loadu_si128((const __m128i*)permuters[3]),
        _mm_loadu_si128((const __m128i*)permuters[2]));
    __m256i shuffled23 = _mm256_shuffle_epi8(src256, perm23);

    __m256i offset23 =
        _mm256_set_m128i(_mm_setzero_si128(), _mm_set1_epi16(scalars[0]));

    __m256i result23 = _mm256_add_epi16(shuffled23, offset23);
    __m256i c2_23 = _mm256_loadu_si256((const __m256i*)(cntl2 + 16));
    __m256i c3_23 = _mm256_loadu_si256((const __m256i*)(cntl3 + 16));
    result23 = _mm256_add_epi16(result23, _mm256_and_si256(c2_23, s2));
    result23 = _mm256_add_epi16(result23, _mm256_and_si256(c3_23, s3));
    _mm256_storeu_si256((__m256i*)(target + 16), result23);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW

#include <immintrin.h>

static inline void volk_16i_branch_4_state_8_u_avx512bw(short* target,
                                                         const short* src0,
                                                         char** permuters,
                                                         const short* cntl2,
                                                         const short* cntl3,
                                                         const short* scalars)
{
    __m128i src128 = _mm_loadu_si128((const __m128i*)src0);
    __m512i src512 = _mm512_broadcast_i32x4(src128);

    __m256i perm_lo = _mm256_set_m128i(
        _mm_loadu_si128((const __m128i*)permuters[1]),
        _mm_loadu_si128((const __m128i*)permuters[0]));
    __m256i perm_hi = _mm256_set_m128i(
        _mm_loadu_si128((const __m128i*)permuters[3]),
        _mm_loadu_si128((const __m128i*)permuters[2]));
    __m512i perm = _mm512_inserti64x4(_mm512_castsi256_si512(perm_lo), perm_hi, 1);

    __m512i shuffled = _mm512_shuffle_epi8(src512, perm);

    __m256i off_lo = _mm256_set_m128i(_mm_set1_epi16(scalars[1]),
                                      _mm_set1_epi16(scalars[0] + scalars[1]));
    __m256i off_hi =
        _mm256_set_m128i(_mm_setzero_si128(), _mm_set1_epi16(scalars[0]));
    __m512i offset = _mm512_inserti64x4(_mm512_castsi256_si512(off_lo), off_hi, 1);

    __m512i result = _mm512_add_epi16(shuffled, offset);

    __m512i s2 = _mm512_set1_epi16(scalars[2]);
    __m512i s3 = _mm512_set1_epi16(scalars[3]);
    __m512i c2 = _mm512_loadu_si512((const __m512i*)cntl2);
    __m512i c3 = _mm512_loadu_si512((const __m512i*)cntl3);
    result = _mm512_add_epi16(result, _mm512_and_si512(c2, s2));
    result = _mm512_add_epi16(result, _mm512_and_si512(c3, s3));

    _mm512_storeu_si512((__m512i*)target, result);
}

#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_NEON

#include <arm_neon.h>

static inline void volk_16i_branch_4_state_8_neon(short* target,
                                                   const short* src0,
                                                   char** permuters,
                                                   const short* cntl2,
                                                   const short* cntl3,
                                                   const short* scalars)
{
    uint8x16_t src_bytes = vld1q_u8((const uint8_t*)src0);
    uint8x8x2_t src_tbl;
    src_tbl.val[0] = vget_low_u8(src_bytes);
    src_tbl.val[1] = vget_high_u8(src_bytes);

    int16x8_t s0 = vdupq_n_s16(scalars[0]);
    int16x8_t s1 = vdupq_n_s16(scalars[1]);
    int16x8_t s2 = vdupq_n_s16(scalars[2]);
    int16x8_t s3 = vdupq_n_s16(scalars[3]);

    /* Group offsets: g0=s0+s1, g1=s1, g2=s0, g3=0 */
    int16x8_t offsets[4] = { vaddq_s16(s0, s1), s1, s0, vdupq_n_s16(0) };

    for (int i = 0; i < 4; i++) {
        uint8x8_t idx_lo = vld1_u8((const uint8_t*)permuters[i]);
        uint8x8_t idx_hi = vld1_u8((const uint8_t*)permuters[i] + 8);
        uint8x8_t res_lo = vtbl2_u8(src_tbl, idx_lo);
        uint8x8_t res_hi = vtbl2_u8(src_tbl, idx_hi);
        int16x8_t permuted = vreinterpretq_s16_u8(vcombine_u8(res_lo, res_hi));

        int16x8_t result = vaddq_s16(permuted, offsets[i]);
        int16x8_t c2 = vld1q_s16(cntl2 + i * 8);
        int16x8_t c3 = vld1q_s16(cntl3 + i * 8);
        result = vaddq_s16(result, vandq_s16(c2, s2));
        result = vaddq_s16(result, vandq_s16(c3, s3));
        vst1q_s16(target + i * 8, result);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV

#include <riscv_vector.h>

static inline void volk_16i_branch_4_state_8_rvv(short* target,
                                                  const short* src0,
                                                  char** permuters,
                                                  const short* cntl2,
                                                  const short* cntl3,
                                                  const short* scalars)
{
    size_t vl = __riscv_vsetvl_e16m1(8);
    vint16m1_t src16 = __riscv_vle16_v_i16m1(src0, vl);

    vint16m1_t s2 = __riscv_vmv_v_x_i16m1(scalars[2], vl);
    vint16m1_t s3 = __riscv_vmv_v_x_i16m1(scalars[3], vl);

    /* Group offsets: g0=s0+s1, g1=s1, g2=s0, g3=0 */
    short off_vals[4] = { (short)(scalars[0] + scalars[1]),
                          scalars[1],
                          scalars[0],
                          0 };

    for (int i = 0; i < 4; i++) {
        /* Extract int16 element indices from byte-level permuter table */
        vuint8mf2_t byte_idx =
            __riscv_vlse8_v_u8mf2((const uint8_t*)permuters[i], 2, vl);
        vuint16m1_t elem_idx =
            __riscv_vsrl(__riscv_vzext_vf2_u16m1(byte_idx, vl), 1, vl);
        vint16m1_t permuted = __riscv_vrgather_vv_i16m1(src16, elem_idx, vl);

        vint16m1_t result =
            __riscv_vadd(permuted, __riscv_vmv_v_x_i16m1(off_vals[i], vl), vl);
        vint16m1_t c2 = __riscv_vle16_v_i16m1(cntl2 + i * 8, vl);
        vint16m1_t c3 = __riscv_vle16_v_i16m1(cntl3 + i * 8, vl);
        result = __riscv_vadd(result, __riscv_vand(c2, s2, vl), vl);
        result = __riscv_vadd(result, __riscv_vand(c3, s3, vl), vl);
        __riscv_vse16(target + i * 8, result, vl);
    }
}

#endif /* LV_HAVE_RVV */

#endif /* INCLUDED_volk_16i_branch_4_state_8_u_H */

#ifndef INCLUDED_volk_16i_branch_4_state_8_a_H
#define INCLUDED_volk_16i_branch_4_state_8_a_H

#ifdef LV_HAVE_SSSE3

#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

static inline void volk_16i_branch_4_state_8_a_ssse3(short* target,
                                                     const short* src0,
                                                     char** permuters,
                                                     const short* cntl2,
                                                     const short* cntl3,
                                                     const short* scalars)
{
    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11;
    __m128i *p_target;
    const __m128i *p_src0, *p_cntl2, *p_cntl3, *p_scalars;

    p_target = (__m128i*)target;
    p_src0 = (const __m128i*)src0;
    p_cntl2 = (const __m128i*)cntl2;
    p_cntl3 = (const __m128i*)cntl3;
    p_scalars = (const __m128i*)scalars;

    xmm0 = _mm_load_si128(p_scalars);

    xmm1 = _mm_shufflelo_epi16(xmm0, 0);
    xmm2 = _mm_shufflelo_epi16(xmm0, 0x55);
    xmm3 = _mm_shufflelo_epi16(xmm0, 0xaa);
    xmm4 = _mm_shufflelo_epi16(xmm0, 0xff);

    xmm1 = _mm_shuffle_epi32(xmm1, 0x00);
    xmm2 = _mm_shuffle_epi32(xmm2, 0x00);
    xmm3 = _mm_shuffle_epi32(xmm3, 0x00);
    xmm4 = _mm_shuffle_epi32(xmm4, 0x00);

    xmm0 = _mm_load_si128((const __m128i*)permuters[0]);
    xmm6 = _mm_load_si128((const __m128i*)permuters[1]);
    xmm8 = _mm_load_si128((const __m128i*)permuters[2]);
    xmm10 = _mm_load_si128((const __m128i*)permuters[3]);

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


#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2

#include <immintrin.h>

static inline void volk_16i_branch_4_state_8_a_avx2(short* target,
                                                     const short* src0,
                                                     char** permuters,
                                                     const short* cntl2,
                                                     const short* cntl3,
                                                     const short* scalars)
{
    __m128i src128 = _mm_load_si128((const __m128i*)src0);
    __m256i src256 = _mm256_broadcastsi128_si256(src128);

    __m256i s2 = _mm256_set1_epi16(scalars[2]);
    __m256i s3 = _mm256_set1_epi16(scalars[3]);

    /* Groups 0 & 1 */
    __m256i perm01 = _mm256_set_m128i(
        _mm_load_si128((const __m128i*)permuters[1]),
        _mm_load_si128((const __m128i*)permuters[0]));
    __m256i shuffled01 = _mm256_shuffle_epi8(src256, perm01);

    __m256i offset01 = _mm256_set_m128i(_mm_set1_epi16(scalars[1]),
                                        _mm_set1_epi16(scalars[0] + scalars[1]));

    __m256i result01 = _mm256_add_epi16(shuffled01, offset01);
    __m256i c2_01 = _mm256_load_si256((const __m256i*)cntl2);
    __m256i c3_01 = _mm256_load_si256((const __m256i*)cntl3);
    result01 = _mm256_add_epi16(result01, _mm256_and_si256(c2_01, s2));
    result01 = _mm256_add_epi16(result01, _mm256_and_si256(c3_01, s3));
    _mm256_store_si256((__m256i*)target, result01);

    /* Groups 2 & 3 */
    __m256i perm23 = _mm256_set_m128i(
        _mm_load_si128((const __m128i*)permuters[3]),
        _mm_load_si128((const __m128i*)permuters[2]));
    __m256i shuffled23 = _mm256_shuffle_epi8(src256, perm23);

    __m256i offset23 =
        _mm256_set_m128i(_mm_setzero_si128(), _mm_set1_epi16(scalars[0]));

    __m256i result23 = _mm256_add_epi16(shuffled23, offset23);
    __m256i c2_23 = _mm256_load_si256((const __m256i*)(cntl2 + 16));
    __m256i c3_23 = _mm256_load_si256((const __m256i*)(cntl3 + 16));
    result23 = _mm256_add_epi16(result23, _mm256_and_si256(c2_23, s2));
    result23 = _mm256_add_epi16(result23, _mm256_and_si256(c3_23, s3));
    _mm256_store_si256((__m256i*)(target + 16), result23);
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW

#include <immintrin.h>

static inline void volk_16i_branch_4_state_8_a_avx512bw(short* target,
                                                         const short* src0,
                                                         char** permuters,
                                                         const short* cntl2,
                                                         const short* cntl3,
                                                         const short* scalars)
{
    __m128i src128 = _mm_load_si128((const __m128i*)src0);
    __m512i src512 = _mm512_broadcast_i32x4(src128);

    __m256i perm_lo = _mm256_set_m128i(
        _mm_load_si128((const __m128i*)permuters[1]),
        _mm_load_si128((const __m128i*)permuters[0]));
    __m256i perm_hi = _mm256_set_m128i(
        _mm_load_si128((const __m128i*)permuters[3]),
        _mm_load_si128((const __m128i*)permuters[2]));
    __m512i perm = _mm512_inserti64x4(_mm512_castsi256_si512(perm_lo), perm_hi, 1);

    __m512i shuffled = _mm512_shuffle_epi8(src512, perm);

    __m256i off_lo = _mm256_set_m128i(_mm_set1_epi16(scalars[1]),
                                      _mm_set1_epi16(scalars[0] + scalars[1]));
    __m256i off_hi =
        _mm256_set_m128i(_mm_setzero_si128(), _mm_set1_epi16(scalars[0]));
    __m512i offset = _mm512_inserti64x4(_mm512_castsi256_si512(off_lo), off_hi, 1);

    __m512i result = _mm512_add_epi16(shuffled, offset);

    __m512i s2 = _mm512_set1_epi16(scalars[2]);
    __m512i s3 = _mm512_set1_epi16(scalars[3]);
    __m512i c2 = _mm512_load_si512((const __m512i*)cntl2);
    __m512i c3 = _mm512_load_si512((const __m512i*)cntl3);
    result = _mm512_add_epi16(result, _mm512_and_si512(c2, s2));
    result = _mm512_add_epi16(result, _mm512_and_si512(c3, s3));

    _mm512_store_si512((__m512i*)target, result);
}

#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_16i_branch_4_state_8_a_H */
