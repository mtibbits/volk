/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_square_dist_32f
 *
 * \b Overview
 *
 * Computes the squared Euclidean distance between a single complex reference
 * sample and each point in a complex vector:
 * target[i] = |src0[0] - points[i]|^2.
 *
 * This kernel is commonly used in constellation decoding for digital
 * modulation schemes such as QAM and PSK. By computing the squared distance
 * from a received sample to every constellation reference point, a
 * minimum-distance detector can identify the most likely transmitted symbol
 * without the cost of a square-root operation.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_square_dist_32f(float* target, const lv_32fc_t* src0,
 *   const lv_32fc_t* points, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The received complex sample (lv_32fc_t). Only the first element is used.
 * \li points: A complex vector of constellation reference points (lv_32fc_t).
 * \li num_points: The number of constellation points.
 *
 * \b Outputs
 * \li target: The squared distance from src0[0] to each constellation point (float).
 *
 * \b Example
 * Compute squared distances from a received sample to four constellation points.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* src = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t), alignment);
 * lv_32fc_t* pts = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * float* out = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * src[0] = lv_cmake(1.0f, 2.0f);
 * pts[0] = lv_cmake(0.0f, 0.0f);  // |1+2j|^2 = 5
 * pts[1] = lv_cmake(1.0f, 1.0f);  // |0+1j|^2 = 1
 * pts[2] = lv_cmake(3.0f, 4.0f);  // |-2-2j|^2 = 8
 * pts[3] = lv_cmake(1.0f, 2.0f);  // |0+0j|^2 = 0
 *
 * volk_32fc_x2_square_dist_32f(out, src, pts, N);
 *
 * // Expected: 5.0, 1.0, 8.0, 0.0
 * printf("Expected: 5.0, 1.0, 8.0, 0.0\n");
 * printf("Result:   %.1f, %.1f, %.1f, %.1f\n", out[0], out[1], out[2], out[3]);
 *
 * volk_free(src);
 * volk_free(pts);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_square_dist_32f_a_H
#define INCLUDED_volk_32fc_x2_square_dist_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_avx2(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;
    __m128 xmm0, xmm9, xmm10;
    __m256 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    lv_32fc_t diff;
    float sq_dist;
    int bound = num_bytes >> 6;
    int leftovers0 = (num_bytes >> 5) & 1;
    int leftovers1 = (num_bytes >> 4) & 1;
    int leftovers2 = (num_bytes >> 3) & 1;
    int i = 0;

    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    xmm1 = _mm256_setzero_ps();
    xmm0 = _mm_load_ps((float*)src0);
    xmm0 = _mm_permute_ps(xmm0, 0b01000100);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 0);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 1);

    for (; i < bound; ++i) {
        xmm2 = _mm256_load_ps((float*)&points[0]);
        xmm3 = _mm256_load_ps((float*)&points[4]);
        points += 8;

        xmm4 = _mm256_sub_ps(xmm1, xmm2);
        xmm5 = _mm256_sub_ps(xmm1, xmm3);
        xmm6 = _mm256_mul_ps(xmm4, xmm4);
        xmm7 = _mm256_mul_ps(xmm5, xmm5);

        xmm4 = _mm256_hadd_ps(xmm6, xmm7);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        _mm256_store_ps(target, xmm4);

        target += 8;
    }

    for (i = 0; i < leftovers0; ++i) {

        xmm2 = _mm256_load_ps((float*)&points[0]);

        xmm4 = _mm256_sub_ps(xmm1, xmm2);

        points += 4;

        xmm6 = _mm256_mul_ps(xmm4, xmm4);

        xmm4 = _mm256_hadd_ps(xmm6, xmm6);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        xmm9 = _mm256_extractf128_ps(xmm4, 1);
        _mm_store_ps(target, xmm9);

        target += 4;
    }

    for (i = 0; i < leftovers1; ++i) {
        xmm9 = _mm_load_ps((float*)&points[0]);

        xmm10 = _mm_sub_ps(xmm0, xmm9);

        points += 2;

        xmm9 = _mm_mul_ps(xmm10, xmm10);

        xmm10 = _mm_hadd_ps(xmm9, xmm9);

        _mm_storeh_pi((__m64*)target, xmm10);

        target += 2;
    }

    for (i = 0; i < leftovers2; ++i) {

        diff = src0[0] - points[0];

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[0] = sq_dist;
    }
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_sse3(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;

    __m128 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    lv_32fc_t diff;
    float sq_dist;
    int bound = num_bytes >> 5;
    int i = 0;

    xmm1 = _mm_setzero_ps();
    xmm1 = _mm_loadl_pi(xmm1, (__m64*)src0);
    xmm1 = _mm_movelh_ps(xmm1, xmm1);

    for (; i < bound; ++i) {
        xmm2 = _mm_load_ps((float*)&points[0]);
        xmm4 = _mm_sub_ps(xmm1, xmm2);
        xmm3 = _mm_load_ps((float*)&points[2]);
        xmm5 = _mm_sub_ps(xmm1, xmm3);

        xmm6 = _mm_mul_ps(xmm4, xmm4);
        xmm7 = _mm_mul_ps(xmm5, xmm5);

        xmm4 = _mm_hadd_ps(xmm6, xmm7);

        _mm_store_ps(target, xmm4);

        points += 4;
        target += 4;
    }

    if (num_bytes >> 4 & 1) {

        xmm2 = _mm_load_ps((float*)&points[0]);

        xmm4 = _mm_sub_ps(xmm1, xmm2);

        points += 2;

        xmm6 = _mm_mul_ps(xmm4, xmm4);

        xmm4 = _mm_hadd_ps(xmm6, xmm6);

        _mm_storeh_pi((__m64*)target, xmm4);

        target += 2;
    }

    if (num_bytes >> 3 & 1) {

        diff = src0[0] - points[0];

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[0] = sq_dist;
    }
}

#endif /*LV_HAVE_SSE3*/


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_32fc_x2_square_dist_32f_neon(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    unsigned int number;

    float32x4x2_t a_vec, b_vec;
    float32x4x2_t diff_vec;
    float32x4_t tmp, tmp1, dist_sq;
    a_vec.val[0] = vdupq_n_f32(lv_creal(src0[0]));
    a_vec.val[1] = vdupq_n_f32(lv_cimag(src0[0]));
    for (number = 0; number < quarter_points; ++number) {
        b_vec = vld2q_f32((float*)points);
        diff_vec.val[0] = vsubq_f32(a_vec.val[0], b_vec.val[0]);
        diff_vec.val[1] = vsubq_f32(a_vec.val[1], b_vec.val[1]);
        tmp = vmulq_f32(diff_vec.val[0], diff_vec.val[0]);
        tmp1 = vmulq_f32(diff_vec.val[1], diff_vec.val[1]);

        dist_sq = vaddq_f32(tmp, tmp1);
        vst1q_f32(target, dist_sq);
        points += 4;
        target += 4;
    }
    for (number = quarter_points * 4; number < num_points; ++number) {
        lv_32fc_t diff = src0[0] - *points++;
        *target++ = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_x2_square_dist_32f_neonv8(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    unsigned int number;

    float32x4x2_t b_vec;
    float32x4_t diff_real, diff_imag, dist_sq;
    float32x4_t a_real = vdupq_n_f32(lv_creal(src0[0]));
    float32x4_t a_imag = vdupq_n_f32(lv_cimag(src0[0]));

    for (number = 0; number < quarter_points; ++number) {
        b_vec = vld2q_f32((float*)points);
        __VOLK_PREFETCH(points + 8);

        diff_real = vsubq_f32(a_real, b_vec.val[0]);
        diff_imag = vsubq_f32(a_imag, b_vec.val[1]);

        /* dist_sq = diff_real^2 + diff_imag^2 using FMA */
        dist_sq = vfmaq_f32(vmulq_f32(diff_real, diff_real), diff_imag, diff_imag);

        vst1q_f32(target, dist_sq);
        points += 4;
        target += 4;
    }

    for (number = quarter_points * 4; number < num_points; ++number) {
        lv_32fc_t diff = src0[0] - *points++;
        *target++ = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_GENERIC
static inline void volk_32fc_x2_square_dist_32f_generic(float* target,
                                                        const lv_32fc_t* src0,
                                                        const lv_32fc_t* points,
                                                        unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;

    lv_32fc_t diff;
    float sq_dist;
    unsigned int i = 0;

    for (; i < (num_bytes >> 3); ++i) {
        diff = src0[0] - points[i];

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[i] = sq_dist;
    }
}

#endif /*LV_HAVE_GENERIC*/


#endif /*INCLUDED_volk_32fc_x2_square_dist_32f_a_H*/

#ifndef INCLUDED_volk_32fc_x2_square_dist_32f_u_H
#define INCLUDED_volk_32fc_x2_square_dist_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_u_avx2(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;
    __m128 xmm0, xmm9;
    __m256 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    lv_32fc_t diff;
    float sq_dist;
    int bound = num_bytes >> 6;
    int leftovers1 = (num_bytes >> 3) & 0b11;
    int i = 0;

    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    xmm1 = _mm256_setzero_ps();
    xmm0 = _mm_loadu_ps((float*)src0);
    xmm0 = _mm_permute_ps(xmm0, 0b01000100);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 0);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 1);

    for (; i < bound; ++i) {
        xmm2 = _mm256_loadu_ps((float*)&points[0]);
        xmm3 = _mm256_loadu_ps((float*)&points[4]);
        points += 8;

        xmm4 = _mm256_sub_ps(xmm1, xmm2);
        xmm5 = _mm256_sub_ps(xmm1, xmm3);
        xmm6 = _mm256_mul_ps(xmm4, xmm4);
        xmm7 = _mm256_mul_ps(xmm5, xmm5);

        xmm4 = _mm256_hadd_ps(xmm6, xmm7);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        _mm256_storeu_ps(target, xmm4);

        target += 8;
    }

    if (num_bytes >> 5 & 1) {

        xmm2 = _mm256_loadu_ps((float*)&points[0]);

        xmm4 = _mm256_sub_ps(xmm1, xmm2);

        points += 4;

        xmm6 = _mm256_mul_ps(xmm4, xmm4);

        xmm4 = _mm256_hadd_ps(xmm6, xmm6);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        xmm9 = _mm256_extractf128_ps(xmm4, 1);
        _mm_storeu_ps(target, xmm9);

        target += 4;
    }

    for (i = 0; i < leftovers1; ++i) {

        diff = src0[0] - points[0];
        points += 1;

        sq_dist = lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff);

        target[0] = sq_dist;
        target += 1;
    }
}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_x2_square_dist_32f_rvv(float* target,
                                                    const lv_32fc_t* src0,
                                                    const lv_32fc_t* points,
                                                    unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t var = __riscv_vfmv_v_f_f32m4(lv_creal(*src0), vlmax);
    vfloat32m4_t vai = __riscv_vfmv_v_f_f32m4(lv_cimag(*src0), vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, target += vl, points += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vb = __riscv_vle64_v_u64m8((const uint64_t*)points, vl);
        vfloat32m4_t vbr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vb, 0, vl));
        vfloat32m4_t vbi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vb, 32, vl));
        vfloat32m4_t vr = __riscv_vfsub(var, vbr, vl);
        vfloat32m4_t vi = __riscv_vfsub(vai, vbi, vl);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(target, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_x2_square_dist_32f_rvvseg(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t var = __riscv_vfmv_v_f_f32m4(lv_creal(*src0), vlmax);
    vfloat32m4_t vai = __riscv_vfmv_v_f_f32m4(lv_cimag(*src0), vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, target += vl, points += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vb = __riscv_vlseg2e32_v_f32m4x2((const float*)points, vl);
        vfloat32m4_t vbr = __riscv_vget_f32m4(vb, 0);
        vfloat32m4_t vbi = __riscv_vget_f32m4(vb, 1);
        vfloat32m4_t vr = __riscv_vfsub(var, vbr, vl);
        vfloat32m4_t vi = __riscv_vfsub(vai, vbi, vl);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(target, v, vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /*INCLUDED_volk_32fc_x2_square_dist_32f_u_H*/
