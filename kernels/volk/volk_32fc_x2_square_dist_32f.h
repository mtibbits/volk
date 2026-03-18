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
 * Calculates the square distance between a single complex input for each
 * point in a complex vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_square_dist_32f(float* target, const lv_32fc_t* src0, const lv_32fc_t*
 * points, unsigned int num_points) { \endcode
 *
 * \b Inputs
 * \li src0: The complex input. Only the first point is used.
 * \li points: A complex vector of reference points.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li target: A vector of squared distances between src0 and the vector of points.
 *
 * \b Example
 * Calculate the distance between an input and reference points in a square
 * 16-qam constellation.
 * \code
 *   int N = 16;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* constellation  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* rx  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float const_vals[] = {-3, -1, 1, 3};
 *
 *   // Generate 16-QAM constellation points
 *   unsigned int jj = 0;
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       constellation[ii] = lv_cmake(const_vals[ii%4], const_vals[jj]);
 *       if((ii+1)%4 == 0) ++jj;
 *   }
 *
 *   *rx = lv_cmake(0.5f, 2.f);
 *
 *   volk_32fc_x2_square_dist_32f(out, rx, constellation, N);
 *
 *   printf("Distance from each constellation point:\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("%.4f  ", out[ii]);
 *       if((ii+1)%4 == 0) printf("\n");
 *   }
 *
 *   volk_free(rx);
 *   volk_free(constellation);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_square_dist_32f_u_H
#define INCLUDED_volk_32fc_x2_square_dist_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

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

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_x2_square_dist_32f_u_sse(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    unsigned int number = 0;

    __m128 xmm_src = _mm_setzero_ps();
    xmm_src = _mm_loadl_pi(xmm_src, (const __m64*)src0);
    xmm_src = _mm_movelh_ps(xmm_src, xmm_src);

    for (; number < quarter_points; ++number) {
        __m128 pts0 = _mm_loadu_ps((const float*)&points[0]);
        __m128 pts1 = _mm_loadu_ps((const float*)&points[2]);

        __m128 diff0 = _mm_sub_ps(xmm_src, pts0);
        __m128 diff1 = _mm_sub_ps(xmm_src, pts1);

        __m128 sq0 = _mm_mul_ps(diff0, diff0);
        __m128 sq1 = _mm_mul_ps(diff1, diff1);

        /* Swap real²↔imag² within each pair, then add to get squared distances */
        __m128 shuf0 = _mm_shuffle_ps(sq0, sq0, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 shuf1 = _mm_shuffle_ps(sq1, sq1, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sum0 = _mm_add_ps(sq0, shuf0);
        __m128 sum1 = _mm_add_ps(sq1, shuf1);

        /* Pack: pick even-index elements from sum0 and sum1 */
        __m128 result = _mm_shuffle_ps(sum0, sum1, _MM_SHUFFLE(2, 0, 2, 0));

        _mm_storeu_ps(target, result);

        points += 4;
        target += 4;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - quarter_points * 4);
}

#endif /*LV_HAVE_SSE*/

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32fc_x2_square_dist_32f_u_sse3(float* target,
                                                        const lv_32fc_t* src0,
                                                        const lv_32fc_t* points,
                                                        unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;

    __m128 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;

    int bound = num_bytes >> 5;
    int i = 0;

    xmm1 = _mm_setzero_ps();
    xmm1 = _mm_loadl_pi(xmm1, (const __m64*)src0);
    xmm1 = _mm_movelh_ps(xmm1, xmm1);

    for (; i < bound; ++i) {
        xmm2 = _mm_loadu_ps((const float*)&points[0]);
        xmm4 = _mm_sub_ps(xmm1, xmm2);
        xmm3 = _mm_loadu_ps((const float*)&points[2]);
        xmm5 = _mm_sub_ps(xmm1, xmm3);

        xmm6 = _mm_mul_ps(xmm4, xmm4);
        xmm7 = _mm_mul_ps(xmm5, xmm5);

        xmm4 = _mm_hadd_ps(xmm6, xmm7);

        _mm_storeu_ps(target, xmm4);

        points += 4;
        target += 4;
    }

    if (num_bytes >> 4 & 1) {
        xmm2 = _mm_loadu_ps((const float*)&points[0]);

        xmm4 = _mm_sub_ps(xmm1, xmm2);

        points += 2;

        xmm6 = _mm_mul_ps(xmm4, xmm4);

        xmm4 = _mm_hadd_ps(xmm6, xmm6);

        _mm_storeh_pi((__m64*)target, xmm4);

        target += 2;
    }

    if (num_bytes >> 3 & 1) {
        volk_32fc_x2_square_dist_32f_generic(target, src0, points, 1);
    }
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_u_avx(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    unsigned int number = 0;

    /* Broadcast src0 [real, imag] across 256-bit register */
    __m128 xmm0 = _mm_setzero_ps();
    xmm0 = _mm_loadl_pi(xmm0, (const __m64*)src0);
    xmm0 = _mm_movelh_ps(xmm0, xmm0);
    __m256 src_vec = _mm256_castps128_ps256(xmm0);
    src_vec = _mm256_insertf128_ps(src_vec, xmm0, 1);

    for (; number < eighth_points; ++number) {
        __m256 pts0 = _mm256_loadu_ps((const float*)&points[0]);
        __m256 pts1 = _mm256_loadu_ps((const float*)&points[4]);
        points += 8;

        __m256 diff0 = _mm256_sub_ps(src_vec, pts0);
        __m256 diff1 = _mm256_sub_ps(src_vec, pts1);

        __m256 sq0 = _mm256_mul_ps(diff0, diff0);
        __m256 sq1 = _mm256_mul_ps(diff1, diff1);

        /* Swap real²↔imag² within each pair, then add */
        __m256 shuf0 = _mm256_shuffle_ps(sq0, sq0, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 shuf1 = _mm256_shuffle_ps(sq1, sq1, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 sum0 = _mm256_add_ps(sq0, shuf0);
        __m256 sum1 = _mm256_add_ps(sq1, shuf1);

        /* Pack even-index elements (in-lane shuffle) */
        __m256 packed = _mm256_shuffle_ps(sum0, sum1, _MM_SHUFFLE(2, 0, 2, 0));
        /* packed = [d0,d1,d4,d5 | d2,d3,d6,d7] — fix cross-lane order */
        __m128 lo = _mm256_castps256_ps128(packed);
        __m128 hi = _mm256_extractf128_ps(packed, 1);
        __m128 out0 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out1 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));

        _mm_storeu_ps(target, out0);
        _mm_storeu_ps(target + 4, out1);

        target += 8;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - eighth_points * 8);
}

#endif /*LV_HAVE_AVX*/

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_u_avx_fma(float* target,
                                                            const lv_32fc_t* src0,
                                                            const lv_32fc_t* points,
                                                            unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    unsigned int number = 0;

    /* Broadcast src0 [real, imag] across 256-bit register */
    __m128 xmm0 = _mm_setzero_ps();
    xmm0 = _mm_loadl_pi(xmm0, (const __m64*)src0);
    xmm0 = _mm_movelh_ps(xmm0, xmm0);
    __m256 src_vec = _mm256_castps128_ps256(xmm0);
    src_vec = _mm256_insertf128_ps(src_vec, xmm0, 1);

    for (; number < eighth_points; ++number) {
        __m256 pts0 = _mm256_loadu_ps((const float*)points);
        __m256 pts1 = _mm256_loadu_ps((const float*)(points + 4));
        points += 8;

        __m256 diff0 = _mm256_sub_ps(src_vec, pts0);
        __m256 diff1 = _mm256_sub_ps(src_vec, pts1);

        /* Deinterleave re/im from the two diff vectors (within 128-bit lanes) */
        __m256 diff_re = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 diff_im = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(3, 1, 3, 1));

        /* dist² = diff_re² + diff_im² using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 dist_sq = _mm256_fmadd_ps(diff_im, diff_im, _mm256_mul_ps(diff_re, diff_re));

        /* Fix cross-lane order */
        __m128 lo = _mm256_castps256_ps128(dist_sq);
        __m128 hi = _mm256_extractf128_ps(dist_sq, 1);
        __m128 out0 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out1 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));

        _mm_storeu_ps(target, out0);
        _mm_storeu_ps(target + 4, out1);

        target += 8;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

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
    xmm0 = _mm_loadu_ps((const float*)src0);
    xmm0 = _mm_permute_ps(xmm0, 0b01000100);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 0);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 1);

    for (; i < bound; ++i) {
        xmm2 = _mm256_loadu_ps((const float*)&points[0]);
        xmm3 = _mm256_loadu_ps((const float*)&points[4]);
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

        xmm2 = _mm256_loadu_ps((const float*)&points[0]);

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

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32fc_x2_square_dist_32f_u_avx2_fma(float* target,
                                                             const lv_32fc_t* src0,
                                                             const lv_32fc_t* points,
                                                             unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    /* Broadcast src0 as interleaved [re,im,re,im,...] */
    const __m256 xmm_src = _mm256_castpd_ps(
        _mm256_broadcast_sd((const double*)src0));

    for (; number < eighthPoints; number++) {
        __m256 pts0 = _mm256_loadu_ps((const float*)points);
        __m256 pts1 = _mm256_loadu_ps((const float*)(points + 4));
        points += 8;

        __m256 diff0 = _mm256_sub_ps(xmm_src, pts0);
        __m256 diff1 = _mm256_sub_ps(xmm_src, pts1);

        __m256 result = _mm256_magnitudesquared_ps_avx2_fma(diff0, diff1);

        _mm256_storeu_ps(target, result);
        target += 8;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - eighthPoints * 8);
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/

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
        b_vec = vld2q_f32((const float*)points);
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
        b_vec = vld2q_f32((const float*)points);
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


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_u_avx512f(float* target,
                                                           const lv_32fc_t* src0,
                                                           const lv_32fc_t* points,
                                                           unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    /* Broadcast src0 real and imag to separate 512-bit vectors */
    const float src0_real = lv_creal(src0[0]);
    const float src0_imag = lv_cimag(src0[0]);
    const __m512 sym_real = _mm512_set1_ps(src0_real);
    const __m512 sym_imag = _mm512_set1_ps(src0_imag);

    /* Indices for deinterleaving complex float pairs into separate re/im */
    const __m512i idx_re =
        _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im =
        _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        /* Load 16 complex points (32 floats) */
        __m512 pts0 = _mm512_loadu_ps((const float*)points);
        __m512 pts1 = _mm512_loadu_ps((const float*)(points + 8));
        points += 16;

        /* Deinterleave into real and imaginary */
        __m512 pts_real = _mm512_permutex2var_ps(pts0, idx_re, pts1);
        __m512 pts_imag = _mm512_permutex2var_ps(pts0, idx_im, pts1);

        /* Compute difference */
        __m512 diff_real = _mm512_sub_ps(sym_real, pts_real);
        __m512 diff_imag = _mm512_sub_ps(sym_imag, pts_imag);

        /* Compute squared distance: real^2 + imag^2 */
        __m512 dist_sq = _mm512_fmadd_ps(diff_real, diff_real,
                                          _mm512_mul_ps(diff_imag, diff_imag));

        _mm512_storeu_ps(target, dist_sq);
        target += 16;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - sixteenthPoints * 16);
}

#endif /*LV_HAVE_AVX512F*/


#endif /*INCLUDED_volk_32fc_x2_square_dist_32f_u_H*/

#ifndef INCLUDED_volk_32fc_x2_square_dist_32f_a_H
#define INCLUDED_volk_32fc_x2_square_dist_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_sse(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    unsigned int number = 0;

    __m128 xmm_src = _mm_setzero_ps();
    xmm_src = _mm_loadl_pi(xmm_src, (const __m64*)src0);
    xmm_src = _mm_movelh_ps(xmm_src, xmm_src);

    for (; number < quarter_points; ++number) {
        __m128 pts0 = _mm_load_ps((const float*)&points[0]);
        __m128 pts1 = _mm_load_ps((const float*)&points[2]);

        __m128 diff0 = _mm_sub_ps(xmm_src, pts0);
        __m128 diff1 = _mm_sub_ps(xmm_src, pts1);

        __m128 sq0 = _mm_mul_ps(diff0, diff0);
        __m128 sq1 = _mm_mul_ps(diff1, diff1);

        /* Swap real²↔imag² within each pair, then add to get squared distances */
        __m128 shuf0 = _mm_shuffle_ps(sq0, sq0, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 shuf1 = _mm_shuffle_ps(sq1, sq1, _MM_SHUFFLE(2, 3, 0, 1));
        __m128 sum0 = _mm_add_ps(sq0, shuf0);
        __m128 sum1 = _mm_add_ps(sq1, shuf1);

        /* Pack: pick even-index elements from sum0 and sum1 */
        __m128 result = _mm_shuffle_ps(sum0, sum1, _MM_SHUFFLE(2, 0, 2, 0));

        _mm_store_ps(target, result);

        points += 4;
        target += 4;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - quarter_points * 4);
}

#endif /*LV_HAVE_SSE*/

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
    xmm1 = _mm_loadl_pi(xmm1, (const __m64*)src0);
    xmm1 = _mm_movelh_ps(xmm1, xmm1);

    for (; i < bound; ++i) {
        xmm2 = _mm_load_ps((const float*)&points[0]);
        xmm4 = _mm_sub_ps(xmm1, xmm2);
        xmm3 = _mm_load_ps((const float*)&points[2]);
        xmm5 = _mm_sub_ps(xmm1, xmm3);

        xmm6 = _mm_mul_ps(xmm4, xmm4);
        xmm7 = _mm_mul_ps(xmm5, xmm5);

        xmm4 = _mm_hadd_ps(xmm6, xmm7);

        _mm_store_ps(target, xmm4);

        points += 4;
        target += 4;
    }

    if (num_bytes >> 4 & 1) {

        xmm2 = _mm_load_ps((const float*)&points[0]);

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

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_avx(float* target,
                                                       const lv_32fc_t* src0,
                                                       const lv_32fc_t* points,
                                                       unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    unsigned int number = 0;

    /* Broadcast src0 [real, imag] across 256-bit register */
    __m128 xmm0 = _mm_setzero_ps();
    xmm0 = _mm_loadl_pi(xmm0, (const __m64*)src0);
    xmm0 = _mm_movelh_ps(xmm0, xmm0);
    __m256 src_vec = _mm256_castps128_ps256(xmm0);
    src_vec = _mm256_insertf128_ps(src_vec, xmm0, 1);

    for (; number < eighth_points; ++number) {
        __m256 pts0 = _mm256_load_ps((const float*)&points[0]);
        __m256 pts1 = _mm256_load_ps((const float*)&points[4]);
        points += 8;

        __m256 diff0 = _mm256_sub_ps(src_vec, pts0);
        __m256 diff1 = _mm256_sub_ps(src_vec, pts1);

        __m256 sq0 = _mm256_mul_ps(diff0, diff0);
        __m256 sq1 = _mm256_mul_ps(diff1, diff1);

        /* Swap real²↔imag² within each pair, then add */
        __m256 shuf0 = _mm256_shuffle_ps(sq0, sq0, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 shuf1 = _mm256_shuffle_ps(sq1, sq1, _MM_SHUFFLE(2, 3, 0, 1));
        __m256 sum0 = _mm256_add_ps(sq0, shuf0);
        __m256 sum1 = _mm256_add_ps(sq1, shuf1);

        /* Pack even-index elements (in-lane shuffle) */
        __m256 packed = _mm256_shuffle_ps(sum0, sum1, _MM_SHUFFLE(2, 0, 2, 0));
        /* packed = [d0,d1,d4,d5 | d2,d3,d6,d7] — fix cross-lane order */
        __m128 lo = _mm256_castps256_ps128(packed);
        __m128 hi = _mm256_extractf128_ps(packed, 1);
        __m128 out0 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out1 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));

        _mm_store_ps(target, out0);
        _mm_store_ps(target + 4, out1);

        target += 8;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - eighth_points * 8);
}

#endif /*LV_HAVE_AVX*/

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_avx_fma(float* target,
                                                            const lv_32fc_t* src0,
                                                            const lv_32fc_t* points,
                                                            unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    unsigned int number = 0;

    /* Broadcast src0 [real, imag] across 256-bit register */
    __m128 xmm0 = _mm_setzero_ps();
    xmm0 = _mm_loadl_pi(xmm0, (const __m64*)src0);
    xmm0 = _mm_movelh_ps(xmm0, xmm0);
    __m256 src_vec = _mm256_castps128_ps256(xmm0);
    src_vec = _mm256_insertf128_ps(src_vec, xmm0, 1);

    for (; number < eighth_points; ++number) {
        __m256 pts0 = _mm256_load_ps((const float*)points);
        __m256 pts1 = _mm256_load_ps((const float*)(points + 4));
        points += 8;

        __m256 diff0 = _mm256_sub_ps(src_vec, pts0);
        __m256 diff1 = _mm256_sub_ps(src_vec, pts1);

        /* Deinterleave re/im from the two diff vectors (within 128-bit lanes) */
        __m256 diff_re = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 diff_im = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(3, 1, 3, 1));

        /* dist² = diff_re² + diff_im² using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 dist_sq = _mm256_fmadd_ps(diff_im, diff_im, _mm256_mul_ps(diff_re, diff_re));

        /* Fix cross-lane order */
        __m128 lo = _mm256_castps256_ps128(dist_sq);
        __m128 hi = _mm256_extractf128_ps(dist_sq, 1);
        __m128 out0 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out1 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));

        _mm_store_ps(target, out0);
        _mm_store_ps(target + 4, out1);

        target += 8;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

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
    xmm0 = _mm_load_ps((const float*)src0);
    xmm0 = _mm_permute_ps(xmm0, 0b01000100);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 0);
    xmm1 = _mm256_insertf128_ps(xmm1, xmm0, 1);

    for (; i < bound; ++i) {
        xmm2 = _mm256_load_ps((const float*)&points[0]);
        xmm3 = _mm256_load_ps((const float*)&points[4]);
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

        xmm2 = _mm256_load_ps((const float*)&points[0]);

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
        xmm9 = _mm_load_ps((const float*)&points[0]);

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

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32fc_x2_square_dist_32f_a_avx2_fma(float* target,
                                                             const lv_32fc_t* src0,
                                                             const lv_32fc_t* points,
                                                             unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    const __m256 xmm_src = _mm256_castpd_ps(
        _mm256_broadcast_sd((const double*)src0));

    for (; number < eighthPoints; number++) {
        __m256 pts0 = _mm256_load_ps((const float*)points);
        __m256 pts1 = _mm256_load_ps((const float*)(points + 4));
        points += 8;

        __m256 diff0 = _mm256_sub_ps(xmm_src, pts0);
        __m256 diff1 = _mm256_sub_ps(xmm_src, pts1);

        __m256 result = _mm256_magnitudesquared_ps_avx2_fma(diff0, diff1);

        _mm256_store_ps(target, result);
        target += 8;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - eighthPoints * 8);
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_x2_square_dist_32f_a_avx512f(float* target,
                                                           const lv_32fc_t* src0,
                                                           const lv_32fc_t* points,
                                                           unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    /* Broadcast src0 real and imag to separate 512-bit vectors */
    const float src0_real = lv_creal(src0[0]);
    const float src0_imag = lv_cimag(src0[0]);
    const __m512 sym_real = _mm512_set1_ps(src0_real);
    const __m512 sym_imag = _mm512_set1_ps(src0_imag);

    /* Indices for deinterleaving complex float pairs into separate re/im */
    const __m512i idx_re =
        _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im =
        _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        /* Load 16 complex points (32 floats) */
        __m512 pts0 = _mm512_load_ps((const float*)points);
        __m512 pts1 = _mm512_load_ps((const float*)(points + 8));
        points += 16;

        /* Deinterleave into real and imaginary */
        __m512 pts_real = _mm512_permutex2var_ps(pts0, idx_re, pts1);
        __m512 pts_imag = _mm512_permutex2var_ps(pts0, idx_im, pts1);

        /* Compute difference */
        __m512 diff_real = _mm512_sub_ps(sym_real, pts_real);
        __m512 diff_imag = _mm512_sub_ps(sym_imag, pts_imag);

        /* Compute squared distance: real^2 + imag^2 */
        __m512 dist_sq = _mm512_fmadd_ps(diff_real, diff_real,
                                          _mm512_mul_ps(diff_imag, diff_imag));

        _mm512_store_ps(target, dist_sq);
        target += 16;
    }

    volk_32fc_x2_square_dist_32f_generic(
        target, src0, points, num_points - sixteenthPoints * 16);
}

#endif /*LV_HAVE_AVX512F*/


#endif /*INCLUDED_volk_32fc_x2_square_dist_32f_a_H*/
