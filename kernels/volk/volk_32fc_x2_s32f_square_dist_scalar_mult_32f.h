/* -*- c++ -*- */
/*
 * Copyright 2012, 2014, 2019 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_s32f_square_dist_scalar_mult_32f
 *
 * \b Overview
 *
 * Computes the scaled squared distance from a single complex sample to each
 * point in a complex reference vector:
 * target[i] = scalar * |src0[0] - points[i]|^2.
 *
 * This kernel is commonly used in digital demodulation to compute soft-decision
 * metrics for constellation decoding. Given a received sample and a set of
 * constellation points (e.g. QAM or PSK symbols), it produces a distance
 * measure for each candidate symbol, scaled by a factor such as SNR or
 * inverse constellation energy for normalization.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_s32f_square_dist_scalar_mult_32f(float* target, const lv_32fc_t* src0, const lv_32fc_t* points, float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The received complex sample (lv_32fc_t). Only the first element is used.
 * \li points: A vector of complex constellation reference points (lv_32fc_t).
 * \li scalar: A float scaling factor applied to each squared distance (e.g. SNR or normalization constant).
 * \li num_points: The number of constellation points.
 *
 * \b Outputs
 * \li target: A vector of scaled squared distances (float), one per constellation point.
 *
 * \b Example
 * Compute scaled squared distances from a received sample to four QPSK constellation points.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* rx = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t), alignment);
 * lv_32fc_t* constellation = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * float* out = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * *rx = lv_cmake(3.0f, 4.0f);
 * constellation[0] = lv_cmake(1.0f, 1.0f);
 * constellation[1] = lv_cmake(1.0f, -1.0f);
 * constellation[2] = lv_cmake(-1.0f, 1.0f);
 * constellation[3] = lv_cmake(-1.0f, -1.0f);
 * float scalar = 0.5f;
 *
 * // Expected: |rx - constellation[i]|^2 * scalar
 * // [0]: (4+9)*0.5 = 6.5, [1]: (4+25)*0.5 = 14.5,
 * // [2]: (16+9)*0.5 = 12.5, [3]: (16+25)*0.5 = 20.5
 *
 * volk_32fc_x2_s32f_square_dist_scalar_mult_32f(out, rx, constellation, scalar, N);
 *
 * printf("Expected: 6.5, 14.5, 12.5, 20.5\n");
 * printf("Result:   %.1f, %.1f, %.1f, %.1f\n", out[0], out[1], out[2], out[3]);
 *
 * volk_free(rx);
 * volk_free(constellation);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_H
#define INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_H

#include <volk/volk_complex.h>


static inline void calculate_scaled_distances(float* target,
                                              const lv_32fc_t symbol,
                                              const lv_32fc_t* points,
                                              const float scalar,
                                              const unsigned int num_points)
{
    lv_32fc_t diff;
    for (unsigned int i = 0; i < num_points; ++i) {
        /*
         * Calculate: |y - x|^2 * SNR_lin
         * Compare C++: *target++ = scalar * std::norm(symbol - *constellation++);
         */
        diff = symbol - *points++;
        *target++ =
            scalar * (lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff));
    }
}


#ifdef LV_HAVE_GENERIC
static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(float* target,
                                                      const lv_32fc_t* src0,
                                                      const lv_32fc_t* points,
                                                      float scalar,
                                                      unsigned int num_points)
{
    const lv_32fc_t symbol = *src0;
    calculate_scaled_distances(target, symbol, points, scalar, num_points);
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>
static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_sse(float* target,
                                                    const lv_32fc_t* src0,
                                                    const lv_32fc_t* points,
                                                    float scalar,
                                                    unsigned int num_points)
{
    const __m128 xmm_scalar = _mm_set1_ps(scalar);
    const __m128 xmm_symbol = _mm_castpd_ps(_mm_load1_pd((const double*)src0));

    for (unsigned i = 0; i < num_points / 4; ++i) {
        __m128 xmm_points0 = _mm_loadu_ps((const float*)points);
        __m128 xmm_points1 = _mm_loadu_ps((const float*)(points + 2));
        points += 4;
        __m128 xmm_result = _mm_scaled_norm_dist_ps_sse(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);
        _mm_storeu_ps((float*)target, xmm_result);
        target += 4;
    }

    calculate_scaled_distances(target, src0[0], points, scalar, num_points % 4);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_sse3(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    __m128 xmm_points0, xmm_points1, xmm_result;

    /*
     * First do 4 values in every loop iteration.
     * There may be up to 3 values left.
     * leftovers0 indicates if at least 2 more are available for SSE execution.
     * leftovers1 indicates if there is a single element left.
     */
    const int quarterPoints = num_points / 4;
    const int leftovers0 = (num_points / 2) - 2 * quarterPoints;
    const int leftovers1 = num_points % 2;

    // load complex value into both parts of the register.
    const __m128 xmm_symbol = _mm_castpd_ps(_mm_load1_pd((const double*)src0));

    // Load scalar into all 4 parts of the register
    const __m128 xmm_scalar = _mm_load1_ps(&scalar);

    for (int i = 0; i < quarterPoints; ++i) {
        xmm_points0 = _mm_loadu_ps((const float*)points);
        xmm_points1 = _mm_loadu_ps((const float*)(points + 2));
        points += 4;
        __VOLK_PREFETCH(points);
        // calculate distances
        xmm_result = _mm_scaled_norm_dist_ps_sse3(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);

        _mm_storeu_ps(target, xmm_result);
        target += 4;
    }

    for (int i = 0; i < leftovers0; ++i) {
        xmm_points0 = _mm_loadu_ps((const float*)points);
        points += 2;

        xmm_points0 = _mm_sub_ps(xmm_symbol, xmm_points0);
        xmm_points0 = _mm_mul_ps(xmm_points0, xmm_points0);
        xmm_points0 = _mm_hadd_ps(xmm_points0, xmm_points0);
        xmm_result = _mm_mul_ps(xmm_points0, xmm_scalar);

        _mm_storeh_pi((__m64*)target, xmm_result);
        target += 2;
    }

    calculate_scaled_distances(target, src0[0], points, scalar, leftovers1);
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_avx(float* target,
                                                    const lv_32fc_t* src0,
                                                    const lv_32fc_t* points,
                                                    float scalar,
                                                    unsigned int num_points)
{
    const int eightsPoints = num_points / 8;
    const int remainder = num_points - 8 * eightsPoints;

    __m256 xmm_points0, xmm_points1, xmm_result;

    // load complex value into all parts of the register.
    const __m256 xmm_symbol = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)src0));

    // Load scalar into all 8 parts of the register
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);

    for (int i = 0; i < eightsPoints; ++i) {
        xmm_points0 = _mm256_loadu_ps((const float*)points);
        xmm_points1 = _mm256_loadu_ps((const float*)(points + 4));
        points += 8;

        xmm_result = _mm256_scaled_norm_dist_ps(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);

        _mm256_storeu_ps(target, xmm_result);
        target += 8;
    }

    const lv_32fc_t symbol = *src0;
    calculate_scaled_distances(target, symbol, points, scalar, remainder);
}

#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_avx_fma(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    /* Broadcast complex symbol and scalar */
    const __m256 xmm_symbol = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)src0));
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);

    for (unsigned int i = 0; i < eighthPoints; ++i) {
        __m256 pts0 = _mm256_loadu_ps((const float*)points);
        __m256 pts1 = _mm256_loadu_ps((const float*)(points + 4));
        points += 8;

        __m256 diff0 = _mm256_sub_ps(xmm_symbol, pts0);
        __m256 diff1 = _mm256_sub_ps(xmm_symbol, pts1);

        /* Deinterleave real/imag within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(3, 1, 3, 1));

        /* Squared magnitude using FMA: re*re + im*im */
        __m256 mag_sq = _mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im));

        /* Scale */
        __m256 scaled = _mm256_mul_ps(mag_sq, xmm_scalar);

        /* Fix cross-lane ordering: [0,1,4,5 | 2,3,6,7] -> [0,1,2,3 | 4,5,6,7] */
        __m128 lo = _mm256_castps256_ps128(scaled);
        __m128 hi = _mm256_extractf128_ps(scaled, 1);
        __m128 first4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 last4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_insertf128_ps(_mm256_castps128_ps256(first4), last4, 1);

        _mm256_storeu_ps(target, result);
        target += 8;
    }

    volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(
        target, src0, points, scalar, num_points - eighthPoints * 8);
}

#endif /*LV_HAVE_AVX && LV_HAVE_FMA*/

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_avx2(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;
    __m128 xmm9, xmm10;
    __m256 xmm4, xmm6;
    __m256 xmm_points0, xmm_points1, xmm_result;

    const unsigned int bound = num_bytes >> 6;

    // load complex value into all parts of the register.
    const __m256 xmm_symbol = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)src0));
    const __m128 xmm128_symbol = _mm256_extractf128_ps(xmm_symbol, 1);

    // Load scalar into all 8 parts of the register
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);
    const __m128 xmm128_scalar = _mm256_extractf128_ps(xmm_scalar, 1);

    // Set permutation constant
    const __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (unsigned int i = 0; i < bound; ++i) {
        xmm_points0 = _mm256_loadu_ps((const float*)points);
        xmm_points1 = _mm256_loadu_ps((const float*)(points + 4));
        points += 8;
        __VOLK_PREFETCH(points);

        xmm_result = _mm256_scaled_norm_dist_ps_avx2(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);

        _mm256_storeu_ps(target, xmm_result);
        target += 8;
    }

    if (num_bytes >> 5 & 1) {
        xmm_points0 = _mm256_loadu_ps((const float*)points);

        xmm4 = _mm256_sub_ps(xmm_symbol, xmm_points0);

        points += 4;

        xmm6 = _mm256_mul_ps(xmm4, xmm4);

        xmm4 = _mm256_hadd_ps(xmm6, xmm6);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        xmm_result = _mm256_mul_ps(xmm4, xmm_scalar);

        xmm9 = _mm256_extractf128_ps(xmm_result, 1);
        _mm_storeu_ps(target, xmm9);
        target += 4;
    }

    if (num_bytes >> 4 & 1) {
        xmm9 = _mm_loadu_ps((const float*)points);

        xmm10 = _mm_sub_ps(xmm128_symbol, xmm9);

        points += 2;

        xmm9 = _mm_mul_ps(xmm10, xmm10);

        xmm10 = _mm_hadd_ps(xmm9, xmm9);

        xmm10 = _mm_mul_ps(xmm10, xmm128_scalar);

        _mm_storeh_pi((__m64*)target, xmm10);
        target += 2;
    }

    calculate_scaled_distances(target, src0[0], points, scalar, (num_bytes >> 3) & 1);
}

#endif /*LV_HAVE_AVX2*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_avx2_fma(float* target,
                                                           const lv_32fc_t* src0,
                                                           const lv_32fc_t* points,
                                                           float scalar,
                                                           unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m256 xmm_symbol = _mm256_castpd_ps(
        _mm256_broadcast_sd((const double*)src0));
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);

    for (unsigned int i = 0; i < eighthPoints; ++i) {
        __m256 pts0 = _mm256_loadu_ps((const float*)points);
        __m256 pts1 = _mm256_loadu_ps((const float*)(points + 4));
        points += 8;

        __m256 result = _mm256_scaled_norm_dist_ps_avx2_fma(
            xmm_symbol, xmm_symbol, pts0, pts1, xmm_scalar);

        _mm256_storeu_ps(target, result);
        target += 8;
    }

    volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(
        target, src0, points, scalar, num_points - eighthPoints * 8);
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_neon(float* target,
                                                   const lv_32fc_t* src0,
                                                   const lv_32fc_t* points,
                                                   float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    // Load the reference symbol real and imag into vectors
    const float32x4_t symbolReal = vdupq_n_f32(lv_creal(*src0));
    const float32x4_t symbolImag = vdupq_n_f32(lv_cimag(*src0));
    const float32x4_t vScalar = vdupq_n_f32(scalar);

    for (; number < quarterPoints; number++) {
        // Load 4 complex points (8 floats) and deinterleave
        float32x4x2_t pts = vld2q_f32((const float*)points);
        points += 4;

        // Calculate difference
        float32x4_t diffReal = vsubq_f32(symbolReal, pts.val[0]);
        float32x4_t diffImag = vsubq_f32(symbolImag, pts.val[1]);

        // Calculate squared magnitude and scale
        float32x4_t result = vmulq_f32(diffReal, diffReal);
        result = vmlaq_f32(result, diffImag, diffImag);
        result = vmulq_f32(result, vScalar);

        vst1q_f32(target, result);
        target += 4;
    }

    // Handle remaining points
    calculate_scaled_distances(
        target, *src0, points, scalar, num_points - quarterPoints * 4);
}

#endif /*LV_HAVE_NEON*/

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_neonv8(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    // Load the reference symbol real and imag into vectors
    const float32x4_t symbolReal = vdupq_n_f32(lv_creal(*src0));
    const float32x4_t symbolImag = vdupq_n_f32(lv_cimag(*src0));
    const float32x4_t vScalar = vdupq_n_f32(scalar);

    for (; number < eighthPoints; number++) {
        __VOLK_PREFETCH(points + 16);

        // Load 8 complex points (16 floats) and deinterleave
        float32x4x2_t pts0 = vld2q_f32((const float*)points);
        float32x4x2_t pts1 = vld2q_f32((const float*)(points + 4));
        points += 8;

        // Calculate difference
        float32x4_t diffReal0 = vsubq_f32(symbolReal, pts0.val[0]);
        float32x4_t diffImag0 = vsubq_f32(symbolImag, pts0.val[1]);
        float32x4_t diffReal1 = vsubq_f32(symbolReal, pts1.val[0]);
        float32x4_t diffImag1 = vsubq_f32(symbolImag, pts1.val[1]);

        // Calculate squared magnitude: real^2 + imag^2 using FMA
        float32x4_t result0 =
            vfmaq_f32(vmulq_f32(diffReal0, diffReal0), diffImag0, diffImag0);
        float32x4_t result1 =
            vfmaq_f32(vmulq_f32(diffReal1, diffReal1), diffImag1, diffImag1);

        // Scale
        result0 = vmulq_f32(result0, vScalar);
        result1 = vmulq_f32(result1, vScalar);

        vst1q_f32(target, result0);
        vst1q_f32(target + 4, result1);
        target += 8;
    }

    // Handle remaining points
    const unsigned int remaining = num_points - eighthPoints * 8;
    calculate_scaled_distances(target, *src0, points, scalar, remaining);
}

#endif /*LV_HAVE_NEONV8*/

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_rvv(float* target,
                                                  const lv_32fc_t* src0,
                                                  const lv_32fc_t* points,
                                                  float scalar,
                                                  unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t var = __riscv_vfmv_v_f_f32m4(lv_creal(*src0), vlmax);
    vfloat32m4_t vai = __riscv_vfmv_v_f_f32m4(lv_cimag(*src0), vlmax);
    vfloat32m4_t vscale = __riscv_vfmv_v_f_f32m4(scalar, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, target += vl, points += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vb = __riscv_vle64_v_u64m8((const uint64_t*)points, vl);
        vfloat32m4_t vbr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vb, 0, vl));
        vfloat32m4_t vbi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vb, 32, vl));
        vfloat32m4_t vr = __riscv_vfsub(var, vbr, vl);
        vfloat32m4_t vi = __riscv_vfsub(vai, vbi, vl);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(target, __riscv_vfmul(v, vscale, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_rvvseg(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t var = __riscv_vfmv_v_f_f32m4(lv_creal(*src0), vlmax);
    vfloat32m4_t vai = __riscv_vfmv_v_f_f32m4(lv_cimag(*src0), vlmax);
    vfloat32m4_t vscale = __riscv_vfmv_v_f_f32m4(scalar, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, target += vl, points += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vb = __riscv_vlseg2e32_v_f32m4x2((const float*)points, vl);
        vfloat32m4_t vbr = __riscv_vget_f32m4(vb, 0);
        vfloat32m4_t vbi = __riscv_vget_f32m4(vb, 1);
        vfloat32m4_t vr = __riscv_vfsub(var, vbr, vl);
        vfloat32m4_t vi = __riscv_vfsub(vai, vbi, vl);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(target, __riscv_vfmul(v, vscale, vl), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_avx512f(float* target,
                                                         const lv_32fc_t* src0,
                                                         const lv_32fc_t* points,
                                                         float scalar,
                                                         unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    /* Broadcast src0 real and imag */
    const float src0_real = lv_creal(src0[0]);
    const float src0_imag = lv_cimag(src0[0]);
    const __m512 sym_real = _mm512_set1_ps(src0_real);
    const __m512 sym_imag = _mm512_set1_ps(src0_imag);
    const __m512 vscalar = _mm512_set1_ps(scalar);

    /* Indices for deinterleaving complex float pairs */
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

        /* Scale */
        _mm512_storeu_ps(target, _mm512_mul_ps(dist_sq, vscalar));
        target += 16;
    }

    volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(
        target, src0, points, scalar, num_points - sixteenthPoints * 16);
}

#endif /*LV_HAVE_AVX512F*/


#endif /*INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_u_H*/

#ifndef INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_H
#define INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_H

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>
static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_sse(float* target,
                                                    const lv_32fc_t* src0,
                                                    const lv_32fc_t* points,
                                                    float scalar,
                                                    unsigned int num_points)
{
    const __m128 xmm_scalar = _mm_set1_ps(scalar);
    const __m128 xmm_symbol = _mm_castpd_ps(_mm_load1_pd((const double*)src0));

    for (unsigned i = 0; i < num_points / 4; ++i) {
        __m128 xmm_points0 = _mm_load_ps((const float*)points);
        __m128 xmm_points1 = _mm_load_ps((const float*)(points + 2));
        points += 4;
        __m128 xmm_result = _mm_scaled_norm_dist_ps_sse(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);
        _mm_store_ps((float*)target, xmm_result);
        target += 4;
    }

    calculate_scaled_distances(target, src0[0], points, scalar, num_points % 4);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_sse3(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    __m128 xmm_points0, xmm_points1, xmm_result;

    /*
     * First do 4 values in every loop iteration.
     * There may be up to 3 values left.
     * leftovers0 indicates if at least 2 more are available for SSE execution.
     * leftovers1 indicates if there is a single element left.
     */
    const int quarterPoints = num_points / 4;
    const int leftovers0 = (num_points / 2) - 2 * quarterPoints;
    const int leftovers1 = num_points % 2;

    // load complex value into both parts of the register.
    const __m128 xmm_symbol = _mm_castpd_ps(_mm_load1_pd((const double*)src0));

    // Load scalar into all 4 parts of the register
    const __m128 xmm_scalar = _mm_load1_ps(&scalar);

    for (int i = 0; i < quarterPoints; ++i) {
        xmm_points0 = _mm_load_ps((const float*)points);
        xmm_points1 = _mm_load_ps((const float*)(points + 2));
        points += 4;
        __VOLK_PREFETCH(points);
        // calculate distances
        xmm_result = _mm_scaled_norm_dist_ps_sse3(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);

        _mm_store_ps(target, xmm_result);
        target += 4;
    }

    for (int i = 0; i < leftovers0; ++i) {
        xmm_points0 = _mm_load_ps((const float*)points);
        points += 2;

        xmm_points0 = _mm_sub_ps(xmm_symbol, xmm_points0);
        xmm_points0 = _mm_mul_ps(xmm_points0, xmm_points0);
        xmm_points0 = _mm_hadd_ps(xmm_points0, xmm_points0);
        xmm_result = _mm_mul_ps(xmm_points0, xmm_scalar);

        _mm_storeh_pi((__m64*)target, xmm_result);
        target += 2;
    }

    calculate_scaled_distances(target, src0[0], points, scalar, leftovers1);
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_avx(float* target,
                                                    const lv_32fc_t* src0,
                                                    const lv_32fc_t* points,
                                                    float scalar,
                                                    unsigned int num_points)
{
    const int eightsPoints = num_points / 8;
    const int remainder = num_points - 8 * eightsPoints;

    __m256 xmm_points0, xmm_points1, xmm_result;

    // load complex value into all parts of the register.
    const __m256 xmm_symbol = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)src0));

    // Load scalar into all 8 parts of the register
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);

    for (int i = 0; i < eightsPoints; ++i) {
        xmm_points0 = _mm256_load_ps((const float*)points);
        xmm_points1 = _mm256_load_ps((const float*)(points + 4));
        points += 8;

        xmm_result = _mm256_scaled_norm_dist_ps(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);

        _mm256_store_ps(target, xmm_result);
        target += 8;
    }

    const lv_32fc_t symbol = *src0;
    calculate_scaled_distances(target, symbol, points, scalar, remainder);
}

#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_avx_fma(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    /* Broadcast complex symbol and scalar */
    const __m256 xmm_symbol = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)src0));
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);

    for (unsigned int i = 0; i < eighthPoints; ++i) {
        __m256 pts0 = _mm256_load_ps((const float*)points);
        __m256 pts1 = _mm256_load_ps((const float*)(points + 4));
        points += 8;

        __m256 diff0 = _mm256_sub_ps(xmm_symbol, pts0);
        __m256 diff1 = _mm256_sub_ps(xmm_symbol, pts1);

        /* Deinterleave real/imag within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(diff0, diff1, _MM_SHUFFLE(3, 1, 3, 1));

        /* Squared magnitude using FMA: re*re + im*im */
        __m256 mag_sq = _mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im));

        /* Scale */
        __m256 scaled = _mm256_mul_ps(mag_sq, xmm_scalar);

        /* Fix cross-lane ordering: [0,1,4,5 | 2,3,6,7] -> [0,1,2,3 | 4,5,6,7] */
        __m128 lo = _mm256_castps256_ps128(scaled);
        __m128 hi = _mm256_extractf128_ps(scaled, 1);
        __m128 first4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 last4 = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_insertf128_ps(_mm256_castps128_ps256(first4), last4, 1);

        _mm256_store_ps(target, result);
        target += 8;
    }

    volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(
        target, src0, points, scalar, num_points - eighthPoints * 8);
}

#endif /*LV_HAVE_AVX && LV_HAVE_FMA*/

#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_avx2(float* target,
                                                     const lv_32fc_t* src0,
                                                     const lv_32fc_t* points,
                                                     float scalar,
                                                     unsigned int num_points)
{
    const unsigned int num_bytes = num_points * 8;
    __m128 xmm9, xmm10;
    __m256 xmm4, xmm6;
    __m256 xmm_points0, xmm_points1, xmm_result;

    const unsigned int bound = num_bytes >> 6;

    // load complex value into all parts of the register.
    const __m256 xmm_symbol = _mm256_castpd_ps(_mm256_broadcast_sd((const double*)src0));
    const __m128 xmm128_symbol = _mm256_extractf128_ps(xmm_symbol, 1);

    // Load scalar into all 8 parts of the register
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);
    const __m128 xmm128_scalar = _mm256_extractf128_ps(xmm_scalar, 1);

    // Set permutation constant
    const __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (unsigned int i = 0; i < bound; ++i) {
        xmm_points0 = _mm256_load_ps((const float*)points);
        xmm_points1 = _mm256_load_ps((const float*)(points + 4));
        points += 8;
        __VOLK_PREFETCH(points);

        xmm_result = _mm256_scaled_norm_dist_ps_avx2(
            xmm_symbol, xmm_symbol, xmm_points0, xmm_points1, xmm_scalar);

        _mm256_store_ps(target, xmm_result);
        target += 8;
    }

    if (num_bytes >> 5 & 1) {
        xmm_points0 = _mm256_load_ps((const float*)points);

        xmm4 = _mm256_sub_ps(xmm_symbol, xmm_points0);

        points += 4;

        xmm6 = _mm256_mul_ps(xmm4, xmm4);

        xmm4 = _mm256_hadd_ps(xmm6, xmm6);
        xmm4 = _mm256_permutevar8x32_ps(xmm4, idx);

        xmm_result = _mm256_mul_ps(xmm4, xmm_scalar);

        xmm9 = _mm256_extractf128_ps(xmm_result, 1);
        _mm_store_ps(target, xmm9);
        target += 4;
    }

    if (num_bytes >> 4 & 1) {
        xmm9 = _mm_load_ps((const float*)points);

        xmm10 = _mm_sub_ps(xmm128_symbol, xmm9);

        points += 2;

        xmm9 = _mm_mul_ps(xmm10, xmm10);

        xmm10 = _mm_hadd_ps(xmm9, xmm9);

        xmm10 = _mm_mul_ps(xmm10, xmm128_scalar);

        _mm_storeh_pi((__m64*)target, xmm10);
        target += 2;
    }

    calculate_scaled_distances(target, src0[0], points, scalar, (num_bytes >> 3) & 1);
}

#endif /*LV_HAVE_AVX2*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_avx2_fma(float* target,
                                                           const lv_32fc_t* src0,
                                                           const lv_32fc_t* points,
                                                           float scalar,
                                                           unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const __m256 xmm_symbol = _mm256_castpd_ps(
        _mm256_broadcast_sd((const double*)src0));
    const __m256 xmm_scalar = _mm256_broadcast_ss(&scalar);

    for (unsigned int i = 0; i < eighthPoints; ++i) {
        __m256 pts0 = _mm256_load_ps((const float*)points);
        __m256 pts1 = _mm256_load_ps((const float*)(points + 4));
        points += 8;

        __m256 result = _mm256_scaled_norm_dist_ps_avx2_fma(
            xmm_symbol, xmm_symbol, pts0, pts1, xmm_scalar);

        _mm256_store_ps(target, result);
        target += 8;
    }

    volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(
        target, src0, points, scalar, num_points - eighthPoints * 8);
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_avx512f(float* target,
                                                         const lv_32fc_t* src0,
                                                         const lv_32fc_t* points,
                                                         float scalar,
                                                         unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    /* Broadcast src0 real and imag */
    const float src0_real = lv_creal(src0[0]);
    const float src0_imag = lv_cimag(src0[0]);
    const __m512 sym_real = _mm512_set1_ps(src0_real);
    const __m512 sym_imag = _mm512_set1_ps(src0_imag);
    const __m512 vscalar = _mm512_set1_ps(scalar);

    /* Indices for deinterleaving complex float pairs */
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

        /* Scale */
        _mm512_store_ps(target, _mm512_mul_ps(dist_sq, vscalar));
        target += 16;
    }

    volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(
        target, src0, points, scalar, num_points - sixteenthPoints * 16);
}

#endif /*LV_HAVE_AVX512F*/


#endif /*INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_H*/
