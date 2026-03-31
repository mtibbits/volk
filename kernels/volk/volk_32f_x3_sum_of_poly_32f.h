/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x3_sum_of_poly_32f
 *
 * \b Overview
 *
 * Evaluates a fourth-order polynomial at each input sample and accumulates
 * the sum of all results, implementing the rectangular method for numerical
 * integration. Input values below a cutoff are clamped before evaluation.
 * To obtain the area under the curve, multiply the result by the bin width.
 *
 * Computes:
 * \f$ \text{target} = \sum_{i} \left( c_0 + c_1 x_i + c_2 x_i^2
 * + c_3 x_i^3 + c_4 x_i^4 \right) \f$
 *
 * where the coefficient array stores {c1, c2, c3, c4, c0}.
 *
 * This kernel is useful in spectral analysis and statistical signal processing
 * where polynomial curve fitting is applied to spectral bins or probability
 * density estimates and the integrated area must be computed efficiently.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x3_sum_of_poly_32f(float* target, float* src0,
 *   float* center_point_array, float* cutoff, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The input sample values (float).
 * \li center_point_array: Polynomial coefficients in order {c1, c2, c3, c4, c0} (float, length 5).
 * \li cutoff: Pointer to the minimum sample value; inputs below this are clamped (float).
 * \li num_points: The number of input samples.
 *
 * \b Outputs
 * \li target: The accumulated polynomial sum (float, scalar).
 *
 * \b Example
 * Evaluate f(x) = 2 + x + x^2 + x^3 + x^4 at four points all equal to 1.0.
 * Each evaluation gives 2+1+1+1+1 = 6, so the sum over 4 points is 24.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * float* src0 = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* coeffs = (float*)volk_malloc(sizeof(float) * 5, alignment);
 * float* result = (float*)volk_malloc(sizeof(float), alignment);
 * float* cutoff = (float*)volk_malloc(sizeof(float), alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     src0[i] = 1.0f;
 * }
 * // coefficients: {c1, c2, c3, c4, c0}
 * coeffs[0] = 1.0f;  // c1
 * coeffs[1] = 1.0f;  // c2
 * coeffs[2] = 1.0f;  // c3
 * coeffs[3] = 1.0f;  // c4
 * coeffs[4] = 2.0f;  // c0
 * *cutoff = 0.0f;
 *
 * float expected = 4.0f * (2.0f + 1.0f + 1.0f + 1.0f + 1.0f);  // 24.0
 *
 * volk_32f_x3_sum_of_poly_32f(result, src0, coeffs, cutoff, N);
 *
 * printf("Expected: %f\n", expected);
 * printf("Result:   %f\n", *result);
 *
 * volk_free(src0);
 * volk_free(coeffs);
 * volk_free(result);
 * volk_free(cutoff);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x3_sum_of_poly_32f_u_H
#define INCLUDED_volk_32f_x3_sum_of_poly_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifndef MAX
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#endif

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x3_sum_of_poly_32f_generic(float* target,
                                                       const float* src0,
                                                       const float* center_point_array,
                                                       const float* cutoff,
                                                       unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    float result[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float fst = 0.0f;
    float sq = 0.0f;
    float thrd = 0.0f;
    float frth = 0.0f;

    unsigned int i = 0;
    unsigned int k = 0;
    for (i = 0; i < eighth_points; ++i) {
        for (k = 0; k < 8; ++k) {
            fst = *src0++;
            fst = MAX(fst, *cutoff);
            sq = fst * fst;
            thrd = fst * sq;
            frth = fst * thrd;
            result[k] += center_point_array[0] * fst + center_point_array[1] * sq;
            result[k] += center_point_array[2] * thrd + center_point_array[3] * frth;
        }
    }
    for (k = 0; k < 8; k += 2) {
        result[k] = result[k] + result[k + 1];
    }

    *target = result[0] + result[2] + result[4] + result[6];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = fst * thrd;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }
    *target += (float)(num_points)*center_point_array[4];
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_u_sse(float* target,
                                                      const float* src0,
                                                      const float* center_point_array,
                                                      const float* cutoff,
                                                      unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    __m128 cpa0 = _mm_load1_ps(&center_point_array[0]);
    __m128 cpa1 = _mm_load1_ps(&center_point_array[1]);
    __m128 cpa2 = _mm_load1_ps(&center_point_array[2]);
    __m128 cpa3 = _mm_load1_ps(&center_point_array[3]);
    __m128 cutoff_vec = _mm_load1_ps(cutoff);
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();

    __m128 x_to_1, x_to_2, x_to_3, x_to_4;

    unsigned int i;
    for (i = 0; i < eighth_points; ++i) {
        // 1st group of 4
        x_to_1 = _mm_loadu_ps(src0);
        x_to_1 = _mm_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm_mul_ps(x_to_1, x_to_1);
        x_to_3 = _mm_mul_ps(x_to_1, x_to_2);
        x_to_4 = _mm_mul_ps(x_to_2, x_to_2);

        x_to_1 = _mm_mul_ps(x_to_1, cpa0);
        x_to_2 = _mm_mul_ps(x_to_2, cpa1);
        x_to_3 = _mm_mul_ps(x_to_3, cpa2);
        x_to_4 = _mm_mul_ps(x_to_4, cpa3);

        x_to_1 = _mm_add_ps(x_to_1, x_to_2);
        x_to_3 = _mm_add_ps(x_to_3, x_to_4);
        acc0 = _mm_add_ps(x_to_1, acc0);
        acc0 = _mm_add_ps(x_to_3, acc0);

        src0 += 4;

        // 2nd group of 4
        x_to_1 = _mm_loadu_ps(src0);
        x_to_1 = _mm_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm_mul_ps(x_to_1, x_to_1);
        x_to_3 = _mm_mul_ps(x_to_1, x_to_2);
        x_to_4 = _mm_mul_ps(x_to_2, x_to_2);

        x_to_1 = _mm_mul_ps(x_to_1, cpa0);
        x_to_2 = _mm_mul_ps(x_to_2, cpa1);
        x_to_3 = _mm_mul_ps(x_to_3, cpa2);
        x_to_4 = _mm_mul_ps(x_to_4, cpa3);

        x_to_1 = _mm_add_ps(x_to_1, x_to_2);
        x_to_3 = _mm_add_ps(x_to_3, x_to_4);
        acc1 = _mm_add_ps(x_to_1, acc1);
        acc1 = _mm_add_ps(x_to_3, acc1);

        src0 += 4;
    }

    // Horizontal reduction via shuffle+add (avoids slow hadd)
    acc0 = _mm_add_ps(acc0, acc1);
    __m128 shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(1, 0, 3, 2));
    acc0 = _mm_add_ps(acc0, shuf);
    shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(0, 1, 0, 1));
    acc0 = _mm_add_ps(acc0, shuf);

    float result;
    _mm_store_ss(&result, acc0);

    // Handle remaining elements via generic
    float tail_result;
    volk_32f_x3_sum_of_poly_32f_generic(
        &tail_result, src0, center_point_array, cutoff, num_points - eighth_points * 8);
    *target = result + tail_result + (float)(eighth_points * 8) * center_point_array[4];
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_u_sse3(float* target,
                                                       const float* src0,
                                                       const float* center_point_array,
                                                       const float* cutoff,
                                                       unsigned int num_points)
{
    float result = 0.0f;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10;

    xmm9 = _mm_setzero_ps();
    xmm1 = _mm_setzero_ps();
    xmm0 = _mm_load1_ps(&center_point_array[0]);
    xmm6 = _mm_load1_ps(&center_point_array[1]);
    xmm7 = _mm_load1_ps(&center_point_array[2]);
    xmm8 = _mm_load1_ps(&center_point_array[3]);
    xmm10 = _mm_load1_ps(cutoff);

    int bound = num_points / 8;
    int leftovers = num_points - 8 * bound;
    int i = 0;
    for (; i < bound; ++i) {
        // 1st
        xmm2 = _mm_loadu_ps(src0);
        xmm2 = _mm_max_ps(xmm10, xmm2);
        xmm3 = _mm_mul_ps(xmm2, xmm2);
        xmm4 = _mm_mul_ps(xmm2, xmm3);
        xmm5 = _mm_mul_ps(xmm3, xmm3);

        xmm2 = _mm_mul_ps(xmm2, xmm0);
        xmm3 = _mm_mul_ps(xmm3, xmm6);
        xmm4 = _mm_mul_ps(xmm4, xmm7);
        xmm5 = _mm_mul_ps(xmm5, xmm8);

        xmm2 = _mm_add_ps(xmm2, xmm3);
        xmm3 = _mm_add_ps(xmm4, xmm5);

        src0 += 4;

        xmm9 = _mm_add_ps(xmm2, xmm9);
        xmm9 = _mm_add_ps(xmm3, xmm9);

        // 2nd
        xmm2 = _mm_loadu_ps(src0);
        xmm2 = _mm_max_ps(xmm10, xmm2);
        xmm3 = _mm_mul_ps(xmm2, xmm2);
        xmm4 = _mm_mul_ps(xmm2, xmm3);
        xmm5 = _mm_mul_ps(xmm3, xmm3);

        xmm2 = _mm_mul_ps(xmm2, xmm0);
        xmm3 = _mm_mul_ps(xmm3, xmm6);
        xmm4 = _mm_mul_ps(xmm4, xmm7);
        xmm5 = _mm_mul_ps(xmm5, xmm8);

        xmm2 = _mm_add_ps(xmm2, xmm3);
        xmm3 = _mm_add_ps(xmm4, xmm5);

        src0 += 4;

        xmm1 = _mm_add_ps(xmm2, xmm1);
        xmm1 = _mm_add_ps(xmm3, xmm1);
    }
    xmm2 = _mm_hadd_ps(xmm9, xmm1);
    xmm3 = _mm_hadd_ps(xmm2, xmm2);
    xmm4 = _mm_hadd_ps(xmm3, xmm3);
    _mm_store_ss(&result, xmm4);

    // Handle remaining elements via generic
    float tail_result;
    volk_32f_x3_sum_of_poly_32f_generic(
        &tail_result, src0, center_point_array, cutoff, leftovers);
    result += tail_result + (float)(num_points - leftovers) * center_point_array[4];
    *target = result;
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_u_avx(float* target,
                                                     const float* src0,
                                                     const float* center_point_array,
                                                     const float* cutoff,
                                                     unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_loadu_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_1 = _mm256_mul_ps(x_to_1, cpa0); // cpa[0] * x^1
        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_3 = _mm256_mul_ps(x_to_3, cpa2); // cpa[2] * x^3
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_add_ps(x_to_1, x_to_2);
        x_to_3 = _mm256_add_ps(x_to_3, x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_storeu_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;

        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }

    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_u_avx_fma(float* target,
                                                         const float* src0,
                                                         const float* center_point_array,
                                                         const float* cutoff,
                                                         unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_loadu_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_fmadd_ps(x_to_1, cpa0, x_to_2);
        x_to_3 = _mm256_fmadd_ps(x_to_3, cpa2, x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_storeu_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }

    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_u_avx2(float* target,
                                                       const float* src0,
                                                       const float* center_point_array,
                                                       const float* cutoff,
                                                       unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_loadu_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_add_ps(_mm256_mul_ps(x_to_1, cpa0), x_to_2);
        x_to_3 = _mm256_add_ps(_mm256_mul_ps(x_to_3, cpa2), x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_storeu_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }

    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_u_avx2_fma(float* target,
                                                           const float* src0,
                                                           const float* center_point_array,
                                                           const float* cutoff,
                                                           unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_loadu_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_fmadd_ps(x_to_1, cpa0, x_to_2);
        x_to_3 = _mm256_fmadd_ps(x_to_3, cpa2, x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_storeu_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }

    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_u_avx512f(float* target,
                                                          const float* src0,
                                                          const float* center_point_array,
                                                          const float* cutoff,
                                                          unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;

    __m512 cpa0 = _mm512_set1_ps(center_point_array[0]);
    __m512 cpa1 = _mm512_set1_ps(center_point_array[1]);
    __m512 cpa2 = _mm512_set1_ps(center_point_array[2]);
    __m512 cpa3 = _mm512_set1_ps(center_point_array[3]);
    __m512 cutoff_vec = _mm512_set1_ps(*cutoff);

    // 4 independent accumulators to saturate FMA pipeline
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    __m512 x_to_1, x_to_2, x_to_3, x_to_4;

    unsigned int i;
    for (i = 0; i < sixteenth_points; ++i) {
        x_to_1 = _mm512_loadu_ps(src0);
        x_to_1 = _mm512_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm512_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm512_mul_ps(x_to_1, x_to_2); // x^3
        x_to_4 = _mm512_mul_ps(x_to_2, x_to_2); // x^4

        // AVX-512F implies FMA
        acc0 = _mm512_fmadd_ps(x_to_1, cpa0, acc0); // cpa[0] * x^1
        acc1 = _mm512_fmadd_ps(x_to_2, cpa1, acc1); // cpa[1] * x^2
        acc2 = _mm512_fmadd_ps(x_to_3, cpa2, acc2); // cpa[2] * x^3
        acc3 = _mm512_fmadd_ps(x_to_4, cpa3, acc3); // cpa[3] * x^4

        src0 += 16;
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    float result = _mm512_reduce_add_ps(acc0);

    // Handle remaining elements via generic
    float tail_result;
    volk_32f_x3_sum_of_poly_32f_generic(
        &tail_result, src0, center_point_array, cutoff, num_points - sixteenth_points * 16);
    *target =
        result + tail_result + (float)(sixteenth_points * 16) * center_point_array[4];
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x3_sum_of_poly_32f_neon(float* __restrict target,
                                                    const float* __restrict src0,
                                                    const float* __restrict center_point_array,
                                                    const float* __restrict cutoff,
                                                    unsigned int num_points)
{
    unsigned int i;
    float zero[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    float accumulator;

    float32x4_t accumulator1_vec, accumulator2_vec, accumulator3_vec, accumulator4_vec;
    accumulator1_vec = vld1q_f32(zero);
    accumulator2_vec = vld1q_f32(zero);
    accumulator3_vec = vld1q_f32(zero);
    accumulator4_vec = vld1q_f32(zero);
    float32x4_t x_to_1, x_to_2, x_to_3, x_to_4;
    float32x4_t cutoff_vector, cpa_0, cpa_1, cpa_2, cpa_3;

    // load the cutoff in to a vector
    cutoff_vector = vdupq_n_f32(*cutoff);
    // ... center point array
    cpa_0 = vdupq_n_f32(center_point_array[0]);
    cpa_1 = vdupq_n_f32(center_point_array[1]);
    cpa_2 = vdupq_n_f32(center_point_array[2]);
    cpa_3 = vdupq_n_f32(center_point_array[3]);

    for (i = 0; i < num_points / 4; ++i) {
        // load x
        x_to_1 = vld1q_f32(src0);

        // Get a vector of max(src0, cutoff)
        x_to_1 = vmaxq_f32(x_to_1, cutoff_vector); // x^1
        x_to_2 = vmulq_f32(x_to_1, x_to_1);        // x^2
        x_to_3 = vmulq_f32(x_to_2, x_to_1);        // x^3
        x_to_4 = vmulq_f32(x_to_3, x_to_1);        // x^4
        x_to_1 = vmulq_f32(x_to_1, cpa_0);
        x_to_2 = vmulq_f32(x_to_2, cpa_1);
        x_to_3 = vmulq_f32(x_to_3, cpa_2);
        x_to_4 = vmulq_f32(x_to_4, cpa_3);
        accumulator1_vec = vaddq_f32(accumulator1_vec, x_to_1);
        accumulator2_vec = vaddq_f32(accumulator2_vec, x_to_2);
        accumulator3_vec = vaddq_f32(accumulator3_vec, x_to_3);
        accumulator4_vec = vaddq_f32(accumulator4_vec, x_to_4);

        src0 += 4;
    }
    accumulator1_vec = vaddq_f32(accumulator1_vec, accumulator2_vec);
    accumulator3_vec = vaddq_f32(accumulator3_vec, accumulator4_vec);
    accumulator1_vec = vaddq_f32(accumulator1_vec, accumulator3_vec);

    __VOLK_ATTR_ALIGNED(32) float res_accumulators[4];
    vst1q_f32(res_accumulators, accumulator1_vec);
    accumulator = res_accumulators[0] + res_accumulators[1] + res_accumulators[2] +
                  res_accumulators[3];

    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    for (i = 4 * (num_points / 4); i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);

        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        // fith = sq * thrd;

        accumulator += (center_point_array[0] * fst + center_point_array[1] * sq +
                        center_point_array[2] * thrd + center_point_array[3] * frth); //+
    }

    *target = accumulator + (float)num_points * center_point_array[4];
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x3_sum_of_poly_32f_neonv8(float* __restrict target,
                                                        const float* __restrict src0,
                                                        const float* __restrict center_point_array,
                                                        const float* __restrict cutoff,
                                                        unsigned int num_points)
{
    unsigned int i;

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    float32x4_t x_to_1, x_to_2, x_to_3, x_to_4;
    float32x4_t cutoff_vector, cpa_0, cpa_1, cpa_2, cpa_3;

    cutoff_vector = vdupq_n_f32(*cutoff);
    cpa_0 = vdupq_n_f32(center_point_array[0]);
    cpa_1 = vdupq_n_f32(center_point_array[1]);
    cpa_2 = vdupq_n_f32(center_point_array[2]);
    cpa_3 = vdupq_n_f32(center_point_array[3]);

    for (i = 0; i < num_points / 4; ++i) {
        x_to_1 = vld1q_f32(src0);

        x_to_1 = vmaxq_f32(x_to_1, cutoff_vector);
        x_to_2 = vmulq_f32(x_to_1, x_to_1);
        x_to_3 = vmulq_f32(x_to_2, x_to_1);
        x_to_4 = vmulq_f32(x_to_3, x_to_1);
        acc0 = vfmaq_f32(acc0, x_to_1, cpa_0);
        acc1 = vfmaq_f32(acc1, x_to_2, cpa_1);
        acc2 = vfmaq_f32(acc2, x_to_3, cpa_2);
        acc3 = vfmaq_f32(acc3, x_to_4, cpa_3);

        src0 += 4;
    }
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    __VOLK_ATTR_ALIGNED(32) float res_accumulators[4];
    vst1q_f32(res_accumulators, acc0);
    float accumulator = res_accumulators[0] + res_accumulators[1] + res_accumulators[2] +
                        res_accumulators[3];

    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    for (i = 4 * (num_points / 4); i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);

        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;

        accumulator += (center_point_array[0] * fst + center_point_array[1] * sq +
                        center_point_array[2] * thrd + center_point_array[3] * frth);
    }

    *target = accumulator + (float)num_points * center_point_array[4];
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32f_x3_sum_of_poly_32f_rvv(float* target,
                                                   const float* src0,
                                                   const float* center_point_array,
                                                   const float* cutoff,
                                                   unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0, vlmax);
    float mul1 = center_point_array[0]; // scalar to avoid register spills
    float mul2 = center_point_array[1];
    vfloat32m4_t vmul3 = __riscv_vfmv_v_f_f32m4(center_point_array[2], vlmax);
    vfloat32m4_t vmul4 = __riscv_vfmv_v_f_f32m4(center_point_array[3], vlmax);
    vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(*cutoff, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, src0 += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(src0, vl);
        vfloat32m4_t v1 = __riscv_vfmax(v, vmax, vl);
        vfloat32m4_t v2 = __riscv_vfmul(v1, v1, vl);
        vfloat32m4_t v3 = __riscv_vfmul(v1, v2, vl);
        vfloat32m4_t v4 = __riscv_vfmul(v2, v2, vl);
        v2 = __riscv_vfmul(v2, mul2, vl);
        v4 = __riscv_vfmul(v4, vmul4, vl);
        v1 = __riscv_vfmadd(v1, mul1, v2, vl);
        v3 = __riscv_vfmadd(v3, vmul3, v4, vl);
        v1 = __riscv_vfadd(v1, v3, vl);
        vsum = __riscv_vfadd_tu(vsum, vsum, v1, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t v = RISCV_SHRINK4(vfadd, f, 32, vsum);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    float sum = __riscv_vfmv_f(__riscv_vfredusum(v, z, vl));
    *target = sum + num_points * center_point_array[4];
}
#endif /*LV_HAVE_RVV*/

#endif /*INCLUDED_volk_32f_x3_sum_of_poly_32f_u_H*/

#ifndef INCLUDED_volk_32f_x3_sum_of_poly_32f_a_H
#define INCLUDED_volk_32f_x3_sum_of_poly_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifndef MAX
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#endif

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_a_sse(float* target,
                                                      const float* src0,
                                                      const float* center_point_array,
                                                      const float* cutoff,
                                                      unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;

    __m128 cpa0 = _mm_load1_ps(&center_point_array[0]);
    __m128 cpa1 = _mm_load1_ps(&center_point_array[1]);
    __m128 cpa2 = _mm_load1_ps(&center_point_array[2]);
    __m128 cpa3 = _mm_load1_ps(&center_point_array[3]);
    __m128 cutoff_vec = _mm_load1_ps(cutoff);
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();

    __m128 x_to_1, x_to_2, x_to_3, x_to_4;

    unsigned int i;
    for (i = 0; i < eighth_points; ++i) {
        // 1st group of 4
        x_to_1 = _mm_load_ps(src0);
        x_to_1 = _mm_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm_mul_ps(x_to_1, x_to_1);
        x_to_3 = _mm_mul_ps(x_to_1, x_to_2);
        x_to_4 = _mm_mul_ps(x_to_2, x_to_2);

        x_to_1 = _mm_mul_ps(x_to_1, cpa0);
        x_to_2 = _mm_mul_ps(x_to_2, cpa1);
        x_to_3 = _mm_mul_ps(x_to_3, cpa2);
        x_to_4 = _mm_mul_ps(x_to_4, cpa3);

        x_to_1 = _mm_add_ps(x_to_1, x_to_2);
        x_to_3 = _mm_add_ps(x_to_3, x_to_4);
        acc0 = _mm_add_ps(x_to_1, acc0);
        acc0 = _mm_add_ps(x_to_3, acc0);

        src0 += 4;

        // 2nd group of 4
        x_to_1 = _mm_load_ps(src0);
        x_to_1 = _mm_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm_mul_ps(x_to_1, x_to_1);
        x_to_3 = _mm_mul_ps(x_to_1, x_to_2);
        x_to_4 = _mm_mul_ps(x_to_2, x_to_2);

        x_to_1 = _mm_mul_ps(x_to_1, cpa0);
        x_to_2 = _mm_mul_ps(x_to_2, cpa1);
        x_to_3 = _mm_mul_ps(x_to_3, cpa2);
        x_to_4 = _mm_mul_ps(x_to_4, cpa3);

        x_to_1 = _mm_add_ps(x_to_1, x_to_2);
        x_to_3 = _mm_add_ps(x_to_3, x_to_4);
        acc1 = _mm_add_ps(x_to_1, acc1);
        acc1 = _mm_add_ps(x_to_3, acc1);

        src0 += 4;
    }

    // Horizontal reduction via shuffle+add (avoids slow hadd)
    acc0 = _mm_add_ps(acc0, acc1);
    __m128 shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(1, 0, 3, 2));
    acc0 = _mm_add_ps(acc0, shuf);
    shuf = _mm_shuffle_ps(acc0, acc0, _MM_SHUFFLE(0, 1, 0, 1));
    acc0 = _mm_add_ps(acc0, shuf);

    float result;
    _mm_store_ss(&result, acc0);

    // Handle remaining elements via generic
    float tail_result;
    volk_32f_x3_sum_of_poly_32f_generic(
        &tail_result, src0, center_point_array, cutoff, num_points - eighth_points * 8);
    *target = result + tail_result + (float)(eighth_points * 8) * center_point_array[4];
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_a_sse3(float* target,
                                                      const float* src0,
                                                      const float* center_point_array,
                                                      const float* cutoff,
                                                      unsigned int num_points)
{
    float result = 0.0f;
    float fst = 0.0f;
    float sq = 0.0f;
    float thrd = 0.0f;
    float frth = 0.0f;

    __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10;

    xmm9 = _mm_setzero_ps();
    xmm1 = _mm_setzero_ps();
    xmm0 = _mm_load1_ps(&center_point_array[0]);
    xmm6 = _mm_load1_ps(&center_point_array[1]);
    xmm7 = _mm_load1_ps(&center_point_array[2]);
    xmm8 = _mm_load1_ps(&center_point_array[3]);
    xmm10 = _mm_load1_ps(cutoff);

    int bound = num_points / 8;
    int leftovers = num_points - 8 * bound;
    int i = 0;
    for (; i < bound; ++i) {
        // 1st
        xmm2 = _mm_load_ps(src0);
        xmm2 = _mm_max_ps(xmm10, xmm2);
        xmm3 = _mm_mul_ps(xmm2, xmm2);
        xmm4 = _mm_mul_ps(xmm2, xmm3);
        xmm5 = _mm_mul_ps(xmm3, xmm3);

        xmm2 = _mm_mul_ps(xmm2, xmm0);
        xmm3 = _mm_mul_ps(xmm3, xmm6);
        xmm4 = _mm_mul_ps(xmm4, xmm7);
        xmm5 = _mm_mul_ps(xmm5, xmm8);

        xmm2 = _mm_add_ps(xmm2, xmm3);
        xmm3 = _mm_add_ps(xmm4, xmm5);

        src0 += 4;

        xmm9 = _mm_add_ps(xmm2, xmm9);
        xmm9 = _mm_add_ps(xmm3, xmm9);

        // 2nd
        xmm2 = _mm_load_ps(src0);
        xmm2 = _mm_max_ps(xmm10, xmm2);
        xmm3 = _mm_mul_ps(xmm2, xmm2);
        xmm4 = _mm_mul_ps(xmm2, xmm3);
        xmm5 = _mm_mul_ps(xmm3, xmm3);

        xmm2 = _mm_mul_ps(xmm2, xmm0);
        xmm3 = _mm_mul_ps(xmm3, xmm6);
        xmm4 = _mm_mul_ps(xmm4, xmm7);
        xmm5 = _mm_mul_ps(xmm5, xmm8);

        xmm2 = _mm_add_ps(xmm2, xmm3);
        xmm3 = _mm_add_ps(xmm4, xmm5);

        src0 += 4;

        xmm1 = _mm_add_ps(xmm2, xmm1);
        xmm1 = _mm_add_ps(xmm3, xmm1);
    }
    xmm2 = _mm_hadd_ps(xmm9, xmm1);
    xmm3 = _mm_hadd_ps(xmm2, xmm2);
    xmm4 = _mm_hadd_ps(xmm3, xmm3);
    _mm_store_ss(&result, xmm4);

    for (i = 0; i < leftovers; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        result += (center_point_array[0] * fst + center_point_array[1] * sq +
                   center_point_array[2] * thrd + center_point_array[3] * frth);
    }

    result += (float)(num_points)*center_point_array[4];
    *target = result;
}


#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_a_avx(float* target,
                                                     const float* src0,
                                                     const float* center_point_array,
                                                     const float* cutoff,
                                                     unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_load_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_1 = _mm256_mul_ps(x_to_1, cpa0); // cpa[0] * x^1
        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_3 = _mm256_mul_ps(x_to_3, cpa2); // cpa[2] * x^3
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_add_ps(x_to_1, x_to_2);
        x_to_3 = _mm256_add_ps(x_to_3, x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_store_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }
    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_a_avx_fma(float* target,
                                                          const float* src0,
                                                          const float* center_point_array,
                                                          const float* cutoff,
                                                          unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_load_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_fmadd_ps(x_to_1, cpa0, x_to_2);
        x_to_3 = _mm256_fmadd_ps(x_to_3, cpa2, x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_store_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }
    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_a_avx2(float* target,
                                                       const float* src0,
                                                       const float* center_point_array,
                                                       const float* cutoff,
                                                       unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_load_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_add_ps(_mm256_mul_ps(x_to_1, cpa0), x_to_2);
        x_to_3 = _mm256_add_ps(_mm256_mul_ps(x_to_3, cpa2), x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_store_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }
    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_a_avx2_fma(float* target,
                                                           const float* src0,
                                                           const float* center_point_array,
                                                           const float* cutoff,
                                                           unsigned int num_points)
{
    const unsigned int eighth_points = num_points / 8;
    float fst = 0.0;
    float sq = 0.0;
    float thrd = 0.0;
    float frth = 0.0;

    __m256 cpa0, cpa1, cpa2, cpa3, cutoff_vec;
    __m256 target_vec;
    __m256 x_to_1, x_to_2, x_to_3, x_to_4;

    cpa0 = _mm256_set1_ps(center_point_array[0]);
    cpa1 = _mm256_set1_ps(center_point_array[1]);
    cpa2 = _mm256_set1_ps(center_point_array[2]);
    cpa3 = _mm256_set1_ps(center_point_array[3]);
    cutoff_vec = _mm256_set1_ps(*cutoff);
    target_vec = _mm256_setzero_ps();

    unsigned int i;

    for (i = 0; i < eighth_points; ++i) {
        x_to_1 = _mm256_load_ps(src0);
        x_to_1 = _mm256_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm256_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm256_mul_ps(x_to_1, x_to_2); // x^3
        // x^1 * x^3 is slightly faster than x^2 * x^2
        x_to_4 = _mm256_mul_ps(x_to_1, x_to_3); // x^4

        x_to_2 = _mm256_mul_ps(x_to_2, cpa1); // cpa[1] * x^2
        x_to_4 = _mm256_mul_ps(x_to_4, cpa3); // cpa[3] * x^4

        x_to_1 = _mm256_fmadd_ps(x_to_1, cpa0, x_to_2);
        x_to_3 = _mm256_fmadd_ps(x_to_3, cpa2, x_to_4);
        // this is slightly faster than result += (x_to_1 + x_to_3)
        target_vec = _mm256_add_ps(x_to_1, target_vec);
        target_vec = _mm256_add_ps(x_to_3, target_vec);

        src0 += 8;
    }

    // the hadd for vector reduction has very very slight impact @ 50k iters
    __VOLK_ATTR_ALIGNED(32) float temp_results[8];
    target_vec = _mm256_hadd_ps(
        target_vec,
        target_vec); // x0+x1 | x2+x3 | x0+x1 | x2+x3 || x4+x5 | x6+x7 | x4+x5 | x6+x7
    _mm256_store_ps(temp_results, target_vec);
    *target = temp_results[0] + temp_results[1] + temp_results[4] + temp_results[5];

    for (i = eighth_points * 8; i < num_points; ++i) {
        fst = *src0++;
        fst = MAX(fst, *cutoff);
        sq = fst * fst;
        thrd = fst * sq;
        frth = sq * sq;
        *target += (center_point_array[0] * fst + center_point_array[1] * sq +
                    center_point_array[2] * thrd + center_point_array[3] * frth);
    }
    *target += (float)(num_points)*center_point_array[4];
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x3_sum_of_poly_32f_a_avx512f(float* target,
                                                          const float* src0,
                                                          const float* center_point_array,
                                                          const float* cutoff,
                                                          unsigned int num_points)
{
    const unsigned int sixteenth_points = num_points / 16;

    __m512 cpa0 = _mm512_set1_ps(center_point_array[0]);
    __m512 cpa1 = _mm512_set1_ps(center_point_array[1]);
    __m512 cpa2 = _mm512_set1_ps(center_point_array[2]);
    __m512 cpa3 = _mm512_set1_ps(center_point_array[3]);
    __m512 cutoff_vec = _mm512_set1_ps(*cutoff);

    // 4 independent accumulators to saturate FMA pipeline
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    __m512 x_to_1, x_to_2, x_to_3, x_to_4;

    unsigned int i;
    for (i = 0; i < sixteenth_points; ++i) {
        x_to_1 = _mm512_load_ps(src0);
        x_to_1 = _mm512_max_ps(x_to_1, cutoff_vec);
        x_to_2 = _mm512_mul_ps(x_to_1, x_to_1); // x^2
        x_to_3 = _mm512_mul_ps(x_to_1, x_to_2); // x^3
        x_to_4 = _mm512_mul_ps(x_to_2, x_to_2); // x^4

        // AVX-512F implies FMA
        acc0 = _mm512_fmadd_ps(x_to_1, cpa0, acc0); // cpa[0] * x^1
        acc1 = _mm512_fmadd_ps(x_to_2, cpa1, acc1); // cpa[1] * x^2
        acc2 = _mm512_fmadd_ps(x_to_3, cpa2, acc2); // cpa[2] * x^3
        acc3 = _mm512_fmadd_ps(x_to_4, cpa3, acc3); // cpa[3] * x^4

        src0 += 16;
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    float result = _mm512_reduce_add_ps(acc0);

    // Handle remaining elements via generic
    float tail_result;
    volk_32f_x3_sum_of_poly_32f_generic(
        &tail_result, src0, center_point_array, cutoff, num_points - sixteenth_points * 16);
    *target =
        result + tail_result + (float)(sixteenth_points * 16) * center_point_array[4];
}
#endif /* LV_HAVE_AVX512F */

#endif /*INCLUDED_volk_32f_x3_sum_of_poly_32f_a_H*/
