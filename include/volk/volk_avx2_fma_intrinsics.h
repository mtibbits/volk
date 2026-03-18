/* -*- c++ -*- */
/*
 * Copyright 2023 - 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This file is intended to hold AVX2 FMA intrinsics.
 * They should be used in VOLK kernels to avoid copy-paste.
 */

#ifndef INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_
#include <immintrin.h>

/*
 * Approximate arctan(x) via polynomial expansion
 * on the interval [-1, 1]
 *
 * Maximum relative error ~6.5e-7
 * Polynomial evaluated via Horner's method
 */
static inline __m256 _mm256_arctan_poly_avx2_fma(const __m256 x)
{
    const __m256 a1 = _mm256_set1_ps(+0x1.ffffeap-1f);
    const __m256 a3 = _mm256_set1_ps(-0x1.55437p-2f);
    const __m256 a5 = _mm256_set1_ps(+0x1.972be6p-3f);
    const __m256 a7 = _mm256_set1_ps(-0x1.1436ap-3f);
    const __m256 a9 = _mm256_set1_ps(+0x1.5785aap-4f);
    const __m256 a11 = _mm256_set1_ps(-0x1.2f3004p-5f);
    const __m256 a13 = _mm256_set1_ps(+0x1.01a37cp-7f);

    const __m256 x_times_x = _mm256_mul_ps(x, x);
    __m256 arctan;
    arctan = a13;
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a11);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a9);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a7);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a5);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a3);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a1);
    arctan = _mm256_mul_ps(x, arctan);

    return arctan;
}

/*
 * Approximate arcsin(x) via polynomial expansion
 * P(u) such that asin(x) = x * P(x^2) on |x| <= 0.5
 *
 * Maximum relative error ~1.5e-6
 * Polynomial evaluated via Horner's method
 */
static inline __m256 _mm256_arcsin_poly_avx2_fma(const __m256 x)
{
    const __m256 c0 = _mm256_set1_ps(0x1.ffffcep-1f);
    const __m256 c1 = _mm256_set1_ps(0x1.55b648p-3f);
    const __m256 c2 = _mm256_set1_ps(0x1.24d192p-4f);
    const __m256 c3 = _mm256_set1_ps(0x1.0a788p-4f);

    const __m256 u = _mm256_mul_ps(x, x);
    __m256 p = c3;
    p = _mm256_fmadd_ps(u, p, c2);
    p = _mm256_fmadd_ps(u, p, c1);
    p = _mm256_fmadd_ps(u, p, c0);

    return _mm256_mul_ps(x, p);
}

/*
 * Minimax polynomial for sin(x) on [-pi/4, pi/4]
 * Coefficients via Remez algorithm (Sollya)
 * Max |error| < 7.3e-9
 * sin(x) = x + x^3 * (s1 + x^2 * (s2 + x^2 * s3))
 */
static inline __m256 _mm256_sin_poly_avx2_fma(const __m256 x)
{
    const __m256 s1 = _mm256_set1_ps(-0x1.555552p-3f);
    const __m256 s2 = _mm256_set1_ps(+0x1.110be2p-7f);
    const __m256 s3 = _mm256_set1_ps(-0x1.9ab22ap-13f);

    const __m256 x2 = _mm256_mul_ps(x, x);
    const __m256 x3 = _mm256_mul_ps(x2, x);

    __m256 poly = _mm256_fmadd_ps(x2, s3, s2);
    poly = _mm256_fmadd_ps(x2, poly, s1);
    return _mm256_fmadd_ps(x3, poly, x);
}

/*
 * Minimax polynomial for cos(x) on [-pi/4, pi/4]
 * Coefficients via Remez algorithm (Sollya)
 * Max |error| < 1.1e-7
 * cos(x) = 1 + x^2 * (c1 + x^2 * (c2 + x^2 * c3))
 */
static inline __m256 _mm256_cos_poly_avx2_fma(const __m256 x)
{
    const __m256 c1 = _mm256_set1_ps(-0x1.fffff4p-2f);
    const __m256 c2 = _mm256_set1_ps(+0x1.554a46p-5f);
    const __m256 c3 = _mm256_set1_ps(-0x1.661be2p-10f);
    const __m256 one = _mm256_set1_ps(1.0f);

    const __m256 x2 = _mm256_mul_ps(x, x);

    __m256 poly = _mm256_fmadd_ps(x2, c3, c2);
    poly = _mm256_fmadd_ps(x2, poly, c1);
    return _mm256_fmadd_ps(x2, poly, one);
}

/*
 * Polynomial coefficients for log2(x)/(x-1) on [1, 2]
 * Generated with Sollya: remez(log2(x)/(x-1), 6, [1+1b-20, 2])
 * Max error: ~1.55e-6
 *
 * Usage: log2(x) ≈ poly(x) * (x - 1) for x ∈ [1, 2]
 * Polynomial evaluated via Horner's method with FMA
 */
static inline __m256 _mm256_log2_poly_avx2_fma(const __m256 x)
{
    const __m256 c0 = _mm256_set1_ps(+0x1.a8a726p+1f);
    const __m256 c1 = _mm256_set1_ps(-0x1.0b7f7ep+2f);
    const __m256 c2 = _mm256_set1_ps(+0x1.05d9ccp+2f);
    const __m256 c3 = _mm256_set1_ps(-0x1.4d476cp+1f);
    const __m256 c4 = _mm256_set1_ps(+0x1.04fc3ap+0f);
    const __m256 c5 = _mm256_set1_ps(-0x1.c97982p-3f);
    const __m256 c6 = _mm256_set1_ps(+0x1.57aa42p-6f);

    // Horner's method with FMA: c0 + x*(c1 + x*(c2 + ...))
    __m256 poly = c6;
    poly = _mm256_fmadd_ps(poly, x, c5);
    poly = _mm256_fmadd_ps(poly, x, c4);
    poly = _mm256_fmadd_ps(poly, x, c3);
    poly = _mm256_fmadd_ps(poly, x, c2);
    poly = _mm256_fmadd_ps(poly, x, c1);
    poly = _mm256_fmadd_ps(poly, x, c0);
    return poly;
}

/*
 * Compute |cplxValue0|² and |cplxValue1|² for 8 interleaved complex floats.
 *
 * Input:  cplxValue0 = [I0,Q0,I1,Q1, I2,Q2,I3,Q3]
 *         cplxValue1 = [I4,Q4,I5,Q5, I6,Q6,I7,Q7]
 * Output: [mag²_0, mag²_1, ..., mag²_7] in sequential order
 *
 * Replaces the hadd-based _mm256_magnitudesquared_ps_avx2 with
 * shuffle + FMA, avoiding the slow 2-cycle-throughput hadd.
 */
static inline __m256 _mm256_magnitudesquared_ps_avx2_fma(const __m256 cplxValue0,
                                                          const __m256 cplxValue1)
{
    const __m256i fix = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    /* Deinterleave re/im (within 128-bit lanes, output is lane-interleaved) */
    const __m256 re = _mm256_shuffle_ps(cplxValue0, cplxValue1, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256 im = _mm256_shuffle_ps(cplxValue0, cplxValue1, _MM_SHUFFLE(3, 1, 3, 1));

    /* mag² = re*re + im*im using FMA (still in lane-interleaved order) */
    const __m256 mag_sq = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));

    /* Fix cross-lane ordering */
    return _mm256_permutevar8x32_ps(mag_sq, fix);
}

/*
 * Compute |y - x|² * scalar for 8 interleaved complex floats.
 *
 * FMA version of _mm256_scaled_norm_dist_ps_avx2.
 */
static inline __m256 _mm256_scaled_norm_dist_ps_avx2_fma(const __m256 symbols0,
                                                          const __m256 symbols1,
                                                          const __m256 points0,
                                                          const __m256 points1,
                                                          const __m256 scalar)
{
    const __m256 diff0 = _mm256_sub_ps(symbols0, points0);
    const __m256 diff1 = _mm256_sub_ps(symbols1, points1);
    const __m256 norms = _mm256_magnitudesquared_ps_avx2_fma(diff0, diff1);
    return _mm256_mul_ps(norms, scalar);
}

/*
 * Newton-Raphson refined reciprocal square root with FMA: 1/sqrt(a)
 * x1 = x0 * (1.5 - 0.5 * a * x0^2)
 *     = x0 * fnmadd(0.5*a, x0^2, 1.5)
 */
static inline __m256 _mm256_rsqrt_nr_avx2_fma(const __m256 a)
{
    const __m256 HALF = _mm256_set1_ps(0.5f);
    const __m256 THREE_HALFS = _mm256_set1_ps(1.5f);

    const __m256 x0 = _mm256_rsqrt_ps(a);

    // Newton-Raphson with FMA: x1 = x0 * (1.5 - 0.5 * a * x0^2)
    const __m256 half_a = _mm256_mul_ps(HALF, a);
    const __m256 x0_sq = _mm256_mul_ps(x0, x0);
    __m256 x1 = _mm256_mul_ps(x0, _mm256_fnmadd_ps(half_a, x0_sq, THREE_HALFS));

    // For +0 and +Inf inputs, x0 is correct but NR produces NaN due to Inf*0
    // AVX2: native 256-bit integer compare
    __m256i a_si = _mm256_castps_si256(a);
    __m256i zero_mask = _mm256_cmpeq_epi32(a_si, _mm256_setzero_si256());
    __m256i inf_mask = _mm256_cmpeq_epi32(a_si, _mm256_set1_epi32(0x7F800000));
    __m256 special_mask = _mm256_castsi256_ps(_mm256_or_si256(zero_mask, inf_mask));
    return _mm256_blendv_ps(x1, x0, special_mask);
}

#endif /* INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_ */
