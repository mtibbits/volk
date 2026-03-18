/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_power_32f
 *
 * \b Overview
 *
 * Takes each input vector value to the specified power and stores the
 * results in the return vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_power_32f(float* cVector, const float* aVector, const float power,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li power: The power to raise the input value to.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * Square the numbers (0,9)
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *   }
 *
 *   // Raise each input value to this power
 *   float scale = 2.0f;
 *
 *   volk_32f_s32f_power_32f(out, increasing, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_power_32f_u_H
#define INCLUDED_volk_32f_s32f_power_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_power_32f_generic(float* cVector,
                                                   const float* aVector,
                                                   const float power,
                                                   unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = powf((*aPtr++), power);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_s32f_power_32f_u_sse2(float* cVector,
                                                  const float* aVector,
                                                  const float power,
                                                  unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128i exp_mask = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias = _mm_set1_epi32(127);
    const __m128 ln2 = _mm_set1_ps(0.6931471805599453f);
    const __m128 one = _mm_set1_ps(1.0f);

    /* Constants for exp */
    const __m128 exp_hi = _mm_set1_ps(88.3762626647949f);
    const __m128 exp_lo = _mm_set1_ps(-88.3762626647949f);
    const __m128 log2EF = _mm_set1_ps(1.44269504088896341f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 exp_C1 = _mm_set1_ps(0.693359375f);
    const __m128 exp_C2 = _mm_set1_ps(-2.12194440e-4f);
    const __m128i pi32_0x7f = _mm_set1_epi32(0x7f);

    const __m128 exp_p0 = _mm_set1_ps(1.9875691500e-4f);
    const __m128 exp_p1 = _mm_set1_ps(1.3981999507e-3f);
    const __m128 exp_p2 = _mm_set1_ps(8.3334519073e-3f);
    const __m128 exp_p3 = _mm_set1_ps(4.1665795894e-2f);
    const __m128 exp_p4 = _mm_set1_ps(1.6666665459e-1f);
    const __m128 exp_p5 = _mm_set1_ps(5.0000001201e-1f);

    const __m128 powerVec = _mm_set1_ps(power);

    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_loadu_ps(aPtr);

        /* Take absolute value for log computation */
        __m128 absVal = _mm_and_ps(aVal, abs_mask);

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m128i aInt = _mm_castps_si128(absVal);
        __m128i exp_i = _mm_srli_epi32(_mm_and_si128(aInt, exp_mask), 23);
        exp_i = _mm_sub_epi32(exp_i, exp_bias);
        __m128 exp_f = _mm_cvtepi32_ps(exp_i);

        __m128 frac = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aInt, mant_mask), one_bits));

        __m128 poly = _mm_log2_poly_sse(frac);
        __m128 logarithm =
            _mm_add_ps(exp_f, _mm_mul_ps(poly, _mm_sub_ps(frac, one)));

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m128 bVal = _mm_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm_max_ps(_mm_min_ps(bVal, exp_hi), exp_lo);

        __m128 fx = _mm_add_ps(_mm_mul_ps(bVal, log2EF), half);

        __m128i emm0 = _mm_cvttps_epi32(fx);
        __m128 tmp = _mm_cvtepi32_ps(emm0);

        __m128 mask = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
        fx = _mm_sub_ps(tmp, mask);

        tmp = _mm_mul_ps(fx, exp_C1);
        __m128 z = _mm_mul_ps(fx, exp_C2);
        bVal = _mm_sub_ps(_mm_sub_ps(bVal, tmp), z);
        z = _mm_mul_ps(bVal, bVal);

        __m128 y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(exp_p0, bVal), exp_p1), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p2), bVal), exp_p3);
        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(y, bVal), exp_p4), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p5), z), bVal);
        y = _mm_add_ps(y, one);

        emm0 = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), pi32_0x7f), 23);

        __m128 pow2n = _mm_castsi128_ps(emm0);
        __m128 cVal = _mm_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            /* Odd integer power: apply sign of input to result */
            __m128 sign_bits = _mm_andnot_ps(abs_mask, aVal);
            cVal = _mm_or_ps(_mm_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            /* Non-integer power: negative inputs produce NaN */
            __m128 neg_mask_v = _mm_cmplt_ps(aVal, _mm_setzero_ps());
            cVal = _mm_or_ps(_mm_andnot_ps(neg_mask_v, cVal),
                             _mm_and_ps(neg_mask_v, _mm_set1_ps(NAN)));
        }
        /* Even integer power: result is already positive, no fixup needed */

        _mm_storeu_ps(cPtr, cVal);

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_s32f_power_32f_u_avx(float* cVector,
                                                 const float* aVector,
                                                 const float power,
                                                 unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* 256-bit float constants */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);
    const __m256 powerVec = _mm256_set1_ps(power);

    /* 128-bit integer constants for bit manipulation */
    const __m128i exp_mask_128 = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask_128 = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits_128 = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias_128 = _mm_set1_epi32(127);
    const __m128i pi32_0x7f_128 = _mm_set1_epi32(0x7f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        /* Absolute value using 256-bit float AND */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Extract log2(|a|): split to 128-bit halves for integer ops */
        __m128 absLo = _mm256_castps256_ps128(absVal);
        __m128 absHi = _mm256_extractf128_ps(absVal, 1);

        __m128i aIntLo = _mm_castps_si128(absLo);
        __m128i aIntHi = _mm_castps_si128(absHi);

        __m128i expLo = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntLo, exp_mask_128), 23), exp_bias_128);
        __m128i expHi = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntHi, exp_mask_128), 23), exp_bias_128);

        __m128 fracLo = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntLo, mant_mask_128), one_bits_128));
        __m128 fracHi = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntHi, mant_mask_128), one_bits_128));

        /* Recombine to 256-bit float for polynomial eval */
        __m256 exp_f = _mm256_cvtepi32_ps(
            _mm256_insertf128_si256(_mm256_castsi128_si256(expLo), expHi, 1));
        __m256 frac = _mm256_insertf128_ps(_mm256_castps128_ps256(fracLo), fracHi, 1);

        /* Evaluate log2 polynomial using 256-bit float arithmetic. */
        __m256 poly = _mm256_log2_poly_avx(frac);

        __m256 logarithm = _mm256_add_ps(
            exp_f, _mm256_mul_ps(poly, _mm256_sub_ps(frac, one)));

        /* ln(|a|) = log2(|a|) * ln(2), then power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, _mm256_mul_ps(logarithm, ln2));

        /* Clamp for exp stability */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        /* Compute floor(bVal * log2e + 0.5) using 256-bit float + SSE truncation */
        __m256 fx = _mm256_add_ps(_mm256_mul_ps(bVal, log2EF), half);

        /* floor(fx): split to 128-bit for cvttps_epi32, recombine */
        __m128 fxLo = _mm256_castps256_ps128(fx);
        __m128 fxHi = _mm256_extractf128_ps(fx, 1);

        __m128i emm0Lo = _mm_cvttps_epi32(fxLo);
        __m128i emm0Hi = _mm_cvttps_epi32(fxHi);
        __m256 tmp = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(emm0Lo)),
            _mm_cvtepi32_ps(emm0Hi), 1);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        /* Reduce bVal */
        tmp = _mm256_sub_ps(bVal, _mm256_mul_ps(fx, exp_C1));
        bVal = _mm256_sub_ps(tmp, _mm256_mul_ps(fx, exp_C2));
        __m256 z = _mm256_mul_ps(bVal, bVal);

        /* Exp polynomial (256-bit float) */
        __m256 y = _mm256_add_ps(_mm256_mul_ps(exp_p0, bVal), exp_p1);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p2);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p3);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p4);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p5);
        y = _mm256_add_ps(_mm256_mul_ps(y, z), bVal);
        y = _mm256_add_ps(y, one);

        /* pow2n = 2^floor(fx): need slli+add on integers, split to 128-bit halves */
        fxLo = _mm256_castps256_ps128(fx);
        fxHi = _mm256_extractf128_ps(fx, 1);
        emm0Lo = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxLo), pi32_0x7f_128), 23);
        emm0Hi = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxHi), pi32_0x7f_128), 23);

        __m256 pow2n = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_castsi128_ps(emm0Lo)),
            _mm_castsi128_ps(emm0Hi), 1);

        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_storeu_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_s32f_power_32f_u_avx_fma(float* cVector,
                                                      const float* aVector,
                                                      const float power,
                                                      unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* 256-bit float constants */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);
    const __m256 powerVec = _mm256_set1_ps(power);

    /* 128-bit integer constants for bit manipulation */
    const __m128i exp_mask_128 = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask_128 = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits_128 = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias_128 = _mm_set1_epi32(127);
    const __m128i pi32_0x7f_128 = _mm_set1_epi32(0x7f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        /* Absolute value using 256-bit float AND */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Extract log2(|a|): split to 128-bit halves for integer ops */
        __m128 absLo = _mm256_castps256_ps128(absVal);
        __m128 absHi = _mm256_extractf128_ps(absVal, 1);

        __m128i aIntLo = _mm_castps_si128(absLo);
        __m128i aIntHi = _mm_castps_si128(absHi);

        __m128i expLo = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntLo, exp_mask_128), 23), exp_bias_128);
        __m128i expHi = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntHi, exp_mask_128), 23), exp_bias_128);

        __m128 fracLo = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntLo, mant_mask_128), one_bits_128));
        __m128 fracHi = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntHi, mant_mask_128), one_bits_128));

        /* Recombine to 256-bit float for polynomial eval */
        __m256 exp_f = _mm256_cvtepi32_ps(
            _mm256_insertf128_si256(_mm256_castsi128_si256(expLo), expHi, 1));
        __m256 frac = _mm256_insertf128_ps(_mm256_castps128_ps256(fracLo), fracHi, 1);

        /* Evaluate log2 polynomial using 256-bit float arithmetic */
        __m256 poly = _mm256_log2_poly_avx(frac);

        __m256 logarithm = _mm256_fmadd_ps(
            poly, _mm256_sub_ps(frac, one), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2), then power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, _mm256_mul_ps(logarithm, ln2));

        /* Clamp for exp stability */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        /* Compute floor(bVal * log2e + 0.5) using FMA */
        __m256 fx = _mm256_fmadd_ps(bVal, log2EF, half);

        /* floor(fx): split to 128-bit for cvttps_epi32, recombine */
        __m128 fxLo = _mm256_castps256_ps128(fx);
        __m128 fxHi = _mm256_extractf128_ps(fx, 1);

        __m128i emm0Lo = _mm_cvttps_epi32(fxLo);
        __m128i emm0Hi = _mm_cvttps_epi32(fxHi);
        __m256 tmp = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(emm0Lo)),
            _mm_cvtepi32_ps(emm0Hi), 1);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        /* Reduce bVal using FMA */
        tmp = _mm256_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm256_fnmadd_ps(fx, exp_C2, tmp);
        __m256 z = _mm256_mul_ps(bVal, bVal);

        /* Exp polynomial (256-bit float) with FMA */
        __m256 y = _mm256_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm256_fmadd_ps(y, bVal, exp_p2);
        y = _mm256_fmadd_ps(y, bVal, exp_p3);
        y = _mm256_fmadd_ps(y, bVal, exp_p4);
        y = _mm256_fmadd_ps(y, bVal, exp_p5);
        y = _mm256_fmadd_ps(y, z, bVal);
        y = _mm256_add_ps(y, one);

        /* pow2n = 2^floor(fx): need slli+add on integers, split to 128-bit halves */
        fxLo = _mm256_castps256_ps128(fx);
        fxHi = _mm256_extractf128_ps(fx, 1);
        emm0Lo = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxLo), pi32_0x7f_128), 23);
        emm0Hi = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxHi), pi32_0x7f_128), 23);

        __m256 pow2n = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_castsi128_ps(emm0Lo)),
            _mm_castsi128_ps(emm0Hi), 1);

        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_storeu_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32f_s32f_power_32f_u_avx2(float* cVector,
                                                  const float* aVector,
                                                  const float power,
                                                  unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 one = _mm256_set1_ps(1.0f);

    /* Constants for exp */
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256i pi32_0x7f = _mm256_set1_epi32(0x7f);

    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);

    const __m256 powerVec = _mm256_set1_ps(power);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        /* Take absolute value for log computation */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m256i aInt = _mm256_castps_si256(absVal);
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aInt, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aInt, mant_mask), one_bits));

        __m256 poly = _mm256_log2_poly_avx2(frac);
        __m256 logarithm =
            _mm256_add_ps(exp_f, _mm256_mul_ps(poly, _mm256_sub_ps(frac, one)));

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm256_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        __m256 fx = _mm256_add_ps(_mm256_mul_ps(bVal, log2EF), half);

        __m256i emm0 = _mm256_cvttps_epi32(fx);
        __m256 tmp = _mm256_cvtepi32_ps(emm0);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_sub_ps(bVal, _mm256_mul_ps(fx, exp_C1));
        bVal = _mm256_sub_ps(tmp, _mm256_mul_ps(fx, exp_C2));
        __m256 z = _mm256_mul_ps(bVal, bVal);

        __m256 y = _mm256_add_ps(_mm256_mul_ps(exp_p0, bVal), exp_p1);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p2);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p3);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p4);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p5);
        y = _mm256_add_ps(_mm256_mul_ps(y, z), bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        __m256 pow2n = _mm256_castsi256_ps(emm0);
        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_storeu_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32f_s32f_power_32f_u_avx2_fma(float* cVector,
                                                      const float* aVector,
                                                      const float power,
                                                      unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 one = _mm256_set1_ps(1.0f);

    /* Constants for exp */
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256i pi32_0x7f = _mm256_set1_epi32(0x7f);

    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);

    const __m256 powerVec = _mm256_set1_ps(power);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        /* Take absolute value for log computation */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Compute log2(|a|) using degree-6 Remez polynomial with FMA */
        __m256i aInt = _mm256_castps_si256(absVal);
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aInt, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aInt, mant_mask), one_bits));

        __m256 poly = _mm256_log2_poly_avx2_fma(frac);
        __m256 logarithm =
            _mm256_fmadd_ps(poly, _mm256_sub_ps(frac, one), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm256_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        __m256 fx = _mm256_fmadd_ps(bVal, log2EF, half);

        __m256i emm0 = _mm256_cvttps_epi32(fx);
        __m256 tmp = _mm256_cvtepi32_ps(emm0);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm256_fnmadd_ps(fx, exp_C2, tmp);
        __m256 z = _mm256_mul_ps(bVal, bVal);

        __m256 y = _mm256_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm256_fmadd_ps(y, bVal, exp_p2);
        y = _mm256_fmadd_ps(y, bVal, exp_p3);
        y = _mm256_fmadd_ps(y, bVal, exp_p4);
        y = _mm256_fmadd_ps(y, bVal, exp_p5);
        y = _mm256_fmadd_ps(y, z, bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        __m256 pow2n = _mm256_castsi256_ps(emm0);
        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_storeu_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_s32f_power_32f_u_avx512f(float* cVector,
                                                     const float* aVector,
                                                     const float power,
                                                     unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m512i abs_mask_i = _mm512_set1_epi32(0x7fffffff);
    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 one = _mm512_set1_ps(1.0f);

    /* Constants for exp */
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2EF = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 exp_C1 = _mm512_set1_ps(0.693359375f);
    const __m512 exp_C2 = _mm512_set1_ps(-2.12194440e-4f);
    const __m512i pi32_0x7f = _mm512_set1_epi32(0x7f);

    const __m512 exp_p0 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 exp_p1 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 exp_p2 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 exp_p3 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 exp_p4 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 exp_p5 = _mm512_set1_ps(5.0000001201e-1f);

    const __m512 powerVec = _mm512_set1_ps(power);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aPtr);

        /* Take absolute value for log computation */
        __m512 absVal = _mm512_castsi512_ps(
            _mm512_and_epi32(_mm512_castps_si512(aVal), abs_mask_i));

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m512i aInt = _mm512_castps_si512(absVal);
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aInt, exp_mask), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        __m512 frac = _mm512_castsi512_ps(
            _mm512_or_epi32(_mm512_and_epi32(aInt, mant_mask), one_bits));

        __m512 poly = _mm512_log2_poly_avx512(frac);
        __m512 logarithm =
            _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm512_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m512 bVal = _mm512_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm512_max_ps(_mm512_min_ps(bVal, exp_hi), exp_lo);

        __m512 fx = _mm512_fmadd_ps(bVal, log2EF, half);

        /* floor(fx) */
        fx = _mm512_floor_ps(fx);

        __m512 tmp = _mm512_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm512_fnmadd_ps(fx, exp_C2, tmp);
        __m512 z = _mm512_mul_ps(bVal, bVal);

        __m512 y = _mm512_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm512_fmadd_ps(y, bVal, exp_p2);
        y = _mm512_fmadd_ps(y, bVal, exp_p3);
        y = _mm512_fmadd_ps(y, bVal, exp_p4);
        y = _mm512_fmadd_ps(y, bVal, exp_p5);
        y = _mm512_fmadd_ps(y, z, bVal);
        y = _mm512_add_ps(y, one);

        __m512i emm0 =
            _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(fx), pi32_0x7f), 23);

        __m512 pow2n = _mm512_castsi512_ps(emm0);
        __m512 cVal = _mm512_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_sub_ps(cVal, neg_k, _mm512_setzero_ps(), cVal);
        } else if (!power_is_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_mov_ps(cVal, neg_k, _mm512_set1_ps(NAN));
        }

        _mm512_storeu_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX512F */


#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_s32f_power_32f_u_avx512dq(float* cVector,
                                                       const float* aVector,
                                                       const float power,
                                                       unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log - using AVX512DQ float-domain bitwise ops where possible */
    const __m512 abs_mask_ps = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512i exp_mask_i = _mm512_set1_epi32(0x7f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 mant_mask_ps = _mm512_castsi512_ps(_mm512_set1_epi32(0x007fffff));
    const __m512 one_ps = _mm512_set1_ps(1.0f);
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);

    /* Constants for exp */
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2EF = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 exp_C1 = _mm512_set1_ps(0.693359375f);
    const __m512 exp_C2 = _mm512_set1_ps(-2.12194440e-4f);
    const __m512i pi32_0x7f = _mm512_set1_epi32(0x7f);

    const __m512 exp_p0 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 exp_p1 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 exp_p2 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 exp_p3 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 exp_p4 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 exp_p5 = _mm512_set1_ps(5.0000001201e-1f);

    const __m512 powerVec = _mm512_set1_ps(power);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aPtr);

        /* Take absolute value using AVX512DQ float-domain AND */
        __m512 absVal = _mm512_and_ps(aVal, abs_mask_ps);

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m512i aInt = _mm512_castps_si512(absVal);
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aInt, exp_mask_i), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        /* Mantissa extraction using AVX512DQ float-domain bitwise ops */
        __m512 frac = _mm512_or_ps(_mm512_and_ps(absVal, mant_mask_ps), one_ps);

        __m512 poly = _mm512_log2_poly_avx512(frac);
        __m512 logarithm =
            _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one_ps), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm512_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m512 bVal = _mm512_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm512_max_ps(_mm512_min_ps(bVal, exp_hi), exp_lo);

        __m512 fx = _mm512_fmadd_ps(bVal, log2EF, half);

        /* floor(fx) */
        fx = _mm512_floor_ps(fx);

        __m512 tmp = _mm512_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm512_fnmadd_ps(fx, exp_C2, tmp);
        __m512 z = _mm512_mul_ps(bVal, bVal);

        __m512 y = _mm512_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm512_fmadd_ps(y, bVal, exp_p2);
        y = _mm512_fmadd_ps(y, bVal, exp_p3);
        y = _mm512_fmadd_ps(y, bVal, exp_p4);
        y = _mm512_fmadd_ps(y, bVal, exp_p5);
        y = _mm512_fmadd_ps(y, z, bVal);
        y = _mm512_add_ps(y, one_ps);

        __m512i emm0 =
            _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(fx), pi32_0x7f), 23);

        __m512 pow2n = _mm512_castsi512_ps(emm0);
        __m512 cVal = _mm512_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_sub_ps(cVal, neg_k, _mm512_setzero_ps(), cVal);
        } else if (!power_is_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_mov_ps(cVal, neg_k, _mm512_set1_ps(NAN));
        }

        _mm512_storeu_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void volk_32f_s32f_power_32f_neonv8(float* cVector,
                                                    const float* aVector,
                                                    const float power,
                                                    unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log */
    const uint32x4_t abs_mask = vdupq_n_u32(0x7fffffff);
    const int32x4_t exp_mask = vdupq_n_s32(0x7f800000);
    const int32x4_t mant_mask = vdupq_n_s32(0x007fffff);
    const int32x4_t one_bits = vdupq_n_s32(0x3f800000);
    const int32x4_t exp_bias = vdupq_n_s32(127);
    const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    /* Constants for exp */
    const float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    const float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    const float32x4_t log2EF = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t exp_C1 = vdupq_n_f32(0.693359375f);
    const float32x4_t exp_C2 = vdupq_n_f32(-2.12194440e-4f);
    const int32x4_t pi32_0x7f = vdupq_n_s32(0x7f);

    const float32x4_t exp_p0 = vdupq_n_f32(1.9875691500e-4f);
    const float32x4_t exp_p1 = vdupq_n_f32(1.3981999507e-3f);
    const float32x4_t exp_p2 = vdupq_n_f32(8.3334519073e-3f);
    const float32x4_t exp_p3 = vdupq_n_f32(4.1665795894e-2f);
    const float32x4_t exp_p4 = vdupq_n_f32(1.6666665459e-1f);
    const float32x4_t exp_p5 = vdupq_n_f32(5.0000001201e-1f);

    const float32x4_t powerVec = vdupq_n_f32(power);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    for (; number < quarter_points; number++) {
        float32x4_t aVal = vld1q_f32(aPtr);

        /* Absolute value */
        float32x4_t absVal = vreinterpretq_f32_u32(
            vandq_u32(vreinterpretq_u32_f32(aVal), abs_mask));

        /* Compute log2(|a|) */
        int32x4_t aInt = vreinterpretq_s32_f32(absVal);
        int32x4_t exp_i = vshrq_n_s32(vandq_s32(aInt, exp_mask), 23);
        exp_i = vsubq_s32(exp_i, exp_bias);
        float32x4_t exp_f = vcvtq_f32_s32(exp_i);

        float32x4_t frac = vreinterpretq_f32_s32(
            vorrq_s32(vandq_s32(aInt, mant_mask), one_bits));

        float32x4_t poly = _vlog2_poly_neonv8(frac);
        float32x4_t logarithm = vfmaq_f32(exp_f, poly, vsubq_f32(frac, one));

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = vmulq_f32(logarithm, ln2);

        /* power * ln(|a|) */
        float32x4_t bVal = vmulq_f32(powerVec, logarithm);

        /* exp(power * ln(|a|)) */
        bVal = vmaxq_f32(vminq_f32(bVal, exp_hi), exp_lo);

        float32x4_t fx = vfmaq_f32(half, bVal, log2EF);

        /* floor(fx) using AArch64 vrndmq_f32 */
        fx = vrndmq_f32(fx);

        float32x4_t tmp = vmulq_f32(fx, exp_C1);
        float32x4_t z = vmulq_f32(fx, exp_C2);
        bVal = vsubq_f32(vsubq_f32(bVal, tmp), z);
        z = vmulq_f32(bVal, bVal);

        float32x4_t y = vfmaq_f32(exp_p1, exp_p0, bVal);
        y = vfmaq_f32(exp_p2, y, bVal);
        y = vfmaq_f32(exp_p3, y, bVal);
        y = vfmaq_f32(exp_p4, y, bVal);
        y = vfmaq_f32(exp_p5, y, bVal);
        y = vfmaq_f32(bVal, y, z);
        y = vaddq_f32(y, one);

        int32x4_t emm0 = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), pi32_0x7f), 23);

        float32x4_t pow2n = vreinterpretq_f32_s32(emm0);
        float32x4_t cVal = vmulq_f32(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            uint32x4_t sign_bits = vbicq_u32(vreinterpretq_u32_f32(aVal), abs_mask);
            cVal = vreinterpretq_f32_u32(
                vorrq_u32(vandq_u32(vreinterpretq_u32_f32(cVal), abs_mask), sign_bits));
        } else if (!power_is_int) {
            uint32x4_t neg_mask = vcltq_f32(aVal, zero);
            cVal = vbslq_f32(neg_mask, vdupq_n_f32(NAN), cVal);
        }

        vst1q_f32(cPtr, cVal);

        aPtr += 4;
        cPtr += 4;
    }

    number = quarter_points * 4;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_power_32f_rvv(float* cVector,
                                                const float* aVector,
                                                const float power,
                                                unsigned int num_points)
{
    /* powf(a, p) = exp(p * ln(a)) = exp(p * ln2 * log2(a)) */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* log2 polynomial coefficients (degree-6 Remez on [1,2]) */
    const float lc0 = +0x1.a8a726p+1f;
    const float lc1 = -0x1.0b7f7ep+2f;
    const float lc2 = +0x1.05d9ccp+2f;
    const float lc3 = -0x1.4d476cp+1f;
    const float lc4 = +0x1.04fc3ap+0f;
    const float lc5 = -0x1.c97982p-3f;
    const float lc6 = +0x1.57aa42p-6f;
    const float ln2_val = 0.6931471805599453f;

    /* exp polynomial coefficients */
    const float exp_hi_val = 88.3762626647949f;
    const float exp_lo_val = -88.3762626647949f;
    const float log2ef = 1.44269504088896341f;
    const float exp_C1_val = 0.693359375f;
    const float exp_C2_val = -2.12194440e-4f;
    const float ep0 = 1.9875691500e-4f;
    const float ep1 = 1.3981999507e-3f;
    const float ep2 = 8.3334519073e-3f;
    const float ep3 = 4.1665795894e-2f;
    const float ep4 = 1.6666665459e-1f;
    const float ep5 = 5.0000001201e-1f;

    size_t n = (size_t)num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t aVal = __riscv_vle32_v_f32m4(aVector, vl);

        /* |a| */
        vfloat32m4_t absVal = __riscv_vfabs(aVal, vl);

        /* log2(|a|): decompose into exponent + mantissa polynomial */
        vint32m4_t aInt = __riscv_vreinterpret_v_f32m4_i32m4(absVal);
        vint32m4_t exp_i = __riscv_vsub(
            __riscv_vsra(__riscv_vand(aInt, 0x7f800000, vl), 23, vl),
            127, vl);
        vfloat32m4_t exp_f = __riscv_vfcvt_f(exp_i, vl);

        vint32m4_t frac_i = __riscv_vor(
            __riscv_vand(aInt, 0x007fffff, vl), 0x3f800000, vl);
        vfloat32m4_t frac = __riscv_vreinterpret_f32m4(frac_i);

        /* Horner's log2 polynomial: lc0 + frac*(lc1 + frac*(...)) */
        vfloat32m4_t poly = __riscv_vfmv_v_f_f32m4(lc6, vl);
        poly = __riscv_vfmadd(poly, frac, __riscv_vfmv_v_f_f32m4(lc5, vl), vl);
        poly = __riscv_vfmadd(poly, frac, __riscv_vfmv_v_f_f32m4(lc4, vl), vl);
        poly = __riscv_vfmadd(poly, frac, __riscv_vfmv_v_f_f32m4(lc3, vl), vl);
        poly = __riscv_vfmadd(poly, frac, __riscv_vfmv_v_f_f32m4(lc2, vl), vl);
        poly = __riscv_vfmadd(poly, frac, __riscv_vfmv_v_f_f32m4(lc1, vl), vl);
        poly = __riscv_vfmadd(poly, frac, __riscv_vfmv_v_f_f32m4(lc0, vl), vl);

        /* log2(|a|) = exponent + poly * (frac - 1) */
        vfloat32m4_t logarithm = __riscv_vfmacc(exp_f,
            __riscv_vfsub(frac, 1.0f, vl), poly, vl);

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = __riscv_vfmul(logarithm, ln2_val, vl);

        /* b = power * ln(|a|) */
        vfloat32m4_t bVal = __riscv_vfmul(logarithm, power, vl);

        /* Clamp to exp range */
        bVal = __riscv_vfmin(__riscv_vfmax(bVal, exp_lo_val, vl), exp_hi_val, vl);

        /* fx = floor(b * log2(e) + 0.5) */
        vfloat32m4_t fx = __riscv_vfmacc(
            __riscv_vfmv_v_f_f32m4(0.5f, vl), bVal,
            __riscv_vfmv_v_f_f32m4(log2ef, vl), vl);
        vint32m4_t fxi = __riscv_vfcvt_rtz_x(fx, vl);
        fx = __riscv_vfcvt_f(fxi, vl);

        /* Reduce: b = b - fx*C1 - fx*C2 */
        bVal = __riscv_vfnmsac(bVal, exp_C1_val, fx, vl);
        bVal = __riscv_vfnmsac(bVal, exp_C2_val, fx, vl);
        vfloat32m4_t z = __riscv_vfmul(bVal, bVal, vl);

        /* Evaluate exp polynomial via Horner */
        vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(ep0, vl);
        y = __riscv_vfmadd(y, bVal, __riscv_vfmv_v_f_f32m4(ep1, vl), vl);
        y = __riscv_vfmadd(y, bVal, __riscv_vfmv_v_f_f32m4(ep2, vl), vl);
        y = __riscv_vfmadd(y, bVal, __riscv_vfmv_v_f_f32m4(ep3, vl), vl);
        y = __riscv_vfmadd(y, bVal, __riscv_vfmv_v_f_f32m4(ep4, vl), vl);
        y = __riscv_vfmadd(y, bVal, __riscv_vfmv_v_f_f32m4(ep5, vl), vl);
        y = __riscv_vfmacc(bVal, y, z, vl);
        y = __riscv_vfadd(y, 1.0f, vl);

        /* Scale by 2^fx */
        vint32m4_t emm0 = __riscv_vsll(
            __riscv_vadd(fxi, 0x7f, vl), 23, vl);
        vfloat32m4_t pow2n = __riscv_vreinterpret_f32m4(emm0);
        vfloat32m4_t cVal = __riscv_vfmul(y, pow2n, vl);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            vuint32m4_t sign = __riscv_vand(
                __riscv_vreinterpret_u32m4(__riscv_vreinterpret_v_f32m4_i32m4(aVal)),
                0x80000000u, vl);
            vuint32m4_t abs_result = __riscv_vand(
                __riscv_vreinterpret_u32m4(__riscv_vreinterpret_v_f32m4_i32m4(cVal)),
                0x7fffffffu, vl);
            cVal = __riscv_vreinterpret_f32m4(
                __riscv_vreinterpret_v_u32m4_i32m4(
                    __riscv_vor(abs_result, sign, vl)));
        } else if (!power_is_int) {
            vbool8_t neg_mask = __riscv_vmflt(aVal, 0.0f, vl);
            cVal = __riscv_vfmerge(cVal, NAN, neg_mask, vl);
        }

        __riscv_vse32(cVector, cVal, vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void volk_32f_s32f_power_32f_neon(float* cVector,
                                                 const float* aVector,
                                                 const float power,
                                                 unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log */
    const uint32x4_t abs_mask = vdupq_n_u32(0x7fffffff);
    const int32x4_t exp_mask = vdupq_n_s32(0x7f800000);
    const int32x4_t mant_mask = vdupq_n_s32(0x007fffff);
    const int32x4_t one_bits = vdupq_n_s32(0x3f800000);
    const int32x4_t exp_bias = vdupq_n_s32(127);
    const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    /* Constants for exp */
    const float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    const float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    const float32x4_t log2EF = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t exp_C1 = vdupq_n_f32(0.693359375f);
    const float32x4_t exp_C2 = vdupq_n_f32(-2.12194440e-4f);
    const int32x4_t pi32_0x7f = vdupq_n_s32(0x7f);

    const float32x4_t exp_p0 = vdupq_n_f32(1.9875691500e-4f);
    const float32x4_t exp_p1 = vdupq_n_f32(1.3981999507e-3f);
    const float32x4_t exp_p2 = vdupq_n_f32(8.3334519073e-3f);
    const float32x4_t exp_p3 = vdupq_n_f32(4.1665795894e-2f);
    const float32x4_t exp_p4 = vdupq_n_f32(1.6666665459e-1f);
    const float32x4_t exp_p5 = vdupq_n_f32(5.0000001201e-1f);

    const float32x4_t powerVec = vdupq_n_f32(power);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    for (; number < quarter_points; number++) {
        float32x4_t aVal = vld1q_f32(aPtr);

        /* Absolute value */
        float32x4_t absVal =
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(aVal), abs_mask));

        /* Compute log2(|a|) */
        int32x4_t aInt = vreinterpretq_s32_f32(absVal);
        int32x4_t exp_i = vshrq_n_s32(vandq_s32(aInt, exp_mask), 23);
        exp_i = vsubq_s32(exp_i, exp_bias);
        float32x4_t exp_f = vcvtq_f32_s32(exp_i);

        float32x4_t frac =
            vreinterpretq_f32_s32(vorrq_s32(vandq_s32(aInt, mant_mask), one_bits));

        float32x4_t poly = _vlog2_poly_f32(frac);
        float32x4_t logarithm = vmlaq_f32(exp_f, poly, vsubq_f32(frac, one));

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = vmulq_f32(logarithm, ln2);

        /* power * ln(|a|) */
        float32x4_t bVal = vmulq_f32(powerVec, logarithm);

        /* exp(power * ln(|a|)) */
        bVal = vmaxq_f32(vminq_f32(bVal, exp_hi), exp_lo);

        float32x4_t fx = vmlaq_f32(half, bVal, log2EF);

        /* floor(fx) for ARMv7: truncate then adjust for negative values */
        int32x4_t fxi = vcvtq_s32_f32(fx);
        float32x4_t fxf = vcvtq_f32_s32(fxi);
        uint32x4_t need_dec = vcgtq_f32(fxf, fx);
        fx = vsubq_f32(
            fxf, vreinterpretq_f32_u32(vandq_u32(need_dec, vreinterpretq_u32_f32(one))));

        float32x4_t tmp = vmulq_f32(fx, exp_C1);
        float32x4_t z = vmulq_f32(fx, exp_C2);
        bVal = vsubq_f32(vsubq_f32(bVal, tmp), z);
        z = vmulq_f32(bVal, bVal);

        float32x4_t y = vmlaq_f32(exp_p1, exp_p0, bVal);
        y = vmlaq_f32(exp_p2, y, bVal);
        y = vmlaq_f32(exp_p3, y, bVal);
        y = vmlaq_f32(exp_p4, y, bVal);
        y = vmlaq_f32(exp_p5, y, bVal);
        y = vmlaq_f32(bVal, y, z);
        y = vaddq_f32(y, one);

        int32x4_t emm0 = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), pi32_0x7f), 23);

        float32x4_t pow2n = vreinterpretq_f32_s32(emm0);
        float32x4_t cVal = vmulq_f32(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            uint32x4_t sign_bits =
                vbicq_u32(vreinterpretq_u32_f32(aVal), abs_mask);
            cVal = vreinterpretq_f32_u32(
                vorrq_u32(vandq_u32(vreinterpretq_u32_f32(cVal), abs_mask), sign_bits));
        } else if (!power_is_int) {
            uint32x4_t neg_mask = vcltq_f32(aVal, zero);
            cVal = vbslq_f32(neg_mask, vdupq_n_f32(NAN), cVal);
        }

        vst1q_f32(cPtr, cVal);

        aPtr += 4;
        cPtr += 4;
    }

    number = quarter_points * 4;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32f_s32f_power_32f_u_H */

#ifndef INCLUDED_volk_32f_s32f_power_32f_a_H
#define INCLUDED_volk_32f_s32f_power_32f_a_H

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_s32f_power_32f_a_sse2(float* cVector,
                                                  const float* aVector,
                                                  const float power,
                                                  unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128i exp_mask = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias = _mm_set1_epi32(127);
    const __m128 ln2 = _mm_set1_ps(0.6931471805599453f);
    const __m128 one = _mm_set1_ps(1.0f);

    /* Constants for exp */
    const __m128 exp_hi = _mm_set1_ps(88.3762626647949f);
    const __m128 exp_lo = _mm_set1_ps(-88.3762626647949f);
    const __m128 log2EF = _mm_set1_ps(1.44269504088896341f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 exp_C1 = _mm_set1_ps(0.693359375f);
    const __m128 exp_C2 = _mm_set1_ps(-2.12194440e-4f);
    const __m128i pi32_0x7f = _mm_set1_epi32(0x7f);

    const __m128 exp_p0 = _mm_set1_ps(1.9875691500e-4f);
    const __m128 exp_p1 = _mm_set1_ps(1.3981999507e-3f);
    const __m128 exp_p2 = _mm_set1_ps(8.3334519073e-3f);
    const __m128 exp_p3 = _mm_set1_ps(4.1665795894e-2f);
    const __m128 exp_p4 = _mm_set1_ps(1.6666665459e-1f);
    const __m128 exp_p5 = _mm_set1_ps(5.0000001201e-1f);

    const __m128 powerVec = _mm_set1_ps(power);

    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_load_ps(aPtr);

        /* Take absolute value for log computation */
        __m128 absVal = _mm_and_ps(aVal, abs_mask);

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m128i aInt = _mm_castps_si128(absVal);
        __m128i exp_i = _mm_srli_epi32(_mm_and_si128(aInt, exp_mask), 23);
        exp_i = _mm_sub_epi32(exp_i, exp_bias);
        __m128 exp_f = _mm_cvtepi32_ps(exp_i);

        __m128 frac = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aInt, mant_mask), one_bits));

        __m128 poly = _mm_log2_poly_sse(frac);
        __m128 logarithm =
            _mm_add_ps(exp_f, _mm_mul_ps(poly, _mm_sub_ps(frac, one)));

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m128 bVal = _mm_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm_max_ps(_mm_min_ps(bVal, exp_hi), exp_lo);

        __m128 fx = _mm_add_ps(_mm_mul_ps(bVal, log2EF), half);

        __m128i emm0 = _mm_cvttps_epi32(fx);
        __m128 tmp = _mm_cvtepi32_ps(emm0);

        __m128 mask = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
        fx = _mm_sub_ps(tmp, mask);

        tmp = _mm_mul_ps(fx, exp_C1);
        __m128 z = _mm_mul_ps(fx, exp_C2);
        bVal = _mm_sub_ps(_mm_sub_ps(bVal, tmp), z);
        z = _mm_mul_ps(bVal, bVal);

        __m128 y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(exp_p0, bVal), exp_p1), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p2), bVal), exp_p3);
        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(y, bVal), exp_p4), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p5), z), bVal);
        y = _mm_add_ps(y, one);

        emm0 = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), pi32_0x7f), 23);

        __m128 pow2n = _mm_castsi128_ps(emm0);
        __m128 cVal = _mm_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m128 sign_bits = _mm_andnot_ps(abs_mask, aVal);
            cVal = _mm_or_ps(_mm_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m128 neg_mask_v = _mm_cmplt_ps(aVal, _mm_setzero_ps());
            cVal = _mm_or_ps(_mm_andnot_ps(neg_mask_v, cVal),
                             _mm_and_ps(neg_mask_v, _mm_set1_ps(NAN)));
        }

        _mm_store_ps(cPtr, cVal);

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_s32f_power_32f_a_avx(float* cVector,
                                                 const float* aVector,
                                                 const float power,
                                                 unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* 256-bit float constants */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);
    const __m256 powerVec = _mm256_set1_ps(power);

    /* 128-bit integer constants for bit manipulation */
    const __m128i exp_mask_128 = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask_128 = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits_128 = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias_128 = _mm_set1_epi32(127);
    const __m128i pi32_0x7f_128 = _mm_set1_epi32(0x7f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        /* Absolute value using 256-bit float AND */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Extract log2(|a|): split to 128-bit halves for integer ops */
        __m128 absLo = _mm256_castps256_ps128(absVal);
        __m128 absHi = _mm256_extractf128_ps(absVal, 1);

        __m128i aIntLo = _mm_castps_si128(absLo);
        __m128i aIntHi = _mm_castps_si128(absHi);

        __m128i expLo = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntLo, exp_mask_128), 23), exp_bias_128);
        __m128i expHi = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntHi, exp_mask_128), 23), exp_bias_128);

        __m128 fracLo = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntLo, mant_mask_128), one_bits_128));
        __m128 fracHi = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntHi, mant_mask_128), one_bits_128));

        /* Recombine to 256-bit float for polynomial eval */
        __m256 exp_f = _mm256_cvtepi32_ps(
            _mm256_insertf128_si256(_mm256_castsi128_si256(expLo), expHi, 1));
        __m256 frac = _mm256_insertf128_ps(_mm256_castps128_ps256(fracLo), fracHi, 1);

        /* Evaluate log2 polynomial using 256-bit float arithmetic. */
        __m256 poly = _mm256_log2_poly_avx(frac);

        __m256 logarithm = _mm256_add_ps(
            exp_f, _mm256_mul_ps(poly, _mm256_sub_ps(frac, one)));

        /* ln(|a|) = log2(|a|) * ln(2), then power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, _mm256_mul_ps(logarithm, ln2));

        /* Clamp for exp stability */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        /* Compute floor(bVal * log2e + 0.5) */
        __m256 fx = _mm256_add_ps(_mm256_mul_ps(bVal, log2EF), half);

        /* floor(fx): split to 128-bit for cvttps_epi32, recombine */
        __m128 fxLo = _mm256_castps256_ps128(fx);
        __m128 fxHi = _mm256_extractf128_ps(fx, 1);

        __m128i emm0Lo = _mm_cvttps_epi32(fxLo);
        __m128i emm0Hi = _mm_cvttps_epi32(fxHi);
        __m256 tmp = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(emm0Lo)),
            _mm_cvtepi32_ps(emm0Hi), 1);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        /* Reduce bVal */
        tmp = _mm256_sub_ps(bVal, _mm256_mul_ps(fx, exp_C1));
        bVal = _mm256_sub_ps(tmp, _mm256_mul_ps(fx, exp_C2));
        __m256 z = _mm256_mul_ps(bVal, bVal);

        /* Exp polynomial (256-bit float) */
        __m256 y = _mm256_add_ps(_mm256_mul_ps(exp_p0, bVal), exp_p1);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p2);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p3);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p4);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p5);
        y = _mm256_add_ps(_mm256_mul_ps(y, z), bVal);
        y = _mm256_add_ps(y, one);

        /* pow2n = 2^floor(fx): split to 128-bit halves for integer ops */
        fxLo = _mm256_castps256_ps128(fx);
        fxHi = _mm256_extractf128_ps(fx, 1);
        emm0Lo = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxLo), pi32_0x7f_128), 23);
        emm0Hi = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxHi), pi32_0x7f_128), 23);

        __m256 pow2n = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_castsi128_ps(emm0Lo)),
            _mm_castsi128_ps(emm0Hi), 1);

        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_store_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>
#include <volk/volk_sse_intrinsics.h>

static inline void volk_32f_s32f_power_32f_a_avx_fma(float* cVector,
                                                      const float* aVector,
                                                      const float power,
                                                      unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* 256-bit float constants */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);
    const __m256 powerVec = _mm256_set1_ps(power);

    /* 128-bit integer constants for bit manipulation */
    const __m128i exp_mask_128 = _mm_set1_epi32(0x7f800000);
    const __m128i mant_mask_128 = _mm_set1_epi32(0x007fffff);
    const __m128i one_bits_128 = _mm_set1_epi32(0x3f800000);
    const __m128i exp_bias_128 = _mm_set1_epi32(127);
    const __m128i pi32_0x7f_128 = _mm_set1_epi32(0x7f);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        /* Absolute value using 256-bit float AND */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Extract log2(|a|): split to 128-bit halves for integer ops */
        __m128 absLo = _mm256_castps256_ps128(absVal);
        __m128 absHi = _mm256_extractf128_ps(absVal, 1);

        __m128i aIntLo = _mm_castps_si128(absLo);
        __m128i aIntHi = _mm_castps_si128(absHi);

        __m128i expLo = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntLo, exp_mask_128), 23), exp_bias_128);
        __m128i expHi = _mm_sub_epi32(
            _mm_srli_epi32(_mm_and_si128(aIntHi, exp_mask_128), 23), exp_bias_128);

        __m128 fracLo = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntLo, mant_mask_128), one_bits_128));
        __m128 fracHi = _mm_castsi128_ps(
            _mm_or_si128(_mm_and_si128(aIntHi, mant_mask_128), one_bits_128));

        /* Recombine to 256-bit float for polynomial eval */
        __m256 exp_f = _mm256_cvtepi32_ps(
            _mm256_insertf128_si256(_mm256_castsi128_si256(expLo), expHi, 1));
        __m256 frac = _mm256_insertf128_ps(_mm256_castps128_ps256(fracLo), fracHi, 1);

        /* Evaluate log2 polynomial using 256-bit float arithmetic */
        __m256 poly = _mm256_log2_poly_avx(frac);

        __m256 logarithm = _mm256_fmadd_ps(
            poly, _mm256_sub_ps(frac, one), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2), then power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, _mm256_mul_ps(logarithm, ln2));

        /* Clamp for exp stability */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        /* Compute floor(bVal * log2e + 0.5) using FMA */
        __m256 fx = _mm256_fmadd_ps(bVal, log2EF, half);

        /* floor(fx): split to 128-bit for cvttps_epi32, recombine */
        __m128 fxLo = _mm256_castps256_ps128(fx);
        __m128 fxHi = _mm256_extractf128_ps(fx, 1);

        __m128i emm0Lo = _mm_cvttps_epi32(fxLo);
        __m128i emm0Hi = _mm_cvttps_epi32(fxHi);
        __m256 tmp = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(emm0Lo)),
            _mm_cvtepi32_ps(emm0Hi), 1);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        /* Reduce bVal using FMA */
        tmp = _mm256_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm256_fnmadd_ps(fx, exp_C2, tmp);
        __m256 z = _mm256_mul_ps(bVal, bVal);

        /* Exp polynomial (256-bit float) with FMA */
        __m256 y = _mm256_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm256_fmadd_ps(y, bVal, exp_p2);
        y = _mm256_fmadd_ps(y, bVal, exp_p3);
        y = _mm256_fmadd_ps(y, bVal, exp_p4);
        y = _mm256_fmadd_ps(y, bVal, exp_p5);
        y = _mm256_fmadd_ps(y, z, bVal);
        y = _mm256_add_ps(y, one);

        /* pow2n = 2^floor(fx): split to 128-bit halves for integer ops */
        fxLo = _mm256_castps256_ps128(fx);
        fxHi = _mm256_extractf128_ps(fx, 1);
        emm0Lo = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxLo), pi32_0x7f_128), 23);
        emm0Hi = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fxHi), pi32_0x7f_128), 23);

        __m256 pow2n = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_castsi128_ps(emm0Lo)),
            _mm_castsi128_ps(emm0Hi), 1);

        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_store_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>
#include <volk/volk_avx2_intrinsics.h>

static inline void volk_32f_s32f_power_32f_a_avx2(float* cVector,
                                                  const float* aVector,
                                                  const float power,
                                                  unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 one = _mm256_set1_ps(1.0f);

    /* Constants for exp */
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256i pi32_0x7f = _mm256_set1_epi32(0x7f);

    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);

    const __m256 powerVec = _mm256_set1_ps(power);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        /* Take absolute value for log computation */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m256i aInt = _mm256_castps_si256(absVal);
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aInt, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aInt, mant_mask), one_bits));

        __m256 poly = _mm256_log2_poly_avx2(frac);
        __m256 logarithm =
            _mm256_add_ps(exp_f, _mm256_mul_ps(poly, _mm256_sub_ps(frac, one)));

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm256_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        __m256 fx = _mm256_add_ps(_mm256_mul_ps(bVal, log2EF), half);

        __m256i emm0 = _mm256_cvttps_epi32(fx);
        __m256 tmp = _mm256_cvtepi32_ps(emm0);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_sub_ps(bVal, _mm256_mul_ps(fx, exp_C1));
        bVal = _mm256_sub_ps(tmp, _mm256_mul_ps(fx, exp_C2));
        __m256 z = _mm256_mul_ps(bVal, bVal);

        __m256 y = _mm256_add_ps(_mm256_mul_ps(exp_p0, bVal), exp_p1);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p2);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p3);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p4);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p5);
        y = _mm256_add_ps(_mm256_mul_ps(y, z), bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        __m256 pow2n = _mm256_castsi256_ps(emm0);
        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_store_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32f_s32f_power_32f_a_avx2_fma(float* cVector,
                                                      const float* aVector,
                                                      const float power,
                                                      unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    const __m256i mant_mask = _mm256_set1_epi32(0x007fffff);
    const __m256i one_bits = _mm256_set1_epi32(0x3f800000);
    const __m256i exp_bias = _mm256_set1_epi32(127);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 one = _mm256_set1_ps(1.0f);

    /* Constants for exp */
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 exp_C1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256i pi32_0x7f = _mm256_set1_epi32(0x7f);

    const __m256 exp_p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 exp_p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 exp_p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 exp_p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 exp_p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 exp_p5 = _mm256_set1_ps(5.0000001201e-1f);

    const __m256 powerVec = _mm256_set1_ps(power);

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        /* Take absolute value for log computation */
        __m256 absVal = _mm256_and_ps(aVal, abs_mask);

        /* Compute log2(|a|) using degree-6 Remez polynomial with FMA */
        __m256i aInt = _mm256_castps_si256(absVal);
        __m256i exp_i = _mm256_srli_epi32(_mm256_and_si256(aInt, exp_mask), 23);
        exp_i = _mm256_sub_epi32(exp_i, exp_bias);
        __m256 exp_f = _mm256_cvtepi32_ps(exp_i);

        __m256 frac = _mm256_castsi256_ps(
            _mm256_or_si256(_mm256_and_si256(aInt, mant_mask), one_bits));

        __m256 poly = _mm256_log2_poly_avx2_fma(frac);
        __m256 logarithm =
            _mm256_fmadd_ps(poly, _mm256_sub_ps(frac, one), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm256_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m256 bVal = _mm256_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        __m256 fx = _mm256_fmadd_ps(bVal, log2EF, half);

        __m256i emm0 = _mm256_cvttps_epi32(fx);
        __m256 tmp = _mm256_cvtepi32_ps(emm0);

        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm256_fnmadd_ps(fx, exp_C2, tmp);
        __m256 z = _mm256_mul_ps(bVal, bVal);

        __m256 y = _mm256_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm256_fmadd_ps(y, bVal, exp_p2);
        y = _mm256_fmadd_ps(y, bVal, exp_p3);
        y = _mm256_fmadd_ps(y, bVal, exp_p4);
        y = _mm256_fmadd_ps(y, bVal, exp_p5);
        y = _mm256_fmadd_ps(y, z, bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        __m256 pow2n = _mm256_castsi256_ps(emm0);
        __m256 cVal = _mm256_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __m256 sign_bits = _mm256_andnot_ps(abs_mask, aVal);
            cVal = _mm256_or_ps(_mm256_and_ps(cVal, abs_mask), sign_bits);
        } else if (!power_is_int) {
            __m256 neg_mask_v = _mm256_cmp_ps(aVal, _mm256_setzero_ps(), _CMP_LT_OS);
            cVal = _mm256_or_ps(_mm256_andnot_ps(neg_mask_v, cVal),
                                _mm256_and_ps(neg_mask_v, _mm256_set1_ps(NAN)));
        }

        _mm256_store_ps(cPtr, cVal);

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_s32f_power_32f_a_avx512f(float* cVector,
                                                     const float* aVector,
                                                     const float power,
                                                     unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log (degree-6 Remez polynomial via helper) */
    const __m512i abs_mask_i = _mm512_set1_epi32(0x7fffffff);
    const __m512i exp_mask = _mm512_set1_epi32(0x7f800000);
    const __m512i mant_mask = _mm512_set1_epi32(0x007fffff);
    const __m512i one_bits = _mm512_set1_epi32(0x3f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 one = _mm512_set1_ps(1.0f);

    /* Constants for exp */
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2EF = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 exp_C1 = _mm512_set1_ps(0.693359375f);
    const __m512 exp_C2 = _mm512_set1_ps(-2.12194440e-4f);
    const __m512i pi32_0x7f = _mm512_set1_epi32(0x7f);

    const __m512 exp_p0 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 exp_p1 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 exp_p2 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 exp_p3 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 exp_p4 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 exp_p5 = _mm512_set1_ps(5.0000001201e-1f);

    const __m512 powerVec = _mm512_set1_ps(power);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aPtr);

        /* Take absolute value for log computation */
        __m512 absVal = _mm512_castsi512_ps(
            _mm512_and_epi32(_mm512_castps_si512(aVal), abs_mask_i));

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m512i aInt = _mm512_castps_si512(absVal);
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aInt, exp_mask), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        __m512 frac = _mm512_castsi512_ps(
            _mm512_or_epi32(_mm512_and_epi32(aInt, mant_mask), one_bits));

        __m512 poly = _mm512_log2_poly_avx512(frac);
        __m512 logarithm =
            _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm512_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m512 bVal = _mm512_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm512_max_ps(_mm512_min_ps(bVal, exp_hi), exp_lo);

        __m512 fx = _mm512_fmadd_ps(bVal, log2EF, half);

        /* floor(fx) */
        fx = _mm512_floor_ps(fx);

        __m512 tmp = _mm512_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm512_fnmadd_ps(fx, exp_C2, tmp);
        __m512 z = _mm512_mul_ps(bVal, bVal);

        __m512 y = _mm512_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm512_fmadd_ps(y, bVal, exp_p2);
        y = _mm512_fmadd_ps(y, bVal, exp_p3);
        y = _mm512_fmadd_ps(y, bVal, exp_p4);
        y = _mm512_fmadd_ps(y, bVal, exp_p5);
        y = _mm512_fmadd_ps(y, z, bVal);
        y = _mm512_add_ps(y, one);

        __m512i emm0 =
            _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(fx), pi32_0x7f), 23);

        __m512 pow2n = _mm512_castsi512_ps(emm0);
        __m512 cVal = _mm512_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_sub_ps(cVal, neg_k, _mm512_setzero_ps(), cVal);
        } else if (!power_is_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_mov_ps(cVal, neg_k, _mm512_set1_ps(NAN));
        }

        _mm512_store_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX512F */


#if LV_HAVE_AVX512F && LV_HAVE_AVX512DQ
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void volk_32f_s32f_power_32f_a_avx512dq(float* cVector,
                                                       const float* aVector,
                                                       const float power,
                                                       unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    /* Precompute sign handling for negative inputs */
    const int power_is_int =
        (power == floorf(power)) && (fabsf(power) < 2147483648.0f);
    const int power_is_odd_int = power_is_int && (((int)power) & 1);

    /* Constants for log - using AVX512DQ float-domain bitwise ops where possible */
    const __m512 abs_mask_ps = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512i exp_mask_i = _mm512_set1_epi32(0x7f800000);
    const __m512i exp_bias = _mm512_set1_epi32(127);
    const __m512 mant_mask_ps = _mm512_castsi512_ps(_mm512_set1_epi32(0x007fffff));
    const __m512 one_ps = _mm512_set1_ps(1.0f);
    const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);

    /* Constants for exp */
    const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
    const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
    const __m512 log2EF = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 exp_C1 = _mm512_set1_ps(0.693359375f);
    const __m512 exp_C2 = _mm512_set1_ps(-2.12194440e-4f);
    const __m512i pi32_0x7f = _mm512_set1_epi32(0x7f);

    const __m512 exp_p0 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 exp_p1 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 exp_p2 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 exp_p3 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 exp_p4 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 exp_p5 = _mm512_set1_ps(5.0000001201e-1f);

    const __m512 powerVec = _mm512_set1_ps(power);

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aPtr);

        /* Take absolute value using AVX512DQ float-domain AND */
        __m512 absVal = _mm512_and_ps(aVal, abs_mask_ps);

        /* Compute log2(|a|) using degree-6 Remez polynomial */
        __m512i aInt = _mm512_castps_si512(absVal);
        __m512i exp_i = _mm512_srli_epi32(_mm512_and_si512(aInt, exp_mask_i), 23);
        exp_i = _mm512_sub_epi32(exp_i, exp_bias);
        __m512 exp_f = _mm512_cvtepi32_ps(exp_i);

        /* Mantissa extraction using AVX512DQ float-domain bitwise ops */
        __m512 frac = _mm512_or_ps(_mm512_and_ps(absVal, mant_mask_ps), one_ps);

        __m512 poly = _mm512_log2_poly_avx512(frac);
        __m512 logarithm =
            _mm512_fmadd_ps(poly, _mm512_sub_ps(frac, one_ps), exp_f);

        /* ln(|a|) = log2(|a|) * ln(2) */
        logarithm = _mm512_mul_ps(logarithm, ln2);

        /* Compute power * ln(|a|) */
        __m512 bVal = _mm512_mul_ps(powerVec, logarithm);

        /* Compute exp(power * ln(|a|)) */
        bVal = _mm512_max_ps(_mm512_min_ps(bVal, exp_hi), exp_lo);

        __m512 fx = _mm512_fmadd_ps(bVal, log2EF, half);

        /* floor(fx) */
        fx = _mm512_floor_ps(fx);

        __m512 tmp = _mm512_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm512_fnmadd_ps(fx, exp_C2, tmp);
        __m512 z = _mm512_mul_ps(bVal, bVal);

        __m512 y = _mm512_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm512_fmadd_ps(y, bVal, exp_p2);
        y = _mm512_fmadd_ps(y, bVal, exp_p3);
        y = _mm512_fmadd_ps(y, bVal, exp_p4);
        y = _mm512_fmadd_ps(y, bVal, exp_p5);
        y = _mm512_fmadd_ps(y, z, bVal);
        y = _mm512_add_ps(y, one_ps);

        __m512i emm0 =
            _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(fx), pi32_0x7f), 23);

        __m512 pow2n = _mm512_castsi512_ps(emm0);
        __m512 cVal = _mm512_mul_ps(y, pow2n);

        /* Handle sign for negative inputs */
        if (power_is_odd_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_sub_ps(cVal, neg_k, _mm512_setzero_ps(), cVal);
        } else if (!power_is_int) {
            __mmask16 neg_k = _mm512_cmplt_ps_mask(aVal, _mm512_setzero_ps());
            cVal = _mm512_mask_mov_ps(cVal, neg_k, _mm512_set1_ps(NAN));
        }

        _mm512_store_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_s32f_power_32f_generic(cPtr, aPtr, power, num_points - number);
}

#endif /* LV_HAVE_AVX512F && LV_HAVE_AVX512DQ */

#endif /* INCLUDED_volk_32f_s32f_power_32f_a_H */
