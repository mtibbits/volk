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

#endif /* INCLUDED_volk_32f_s32f_power_32f_a_H */
