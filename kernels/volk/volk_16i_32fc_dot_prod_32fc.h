/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_32fc_dot_prod_32fc
 *
 * \b Overview
 *
 * Computes the dot product between a vector of 16-bit integers and a
 * vector of complex floats. Each short input is converted to float,
 * multiplied by the corresponding complex tap, and the products are
 * summed into a single complex result.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_32fc_dot_prod_32fc(lv_32fc_t* result, const short* input, const
 * lv_32fc_t* taps, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: vector of 16-bit integer samples (short).
 * \li taps: vector of complex float taps (lv_32fc_t).
 * \li num_points: number of elements in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a complex float to store the dot product result (lv_32fc_t).
 *
 * \b Example
 * Compute a weighted sum of short-valued samples with complex coefficients.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main() {
 *     unsigned int num_points = 8;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     // Allocate aligned memory
 *     short* input = (short*)volk_malloc(sizeof(short) * num_points, alignment);
 *     lv_32fc_t* taps =
 *         (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * num_points, alignment);
 *     lv_32fc_t result;
 *
 *     // Initialize input samples (e.g., a short ramp)
 *     for (unsigned int i = 0; i < num_points; i++) {
 *       input[i] = (short)(i * 100);
 *     }
 *
 *     // Initialize complex taps (e.g., a simple low-pass-like filter)
 *     for (unsigned int i = 0; i < num_points; i++) {
 *       taps[i] = lv_cmake(0.125f, 0.0f);
 *     }
 *
 *     // Compute the dot product: result = sum(input[i] * taps[i])
 *     volk_16i_32fc_dot_prod_32fc(&result, input, taps, num_points);
 *
 *     printf("Dot product = (%f, %f)\n", lv_creal(result), lv_cimag(result));
 *
 *     volk_free(input);
 *     volk_free(taps);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_16i_32fc_dot_prod_32fc_u_H
#define INCLUDED_volk_16i_32fc_dot_prod_32fc_u_H

#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_16i_32fc_dot_prod_32fc_generic(lv_32fc_t* result,
                                                       const short* input,
                                                       const lv_32fc_t* taps,
                                                       unsigned int num_points)
{

    static const int N_UNROLL = 4;

    lv_32fc_t acc0 = 0;
    lv_32fc_t acc1 = 0;
    lv_32fc_t acc2 = 0;
    lv_32fc_t acc3 = 0;

    unsigned i = 0;
    unsigned n = (num_points / N_UNROLL) * N_UNROLL;

    for (i = 0; i < n; i += N_UNROLL) {
        acc0 += taps[i + 0] * (float)input[i + 0];
        acc1 += taps[i + 1] * (float)input[i + 1];
        acc2 += taps[i + 2] * (float)input[i + 2];
        acc3 += taps[i + 3] * (float)input[i + 3];
    }

    for (; i < num_points; i++) {
        acc0 += taps[i] * (float)input[i];
    }

    *result = acc0 + acc1 + acc2 + acc3;
}

#endif /*LV_HAVE_GENERIC*/


#if LV_HAVE_SSE && LV_HAVE_MMX

static inline void volk_16i_32fc_dot_prod_32fc_u_sse(lv_32fc_t* result,
                                                     const short* input,
                                                     const lv_32fc_t* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m64 m0, m1;
    __m128 f0, f1, f2, f3;
    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {

        m0 = _mm_set_pi16(*(aPtr + 3), *(aPtr + 2), *(aPtr + 1), *(aPtr + 0));
        m1 = _mm_set_pi16(*(aPtr + 7), *(aPtr + 6), *(aPtr + 5), *(aPtr + 4));
        f0 = _mm_cvtpi16_ps(m0);
        f1 = _mm_cvtpi16_ps(m0);
        f2 = _mm_cvtpi16_ps(m1);
        f3 = _mm_cvtpi16_ps(m1);

        a0Val = _mm_unpacklo_ps(f0, f1);
        a1Val = _mm_unpackhi_ps(f0, f1);
        a2Val = _mm_unpacklo_ps(f2, f3);
        a3Val = _mm_unpackhi_ps(f2, f3);

        b0Val = _mm_loadu_ps(bPtr);
        b1Val = _mm_loadu_ps(bPtr + 4);
        b2Val = _mm_loadu_ps(bPtr + 8);
        b3Val = _mm_loadu_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    _mm_empty(); // clear the mmx technology state

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_SSE && LV_HAVE_MMX*/


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_u_sse2(lv_32fc_t* result,
                                                       const short* input,
                                                       const lv_32fc_t* taps,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values */
        __m128i v = _mm_loadu_si128((const __m128i*)aPtr);

        /* Sign-extend lower 4 int16 to int32 (no SSE4.1 pmovsxwd) */
        __m128i lo32 = _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16);
        /* Sign-extend upper 4 int16 to int32 */
        __m128i hi32 = _mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16);

        /* Convert int32 to float */
        __m128 f0 = _mm_cvtepi32_ps(lo32);
        __m128 f1 = _mm_cvtepi32_ps(hi32);

        /* Duplicate each float for complex multiply:
         * f0=[s0,s1,s2,s3] -> a0=[s0,s0,s1,s1], a1=[s2,s2,s3,s3] */
        __m128 a0Val = _mm_unpacklo_ps(f0, f0);
        __m128 a1Val = _mm_unpackhi_ps(f0, f0);
        __m128 a2Val = _mm_unpacklo_ps(f1, f1);
        __m128 a3Val = _mm_unpackhi_ps(f1, f1);

        /* Load 8 complex taps (16 floats) */
        __m128 b0Val = _mm_loadu_ps(bPtr);
        __m128 b1Val = _mm_loadu_ps(bPtr + 4);
        __m128 b2Val = _mm_loadu_ps(bPtr + 8);
        __m128 b3Val = _mm_loadu_ps(bPtr + 12);

        dotProdVal0 = _mm_add_ps(_mm_mul_ps(a0Val, b0Val), dotProdVal0);
        dotProdVal1 = _mm_add_ps(_mm_mul_ps(a1Val, b1Val), dotProdVal1);
        dotProdVal2 = _mm_add_ps(_mm_mul_ps(a2Val, b2Val), dotProdVal2);
        dotProdVal3 = _mm_add_ps(_mm_mul_ps(a3Val, b3Val), dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, dotProdVal0);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_SSE2*/


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_u_sse4_1(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values */
        __m128i v = _mm_loadu_si128((const __m128i*)aPtr);

        /* Sign-extend int16 to int32 using SSE4.1 pmovsxwd */
        __m128i lo32 = _mm_cvtepi16_epi32(v);
        __m128i hi32 = _mm_cvtepi16_epi32(_mm_srli_si128(v, 8));

        /* Convert int32 to float */
        __m128 f0 = _mm_cvtepi32_ps(lo32);
        __m128 f1 = _mm_cvtepi32_ps(hi32);

        /* Duplicate each float for complex multiply */
        __m128 a0Val = _mm_unpacklo_ps(f0, f0);
        __m128 a1Val = _mm_unpackhi_ps(f0, f0);
        __m128 a2Val = _mm_unpacklo_ps(f1, f1);
        __m128 a3Val = _mm_unpackhi_ps(f1, f1);

        /* Load 8 complex taps (16 floats) */
        __m128 b0Val = _mm_loadu_ps(bPtr);
        __m128 b1Val = _mm_loadu_ps(bPtr + 4);
        __m128 b2Val = _mm_loadu_ps(bPtr + 8);
        __m128 b3Val = _mm_loadu_ps(bPtr + 12);

        dotProdVal0 = _mm_add_ps(_mm_mul_ps(a0Val, b0Val), dotProdVal0);
        dotProdVal1 = _mm_add_ps(_mm_mul_ps(a1Val, b1Val), dotProdVal1);
        dotProdVal2 = _mm_add_ps(_mm_mul_ps(a2Val, b2Val), dotProdVal2);
        dotProdVal3 = _mm_add_ps(_mm_mul_ps(a3Val, b3Val), dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, dotProdVal0);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_SSE4_1*/


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_u_avx(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values into 128-bit register */
        __m128i m0 = _mm_loadu_si128((const __m128i*)aPtr);

        /* Convert int16 -> int32 -> float using SSE4.1 */
        __m128 lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(m0));
        __m128 hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(m0, 8)));

        /* Duplicate each float for complex multiply:
         * [s0,s1,s2,s3] -> [s0,s0,s1,s1] and [s2,s2,s3,s3]
         * [s4,s5,s6,s7] -> [s4,s4,s5,s5] and [s6,s6,s7,s7] */
        __m128 dup_lo0 = _mm_unpacklo_ps(lo, lo);
        __m128 dup_lo1 = _mm_unpackhi_ps(lo, lo);
        __m128 dup_hi0 = _mm_unpacklo_ps(hi, hi);
        __m128 dup_hi1 = _mm_unpackhi_ps(hi, hi);

        /* Combine into 256-bit vectors */
        __m256 a0Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_lo0), dup_lo1, 1);
        __m256 a1Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_hi0), dup_hi1, 1);

        /* Load 8 complex taps (16 floats) in two 256-bit registers */
        __m256 b0Val = _mm256_loadu_ps(bPtr);
        __m256 b1Val = _mm256_loadu_ps(bPtr + 8);

        /* Multiply and accumulate */
        dotProdVal0 = _mm256_add_ps(dotProdVal0, _mm256_mul_ps(a0Val, b0Val));
        dotProdVal1 = _mm256_add_ps(dotProdVal1, _mm256_mul_ps(a1Val, b1Val));

        aPtr += 8;
        bPtr += 16;
    }

    /* Combine accumulators */
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);

    /* Horizontal reduction: 256-bit -> 128-bit */
    __m128 hi128 = _mm256_extractf128_ps(dotProdVal0, 1);
    __m128 lo128 = _mm256_castps256_ps128(dotProdVal0);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, sum128);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    /* Handle tail */
    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX*/


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_u_avx_fma(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values into 128-bit register */
        __m128i m0 = _mm_loadu_si128((const __m128i*)aPtr);

        /* Convert int16 -> int32 -> float using SSE4.1 */
        __m128 lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(m0));
        __m128 hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(m0, 8)));

        /* Duplicate each float for complex multiply */
        __m128 dup_lo0 = _mm_unpacklo_ps(lo, lo);
        __m128 dup_lo1 = _mm_unpackhi_ps(lo, lo);
        __m128 dup_hi0 = _mm_unpacklo_ps(hi, hi);
        __m128 dup_hi1 = _mm_unpackhi_ps(hi, hi);

        /* Combine into 256-bit vectors */
        __m256 a0Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_lo0), dup_lo1, 1);
        __m256 a1Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_hi0), dup_hi1, 1);

        /* Load 8 complex taps (16 floats) in two 256-bit registers */
        __m256 b0Val = _mm256_loadu_ps(bPtr);
        __m256 b1Val = _mm256_loadu_ps(bPtr + 8);

        /* FMA accumulate */
        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);

        aPtr += 8;
        bPtr += 16;
    }

    /* Combine accumulators */
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);

    /* Horizontal reduction: 256-bit -> 128-bit */
    __m128 hi128 = _mm256_extractf128_ps(dotProdVal0, 1);
    __m128 lo128 = _mm256_castps256_ps128(dotProdVal0);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, sum128);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    /* Handle tail */
    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX && LV_HAVE_FMA*/


#ifdef LV_HAVE_AVX2

static inline void volk_16i_32fc_dot_prod_32fc_u_avx2(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_loadu_si128((__m128i const*)aPtr);
        m1 = _mm_loadu_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_loadu_ps(bPtr);
        b1Val = _mm256_loadu_ps(bPtr + 8);
        b2Val = _mm256_loadu_ps(bPtr + 16);
        b3Val = _mm256_loadu_ps(bPtr + 24);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_AVX2*/


#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void volk_16i_32fc_dot_prod_32fc_u_avx2_fma(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_loadu_si128((__m128i const*)aPtr);
        m1 = _mm_loadu_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_loadu_ps(bPtr);
        b1Val = _mm256_loadu_ps(bPtr + 8);
        b2Val = _mm256_loadu_ps(bPtr + 16);
        b3Val = _mm256_loadu_ps(bPtr + 24);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_16i_32fc_dot_prod_32fc_neon(lv_32fc_t* result,
                                                    const short* input,
                                                    const lv_32fc_t* taps,
                                                    unsigned int num_points)
{

    unsigned ii;
    unsigned quarter_points = num_points / 4;
    const lv_32fc_t* tapsPtr = taps;
    const short* inputPtr = input;
    lv_32fc_t accumulator_vec[4];

    float32x4x2_t tapsVal, accumulator_val;
    int16x4_t input16;
    int32x4_t input32;
    float32x4_t input_float, prod_re, prod_im;

    accumulator_val.val[0] = vdupq_n_f32(0.0);
    accumulator_val.val[1] = vdupq_n_f32(0.0);

    for (ii = 0; ii < quarter_points; ++ii) {
        tapsVal = vld2q_f32((const float*)tapsPtr);
        input16 = vld1_s16(inputPtr);
        // widen 16-bit int to 32-bit int
        input32 = vmovl_s16(input16);
        // convert 32-bit int to float with scale
        input_float = vcvtq_f32_s32(input32);

        prod_re = vmulq_f32(input_float, tapsVal.val[0]);
        prod_im = vmulq_f32(input_float, tapsVal.val[1]);

        accumulator_val.val[0] = vaddq_f32(prod_re, accumulator_val.val[0]);
        accumulator_val.val[1] = vaddq_f32(prod_im, accumulator_val.val[1]);

        tapsPtr += 4;
        inputPtr += 4;
    }
    vst2q_f32((float*)accumulator_vec, accumulator_val);
    accumulator_vec[0] += accumulator_vec[1];
    accumulator_vec[2] += accumulator_vec[3];
    accumulator_vec[0] += accumulator_vec[2];

    for (ii = quarter_points * 4; ii < num_points; ++ii) {
        accumulator_vec[0] += *(tapsPtr++) * (float)(*(inputPtr++));
    }

    *result = accumulator_vec[0];
}

#endif /*LV_HAVE_NEON*/

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16i_32fc_dot_prod_32fc_neonv8(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const short* inputPtr = input;
    const lv_32fc_t* tapsPtr = taps;

    /* Use 2 independent real/imag accumulators for FMA pipelining */
    float32x4_t real_acc0 = vdupq_n_f32(0);
    float32x4_t imag_acc0 = vdupq_n_f32(0);
    float32x4_t real_acc1 = vdupq_n_f32(0);
    float32x4_t imag_acc1 = vdupq_n_f32(0);

    for (unsigned int number = 0; number < eighthPoints; number++) {
        /* Load 8 int16 values and convert to float */
        int16x8_t input16 = vld1q_s16(inputPtr);
        float32x4_t input_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input16)));
        float32x4_t input_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input16)));

        /* Load 8 complex taps deinterleaved */
        float32x4x2_t taps0 = vld2q_f32((const float*)tapsPtr);
        float32x4x2_t taps1 = vld2q_f32((const float*)(tapsPtr + 4));
        __VOLK_PREFETCH(inputPtr + 16);
        __VOLK_PREFETCH(tapsPtr + 16);

        /* FMA: acc += input * taps */
        real_acc0 = vfmaq_f32(real_acc0, input_lo, taps0.val[0]);
        imag_acc0 = vfmaq_f32(imag_acc0, input_lo, taps0.val[1]);
        real_acc1 = vfmaq_f32(real_acc1, input_hi, taps1.val[0]);
        imag_acc1 = vfmaq_f32(imag_acc1, input_hi, taps1.val[1]);

        inputPtr += 8;
        tapsPtr += 8;
    }

    /* Combine accumulators */
    real_acc0 = vaddq_f32(real_acc0, real_acc1);
    imag_acc0 = vaddq_f32(imag_acc0, imag_acc1);

    /* Horizontal sum */
    float real_sum = vaddvq_f32(real_acc0);
    float imag_sum = vaddvq_f32(imag_acc0);

    lv_32fc_t returnValue = lv_cmake(real_sum, imag_sum);

    /* Handle remainder */
    const float* bPtr = (const float*)tapsPtr;
    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        returnValue += lv_cmake(inputPtr[0] * bPtr[0], inputPtr[0] * bPtr[1]);
        inputPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}
#endif /*LV_HAVE_NEONV8*/

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_16i_32fc_dot_prod_32fc_rvv(lv_32fc_t* result,
                                                   const short* input,
                                                   const lv_32fc_t* taps,
                                                   unsigned int num_points)
{
    vfloat32m4_t vsumr = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)taps, vl);
        vfloat32m4_t vr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 0, vl));
        vfloat32m4_t vi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 32, vl));
        vfloat32m4_t v =
            __riscv_vfwcvt_f(__riscv_vle16_v_i16m2((const int16_t*)input, vl), vl);
        vsumr = __riscv_vfmacc_tu(vsumr, vr, v, vl);
        vsumi = __riscv_vfmacc_tu(vsumi, vi, v, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vr = RISCV_SHRINK4(vfadd, f, 32, vsumr);
    vfloat32m1_t vi = RISCV_SHRINK4(vfadd, f, 32, vsumi);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    *result = lv_cmake(__riscv_vfmv_f(__riscv_vfredusum(vr, z, vl)),
                       __riscv_vfmv_f(__riscv_vfredusum(vi, z, vl)));
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_16i_32fc_dot_prod_32fc_rvvseg(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{
    vfloat32m4_t vsumr = __riscv_vfmv_v_f_f32m4(0, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)taps, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0);
        vfloat32m4_t vi = __riscv_vget_f32m4(vc, 1);
        vfloat32m4_t v =
            __riscv_vfwcvt_f(__riscv_vle16_v_i16m2((const int16_t*)input, vl), vl);
        vsumr = __riscv_vfmacc_tu(vsumr, vr, v, vl);
        vsumi = __riscv_vfmacc_tu(vsumi, vi, v, vl);
    }
    size_t vl = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vr = RISCV_SHRINK4(vfadd, f, 32, vsumr);
    vfloat32m1_t vi = RISCV_SHRINK4(vfadd, f, 32, vsumi);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vl);
    *result = lv_cmake(__riscv_vfmv_f(__riscv_vfredusum(vr, z, vl)),
                       __riscv_vfmv_f(__riscv_vfredusum(vi, z, vl)));
}
#endif /*LV_HAVE_RVVSEG*/


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_u_avx512f(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m512 dotProdVal0 = _mm512_setzero_ps();
    __m512 dotProdVal1 = _mm512_setzero_ps();
    __m512 dotProdVal2 = _mm512_setzero_ps();
    __m512 dotProdVal3 = _mm512_setzero_ps();

    for (; number < thirtysecondPoints; number++) {
        /* Load 32 int16 values in two groups of 16 */
        __m256i m0 = _mm256_loadu_si256((const __m256i*)aPtr);
        __m256i m1 = _mm256_loadu_si256((const __m256i*)(aPtr + 16));

        /* Convert int16 -> int32 -> float */
        __m512 g0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(m0));
        __m512 g1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(m1));

        /* Duplicate each float for complex multiply:
         * g0 = [s0,s1,s2,...,s15]
         * We need [s0,s0,s1,s1,...,s7,s7] and [s8,s8,...,s15,s15]
         * Use permutexvar to duplicate */
        const __m512i duplo_idx =
            _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
        const __m512i duphi_idx =
            _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);

        __m512 a0Val = _mm512_permutexvar_ps(duplo_idx, g0);
        __m512 a1Val = _mm512_permutexvar_ps(duphi_idx, g0);
        __m512 a2Val = _mm512_permutexvar_ps(duplo_idx, g1);
        __m512 a3Val = _mm512_permutexvar_ps(duphi_idx, g1);

        /* Load 32 complex taps (64 floats) */
        __m512 b0Val = _mm512_loadu_ps(bPtr);
        __m512 b1Val = _mm512_loadu_ps(bPtr + 16);
        __m512 b2Val = _mm512_loadu_ps(bPtr + 32);
        __m512 b3Val = _mm512_loadu_ps(bPtr + 48);

        /* FMA: dotProd += a * b */
        dotProdVal0 = _mm512_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm512_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm512_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm512_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 64;
    }

    /* Combine the 4 accumulators */
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal2 = _mm512_add_ps(dotProdVal2, dotProdVal3);
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal2);

    /* Horizontal reduction: 16 floats -> 8 complex pairs -> sum */
    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];
    _mm512_store_ps(dotProductVector, dotProdVal0);

    for (int i = 0; i < 16; i += 2) {
        returnValue += lv_cmake(dotProductVector[i], dotProductVector[i + 1]);
    }

    /* Handle tail */
    number = thirtysecondPoints * 32;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX512F*/


#ifdef LV_HAVE_AVX512VNNI
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_u_avx512vnni(lv_32fc_t* result,
                                                             const short* input,
                                                             const lv_32fc_t* taps,
                                                             unsigned int num_points)
{
    /*
     * VNNI approach: quantize float taps to int16 with a global scale factor,
     * then use _mm512_dpwssds_epi32 for int16 x int16 multiply-accumulate.
     * Each dpwssds processes 32 int16 pairs into 16 int32 partial sums.
     *
     * Precision note: quantizing float taps to int16 introduces error
     * proportional to the dynamic range of the tap values. For typical
     * normalized filter taps this is acceptable (~0.006% for uniform magnitude).
     */
    const float* tapsF = (const float*)taps;

    /* Find max tap magnitude for quantization scale using AVX-512 */
    __m512 max_vec = _mm512_setzero_ps();
    const __m512i abs_mask_i = _mm512_set1_epi32(0x7fffffff);
    unsigned int i;
    const unsigned int num_floats = num_points * 2;
    const unsigned int num_floats_16 = (num_floats / 16) * 16;
    for (i = 0; i < num_floats_16; i += 16) {
        __m512 v = _mm512_loadu_ps(tapsF + i);
        __m512 abs_v = _mm512_castsi512_ps(
            _mm512_and_epi32(_mm512_castps_si512(v), abs_mask_i));
        max_vec = _mm512_max_ps(max_vec, abs_v);
    }
    float max_tap_mag = _mm512_reduce_max_ps(max_vec);
    for (; i < num_floats; i++) {
        float m = fabsf(tapsF[i]);
        if (m > max_tap_mag)
            max_tap_mag = m;
    }

    if (max_tap_mag == 0.0f) {
        *result = lv_cmake(0.0f, 0.0f);
        return;
    }

    const float tap_scale = 16383.0f / max_tap_mag;
    const float inv_tap_scale = max_tap_mag / 16383.0f;
    const __m512 tap_scale_vec = _mm512_set1_ps(tap_scale);
    const __m512 inv_scale_vec = _mm512_set1_ps(inv_tap_scale);

    /* Deinterleave index vectors for separating re/im from complex taps */
    const __m512i idx_re =
        _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im =
        _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    __m512 re_acc0 = _mm512_setzero_ps();
    __m512 re_acc1 = _mm512_setzero_ps();
    __m512 im_acc0 = _mm512_setzero_ps();
    __m512 im_acc1 = _mm512_setzero_ps();

    const short* aPtr = input;
    const float* bPtr = tapsF;
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    for (; number < thirtysecondPoints; number++) {
        /* Load 32 int16 input values */
        __m512i in = _mm512_loadu_si512((const __m512i*)aPtr);

        /* Load 32 complex taps (64 floats) and deinterleave re/im */
        __m512 t0 = _mm512_loadu_ps(bPtr);
        __m512 t1 = _mm512_loadu_ps(bPtr + 16);
        __m512 t2 = _mm512_loadu_ps(bPtr + 32);
        __m512 t3 = _mm512_loadu_ps(bPtr + 48);

        __m512 re_lo = _mm512_permutex2var_ps(t0, idx_re, t1);
        __m512 im_lo = _mm512_permutex2var_ps(t0, idx_im, t1);
        __m512 re_hi = _mm512_permutex2var_ps(t2, idx_re, t3);
        __m512 im_hi = _mm512_permutex2var_ps(t2, idx_im, t3);

        /* Quantize taps to int32, then narrow to int16 */
        __m512i re_lo_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(re_lo, tap_scale_vec));
        __m512i re_hi_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(re_hi, tap_scale_vec));
        __m512i im_lo_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(im_lo, tap_scale_vec));
        __m512i im_hi_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(im_hi, tap_scale_vec));

        /* Narrow int32 to int16 with saturation (AVX-512F) */
        __m256i re_lo_i16 = _mm512_cvtsepi32_epi16(re_lo_i32);
        __m256i re_hi_i16 = _mm512_cvtsepi32_epi16(re_hi_i32);
        __m256i im_lo_i16 = _mm512_cvtsepi32_epi16(im_lo_i32);
        __m256i im_hi_i16 = _mm512_cvtsepi32_epi16(im_hi_i32);

        __m512i re_i16 =
            _mm512_inserti64x4(_mm512_castsi256_si512(re_lo_i16), re_hi_i16, 1);
        __m512i im_i16 =
            _mm512_inserti64x4(_mm512_castsi256_si512(im_lo_i16), im_hi_i16, 1);

        /* VNNI: int16 multiply-accumulate with saturation
         * Each lane j computes: in[2j]*tap[2j] + in[2j+1]*tap[2j+1]
         * giving 16 int32 partial sums from 32 input pairs */
        __m512i re_prod = _mm512_dpwssds_epi32(_mm512_setzero_si512(), in, re_i16);
        __m512i im_prod = _mm512_dpwssds_epi32(_mm512_setzero_si512(), in, im_i16);

        /* Convert to float, rescale, and accumulate */
        re_acc0 = _mm512_add_ps(
            re_acc0, _mm512_mul_ps(_mm512_cvtepi32_ps(re_prod), inv_scale_vec));
        im_acc0 = _mm512_add_ps(
            im_acc0, _mm512_mul_ps(_mm512_cvtepi32_ps(im_prod), inv_scale_vec));

        aPtr += 32;
        bPtr += 64;
    }

    /* Reduce 512-bit accumulators to scalar */
    float re_sum = _mm512_reduce_add_ps(_mm512_add_ps(re_acc0, re_acc1));
    float im_sum = _mm512_reduce_add_ps(_mm512_add_ps(im_acc0, im_acc1));

    lv_32fc_t returnValue = lv_cmake(re_sum, im_sum);

    /* Handle tail via generic */
    unsigned int processed = thirtysecondPoints * 32;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + processed, taps + processed, num_points - processed);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX512VNNI*/


#endif /* INCLUDED_volk_16i_32fc_dot_prod_32fc_u_H */

#ifndef INCLUDED_volk_16i_32fc_dot_prod_32fc_a_H
#define INCLUDED_volk_16i_32fc_dot_prod_32fc_a_H

#if LV_HAVE_SSE && LV_HAVE_MMX


static inline void volk_16i_32fc_dot_prod_32fc_a_sse(lv_32fc_t* result,
                                                     const short* input,
                                                     const lv_32fc_t* taps,
                                                     unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m64 m0, m1;
    __m128 f0, f1, f2, f3;
    __m128 a0Val, a1Val, a2Val, a3Val;
    __m128 b0Val, b1Val, b2Val, b3Val;
    __m128 c0Val, c1Val, c2Val, c3Val;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {

        m0 = _mm_set_pi16(*(aPtr + 3), *(aPtr + 2), *(aPtr + 1), *(aPtr + 0));
        m1 = _mm_set_pi16(*(aPtr + 7), *(aPtr + 6), *(aPtr + 5), *(aPtr + 4));
        f0 = _mm_cvtpi16_ps(m0);
        f1 = _mm_cvtpi16_ps(m0);
        f2 = _mm_cvtpi16_ps(m1);
        f3 = _mm_cvtpi16_ps(m1);

        a0Val = _mm_unpacklo_ps(f0, f1);
        a1Val = _mm_unpackhi_ps(f0, f1);
        a2Val = _mm_unpacklo_ps(f2, f3);
        a3Val = _mm_unpackhi_ps(f2, f3);

        b0Val = _mm_load_ps(bPtr);
        b1Val = _mm_load_ps(bPtr + 4);
        b2Val = _mm_load_ps(bPtr + 8);
        b3Val = _mm_load_ps(bPtr + 12);

        c0Val = _mm_mul_ps(a0Val, b0Val);
        c1Val = _mm_mul_ps(a1Val, b1Val);
        c2Val = _mm_mul_ps(a2Val, b2Val);
        c3Val = _mm_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    _mm_empty(); // clear the mmx technology state

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

    _mm_store_ps(dotProductVector,
                 dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}

#endif /*LV_HAVE_SSE && LV_HAVE_MMX*/

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_a_sse2(lv_32fc_t* result,
                                                       const short* input,
                                                       const lv_32fc_t* taps,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values */
        __m128i v = _mm_load_si128((const __m128i*)aPtr);

        /* Sign-extend lower 4 int16 to int32 (no SSE4.1 pmovsxwd) */
        __m128i lo32 = _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16);
        /* Sign-extend upper 4 int16 to int32 */
        __m128i hi32 = _mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16);

        /* Convert int32 to float */
        __m128 f0 = _mm_cvtepi32_ps(lo32);
        __m128 f1 = _mm_cvtepi32_ps(hi32);

        /* Duplicate each float for complex multiply */
        __m128 a0Val = _mm_unpacklo_ps(f0, f0);
        __m128 a1Val = _mm_unpackhi_ps(f0, f0);
        __m128 a2Val = _mm_unpacklo_ps(f1, f1);
        __m128 a3Val = _mm_unpackhi_ps(f1, f1);

        /* Load 8 complex taps (16 floats) */
        __m128 b0Val = _mm_load_ps(bPtr);
        __m128 b1Val = _mm_load_ps(bPtr + 4);
        __m128 b2Val = _mm_load_ps(bPtr + 8);
        __m128 b3Val = _mm_load_ps(bPtr + 12);

        dotProdVal0 = _mm_add_ps(_mm_mul_ps(a0Val, b0Val), dotProdVal0);
        dotProdVal1 = _mm_add_ps(_mm_mul_ps(a1Val, b1Val), dotProdVal1);
        dotProdVal2 = _mm_add_ps(_mm_mul_ps(a2Val, b2Val), dotProdVal2);
        dotProdVal3 = _mm_add_ps(_mm_mul_ps(a3Val, b3Val), dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, dotProdVal0);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_SSE2*/

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_a_sse4_1(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128 dotProdVal0 = _mm_setzero_ps();
    __m128 dotProdVal1 = _mm_setzero_ps();
    __m128 dotProdVal2 = _mm_setzero_ps();
    __m128 dotProdVal3 = _mm_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values */
        __m128i v = _mm_load_si128((const __m128i*)aPtr);

        /* Sign-extend int16 to int32 using SSE4.1 pmovsxwd */
        __m128i lo32 = _mm_cvtepi16_epi32(v);
        __m128i hi32 = _mm_cvtepi16_epi32(_mm_srli_si128(v, 8));

        /* Convert int32 to float */
        __m128 f0 = _mm_cvtepi32_ps(lo32);
        __m128 f1 = _mm_cvtepi32_ps(hi32);

        /* Duplicate each float for complex multiply */
        __m128 a0Val = _mm_unpacklo_ps(f0, f0);
        __m128 a1Val = _mm_unpackhi_ps(f0, f0);
        __m128 a2Val = _mm_unpacklo_ps(f1, f1);
        __m128 a3Val = _mm_unpackhi_ps(f1, f1);

        /* Load 8 complex taps (16 floats) */
        __m128 b0Val = _mm_load_ps(bPtr);
        __m128 b1Val = _mm_load_ps(bPtr + 4);
        __m128 b2Val = _mm_load_ps(bPtr + 8);
        __m128 b3Val = _mm_load_ps(bPtr + 12);

        dotProdVal0 = _mm_add_ps(_mm_mul_ps(a0Val, b0Val), dotProdVal0);
        dotProdVal1 = _mm_add_ps(_mm_mul_ps(a1Val, b1Val), dotProdVal1);
        dotProdVal2 = _mm_add_ps(_mm_mul_ps(a2Val, b2Val), dotProdVal2);
        dotProdVal3 = _mm_add_ps(_mm_mul_ps(a3Val, b3Val), dotProdVal3);

        aPtr += 8;
        bPtr += 16;
    }

    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, dotProdVal0);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_SSE4_1*/

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_a_avx(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values into 128-bit register */
        __m128i m0 = _mm_load_si128((const __m128i*)aPtr);

        /* Convert int16 -> int32 -> float using SSE4.1 */
        __m128 lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(m0));
        __m128 hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(m0, 8)));

        /* Duplicate each float for complex multiply:
         * [s0,s1,s2,s3] -> [s0,s0,s1,s1] and [s2,s2,s3,s3]
         * [s4,s5,s6,s7] -> [s4,s4,s5,s5] and [s6,s6,s7,s7] */
        __m128 dup_lo0 = _mm_unpacklo_ps(lo, lo);
        __m128 dup_lo1 = _mm_unpackhi_ps(lo, lo);
        __m128 dup_hi0 = _mm_unpacklo_ps(hi, hi);
        __m128 dup_hi1 = _mm_unpackhi_ps(hi, hi);

        /* Combine into 256-bit vectors */
        __m256 a0Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_lo0), dup_lo1, 1);
        __m256 a1Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_hi0), dup_hi1, 1);

        /* Load 8 complex taps (16 floats) in two 256-bit registers */
        __m256 b0Val = _mm256_load_ps(bPtr);
        __m256 b1Val = _mm256_load_ps(bPtr + 8);

        /* Multiply and accumulate */
        dotProdVal0 = _mm256_add_ps(dotProdVal0, _mm256_mul_ps(a0Val, b0Val));
        dotProdVal1 = _mm256_add_ps(dotProdVal1, _mm256_mul_ps(a1Val, b1Val));

        aPtr += 8;
        bPtr += 16;
    }

    /* Combine accumulators */
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);

    /* Horizontal reduction: 256-bit -> 128-bit */
    __m128 hi128 = _mm256_extractf128_ps(dotProdVal0, 1);
    __m128 lo128 = _mm256_castps256_ps128(dotProdVal0);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, sum128);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    /* Handle tail */
    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX*/


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_a_avx_fma(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {
        /* Load 8 int16 values into 128-bit register */
        __m128i m0 = _mm_load_si128((const __m128i*)aPtr);

        /* Convert int16 -> int32 -> float using SSE4.1 */
        __m128 lo = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(m0));
        __m128 hi = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(m0, 8)));

        /* Duplicate each float for complex multiply */
        __m128 dup_lo0 = _mm_unpacklo_ps(lo, lo);
        __m128 dup_lo1 = _mm_unpackhi_ps(lo, lo);
        __m128 dup_hi0 = _mm_unpacklo_ps(hi, hi);
        __m128 dup_hi1 = _mm_unpackhi_ps(hi, hi);

        /* Combine into 256-bit vectors */
        __m256 a0Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_lo0), dup_lo1, 1);
        __m256 a1Val = _mm256_insertf128_ps(_mm256_castps128_ps256(dup_hi0), dup_hi1, 1);

        /* Load 8 complex taps (16 floats) in two 256-bit registers */
        __m256 b0Val = _mm256_load_ps(bPtr);
        __m256 b1Val = _mm256_load_ps(bPtr + 8);

        /* FMA accumulate */
        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);

        aPtr += 8;
        bPtr += 16;
    }

    /* Combine accumulators */
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);

    /* Horizontal reduction: 256-bit -> 128-bit */
    __m128 hi128 = _mm256_extractf128_ps(dotProdVal0, 1);
    __m128 lo128 = _mm256_castps256_ps128(dotProdVal0);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];
    _mm_store_ps(dotProductVector, sum128);

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);

    /* Handle tail */
    number = eighthPoints * 8;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX && LV_HAVE_FMA*/


#ifdef LV_HAVE_AVX2

static inline void volk_16i_32fc_dot_prod_32fc_a_avx2(lv_32fc_t* result,
                                                      const short* input,
                                                      const lv_32fc_t* taps,
                                                      unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;
    __m256 c0Val, c1Val, c2Val, c3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_load_si128((__m128i const*)aPtr);
        m1 = _mm_load_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_load_ps(bPtr);
        b1Val = _mm256_load_ps(bPtr + 8);
        b2Val = _mm256_load_ps(bPtr + 16);
        b3Val = _mm256_load_ps(bPtr + 24);

        c0Val = _mm256_mul_ps(a0Val, b0Val);
        c1Val = _mm256_mul_ps(a1Val, b1Val);
        c2Val = _mm256_mul_ps(a2Val, b2Val);
        c3Val = _mm256_mul_ps(a3Val, b3Val);

        dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
        dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
        dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
        dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}


#endif /*LV_HAVE_AVX2*/

#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void volk_16i_32fc_dot_prod_32fc_a_avx2_fma(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m128i m0, m1;
    __m256i f0, f1;
    __m256 g0, g1, h0, h1, h2, h3;
    __m256 a0Val, a1Val, a2Val, a3Val;
    __m256 b0Val, b1Val, b2Val, b3Val;

    __m256 dotProdVal0 = _mm256_setzero_ps();
    __m256 dotProdVal1 = _mm256_setzero_ps();
    __m256 dotProdVal2 = _mm256_setzero_ps();
    __m256 dotProdVal3 = _mm256_setzero_ps();

    for (; number < sixteenthPoints; number++) {

        m0 = _mm_load_si128((__m128i const*)aPtr);
        m1 = _mm_load_si128((__m128i const*)(aPtr + 8));

        f0 = _mm256_cvtepi16_epi32(m0);
        g0 = _mm256_cvtepi32_ps(f0);
        f1 = _mm256_cvtepi16_epi32(m1);
        g1 = _mm256_cvtepi32_ps(f1);

        h0 = _mm256_unpacklo_ps(g0, g0);
        h1 = _mm256_unpackhi_ps(g0, g0);
        h2 = _mm256_unpacklo_ps(g1, g1);
        h3 = _mm256_unpackhi_ps(g1, g1);

        a0Val = _mm256_permute2f128_ps(h0, h1, 0x20);
        a1Val = _mm256_permute2f128_ps(h0, h1, 0x31);
        a2Val = _mm256_permute2f128_ps(h2, h3, 0x20);
        a3Val = _mm256_permute2f128_ps(h2, h3, 0x31);

        b0Val = _mm256_load_ps(bPtr);
        b1Val = _mm256_load_ps(bPtr + 8);
        b2Val = _mm256_load_ps(bPtr + 16);
        b3Val = _mm256_load_ps(bPtr + 24);

        dotProdVal0 = _mm256_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm256_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm256_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm256_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 16;
        bPtr += 32;
    }

    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
    dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

    __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

    _mm256_store_ps(dotProductVector,
                    dotProdVal0); // Store the results back into the dot product vector

    returnValue += lv_cmake(dotProductVector[0], dotProductVector[1]);
    returnValue += lv_cmake(dotProductVector[2], dotProductVector[3]);
    returnValue += lv_cmake(dotProductVector[4], dotProductVector[5]);
    returnValue += lv_cmake(dotProductVector[6], dotProductVector[7]);

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        returnValue += lv_cmake(aPtr[0] * bPtr[0], aPtr[0] * bPtr[1]);
        aPtr += 1;
        bPtr += 2;
    }

    *result = returnValue;
}


#endif /*LV_HAVE_AVX2 && LV_HAVE_FMA*/


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_a_avx512f(lv_32fc_t* result,
                                                          const short* input,
                                                          const lv_32fc_t* taps,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    lv_32fc_t returnValue = lv_cmake(0.0f, 0.0f);
    const short* aPtr = input;
    const float* bPtr = (const float*)taps;

    __m512 dotProdVal0 = _mm512_setzero_ps();
    __m512 dotProdVal1 = _mm512_setzero_ps();
    __m512 dotProdVal2 = _mm512_setzero_ps();
    __m512 dotProdVal3 = _mm512_setzero_ps();

    for (; number < thirtysecondPoints; number++) {
        /* Load 32 int16 values in two groups of 16 */
        __m256i m0 = _mm256_load_si256((const __m256i*)aPtr);
        __m256i m1 = _mm256_load_si256((const __m256i*)(aPtr + 16));

        /* Convert int16 -> int32 -> float */
        __m512 g0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(m0));
        __m512 g1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(m1));

        /* Duplicate each float for complex multiply */
        const __m512i duplo_idx =
            _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
        const __m512i duphi_idx =
            _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);

        __m512 a0Val = _mm512_permutexvar_ps(duplo_idx, g0);
        __m512 a1Val = _mm512_permutexvar_ps(duphi_idx, g0);
        __m512 a2Val = _mm512_permutexvar_ps(duplo_idx, g1);
        __m512 a3Val = _mm512_permutexvar_ps(duphi_idx, g1);

        /* Load 32 complex taps (64 floats) */
        __m512 b0Val = _mm512_load_ps(bPtr);
        __m512 b1Val = _mm512_load_ps(bPtr + 16);
        __m512 b2Val = _mm512_load_ps(bPtr + 32);
        __m512 b3Val = _mm512_load_ps(bPtr + 48);

        /* FMA: dotProd += a * b */
        dotProdVal0 = _mm512_fmadd_ps(a0Val, b0Val, dotProdVal0);
        dotProdVal1 = _mm512_fmadd_ps(a1Val, b1Val, dotProdVal1);
        dotProdVal2 = _mm512_fmadd_ps(a2Val, b2Val, dotProdVal2);
        dotProdVal3 = _mm512_fmadd_ps(a3Val, b3Val, dotProdVal3);

        aPtr += 32;
        bPtr += 64;
    }

    /* Combine the 4 accumulators */
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal1);
    dotProdVal2 = _mm512_add_ps(dotProdVal2, dotProdVal3);
    dotProdVal0 = _mm512_add_ps(dotProdVal0, dotProdVal2);

    /* Horizontal reduction */
    __VOLK_ATTR_ALIGNED(64) float dotProductVector[16];
    _mm512_store_ps(dotProductVector, dotProdVal0);

    for (int i = 0; i < 16; i += 2) {
        returnValue += lv_cmake(dotProductVector[i], dotProductVector[i + 1]);
    }

    /* Handle tail */
    number = thirtysecondPoints * 32;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + number, taps + number, num_points - number);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX512F*/


#ifdef LV_HAVE_AVX512VNNI
#include <immintrin.h>

static inline void volk_16i_32fc_dot_prod_32fc_a_avx512vnni(lv_32fc_t* result,
                                                             const short* input,
                                                             const lv_32fc_t* taps,
                                                             unsigned int num_points)
{
    const float* tapsF = (const float*)taps;

    /* Find max tap magnitude for quantization scale using AVX-512 */
    __m512 max_vec = _mm512_setzero_ps();
    const __m512i abs_mask_i = _mm512_set1_epi32(0x7fffffff);
    unsigned int i;
    const unsigned int num_floats = num_points * 2;
    const unsigned int num_floats_16 = (num_floats / 16) * 16;
    for (i = 0; i < num_floats_16; i += 16) {
        __m512 v = _mm512_load_ps(tapsF + i);
        __m512 abs_v = _mm512_castsi512_ps(
            _mm512_and_epi32(_mm512_castps_si512(v), abs_mask_i));
        max_vec = _mm512_max_ps(max_vec, abs_v);
    }
    float max_tap_mag = _mm512_reduce_max_ps(max_vec);
    for (; i < num_floats; i++) {
        float m = fabsf(tapsF[i]);
        if (m > max_tap_mag)
            max_tap_mag = m;
    }

    if (max_tap_mag == 0.0f) {
        *result = lv_cmake(0.0f, 0.0f);
        return;
    }

    const float tap_scale = 16383.0f / max_tap_mag;
    const float inv_tap_scale = max_tap_mag / 16383.0f;
    const __m512 tap_scale_vec = _mm512_set1_ps(tap_scale);
    const __m512 inv_scale_vec = _mm512_set1_ps(inv_tap_scale);

    const __m512i idx_re =
        _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im =
        _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    __m512 re_acc0 = _mm512_setzero_ps();
    __m512 re_acc1 = _mm512_setzero_ps();
    __m512 im_acc0 = _mm512_setzero_ps();
    __m512 im_acc1 = _mm512_setzero_ps();

    const short* aPtr = input;
    const float* bPtr = tapsF;
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    for (; number < thirtysecondPoints; number++) {
        __m512i in = _mm512_load_si512((const __m512i*)aPtr);

        __m512 t0 = _mm512_load_ps(bPtr);
        __m512 t1 = _mm512_load_ps(bPtr + 16);
        __m512 t2 = _mm512_load_ps(bPtr + 32);
        __m512 t3 = _mm512_load_ps(bPtr + 48);

        __m512 re_lo = _mm512_permutex2var_ps(t0, idx_re, t1);
        __m512 im_lo = _mm512_permutex2var_ps(t0, idx_im, t1);
        __m512 re_hi = _mm512_permutex2var_ps(t2, idx_re, t3);
        __m512 im_hi = _mm512_permutex2var_ps(t2, idx_im, t3);

        __m512i re_lo_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(re_lo, tap_scale_vec));
        __m512i re_hi_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(re_hi, tap_scale_vec));
        __m512i im_lo_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(im_lo, tap_scale_vec));
        __m512i im_hi_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(im_hi, tap_scale_vec));

        __m256i re_lo_i16 = _mm512_cvtsepi32_epi16(re_lo_i32);
        __m256i re_hi_i16 = _mm512_cvtsepi32_epi16(re_hi_i32);
        __m256i im_lo_i16 = _mm512_cvtsepi32_epi16(im_lo_i32);
        __m256i im_hi_i16 = _mm512_cvtsepi32_epi16(im_hi_i32);

        __m512i re_i16 =
            _mm512_inserti64x4(_mm512_castsi256_si512(re_lo_i16), re_hi_i16, 1);
        __m512i im_i16 =
            _mm512_inserti64x4(_mm512_castsi256_si512(im_lo_i16), im_hi_i16, 1);

        __m512i re_prod = _mm512_dpwssds_epi32(_mm512_setzero_si512(), in, re_i16);
        __m512i im_prod = _mm512_dpwssds_epi32(_mm512_setzero_si512(), in, im_i16);

        re_acc0 = _mm512_add_ps(
            re_acc0, _mm512_mul_ps(_mm512_cvtepi32_ps(re_prod), inv_scale_vec));
        im_acc0 = _mm512_add_ps(
            im_acc0, _mm512_mul_ps(_mm512_cvtepi32_ps(im_prod), inv_scale_vec));

        aPtr += 32;
        bPtr += 64;
    }

    float re_sum = _mm512_reduce_add_ps(_mm512_add_ps(re_acc0, re_acc1));
    float im_sum = _mm512_reduce_add_ps(_mm512_add_ps(im_acc0, im_acc1));

    lv_32fc_t returnValue = lv_cmake(re_sum, im_sum);

    unsigned int processed = thirtysecondPoints * 32;
    lv_32fc_t tail_result;
    volk_16i_32fc_dot_prod_32fc_generic(
        &tail_result, input + processed, taps + processed, num_points - processed);
    returnValue += tail_result;

    *result = returnValue;
}

#endif /*LV_HAVE_AVX512VNNI*/


#endif /* INCLUDED_volk_16i_32fc_dot_prod_32fc_a_H */
