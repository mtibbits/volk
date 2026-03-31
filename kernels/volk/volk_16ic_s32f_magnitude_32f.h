/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_s32f_magnitude_32f
 *
 * \b Overview
 *
 * Computes the magnitude of each complex 16-bit integer sample and divides
 * by a scalar, storing the results as 32-bit floats:
 * magnitudeVector[i] = sqrt(real[i]^2 + imag[i]^2) / scalar.
 *
 * This kernel is useful in signal processing pipelines where received
 * complex samples are represented as fixed-point 16-bit I/Q pairs and
 * need to be converted to a floating-point envelope (magnitude) estimate.
 * The scalar divisor allows normalization — for example, removing ADC
 * full-scale gain so that downstream stages (AGC, power measurement,
 * spectral analysis) operate on calibrated amplitude values.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_s32f_magnitude_32f(float* magnitudeVector, const lv_16sc_t*
 * complexVector, const float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of 16-bit I/Q samples (lv_16sc_t).
 * \li scalar: The value to divide each magnitude by (e.g. ADC full-scale).
 * \li num_points: The number of complex samples.
 *
 * \b Outputs
 * \li magnitudeVector: The scaled magnitude of each complex sample (float).
 *
 * \b Example
 * Compute scaled magnitude of four complex samples using a 3-4-5 triangle.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_16sc_t* complexVector =
 *     (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t) * N, alignment);
 * float* magnitudeVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     complexVector[i] = lv_cmake((int16_t)3, (int16_t)4);
 * }
 * float scalar = 5.0f;
 *
 * // Expected: sqrt(3^2 + 4^2) / 5 = 5 / 5 = 1.0 for each element
 *
 * volk_16ic_s32f_magnitude_32f(magnitudeVector, complexVector, scalar, N);
 *
 * printf("Expected: 1.000000\n");
 * printf("Result:   %f\n", magnitudeVector[0]);
 *
 * volk_free(complexVector);
 * volk_free(magnitudeVector);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_s32f_magnitude_32f_u_H
#define INCLUDED_volk_16ic_s32f_magnitude_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_s32f_magnitude_32f_generic(float* magnitudeVector,
                                                        const lv_16sc_t* complexVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    const float invScalar = 1.0f / scalar;
    for (number = 0; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) * invScalar;
        float imag = ((float)(*complexVectorPtr++)) * invScalar;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_sse(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);

    __m128 cplxValue1, cplxValue2, result, re, im;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

    for (; number < quarterPoints; number++) {
        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        re = _mm_shuffle_ps(cplxValue1, cplxValue2, 0x88);
        im = _mm_shuffle_ps(cplxValue1, cplxValue2, 0xdd);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(re, invScalar);
        cplxValue2 = _mm_mul_ps(im, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_add_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        _mm_storeu_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}

#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_sse2(float* magnitudeVector,
                                                        const lv_16sc_t* complexVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    for (; number < quarterPoints; number++) {
        /* Load 4 complex int16 samples (8 x int16) */
        __m128i raw = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        /* Sign-extend int16 -> int32 using SSE2 arithmetic shift */
        __m128i sign = _mm_srai_epi16(raw, 15);
        __m128i lo32 = _mm_unpacklo_epi16(raw, sign); /* [I0,Q0,I1,Q1] as int32 */
        __m128i hi32 = _mm_unpackhi_epi16(raw, sign); /* [I2,Q2,I3,Q3] as int32 */

        /* Convert to float */
        __m128 flo = _mm_cvtepi32_ps(lo32);
        __m128 fhi = _mm_cvtepi32_ps(hi32);

        /* Deinterleave real and imaginary */
        __m128 re = _mm_shuffle_ps(flo, fhi, 0x88); /* [I0,I1,I2,I3] */
        __m128 im = _mm_shuffle_ps(flo, fhi, 0xdd); /* [Q0,Q1,Q2,Q3] */

        /* Scale by 1/scalar */
        re = _mm_mul_ps(re, invScalar);
        im = _mm_mul_ps(im, invScalar);

        /* mag = sqrt(re^2 + im^2) */
        __m128 reSquared = _mm_mul_ps(re, re);
        __m128 imSquared = _mm_mul_ps(im, im);
        __m128 magSquared = _mm_add_ps(reSquared, imSquared);
        __m128 result = _mm_sqrt_ps(magSquared);

        _mm_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_sse3(float* magnitudeVector,
                                                        const lv_16sc_t* complexVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        _mm_storeu_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_avx(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < eighthPoints; number++) {
        /* Load 8 complex int16 samples as two 128-bit halves.
         * Each 128-bit register holds 4 complex samples packed as int32 per sample. */
        __m128i v0 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        __m128i v1 = _mm_loadu_si128((const __m128i*)(complexVectorPtr + 8));
        complexVectorPtr += 16;

        /* Sign-extend: real = lower 16 bits, imag = upper 16 bits */
        __m128i i0 = _mm_srai_epi32(_mm_slli_epi32(v0, 16), 16);
        __m128i q0 = _mm_srai_epi32(v0, 16);
        __m128i i1 = _mm_srai_epi32(_mm_slli_epi32(v1, 16), 16);
        __m128i q1 = _mm_srai_epi32(v1, 16);

        /* Convert to float and scale; combine pairs into 256-bit vectors */
        __m256 re = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(i1), _mm_cvtepi32_ps(i0)), invScalar);
        __m256 im = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(q1), _mm_cvtepi32_ps(q0)), invScalar);

        /* mag = sqrt(re^2 + im^2) */
        __m256 reSquared = _mm256_mul_ps(re, re);
        __m256 imSquared = _mm256_mul_ps(im, im);
        __m256 magSquared = _mm256_add_ps(reSquared, imSquared);
        __m256 result = _mm256_sqrt_ps(magSquared);

        _mm256_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_avx_fma(float* magnitudeVector,
                                                            const lv_16sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < eighthPoints; number++) {
        /* Load 8 complex int16 samples as two 128-bit halves. */
        __m128i v0 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        __m128i v1 = _mm_loadu_si128((const __m128i*)(complexVectorPtr + 8));
        complexVectorPtr += 16;

        /* Sign-extend: real = lower 16 bits, imag = upper 16 bits */
        __m128i i0 = _mm_srai_epi32(_mm_slli_epi32(v0, 16), 16);
        __m128i q0 = _mm_srai_epi32(v0, 16);
        __m128i i1 = _mm_srai_epi32(_mm_slli_epi32(v1, 16), 16);
        __m128i q1 = _mm_srai_epi32(v1, 16);

        /* Convert to float and scale; combine pairs into 256-bit vectors */
        __m256 re = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(i1), _mm_cvtepi32_ps(i0)), invScalar);
        __m256 im = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(q1), _mm_cvtepi32_ps(q0)), invScalar);

        /* mag = sqrt(re^2 + im^2) using FMA */
        __m256 magSquared = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));
        __m256 result = _mm256_sqrt_ps(magSquared);

        _mm256_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_avx2(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    __m256 cplxValue1, cplxValue2, result;
    __m256i int1, int2;
    __m128i short1, short2;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (; number < eighthPoints; number++) {

        int1 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        short1 = _mm256_extracti128_si256(int1, 0);
        short2 = _mm256_extracti128_si256(int1, 1);

        int1 = _mm256_cvtepi16_epi32(short1);
        int2 = _mm256_cvtepi16_epi32(short2);
        cplxValue1 = _mm256_cvtepi32_ps(int1);
        cplxValue2 = _mm256_cvtepi32_ps(int2);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values
        result = _mm256_permutevar8x32_ps(result, idx);

        result = _mm256_sqrt_ps(result); // Square root the values

        _mm256_storeu_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_16ic_s32f_magnitude_32f_u_avx2_fma(float* magnitudeVector,
                                                            const lv_16sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < eighthPoints; number++) {
        __m256i raw = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        __m128i lo_half = _mm256_extracti128_si256(raw, 0);
        __m128i hi_half = _mm256_extracti128_si256(raw, 1);

        __m256 cplxValue1 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_half)), invScalar);
        __m256 cplxValue2 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_half)), invScalar);

        __m256 result = _mm256_magnitudesquared_ps_avx2_fma(cplxValue1, cplxValue2);

        result = _mm256_sqrt_ps(result);

        _mm256_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_u_avx512f(float* magnitudeVector,
                                                           const lv_16sc_t* complexVector,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m512 vInvScalar = _mm512_set1_ps(fInvScalar);

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        /* Load 16 complex int16 samples = 512 bits */
        __m512i raw = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 32;

        /* Split into two 256-bit halves and widen int16 -> int32 */
        __m256i lo_half = _mm512_castsi512_si256(raw);
        __m256i hi_half = _mm512_extracti64x4_epi64(raw, 1);
        __m512i wide_lo = _mm512_cvtepi16_epi32(lo_half);
        __m512i wide_hi = _mm512_cvtepi16_epi32(hi_half);

        /* Convert int32 -> float */
        __m512 flt_lo = _mm512_cvtepi32_ps(wide_lo);
        __m512 flt_hi = _mm512_cvtepi32_ps(wide_hi);

        /* Scale by 1/scalar */
        flt_lo = _mm512_mul_ps(flt_lo, vInvScalar);
        flt_hi = _mm512_mul_ps(flt_hi, vInvScalar);

        /* Deinterleave re/im */
        __m512 re = _mm512_permutex2var_ps(flt_lo, idx_re, flt_hi);
        __m512 im = _mm512_permutex2var_ps(flt_lo, idx_im, flt_hi);

        /* mag = sqrt(re^2 + im^2) */
        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 magSquared = _mm512_add_ps(reSquared, imSquared);
        __m512 result = _mm512_sqrt_ps(magSquared);

        _mm512_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_s32f_magnitude_32f_neon(float* magnitudeVector,
                                                     const lv_16sc_t* complexVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < quarter_points; number++) {
        int16x4x2_t input = vld2_s16(complexVectorPtr);
        complexVectorPtr += 8;

        int32x4_t realInt = vmovl_s16(input.val[0]);
        int32x4_t imagInt = vmovl_s16(input.val[1]);

        float32x4_t realFloat = vcvtq_f32_s32(realInt);
        float32x4_t imagFloat = vcvtq_f32_s32(imagInt);

        realFloat = vmulq_f32(realFloat, vInvScalar);
        imagFloat = vmulq_f32(imagFloat, vInvScalar);

        float32x4_t realSquared = vmulq_f32(realFloat, realFloat);
        float32x4_t imagSquared = vmulq_f32(imagFloat, imagFloat);
        float32x4_t sumSquared = vaddq_f32(realSquared, imagSquared);

        /* Use reciprocal square root estimate with Newton-Raphson refinement */
        float32x4_t rsqrt = vrsqrteq_f32(sumSquared);
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(sumSquared, rsqrt), rsqrt));
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(sumSquared, rsqrt), rsqrt));
        float32x4_t result = vmulq_f32(sumSquared, rsqrt);

        /* Handle zero case - if sumSquared is 0, result should be 0 */
        uint32x4_t zero_mask = vceqq_f32(sumSquared, vdupq_n_f32(0.0f));
        result = vbslq_f32(zero_mask, sumSquared, result);

        vst1q_f32(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarter_points * 4;
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) * invScalar;
        float imag = ((float)(*complexVectorPtr++)) * invScalar;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_s32f_magnitude_32f_neonv8(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < eighth_points; number++) {
        int16x8x2_t input = vld2q_s16(complexVectorPtr);
        complexVectorPtr += 16;
        __VOLK_PREFETCH(complexVectorPtr + 16);

        /* First 4 elements */
        int32x4_t realInt0 = vmovl_s16(vget_low_s16(input.val[0]));
        int32x4_t imagInt0 = vmovl_s16(vget_low_s16(input.val[1]));

        float32x4_t realFloat0 = vcvtq_f32_s32(realInt0);
        float32x4_t imagFloat0 = vcvtq_f32_s32(imagInt0);

        realFloat0 = vmulq_f32(realFloat0, vInvScalar);
        imagFloat0 = vmulq_f32(imagFloat0, vInvScalar);

        float32x4_t sumSquared0 =
            vfmaq_f32(vmulq_f32(imagFloat0, imagFloat0), realFloat0, realFloat0);
        float32x4_t result0 = vsqrtq_f32(sumSquared0);

        /* Second 4 elements */
        int32x4_t realInt1 = vmovl_s16(vget_high_s16(input.val[0]));
        int32x4_t imagInt1 = vmovl_s16(vget_high_s16(input.val[1]));

        float32x4_t realFloat1 = vcvtq_f32_s32(realInt1);
        float32x4_t imagFloat1 = vcvtq_f32_s32(imagInt1);

        realFloat1 = vmulq_f32(realFloat1, vInvScalar);
        imagFloat1 = vmulq_f32(imagFloat1, vInvScalar);

        float32x4_t sumSquared1 =
            vfmaq_f32(vmulq_f32(imagFloat1, imagFloat1), realFloat1, realFloat1);
        float32x4_t result1 = vsqrtq_f32(sumSquared1);

        vst1q_f32(magnitudeVectorPtr, result0);
        vst1q_f32(magnitudeVectorPtr + 4, result1);
        magnitudeVectorPtr += 8;
    }

    number = eighth_points * 8;
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) * invScalar;
        float imag = ((float)(*complexVectorPtr++)) * invScalar;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_s32f_magnitude_32f_rvv(float* magnitudeVector,
                                                    const lv_16sc_t* complexVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint32m8_t vc = __riscv_vle32_v_i32m8((const int32_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vnsra(vc, 0, vl);
        vint16m4_t vi = __riscv_vnsra(vc, 16, vl);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), 1.0f / scalar, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), 1.0f / scalar, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        __riscv_vse32(magnitudeVector, __riscv_vfsqrt(vf, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_16ic_s32f_magnitude_32f_rvvseg(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4x2_t vc = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vget_i16m4(vc, 0);
        vint16m4_t vi = __riscv_vget_i16m4(vc, 1);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), 1.0f / scalar, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), 1.0f / scalar, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        __riscv_vse32(magnitudeVector, __riscv_vfsqrt(vf, vl), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_16ic_s32f_magnitude_32f_u_H */

#ifndef INCLUDED_volk_16ic_s32f_magnitude_32f_a_H
#define INCLUDED_volk_16ic_s32f_magnitude_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_sse(float* magnitudeVector,
                                                      const lv_16sc_t* complexVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);

    __m128 cplxValue1, cplxValue2, result, re, im;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

    for (; number < quarterPoints; number++) {
        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        re = _mm_shuffle_ps(cplxValue1, cplxValue2, 0x88);
        im = _mm_shuffle_ps(cplxValue1, cplxValue2, 0xdd);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(re, invScalar);
        cplxValue2 = _mm_mul_ps(im, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_add_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        _mm_store_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) * iScalar;
        float val1Imag = (float)(*complexVectorPtr++) * iScalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}


#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_sse2(float* magnitudeVector,
                                                        const lv_16sc_t* complexVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    for (; number < quarterPoints; number++) {
        /* Load 4 complex int16 samples (8 x int16) */
        __m128i raw = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        /* Sign-extend int16 -> int32 using SSE2 arithmetic shift */
        __m128i sign = _mm_srai_epi16(raw, 15);
        __m128i lo32 = _mm_unpacklo_epi16(raw, sign); /* [I0,Q0,I1,Q1] as int32 */
        __m128i hi32 = _mm_unpackhi_epi16(raw, sign); /* [I2,Q2,I3,Q3] as int32 */

        /* Convert to float */
        __m128 flo = _mm_cvtepi32_ps(lo32);
        __m128 fhi = _mm_cvtepi32_ps(hi32);

        /* Deinterleave real and imaginary */
        __m128 re = _mm_shuffle_ps(flo, fhi, 0x88); /* [I0,I1,I2,I3] */
        __m128 im = _mm_shuffle_ps(flo, fhi, 0xdd); /* [Q0,Q1,Q2,Q3] */

        /* Scale by 1/scalar */
        re = _mm_mul_ps(re, invScalar);
        im = _mm_mul_ps(im, invScalar);

        /* mag = sqrt(re^2 + im^2) */
        __m128 reSquared = _mm_mul_ps(re, re);
        __m128 imSquared = _mm_mul_ps(im, im);
        __m128 magSquared = _mm_add_ps(reSquared, imSquared);
        __m128 result = _mm_sqrt_ps(magSquared);

        _mm_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_sse3(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        _mm_store_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_avx(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < eighthPoints; number++) {
        /* Load 8 complex int16 samples as two 128-bit halves.
         * Each 128-bit register holds 4 complex samples packed as int32 per sample. */
        __m128i v0 = _mm_load_si128((const __m128i*)complexVectorPtr);
        __m128i v1 = _mm_load_si128((const __m128i*)(complexVectorPtr + 8));
        complexVectorPtr += 16;

        /* Sign-extend: real = lower 16 bits, imag = upper 16 bits */
        __m128i i0 = _mm_srai_epi32(_mm_slli_epi32(v0, 16), 16);
        __m128i q0 = _mm_srai_epi32(v0, 16);
        __m128i i1 = _mm_srai_epi32(_mm_slli_epi32(v1, 16), 16);
        __m128i q1 = _mm_srai_epi32(v1, 16);

        /* Convert to float and scale; combine pairs into 256-bit vectors */
        __m256 re = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(i1), _mm_cvtepi32_ps(i0)), invScalar);
        __m256 im = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(q1), _mm_cvtepi32_ps(q0)), invScalar);

        /* mag = sqrt(re^2 + im^2) */
        __m256 reSquared = _mm256_mul_ps(re, re);
        __m256 imSquared = _mm256_mul_ps(im, im);
        __m256 magSquared = _mm256_add_ps(reSquared, imSquared);
        __m256 result = _mm256_sqrt_ps(magSquared);

        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_avx_fma(float* magnitudeVector,
                                                            const lv_16sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < eighthPoints; number++) {
        /* Load 8 complex int16 samples as two 128-bit halves. */
        __m128i v0 = _mm_load_si128((const __m128i*)complexVectorPtr);
        __m128i v1 = _mm_load_si128((const __m128i*)(complexVectorPtr + 8));
        complexVectorPtr += 16;

        /* Sign-extend: real = lower 16 bits, imag = upper 16 bits */
        __m128i i0 = _mm_srai_epi32(_mm_slli_epi32(v0, 16), 16);
        __m128i q0 = _mm_srai_epi32(v0, 16);
        __m128i i1 = _mm_srai_epi32(_mm_slli_epi32(v1, 16), 16);
        __m128i q1 = _mm_srai_epi32(v1, 16);

        /* Convert to float and scale; combine pairs into 256-bit vectors */
        __m256 re = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(i1), _mm_cvtepi32_ps(i0)), invScalar);
        __m256 im = _mm256_mul_ps(
            _mm256_set_m128(_mm_cvtepi32_ps(q1), _mm_cvtepi32_ps(q0)), invScalar);

        /* mag = sqrt(re^2 + im^2) using FMA */
        __m256 magSquared = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));
        __m256 result = _mm256_sqrt_ps(magSquared);

        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_avx2(float* magnitudeVector,
                                                       const lv_16sc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    __m256 cplxValue1, cplxValue2, result;
    __m256i int1, int2;
    __m128i short1, short2;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    for (; number < eighthPoints; number++) {

        int1 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        short1 = _mm256_extracti128_si256(int1, 0);
        short2 = _mm256_extracti128_si256(int1, 1);

        int1 = _mm256_cvtepi16_epi32(short1);
        int2 = _mm256_cvtepi16_epi32(short2);
        cplxValue1 = _mm256_cvtepi32_ps(int1);
        cplxValue2 = _mm256_cvtepi32_ps(int2);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values
        result = _mm256_permutevar8x32_ps(result, idx);

        result = _mm256_sqrt_ps(result); // Square root the values

        _mm256_store_ps(magnitudeVectorPtr, result);

        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_16ic_s32f_magnitude_32f_a_avx2_fma(float* magnitudeVector,
                                                            const lv_16sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < eighthPoints; number++) {
        __m256i raw = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        __m128i lo_half = _mm256_extracti128_si256(raw, 0);
        __m128i hi_half = _mm256_extracti128_si256(raw, 1);

        __m256 cplxValue1 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_half)), invScalar);
        __m256 cplxValue2 = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_half)), invScalar);

        __m256 result = _mm256_magnitudesquared_ps_avx2_fma(cplxValue1, cplxValue2);

        result = _mm256_sqrt_ps(result);

        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_s32f_magnitude_32f_a_avx512f(float* magnitudeVector,
                                                           const lv_16sc_t* complexVector,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / scalar;
    __m512 vInvScalar = _mm512_set1_ps(fInvScalar);

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        /* Load 16 complex int16 samples = 512 bits */
        __m512i raw = _mm512_load_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 32;

        /* Split into two 256-bit halves and widen int16 -> int32 */
        __m256i lo_half = _mm512_castsi512_si256(raw);
        __m256i hi_half = _mm512_extracti64x4_epi64(raw, 1);
        __m512i wide_lo = _mm512_cvtepi16_epi32(lo_half);
        __m512i wide_hi = _mm512_cvtepi16_epi32(hi_half);

        /* Convert int32 -> float */
        __m512 flt_lo = _mm512_cvtepi32_ps(wide_lo);
        __m512 flt_hi = _mm512_cvtepi32_ps(wide_hi);

        /* Scale by 1/scalar */
        flt_lo = _mm512_mul_ps(flt_lo, vInvScalar);
        flt_hi = _mm512_mul_ps(flt_hi, vInvScalar);

        /* Deinterleave re/im */
        __m512 re = _mm512_permutex2var_ps(flt_lo, idx_re, flt_hi);
        __m512 im = _mm512_permutex2var_ps(flt_lo, idx_im, flt_hi);

        /* mag = sqrt(re^2 + im^2) */
        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 magSquared = _mm512_add_ps(reSquared, imSquared);
        __m512 result = _mm512_sqrt_ps(magSquared);

        _mm512_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_16ic_s32f_magnitude_32f_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#endif /* INCLUDED_volk_16ic_s32f_magnitude_32f_a_H */
