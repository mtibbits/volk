/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32i_s32f_convert_32f
 *
 * \b Overview
 *
 * Converts samples from 32-bit integers to floating point values and divides
 * each by a scalar: out[i] = (float)in[i] / scalar. This operation is commonly
 * used in SDR receivers to normalize fixed-point ADC samples into a
 * floating-point range suitable for downstream DSP, such as filtering,
 * demodulation, or spectral analysis. The scalar typically represents the
 * full-scale value of the converter.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32i_s32f_convert_32f(float* outputVector, const int32_t* inputVector, const
 * float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 32-bit integer samples (int32_t).
 * \li scalar: The divisor applied after converting each sample to float (float).
 * \li num_points: The number of samples to convert.
 *
 * \b Outputs
 * \li outputVector: The output vector of normalized float samples (float).
 *
 * \b Example
 * Convert integers to floats scaled by 1/100.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * int32_t* in = (int32_t*)volk_malloc(sizeof(int32_t) * N, alignment);
 * float* out = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * in[0] = 100; in[1] = 200; in[2] = -300; in[3] = 400;
 * float scalar = 100.0f;
 *
 * // Expected: 100/100=1.0, 200/100=2.0, -300/100=-3.0, 400/100=4.0
 * float expected = 1.0f + 2.0f + (-3.0f) + 4.0f; // sum = 4.0
 *
 * volk_32i_s32f_convert_32f(out, in, scalar, N);
 *
 * float actual = out[0] + out[1] + out[2] + out[3];
 * printf("Expected sum: %f\n", expected);
 * printf("Actual sum:   %f\n", actual);
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32i_s32f_convert_32f_u_H
#define INCLUDED_volk_32i_s32f_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_u_avx512f(float* outputVector,
                                                       const int32_t* inputVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int onesixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    int32_t* inputPtr = (int32_t*)inputVector;
    __m512i inputVal;
    __m512 ret;

    for (; number < onesixteenthPoints; number++) {
        // Load the values
        inputVal = _mm512_loadu_si512((__m512i*)inputPtr);

        ret = _mm512_cvtepi32_ps(inputVal);
        ret = _mm512_mul_ps(ret, invScalar);

        _mm512_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 16;
        inputPtr += 16;
    }

    number = onesixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_u_avx2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEightPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    int32_t* inputPtr = (int32_t*)inputVector;
    __m256i inputVal;
    __m256 ret;

    for (; number < oneEightPoints; number++) {
        // Load the 4 values
        inputVal = _mm256_loadu_si256((__m256i*)inputPtr);

        ret = _mm256_cvtepi32_ps(inputVal);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;
        inputPtr += 8;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32i_s32f_convert_32f_u_sse2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    int32_t* inputPtr = (int32_t*)inputVector;
    __m128i inputVal;
    __m128 ret;

    for (; number < quarterPoints; number++) {
        // Load the 4 values
        inputVal = _mm_loadu_si128((__m128i*)inputPtr);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);

        _mm_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;
        inputPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_32i_s32f_convert_32f_generic(float* outputVector,
                                                     const int32_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    float* outputVectorPtr = outputVector;
    const int32_t* inputVectorPtr = inputVector;
    unsigned int number = 0;
    const float iScalar = 1.0 / scalar;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
    }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32i_s32f_convert_32f_u_H */


#ifndef INCLUDED_volk_32i_s32f_convert_32f_a_H
#define INCLUDED_volk_32i_s32f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_a_avx512f(float* outputVector,
                                                       const int32_t* inputVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int onesixteenthPoints = num_points / 16;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    int32_t* inputPtr = (int32_t*)inputVector;
    __m512i inputVal;
    __m512 ret;

    for (; number < onesixteenthPoints; number++) {
        // Load the values
        inputVal = _mm512_load_si512((__m512i*)inputPtr);

        ret = _mm512_cvtepi32_ps(inputVal);
        ret = _mm512_mul_ps(ret, invScalar);

        _mm512_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 16;
        inputPtr += 16;
    }

    number = onesixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_s32f_convert_32f_a_avx2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEightPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    int32_t* inputPtr = (int32_t*)inputVector;
    __m256i inputVal;
    __m256 ret;

    for (; number < oneEightPoints; number++) {
        // Load the 4 values
        inputVal = _mm256_load_si256((__m256i*)inputPtr);

        ret = _mm256_cvtepi32_ps(inputVal);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;
        inputPtr += 8;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32i_s32f_convert_32f_a_sse2(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* outputVectorPtr = outputVector;
    const float iScalar = 1.0 / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    int32_t* inputPtr = (int32_t*)inputVector;
    __m128i inputVal;
    __m128 ret;

    for (; number < quarterPoints; number++) {
        // Load the 4 values
        inputVal = _mm_load_si128((__m128i*)inputPtr);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);

        _mm_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;
        inputPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32i_s32f_convert_32f_neon(float* outputVector,
                                                  const int32_t* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* outputVectorPtr = outputVector;
    const int32_t* inputPtr = inputVector;
    const float iScalar = 1.0f / scalar;
    float32x4_t invScalar = vdupq_n_f32(iScalar);

    for (; number < quarterPoints; number++) {
        int32x4_t inputVal = vld1q_s32(inputPtr);
        float32x4_t ret = vcvtq_f32_s32(inputVal);
        ret = vmulq_f32(ret, invScalar);
        vst1q_f32(outputVectorPtr, ret);

        inputPtr += 4;
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32i_s32f_convert_32f_neonv8(float* outputVector,
                                                    const int32_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* outputVectorPtr = outputVector;
    const int32_t* inputPtr = inputVector;
    const float iScalar = 1.0f / scalar;
    float32x4_t invScalar = vdupq_n_f32(iScalar);

    for (; number < eighthPoints; number++) {
        int32x4_t inputVal0 = vld1q_s32(inputPtr);
        int32x4_t inputVal1 = vld1q_s32(inputPtr + 4);
        __VOLK_PREFETCH(inputPtr + 8);
        inputPtr += 8;

        float32x4_t ret0 = vcvtq_f32_s32(inputVal0);
        float32x4_t ret1 = vcvtq_f32_s32(inputVal1);

        ret0 = vmulq_f32(ret0, invScalar);
        ret1 = vmulq_f32(ret1, invScalar);

        vst1q_f32(outputVectorPtr, ret0);
        vst1q_f32(outputVectorPtr + 4, ret1);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * iScalar;
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32i_s32f_convert_32f_rvv(float* outputVector,
                                                 const int32_t* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vfcvt_f(__riscv_vle32_v_i32m8(inputVector, vl), vl);
        __riscv_vse32(outputVector, __riscv_vfmul(v, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32i_s32f_convert_32f_a_H */
