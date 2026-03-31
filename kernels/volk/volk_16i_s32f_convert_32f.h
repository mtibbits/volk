/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_s32f_convert_32f
 *
 * \b Overview
 *
 * Converts 16-bit integer samples to scaled 32-bit floating point values by
 * dividing each sample by a scalar: out[i] = (float)in[i] / scalar.
 *
 * This kernel is commonly used when ingesting fixed-point ADC samples into a
 * floating-point signal processing pipeline. The scalar divisor normalizes the
 * integer sample range to a desired floating-point amplitude, which is essential
 * for correct operation of downstream blocks such as filters, demodulators, and
 * spectral analysis routines.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_s32f_convert_32f(float* outputVector, const int16_t* inputVector, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 16-bit integer samples (int16_t).
 * \li scalar: The divisor applied to each converted sample (float).
 * \li num_points: The number of samples to convert.
 *
 * \b Outputs
 * \li outputVector: The output vector of scaled 32-bit floating point values (float).
 *
 * \b Example
 * Convert four int16 samples with a scalar divisor of 2 and verify the result.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * int16_t* input = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 * float* output = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *   input[i] = (int16_t)((i + 1) * 100);  // 100, 200, 300, 400
 * }
 * float scalar = 2.0f;
 *
 * // Expected: 100/2=50, 200/2=100, 300/2=150, 400/2=200
 *
 * volk_16i_s32f_convert_32f(output, input, scalar, N);
 *
 * printf("Expected: 50, 100, 150, 200\n");
 * printf("Result:   %1.0f, %1.0f, %1.0f, %1.0f\n",
 *        output[0], output[1], output[2], output[3]);
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_s32f_convert_32f_u_H
#define INCLUDED_volk_16i_s32f_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16i_s32f_convert_32f_generic(float* outputVector,
                                                     const int16_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    const int16_t* inputVectorPtr = inputVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * reciprocal;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16i_s32f_convert_32f_u_sse(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128 ret;

    for (; number < quarterPoints; number++) {
        ret = _mm_set_ps((float)(inputPtr[3]),
                         (float)(inputPtr[2]),
                         (float)(inputPtr[1]),
                         (float)(inputPtr[0]));

        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);

        inputPtr += 4;
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * reciprocal;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_s32f_convert_32f_u_sse2(float* outputVector,
                                                     const int16_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal, sign, lo32, hi32;
    __m128 ret;

    for (; number < eighthPoints; number++) {

        // Load 8 int16 values
        inputVal = _mm_loadu_si128((const __m128i*)inputPtr);

        // Sign-extend int16 to int32 (SSE2 pattern)
        sign = _mm_srai_epi16(inputVal, 15);
        lo32 = _mm_unpacklo_epi16(inputVal, sign);
        hi32 = _mm_unpackhi_epi16(inputVal, sign);

        // Convert int32 to float and scale
        ret = _mm_cvtepi32_ps(lo32);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        ret = _mm_cvtepi32_ps(hi32);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16i_s32f_convert_32f_generic(
        outputVector + number, inputVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_s32f_convert_32f_u_sse4_1(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal;
    __m128i inputVal2;
    __m128 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_loadu_si128((const __m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal, inputVal2;
    __m128 ret;
    __m256 output;
    __m256 dummy = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_loadu_si128((const __m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(dummy, ret, 0);

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(output, ret, 1);

        _mm256_storeu_ps(outputVectorPtr, output);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx2(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m256 invScalar = _mm256_set1_ps(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal;
    __m256i inputVal2;
    __m256 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_loadu_si128((const __m128i*)inputPtr);

        // Convert
        inputVal2 = _mm256_cvtepi16_epi32(inputVal);

        ret = _mm256_cvtepi32_ps(inputVal2);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx2_fma(float* outputVector,
                                                         const int16_t* inputVector,
                                                         const float scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m256 invScalar = _mm256_set1_ps(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal;
    __m256i inputVal2;
    __m256 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_loadu_si128((const __m128i*)inputPtr);

        // Convert int16 → int32 → float, then scale
        inputVal2 = _mm256_cvtepi16_epi32(inputVal);
        ret = _mm256_cvtepi32_ps(inputVal2);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;
        inputPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16i_s32f_convert_32f_generic(
        outputVector + number, inputVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_u_avx512(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m512 invScalar = _mm512_set1_ps(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m256i inputVal;
    __m512i inputVal2;
    __m512 ret;

    for (; number < sixteenthPoints; number++) {

        // Load 16 int16 values
        inputVal = _mm256_loadu_si256((const __m256i*)inputPtr);

        // Convert int16 → int32 → float
        inputVal2 = _mm512_cvtepi16_epi32(inputVal);
        ret = _mm512_cvtepi32_ps(inputVal2);
        ret = _mm512_mul_ps(ret, invScalar);

        _mm512_storeu_ps(outputVectorPtr, ret);

        outputVectorPtr += 16;
        inputPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_s32f_convert_32f_neon(float* outputVector,
                                                  const int16_t* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    float* outputPtr = outputVector;
    const int16_t* inputPtr = inputVector;
    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    int16x4x2_t input16;
    int32x4_t input32_0, input32_1;
    float32x4_t input_float_0, input_float_1;
    float32x4x2_t output_float;
    float32x4_t inv_scale;

    inv_scale = vdupq_n_f32(reciprocal);

    // the generic disassembles to a 128-bit load
    // and duplicates every instruction to operate on 64-bits
    // at a time. This is only possible with lanes, which is faster
    // than just doing a vld1_s16, but still slower.
    for (number = 0; number < eighth_points; number++) {
        input16 = vld2_s16(inputPtr);
        // widen 16-bit int to 32-bit int
        input32_0 = vmovl_s16(input16.val[0]);
        input32_1 = vmovl_s16(input16.val[1]);
        // convert 32-bit int to float with scale
        input_float_0 = vcvtq_f32_s32(input32_0);
        input_float_1 = vcvtq_f32_s32(input32_1);
        output_float.val[0] = vmulq_f32(input_float_0, inv_scale);
        output_float.val[1] = vmulq_f32(input_float_1, inv_scale);
        vst2q_f32(outputPtr, output_float);
        inputPtr += 8;
        outputPtr += 8;
    }

    for (number = eighth_points * 8; number < num_points; number++) {
        *outputPtr++ = ((float)(*inputPtr++)) * reciprocal;
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16i_s32f_convert_32f_neonv8(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int n = num_points;
    float* out = outputVector;
    const int16_t* in = inputVector;

    const float reciprocal = 1.0f / scalar;
    const float32x4_t inv_scale = vdupq_n_f32(reciprocal);

    /* Process 8 int16 values per iteration using 64-bit loads */
    while (n >= 8) {
        int16x4_t v0 = vld1_s16(in);
        int16x4_t v1 = vld1_s16(in + 4);
        __VOLK_PREFETCH(in + 16);

        /* Widen int16 to int32, convert to float, scale */
        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(v0)), inv_scale);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(v1)), inv_scale);

        vst1q_f32(out, f0);
        vst1q_f32(out + 4, f1);

        in += 8;
        out += 8;
        n -= 8;
    }

    /* Process remaining 4 values */
    if (n >= 4) {
        int16x4_t v0 = vld1_s16(in);
        vst1q_f32(out, vmulq_f32(vcvtq_f32_s32(vmovl_s16(v0)), inv_scale));
        in += 4;
        out += 4;
        n -= 4;
    }

    /* Scalar tail */
    while (n > 0) {
        *out++ = ((float)(*in++)) * reciprocal;
        n--;
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_s32f_convert_32f_rvv(float* outputVector,
                                                 const int16_t* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vfloat32m8_t v = __riscv_vfwcvt_f(__riscv_vle16_v_i16m4(inputVector, vl), vl);
        __riscv_vse32(outputVector, __riscv_vfmul(v, 1.0f / scalar, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_ORC

extern void volk_16i_s32f_convert_32f_a_orc_impl(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   int num_points);

static inline void volk_16i_s32f_convert_32f_u_orc(float* outputVector,
                                                     const int16_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    float invScalar = 1.0f / scalar;
    volk_16i_s32f_convert_32f_a_orc_impl(outputVector, inputVector, invScalar, num_points);
}

#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_16i_s32f_convert_32f_u_H */
#ifndef INCLUDED_volk_16i_s32f_convert_32f_a_H
#define INCLUDED_volk_16i_s32f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16i_s32f_convert_32f_a_sse(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128 ret;

    for (; number < quarterPoints; number++) {
        ret = _mm_set_ps((float)(inputPtr[3]),
                         (float)(inputPtr[2]),
                         (float)(inputPtr[1]),
                         (float)(inputPtr[0]));

        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);

        inputPtr += 4;
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        outputVector[number] = (float)(inputVector[number]) * reciprocal;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_s32f_convert_32f_a_sse2(float* outputVector,
                                                     const int16_t* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal, sign, lo32, hi32;
    __m128 ret;

    for (; number < eighthPoints; number++) {

        // Load 8 int16 values (aligned)
        inputVal = _mm_load_si128((const __m128i*)inputPtr);

        // Sign-extend int16 to int32 (SSE2 pattern)
        sign = _mm_srai_epi16(inputVal, 15);
        lo32 = _mm_unpacklo_epi16(inputVal, sign);
        hi32 = _mm_unpackhi_epi16(inputVal, sign);

        // Convert int32 to float and scale
        ret = _mm_cvtepi32_ps(lo32);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        ret = _mm_cvtepi32_ps(hi32);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16i_s32f_convert_32f_generic(
        outputVector + number, inputVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_16i_s32f_convert_32f_a_sse4_1(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal;
    __m128i inputVal2;
    __m128 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_load_si128((const __m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);
        outputVectorPtr += 4;

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        _mm_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 4;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx(float* outputVector,
                                                   const int16_t* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m128 invScalar = _mm_set_ps1(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal, inputVal2;
    __m128 ret;
    __m256 output;
    __m256 dummy = _mm256_setzero_ps();

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_load_si128((const __m128i*)inputPtr);

        // Shift the input data to the right by 64 bits ( 8 bytes )
        inputVal2 = _mm_srli_si128(inputVal, 8);

        // Convert the lower 4 values into 32 bit words
        inputVal = _mm_cvtepi16_epi32(inputVal);
        inputVal2 = _mm_cvtepi16_epi32(inputVal2);

        ret = _mm_cvtepi32_ps(inputVal);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(dummy, ret, 0);

        ret = _mm_cvtepi32_ps(inputVal2);
        ret = _mm_mul_ps(ret, invScalar);
        output = _mm256_insertf128_ps(output, ret, 1);

        _mm256_store_ps(outputVectorPtr, output);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx2(float* outputVector,
                                                    const int16_t* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m256 invScalar = _mm256_set1_ps(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal;
    __m256i inputVal2;
    __m256 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values
        inputVal = _mm_load_si128((const __m128i*)inputPtr);

        // Convert
        inputVal2 = _mm256_cvtepi16_epi32(inputVal);

        ret = _mm256_cvtepi32_ps(inputVal2);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;

        inputPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx2_fma(float* outputVector,
                                                         const int16_t* inputVector,
                                                         const float scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m256 invScalar = _mm256_set1_ps(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m128i inputVal;
    __m256i inputVal2;
    __m256 ret;

    for (; number < eighthPoints; number++) {

        // Load the 8 values (aligned)
        inputVal = _mm_load_si128((const __m128i*)inputPtr);

        // Convert int16 → int32 → float, then scale
        inputVal2 = _mm256_cvtepi16_epi32(inputVal);
        ret = _mm256_cvtepi32_ps(inputVal2);
        ret = _mm256_mul_ps(ret, invScalar);

        _mm256_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 8;
        inputPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16i_s32f_convert_32f_generic(
        outputVector + number, inputVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16i_s32f_convert_32f_a_avx512(float* outputVector,
                                                      const int16_t* inputVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float reciprocal = 1.0f / scalar;
    float* outputVectorPtr = outputVector;
    __m512 invScalar = _mm512_set1_ps(reciprocal);
    const int16_t* inputPtr = inputVector;
    __m256i inputVal;
    __m512i inputVal2;
    __m512 ret;

    for (; number < sixteenthPoints; number++) {

        // Load 16 int16 values
        inputVal = _mm256_load_si256((const __m256i*)inputPtr);

        // Convert int16 → int32 → float
        inputVal2 = _mm512_cvtepi16_epi32(inputVal);
        ret = _mm512_cvtepi32_ps(inputVal2);
        ret = _mm512_mul_ps(ret, invScalar);

        _mm512_store_ps(outputVectorPtr, ret);

        outputVectorPtr += 16;
        inputPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = ((float)(inputVector[number])) * reciprocal;
    }
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_16i_s32f_convert_32f_a_H */
