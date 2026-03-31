/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8i_convert_16i
 *
 * \b Overview
 *
 * Convert 8-bit signed integer samples to 16-bit signed integers with scaling.
 * Each input value is multiplied by 256 (left-shifted by 8 bits), mapping the
 * full 8-bit range into the upper byte of the 16-bit output:
 * out[i] = (int16_t)in[i] * 256.
 *
 * This kernel is commonly used in SDR receive pipelines where digitized samples
 * arrive as 8-bit I/Q values (e.g. from an RTL-SDR or similar front-end) and
 * must be promoted to 16-bit precision for downstream processing such as
 * filtering, mixing, or demodulation while preserving the full dynamic range.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8i_convert_16i(int16_t* outputVector, const int8_t* inputVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 8-bit signed integer samples (int8_t).
 * \li num_points: The number of samples to convert.
 *
 * \b Outputs
 * \li outputVector: The output vector of scaled 16-bit signed integers (int16_t).
 *
 * \b Example
 * Convert a small array of 8-bit values and verify the scaling.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * int8_t* input = (int8_t*)volk_malloc(sizeof(int8_t) * N, alignment);
 * int16_t* output = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 *
 * input[0] = 1;
 * input[1] = -1;
 * input[2] = 127;
 * input[3] = -128;
 *
 * volk_8i_convert_16i(output, input, N);
 *
 * // Expected: each value * 256 -> 256, -256, 32512, -32768
 * printf("Expected: 256, -256, 32512, -32768\n");
 * printf("Result:   %d, %d, %d, %d\n", output[0], output[1], output[2], output[3]);
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */

#ifndef INCLUDED_volk_8i_convert_16i_u_H
#define INCLUDED_volk_8i_convert_16i_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_convert_16i_u_avx2(int16_t* outputVector,
                                              const int8_t* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m128i* inputVectorPtr = (const __m128i*)inputVector;
    __m256i* outputVectorPtr = (__m256i*)outputVector;
    __m128i inputVal;
    __m256i ret;

    for (; number < sixteenthPoints; number++) {
        inputVal = _mm_loadu_si128(inputVectorPtr);
        ret = _mm256_cvtepi8_epi16(inputVal);
        ret = _mm256_slli_epi16(ret, 8); // Multiply by 256
        _mm256_storeu_si256(outputVectorPtr, ret);

        outputVectorPtr++;
        inputVectorPtr++;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (int16_t)(inputVector[number]) * 256;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8i_convert_16i_u_avx512bw(int16_t* outputVector,
                                                  const int8_t* inputVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    const __m256i* inputVectorPtr = (const __m256i*)inputVector;
    __m512i* outputVectorPtr = (__m512i*)outputVector;
    __m256i inputVal;
    __m512i ret;

    for (; number < thirtysecondPoints; number++) {
        inputVal = _mm256_loadu_si256(inputVectorPtr);
        ret = _mm512_cvtepi8_epi16(inputVal);
        ret = _mm512_slli_epi16(ret, 8); // Multiply by 256
        _mm512_storeu_si512(outputVectorPtr, ret);

        outputVectorPtr++;
        inputVectorPtr++;
    }

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        outputVector[number] = (int16_t)(inputVector[number]) * 256;
    }
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_convert_16i_u_sse4_1(int16_t* outputVector,
                                                const int8_t* inputVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m128i* inputVectorPtr = (const __m128i*)inputVector;
    __m128i* outputVectorPtr = (__m128i*)outputVector;
    __m128i inputVal;
    __m128i ret;

    for (; number < sixteenthPoints; number++) {
        inputVal = _mm_loadu_si128(inputVectorPtr);
        ret = _mm_cvtepi8_epi16(inputVal);
        ret = _mm_slli_epi16(ret, 8); // Multiply by 256
        _mm_storeu_si128(outputVectorPtr, ret);

        outputVectorPtr++;

        inputVal = _mm_srli_si128(inputVal, 8);
        ret = _mm_cvtepi8_epi16(inputVal);
        ret = _mm_slli_epi16(ret, 8); // Multiply by 256
        _mm_storeu_si128(outputVectorPtr, ret);

        outputVectorPtr++;

        inputVectorPtr++;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (int16_t)(inputVector[number]) * 256;
    }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_GENERIC

static inline void volk_8i_convert_16i_generic(int16_t* outputVector,
                                               const int8_t* inputVector,
                                               unsigned int num_points)
{
    int16_t* outputVectorPtr = outputVector;
    const int8_t* inputVectorPtr = inputVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *outputVectorPtr++ = ((int16_t)(*inputVectorPtr++)) * 256;
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_VOLK_8s_CONVERT_16s_UNALIGNED8_H */


#ifndef INCLUDED_volk_8i_convert_16i_a_H
#define INCLUDED_volk_8i_convert_16i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8i_convert_16i_a_avx2(int16_t* outputVector,
                                              const int8_t* inputVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m128i* inputVectorPtr = (const __m128i*)inputVector;
    __m256i* outputVectorPtr = (__m256i*)outputVector;
    __m128i inputVal;
    __m256i ret;

    for (; number < sixteenthPoints; number++) {
        inputVal = _mm_load_si128(inputVectorPtr);
        ret = _mm256_cvtepi8_epi16(inputVal);
        ret = _mm256_slli_epi16(ret, 8); // Multiply by 256
        _mm256_store_si256(outputVectorPtr, ret);

        outputVectorPtr++;
        inputVectorPtr++;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (int16_t)(inputVector[number]) * 256;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8i_convert_16i_a_avx512bw(int16_t* outputVector,
                                                  const int8_t* inputVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    const __m256i* inputVectorPtr = (const __m256i*)inputVector;
    __m512i* outputVectorPtr = (__m512i*)outputVector;
    __m256i inputVal;
    __m512i ret;

    for (; number < thirtysecondPoints; number++) {
        inputVal = _mm256_load_si256(inputVectorPtr);
        ret = _mm512_cvtepi8_epi16(inputVal);
        ret = _mm512_slli_epi16(ret, 8); // Multiply by 256
        _mm512_store_si512(outputVectorPtr, ret);

        outputVectorPtr++;
        inputVectorPtr++;
    }

    number = thirtysecondPoints * 32;
    for (; number < num_points; number++) {
        outputVector[number] = (int16_t)(inputVector[number]) * 256;
    }
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8i_convert_16i_a_sse4_1(int16_t* outputVector,
                                                const int8_t* inputVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const __m128i* inputVectorPtr = (const __m128i*)inputVector;
    __m128i* outputVectorPtr = (__m128i*)outputVector;
    __m128i inputVal;
    __m128i ret;

    for (; number < sixteenthPoints; number++) {
        inputVal = _mm_load_si128(inputVectorPtr);
        ret = _mm_cvtepi8_epi16(inputVal);
        ret = _mm_slli_epi16(ret, 8); // Multiply by 256
        _mm_store_si128(outputVectorPtr, ret);

        outputVectorPtr++;

        inputVal = _mm_srli_si128(inputVal, 8);
        ret = _mm_cvtepi8_epi16(inputVal);
        ret = _mm_slli_epi16(ret, 8); // Multiply by 256
        _mm_store_si128(outputVectorPtr, ret);

        outputVectorPtr++;

        inputVectorPtr++;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        outputVector[number] = (int16_t)(inputVector[number]) * 256;
    }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8i_convert_16i_neon(int16_t* outputVector,
                                            const int8_t* inputVector,
                                            unsigned int num_points)
{
    int16_t* outputVectorPtr = outputVector;
    const int8_t* inputVectorPtr = inputVector;
    unsigned int number;
    const unsigned int eighth_points = num_points / 8;

    int8x8_t input_vec;
    int16x8_t converted_vec;

    // NEON doesn't have a concept of 8 bit registers, so we are really
    // dealing with the low half of 16-bit registers. Since this requires
    // a move instruction we likely do better with ASM here.
    for (number = 0; number < eighth_points; ++number) {
        input_vec = vld1_s8(inputVectorPtr);
        converted_vec = vmovl_s8(input_vec);
        // converted_vec = vmulq_s16(converted_vec, scale_factor);
        converted_vec = vshlq_n_s16(converted_vec, 8);
        vst1q_s16(outputVectorPtr, converted_vec);

        inputVectorPtr += 8;
        outputVectorPtr += 8;
    }

    for (number = eighth_points * 8; number < num_points; number++) {
        *outputVectorPtr++ = ((int16_t)(*inputVectorPtr++)) * 256;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_8i_convert_16i_neonv8(int16_t* outputVector,
                                              const int8_t* inputVector,
                                              unsigned int num_points)
{
    int16_t* outputVectorPtr = outputVector;
    const int8_t* inputVectorPtr = inputVector;
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        int8x16_t in = vld1q_s8(inputVectorPtr);
        __VOLK_PREFETCH(inputVectorPtr + 32);

        int16x8_t out_lo = vshll_n_s8(vget_low_s8(in), 8);
        int16x8_t out_hi = vshll_n_s8(vget_high_s8(in), 8);

        vst1q_s16(outputVectorPtr, out_lo);
        vst1q_s16(outputVectorPtr + 8, out_hi);

        inputVectorPtr += 16;
        outputVectorPtr += 16;
    }

    for (unsigned int number = sixteenthPoints * 16; number < num_points; number++) {
        *outputVectorPtr++ = ((int16_t)(*inputVectorPtr++)) * 256;
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_ORC
extern void volk_8i_convert_16i_a_orc_impl(int16_t* outputVector,
                                           const int8_t* inputVector,
                                           int num_points);

static inline void volk_8i_convert_16i_u_orc(int16_t* outputVector,
                                             const int8_t* inputVector,
                                             unsigned int num_points)
{
    volk_8i_convert_16i_a_orc_impl(outputVector, inputVector, num_points);
}
#endif /* LV_HAVE_ORC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8i_convert_16i_rvv(int16_t* outputVector,
                                           const int8_t* inputVector,
                                           unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e8m4(n);
        vint16m8_t v = __riscv_vsext_vf2(__riscv_vle8_v_i8m4(inputVector, vl), vl);
        __riscv_vse16(outputVector, __riscv_vsll(v, 8, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_VOLK_8s_CONVERT_16s_ALIGNED8_H */
