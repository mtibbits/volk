/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8ic_deinterleave_real_16i
 *
 * \b Overview
 *
 * Deinterleaves the complex 8-bit char vector into just the real (I)
 * component and converts each sample to a 16-bit short by multiplying
 * by 128. The imaginary (Q) component is discarded.
 *
 * This kernel is useful in receiver front-ends where baseband I/Q samples
 * arrive as interleaved 8-bit pairs and only the in-phase channel is needed
 * for further processing such as real-valued demodulation or power detection.
 * The 128x scaling maps the 8-bit dynamic range into the upper bits of a
 * 16-bit word, preserving resolution for downstream DSP stages.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8ic_deinterleave_real_16i(int16_t* iBuffer, const lv_8sc_t* complexVector,
 * unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of interleaved I/Q samples (lv_8sc_t).
 * \li num_points: The number of complex samples to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The deinterleaved real (I) output samples scaled by 128 (int16_t).
 *
 * \b Example
 * Deinterleave 4 complex 8-bit samples and verify the real output is scaled by 128.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_8sc_t* complexVector = (lv_8sc_t*)volk_malloc(sizeof(lv_8sc_t) * N, alignment);
 * int16_t* iBuffer = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *   complexVector[i] = lv_cmake((int8_t)(i + 1), (int8_t)(-(i + 1)));
 * }
 *
 * // Expected: real part * 128, so first element = 1 * 128 = 128
 * int16_t expected = 1 * 128;
 *
 * volk_8ic_deinterleave_real_16i(iBuffer, complexVector, N);
 *
 * printf("Expected: %d\n", expected);
 * printf("Result:   %d\n", iBuffer[0]);
 *
 * volk_free(complexVector);
 * volk_free(iBuffer);
 * \endcode
 */

#ifndef INCLUDED_volk_8ic_deinterleave_real_16i_a_H
#define INCLUDED_volk_8ic_deinterleave_real_16i_a_H

#include <inttypes.h>
#include <stdio.h>


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_avx2(int16_t* iBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m256i moveMask = _mm256_set_epi8(0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0);
    __m256i complexVal, outputVal;
    __m128i outputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);
        complexVal = _mm256_permute4x64_epi64(complexVal, 0xd8);

        outputVal0 = _mm256_extractf128_si256(complexVal, 0);

        outputVal = _mm256_cvtepi8_epi16(outputVal0);
        outputVal = _mm256_slli_epi16(outputVal, 7);

        _mm256_store_si256((__m256i*)iBufferPtr, outputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int16_t)*complexVectorPtr++) * 128;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_sse4_1(int16_t* iBuffer,
                                                           const lv_8sc_t* complexVector,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i complexVal, outputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal = _mm_shuffle_epi8(complexVal, moveMask);

        outputVal = _mm_cvtepi8_epi16(complexVal);
        outputVal = _mm_slli_epi16(outputVal, 7);

        _mm_store_si128((__m128i*)iBufferPtr, outputVal);
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int16_t)*complexVectorPtr++) * 128;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_avx(int16_t* iBuffer,
                                                        const lv_8sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i complexVal, outputVal;
    __m128i complexVal1, complexVal0, outputVal1, outputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal1 = _mm256_extractf128_si256(complexVal, 1);
        complexVal0 = _mm256_extractf128_si256(complexVal, 0);

        outputVal1 = _mm_shuffle_epi8(complexVal1, moveMask);
        outputVal0 = _mm_shuffle_epi8(complexVal0, moveMask);

        outputVal1 = _mm_cvtepi8_epi16(outputVal1);
        outputVal1 = _mm_slli_epi16(outputVal1, 7);
        outputVal0 = _mm_cvtepi8_epi16(outputVal0);
        outputVal0 = _mm_slli_epi16(outputVal0, 7);

        __m256i dummy = _mm256_setzero_si256();
        outputVal = _mm256_insertf128_si256(dummy, outputVal0, 0);
        outputVal = _mm256_insertf128_si256(outputVal, outputVal1, 1);
        _mm256_store_si256((__m256i*)iBufferPtr, outputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int16_t)*complexVectorPtr++) * 128;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_GENERIC

static inline void volk_8ic_deinterleave_real_16i_generic(int16_t* iBuffer,
                                                          const lv_8sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = ((int16_t)(*complexVectorPtr++)) * 128;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_8ic_deinterleave_real_16i_a_H */

#ifndef INCLUDED_volk_8ic_deinterleave_real_16i_u_H
#define INCLUDED_volk_8ic_deinterleave_real_16i_u_H

#include <inttypes.h>
#include <stdio.h>


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_avx2(int16_t* iBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m256i moveMask = _mm256_set_epi8(0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       0x80,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0);
    __m256i complexVal, outputVal;
    __m128i outputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);
        complexVal = _mm256_permute4x64_epi64(complexVal, 0xd8);

        outputVal0 = _mm256_extractf128_si256(complexVal, 0);

        outputVal = _mm256_cvtepi8_epi16(outputVal0);
        outputVal = _mm256_slli_epi16(outputVal, 7);

        _mm256_storeu_si256((__m256i*)iBufferPtr, outputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int16_t)*complexVectorPtr++) * 128;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8ic_deinterleave_real_16i_neon(int16_t* iBuffer,
                                                       const lv_8sc_t* complexVector,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    for (; number < eighth_points; number++) {
        int8x8x2_t input = vld2_s8(complexVectorPtr);
        complexVectorPtr += 16;

        int16x8_t iVal = vshll_n_s8(input.val[0], 7);

        vst1q_s16(iBufferPtr, iVal);
        iBufferPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int16_t)*complexVectorPtr++) * 128;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8ic_deinterleave_real_16i_rvv(int16_t* iBuffer,
                                                      const lv_8sc_t* complexVector,
                                                      unsigned int num_points)
{
    const int16_t* in = (const int16_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e16m8(n);
        vint16m8_t v = __riscv_vle16_v_i16m8(in, vl);
        __riscv_vse16(iBuffer, __riscv_vsra(__riscv_vsll(v, 8, vl), 1, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_8ic_deinterleave_real_16i_u_H */
