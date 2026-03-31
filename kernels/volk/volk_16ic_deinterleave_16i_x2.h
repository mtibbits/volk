/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_deinterleave_16i_x2
 *
 * \b Overview
 *
 * Deinterleaves a complex 16-bit integer vector into separate in-phase (I) and
 * quadrature (Q) sample streams. Each interleaved lv_16sc_t element (real, imag)
 * is split so that iBuffer receives the real components and qBuffer receives the
 * imaginary components.
 *
 * Deinterleaving is a fundamental operation in digital radio receivers where an
 * ADC or front-end delivers interleaved I/Q sample pairs. Separating the
 * components enables independent processing such as matched filtering, carrier
 * recovery, or IQ imbalance correction on each branch.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_deinterleave_16i_x2(int16_t* iBuffer, int16_t* qBuffer, const lv_16sc_t* complexVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The interleaved complex input samples (lv_16sc_t).
 * \li num_points: The number of complex samples to deinterleave.
 *
 * \b Outputs
 * \li iBuffer: The in-phase (real) output samples (int16_t).
 * \li qBuffer: The quadrature (imaginary) output samples (int16_t).
 *
 * \b Example
 * Deinterleave 4 complex samples and verify the first I and Q values.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_16sc_t* complexVector = (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t) * N, alignment);
 * int16_t* iBuffer = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 * int16_t* qBuffer = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *   complexVector[i] = (lv_16sc_t){ (int16_t)(i + 1), (int16_t)(-(i + 1)) };
 * }
 * // complexVector = [(1,-1), (2,-2), (3,-3), (4,-4)]
 *
 * volk_16ic_deinterleave_16i_x2(iBuffer, qBuffer, complexVector, N);
 *
 * printf("Expected I[0]: 1, Q[0]: -1\n");
 * printf("Result   I[0]: %d, Q[0]: %d\n", iBuffer[0], qBuffer[0]);
 *
 * volk_free(complexVector);
 * volk_free(iBuffer);
 * volk_free(qBuffer);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_deinterleave_16i_x2_u_H
#define INCLUDED_volk_16ic_deinterleave_16i_x2_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_deinterleave_16i_x2_generic(int16_t* iBuffer,
                                                         int16_t* qBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    unsigned int number;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_sse2(int16_t* iBuffer,
                                                         int16_t* qBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    __m128i complexVal1, complexVal2, iComplexVal1, iComplexVal2, qComplexVal1,
        qComplexVal2, iOutputVal, qOutputVal;
    __m128i lowMask = _mm_set_epi32(0x0, 0x0, -1, -1);
    __m128i highMask = _mm_set_epi32(-1, -1, 0x0, 0x0);

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;
        complexVal2 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        iComplexVal1 = _mm_shufflelo_epi16(complexVal1, _MM_SHUFFLE(3, 1, 2, 0));
        iComplexVal1 = _mm_shufflehi_epi16(iComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));
        iComplexVal1 = _mm_shuffle_epi32(iComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal2 = _mm_shufflelo_epi16(complexVal2, _MM_SHUFFLE(3, 1, 2, 0));
        iComplexVal2 = _mm_shufflehi_epi16(iComplexVal2, _MM_SHUFFLE(3, 1, 2, 0));
        iComplexVal2 = _mm_shuffle_epi32(iComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        iOutputVal = _mm_or_si128(_mm_and_si128(iComplexVal1, lowMask),
                                  _mm_and_si128(iComplexVal2, highMask));
        _mm_storeu_si128((__m128i*)iBufferPtr, iOutputVal);

        qComplexVal1 = _mm_shufflelo_epi16(complexVal1, _MM_SHUFFLE(2, 0, 3, 1));
        qComplexVal1 = _mm_shufflehi_epi16(qComplexVal1, _MM_SHUFFLE(2, 0, 3, 1));
        qComplexVal1 = _mm_shuffle_epi32(qComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        qComplexVal2 = _mm_shufflelo_epi16(complexVal2, _MM_SHUFFLE(2, 0, 3, 1));
        qComplexVal2 = _mm_shufflehi_epi16(qComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));
        qComplexVal2 = _mm_shuffle_epi32(qComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        qOutputVal = _mm_or_si128(_mm_and_si128(qComplexVal1, lowMask),
                                  _mm_and_si128(qComplexVal2, highMask));
        _mm_storeu_si128((__m128i*)qBufferPtr, qOutputVal);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_ssse3(int16_t* iBuffer,
                                                          int16_t* qBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    __m128i iMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);
    __m128i iMoveMask2 = _mm_set_epi8(
        13, 12, 9, 8, 5, 4, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    __m128i qMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 15, 14, 11, 10, 7, 6, 3, 2);
    __m128i qMoveMask2 = _mm_set_epi8(
        15, 14, 11, 10, 7, 6, 3, 2, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    __m128i complexVal1, complexVal2, iOutputVal, qOutputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        iOutputVal = _mm_or_si128(_mm_shuffle_epi8(complexVal1, iMoveMask1),
                                  _mm_shuffle_epi8(complexVal2, iMoveMask2));
        qOutputVal = _mm_or_si128(_mm_shuffle_epi8(complexVal1, qMoveMask1),
                                  _mm_shuffle_epi8(complexVal2, qMoveMask2));

        _mm_storeu_si128((__m128i*)iBufferPtr, iOutputVal);
        _mm_storeu_si128((__m128i*)qBufferPtr, qOutputVal);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_avx2(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    __m256i MoveMask = _mm256_set_epi8(15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0,
                                       15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0);

    __m256i iMove2, iMove1;
    __m256i complexVal1, complexVal2, iOutputVal, qOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        iMove2 = _mm256_shuffle_epi8(complexVal2, MoveMask);
        iMove1 = _mm256_shuffle_epi8(complexVal1, MoveMask);

        iOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x08),
                                               _mm256_permute4x64_epi64(iMove2, 0x80),
                                               0x30);
        qOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x0d),
                                               _mm256_permute4x64_epi64(iMove2, 0xd0),
                                               0x30);

        _mm256_storeu_si256((__m256i*)iBufferPtr, iOutputVal);
        _mm256_storeu_si256((__m256i*)qBufferPtr, qOutputVal);

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    const int16_t* int16ComplexVectorPtr = (const int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *int16ComplexVectorPtr++;
        *qBufferPtr++ = *int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_avx512f(
    int16_t* iBuffer,
    int16_t* qBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    const __m512i lowMask = _mm512_set1_epi32(0x0000FFFF);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal0 = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        __m512i complexVal1 =
            _mm512_loadu_si512((const __m512i*)(complexVectorPtr + 32));
        complexVectorPtr += 64;

        __m256i iLo = _mm512_cvtepi32_epi16(_mm512_and_epi32(complexVal0, lowMask));
        __m256i iHi = _mm512_cvtepi32_epi16(_mm512_and_epi32(complexVal1, lowMask));
        __m256i qLo = _mm512_cvtepi32_epi16(_mm512_srli_epi32(complexVal0, 16));
        __m256i qHi = _mm512_cvtepi32_epi16(_mm512_srli_epi32(complexVal1, 16));

        __m512i iOut = _mm512_inserti64x4(_mm512_castsi256_si512(iLo), iHi, 1);
        __m512i qOut = _mm512_inserti64x4(_mm512_castsi256_si512(qLo), qHi, 1);

        _mm512_storeu_si512((__m512i*)iBufferPtr, iOut);
        _mm512_storeu_si512((__m512i*)qBufferPtr, qOut);
        iBufferPtr += 32;
        qBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_avx512bw(
    int16_t* iBuffer,
    int16_t* qBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    /* Shuffle mask: within each 128-bit lane, group I bytes (0,1,4,5,8,9,12,13)
     * into the low 8 bytes and Q bytes (2,3,6,7,10,11,14,15) into the high 8 bytes */
    const __m512i moveMask = _mm512_set_epi8(
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);

    /* After shuffle, each 128-bit lane has [I0-3 | Q0-3] as 64-bit qwords.
     * Gather even qwords (I) and odd qwords (Q) across both registers. */
    const __m512i iIdx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i qIdx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal0 = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        __m512i complexVal1 =
            _mm512_loadu_si512((const __m512i*)(complexVectorPtr + 64));
        complexVectorPtr += 128;

        __m512i shuffled0 = _mm512_shuffle_epi8(complexVal0, moveMask);
        __m512i shuffled1 = _mm512_shuffle_epi8(complexVal1, moveMask);

        __m512i iOut = _mm512_permutex2var_epi64(shuffled0, iIdx, shuffled1);
        __m512i qOut = _mm512_permutex2var_epi64(shuffled0, qIdx, shuffled1);

        _mm512_storeu_si512((__m512i*)iBufferPtr, iOut);
        _mm512_storeu_si512((__m512i*)qBufferPtr, qOut);
        iBufferPtr += 32;
        qBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_AVX512VBMI
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_avx512vbmi(
    int16_t* iBuffer,
    int16_t* qBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    /* Index vectors to extract I (bytes 0,1 of each 4-byte sample) and
     * Q (bytes 2,3) from the concatenation of two 512-bit registers.
     * Each register holds 16 complex samples (64 bytes); two registers
     * hold 32 samples total. */
    const __m512i iIdx = _mm512_set_epi8(
        125, 124, 121, 120, 117, 116, 113, 112,
        109, 108, 105, 104, 101, 100, 97, 96,
        93, 92, 89, 88, 85, 84, 81, 80,
        77, 76, 73, 72, 69, 68, 65, 64,
        61, 60, 57, 56, 53, 52, 49, 48,
        45, 44, 41, 40, 37, 36, 33, 32,
        29, 28, 25, 24, 21, 20, 17, 16,
        13, 12, 9, 8, 5, 4, 1, 0);

    const __m512i qIdx = _mm512_set_epi8(
        127, 126, 123, 122, 119, 118, 115, 114,
        111, 110, 107, 106, 103, 102, 99, 98,
        95, 94, 91, 90, 87, 86, 83, 82,
        79, 78, 75, 74, 71, 70, 67, 66,
        63, 62, 59, 58, 55, 54, 51, 50,
        47, 46, 43, 42, 39, 38, 35, 34,
        31, 30, 27, 26, 23, 22, 19, 18,
        15, 14, 11, 10, 7, 6, 3, 2);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal0 = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        __m512i complexVal1 =
            _mm512_loadu_si512((const __m512i*)(complexVectorPtr + 64));
        complexVectorPtr += 128;

        __m512i iOut = _mm512_permutex2var_epi8(complexVal0, iIdx, complexVal1);
        __m512i qOut = _mm512_permutex2var_epi8(complexVal0, qIdx, complexVal1);

        _mm512_storeu_si512((__m512i*)iBufferPtr, iOut);
        _mm512_storeu_si512((__m512i*)qBufferPtr, qOut);
        iBufferPtr += 32;
        qBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI */

#ifdef LV_HAVE_AVX512VBMI2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_u_avx512vbmi2(
    int16_t* iBuffer,
    int16_t* qBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    const __mmask32 iMask = 0x55555555;
    const __mmask32 qMask = (__mmask32)0xAAAAAAAAu;

    const unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        __m512i data = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 32;

        _mm512_mask_compressstoreu_epi16(iBufferPtr, iMask, data);
        _mm512_mask_compressstoreu_epi16(qBufferPtr, qMask, data);
        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI2 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_16i_x2_neon(int16_t* iBuffer,
                                                      int16_t* qBuffer,
                                                      const lv_16sc_t* complexVector,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    int16x8x2_t complexVal;

    for (; number < eighthPoints; number++) {
        complexVal = vld2q_s16(complexVectorPtr);
        vst1q_s16(iBufferPtr, complexVal.val[0]);
        vst1q_s16(qBufferPtr, complexVal.val[1]);
        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_16i_x2_neonv8(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    int16x8x2_t complexVal0, complexVal1;

    for (; number < sixteenthPoints; number++) {
        complexVal0 = vld2q_s16(complexVectorPtr);
        complexVal1 = vld2q_s16(complexVectorPtr + 16);
        __VOLK_PREFETCH(complexVectorPtr + 32);

        vst1q_s16(iBufferPtr, complexVal0.val[0]);
        vst1q_s16(iBufferPtr + 8, complexVal1.val[0]);
        vst1q_s16(qBufferPtr, complexVal0.val[1]);
        vst1q_s16(qBufferPtr + 8, complexVal1.val[1]);

        complexVectorPtr += 32;
        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_deinterleave_16i_x2_rvv(int16_t* iBuffer,
                                                     int16_t* qBuffer,
                                                     const lv_16sc_t* complexVector,
                                                     unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vuint32m8_t vc = __riscv_vle32_v_u32m8((const uint32_t*)complexVector, vl);
        vuint16m4_t vr = __riscv_vnsrl(vc, 0, vl);
        vuint16m4_t vi = __riscv_vnsrl(vc, 16, vl);
        __riscv_vse16((uint16_t*)iBuffer, vr, vl);
        __riscv_vse16((uint16_t*)qBuffer, vi, vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_16ic_deinterleave_16i_x2_rvvseg(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vuint16m4x2_t vc =
            __riscv_vlseg2e16_v_u16m4x2((const uint16_t*)complexVector, vl);
        vuint16m4_t vr = __riscv_vget_u16m4(vc, 0);
        vuint16m4_t vi = __riscv_vget_u16m4(vc, 1);
        __riscv_vse16((uint16_t*)iBuffer, vr, vl);
        __riscv_vse16((uint16_t*)qBuffer, vi, vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#ifdef LV_HAVE_ORC

extern void volk_16ic_deinterleave_16i_x2_a_orc_impl(int16_t* iBuffer,
                                                     int16_t* qBuffer,
                                                     const lv_16sc_t* complexVector,
                                                     int num_points);
static inline void volk_16ic_deinterleave_16i_x2_u_orc(int16_t* iBuffer,
                                                       int16_t* qBuffer,
                                                       const lv_16sc_t* complexVector,
                                                       unsigned int num_points)
{
    volk_16ic_deinterleave_16i_x2_a_orc_impl(iBuffer, qBuffer, complexVector, num_points);
}
#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_16ic_deinterleave_16i_x2_u_H */


#ifndef INCLUDED_volk_16ic_deinterleave_16i_x2_a_H
#define INCLUDED_volk_16ic_deinterleave_16i_x2_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_sse2(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;
    __m128i complexVal1, complexVal2, iComplexVal1, iComplexVal2, qComplexVal1,
        qComplexVal2, iOutputVal, qOutputVal;
    __m128i lowMask = _mm_set_epi32(0x0, 0x0, -1, -1);
    __m128i highMask = _mm_set_epi32(-1, -1, 0x0, 0x0);

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;
        complexVal2 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        iComplexVal1 = _mm_shufflelo_epi16(complexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal1 = _mm_shufflehi_epi16(iComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal1 = _mm_shuffle_epi32(iComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal2 = _mm_shufflelo_epi16(complexVal2, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal2 = _mm_shufflehi_epi16(iComplexVal2, _MM_SHUFFLE(3, 1, 2, 0));

        iComplexVal2 = _mm_shuffle_epi32(iComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        iOutputVal = _mm_or_si128(_mm_and_si128(iComplexVal1, lowMask),
                                  _mm_and_si128(iComplexVal2, highMask));

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);

        qComplexVal1 = _mm_shufflelo_epi16(complexVal1, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal1 = _mm_shufflehi_epi16(qComplexVal1, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal1 = _mm_shuffle_epi32(qComplexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        qComplexVal2 = _mm_shufflelo_epi16(complexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal2 = _mm_shufflehi_epi16(qComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        qComplexVal2 = _mm_shuffle_epi32(qComplexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        qOutputVal = _mm_or_si128(_mm_and_si128(qComplexVal1, lowMask),
                                  _mm_and_si128(qComplexVal2, highMask));

        _mm_store_si128((__m128i*)qBufferPtr, qOutputVal);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_ssse3(int16_t* iBuffer,
                                                         int16_t* qBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    __m128i iMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);
    __m128i iMoveMask2 = _mm_set_epi8(
        13, 12, 9, 8, 5, 4, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    __m128i qMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 15, 14, 11, 10, 7, 6, 3, 2);
    __m128i qMoveMask2 = _mm_set_epi8(
        15, 14, 11, 10, 7, 6, 3, 2, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    __m128i complexVal1, complexVal2, iOutputVal, qOutputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        iOutputVal = _mm_or_si128(_mm_shuffle_epi8(complexVal1, iMoveMask1),
                                  _mm_shuffle_epi8(complexVal2, iMoveMask2));
        qOutputVal = _mm_or_si128(_mm_shuffle_epi8(complexVal1, qMoveMask1),
                                  _mm_shuffle_epi8(complexVal2, qMoveMask2));

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);
        _mm_store_si128((__m128i*)qBufferPtr, qOutputVal);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    const int16_t* int16ComplexVectorPtr = (const int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *int16ComplexVectorPtr++;
        *qBufferPtr++ = *int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_avx2(int16_t* iBuffer,
                                                        int16_t* qBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    __m256i MoveMask = _mm256_set_epi8(15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0,
                                       15,
                                       14,
                                       11,
                                       10,
                                       7,
                                       6,
                                       3,
                                       2,
                                       13,
                                       12,
                                       9,
                                       8,
                                       5,
                                       4,
                                       1,
                                       0);

    __m256i iMove2, iMove1;
    __m256i complexVal1, complexVal2, iOutputVal, qOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        iMove2 = _mm256_shuffle_epi8(complexVal2, MoveMask);
        iMove1 = _mm256_shuffle_epi8(complexVal1, MoveMask);

        iOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x08),
                                               _mm256_permute4x64_epi64(iMove2, 0x80),
                                               0x30);
        qOutputVal = _mm256_permute2x128_si256(_mm256_permute4x64_epi64(iMove1, 0x0d),
                                               _mm256_permute4x64_epi64(iMove2, 0xd0),
                                               0x30);

        _mm256_store_si256((__m256i*)iBufferPtr, iOutputVal);
        _mm256_store_si256((__m256i*)qBufferPtr, qOutputVal);

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    const int16_t* int16ComplexVectorPtr = (const int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *int16ComplexVectorPtr++;
        *qBufferPtr++ = *int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_avx512f(
    int16_t* iBuffer,
    int16_t* qBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    const __m512i lowMask = _mm512_set1_epi32(0x0000FFFF);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal0 = _mm512_load_si512((const __m512i*)complexVectorPtr);
        __m512i complexVal1 =
            _mm512_load_si512((const __m512i*)(complexVectorPtr + 32));
        complexVectorPtr += 64;

        __m256i iLo = _mm512_cvtepi32_epi16(_mm512_and_epi32(complexVal0, lowMask));
        __m256i iHi = _mm512_cvtepi32_epi16(_mm512_and_epi32(complexVal1, lowMask));
        __m256i qLo = _mm512_cvtepi32_epi16(_mm512_srli_epi32(complexVal0, 16));
        __m256i qHi = _mm512_cvtepi32_epi16(_mm512_srli_epi32(complexVal1, 16));

        __m512i iOut = _mm512_inserti64x4(_mm512_castsi256_si512(iLo), iHi, 1);
        __m512i qOut = _mm512_inserti64x4(_mm512_castsi256_si512(qLo), qHi, 1);

        _mm512_store_si512((__m512i*)iBufferPtr, iOut);
        _mm512_store_si512((__m512i*)qBufferPtr, qOut);
        iBufferPtr += 32;
        qBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_avx512bw(
    int16_t* iBuffer,
    int16_t* qBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    /* Shuffle mask: within each 128-bit lane, group I bytes (0,1,4,5,8,9,12,13)
     * into the low 8 bytes and Q bytes (2,3,6,7,10,11,14,15) into the high 8 bytes */
    const __m512i moveMask = _mm512_set_epi8(
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
        15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);

    /* After shuffle, each 128-bit lane has [I0-3 | Q0-3] as 64-bit qwords.
     * Gather even qwords (I) and odd qwords (Q) across both registers. */
    const __m512i iIdx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i qIdx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal0 = _mm512_load_si512((const __m512i*)complexVectorPtr);
        __m512i complexVal1 =
            _mm512_load_si512((const __m512i*)(complexVectorPtr + 64));
        complexVectorPtr += 128;

        __m512i shuffled0 = _mm512_shuffle_epi8(complexVal0, moveMask);
        __m512i shuffled1 = _mm512_shuffle_epi8(complexVal1, moveMask);

        __m512i iOut = _mm512_permutex2var_epi64(shuffled0, iIdx, shuffled1);
        __m512i qOut = _mm512_permutex2var_epi64(shuffled0, qIdx, shuffled1);

        _mm512_store_si512((__m512i*)iBufferPtr, iOut);
        _mm512_store_si512((__m512i*)qBufferPtr, qOut);
        iBufferPtr += 32;
        qBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_AVX512VBMI2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_16i_x2_a_avx512vbmi2(
    int16_t* iBuffer,
    int16_t* qBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    int16_t* qBufferPtr = qBuffer;

    const __mmask32 iMask = 0x55555555;
    const __mmask32 qMask = (__mmask32)0xAAAAAAAAu;

    const unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        __m512i data = _mm512_load_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 32;

        __m512i iVals = _mm512_maskz_compress_epi16(iMask, data);
        __m512i qVals = _mm512_maskz_compress_epi16(qMask, data);
        _mm256_store_si256((__m256i*)iBufferPtr, _mm512_castsi512_si256(iVals));
        _mm256_store_si256((__m256i*)qBufferPtr, _mm512_castsi512_si256(qVals));
        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_16ic_deinterleave_16i_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI2 */

#endif /* INCLUDED_volk_16ic_deinterleave_16i_x2_a_H */
