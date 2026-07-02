/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_deinterleave_real_16i
 *
 * \b Overview
 *
 * Deinterleaves a complex 16-bit integer vector and extracts the real (inphase)
 * component. For each complex sample \f$z_k = I_k + jQ_k\f$, the output is
 * \f$I_k\f$.
 *
 * This operation is common in receiver signal processing pipelines where complex
 * baseband samples arrive interleaved (I, Q, I, Q, ...) and only the inphase
 * component is needed — for example, after downconversion when the signal of
 * interest is real-valued, or as a preprocessing step before further demodulation.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_deinterleave_real_16i(int16_t* iBuffer, const lv_16sc_t* complexVector,
 * unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of interleaved I/Q samples (lv_16sc_t).
 * \li num_points: The number of complex samples to deinterleave.
 *
 * \b Outputs
 * \li iBuffer: The real (inphase) output samples (int16_t).
 *
 * \b Example
 * Extract the real part from four complex 16-bit samples.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_16sc_t* complexVector =
 *     (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t) * N, alignment);
 * int16_t* iBuffer = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 *
 * // Each complex sample has real and imaginary parts
 * complexVector[0] = (lv_16sc_t){ 100, -50 };
 * complexVector[1] = (lv_16sc_t){ 200, -100 };
 * complexVector[2] = (lv_16sc_t){ 300, -150 };
 * complexVector[3] = (lv_16sc_t){ 400, -200 };
 *
 * volk_16ic_deinterleave_real_16i(iBuffer, complexVector, N);
 *
 * // Expected output: iBuffer = {100, 200, 300, 400}
 * printf("Expected: %d %d %d %d\n", 100, 200, 300, 400);
 * printf("Result:   %d %d %d %d\n", iBuffer[0], iBuffer[1], iBuffer[2], iBuffer[3]);
 *
 * volk_free(complexVector);
 * volk_free(iBuffer);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_deinterleave_real_16i_a_H
#define INCLUDED_volk_16ic_deinterleave_real_16i_a_H

#include <inttypes.h>
#include <stdio.h>


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_real_16i_a_avx2(int16_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    __m256i iMoveMask1 = _mm256_set_epi8(0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0);
    __m256i iMoveMask2 = _mm256_set_epi8(13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80);

    __m256i complexVal1, complexVal2, iOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm256_load_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal1 = _mm256_shuffle_epi8(complexVal1, iMoveMask1);
        complexVal2 = _mm256_shuffle_epi8(complexVal2, iMoveMask2);

        iOutputVal = _mm256_or_si256(complexVal1, complexVal2);
        iOutputVal = _mm256_permute4x64_epi64(iOutputVal, 0xd8);

        _mm256_store_si256((__m256i*)iBufferPtr, iOutputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_16ic_deinterleave_real_16i_a_ssse3(int16_t* iBuffer,
                                                           const lv_16sc_t* complexVector,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    __m128i iMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);
    __m128i iMoveMask2 = _mm_set_epi8(
        13, 12, 9, 8, 5, 4, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    __m128i complexVal1, complexVal2, iOutputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;
        complexVal2 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        complexVal1 = _mm_shuffle_epi8(complexVal1, iMoveMask1);
        complexVal2 = _mm_shuffle_epi8(complexVal2, iMoveMask2);

        iOutputVal = _mm_or_si128(complexVal1, complexVal2);

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);

        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSSE3 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_deinterleave_real_16i_a_sse2(int16_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m128i complexVal1, complexVal2, iOutputVal;
    __m128i lowMask = _mm_set_epi32(0x0, 0x0, 0xFFFFFFFF, 0xFFFFFFFF);
    __m128i highMask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0);

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal1 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;
        complexVal2 = _mm_load_si128((__m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        complexVal1 = _mm_shufflelo_epi16(complexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        complexVal1 = _mm_shufflehi_epi16(complexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        complexVal1 = _mm_shuffle_epi32(complexVal1, _MM_SHUFFLE(3, 1, 2, 0));

        complexVal2 = _mm_shufflelo_epi16(complexVal2, _MM_SHUFFLE(3, 1, 2, 0));

        complexVal2 = _mm_shufflehi_epi16(complexVal2, _MM_SHUFFLE(3, 1, 2, 0));

        complexVal2 = _mm_shuffle_epi32(complexVal2, _MM_SHUFFLE(2, 0, 3, 1));

        iOutputVal = _mm_or_si128(_mm_and_si128(complexVal1, lowMask),
                                  _mm_and_si128(complexVal2, highMask));

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);

        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_deinterleave_real_16i_generic(int16_t* iBuffer,
                                                           const lv_16sc_t* complexVector,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_real_16i_neon(int16_t* iBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    int16x8x2_t complexVal;

    for (; number < eighthPoints; number++) {
        complexVal = vld2q_s16(complexVectorPtr);
        vst1q_s16(iBufferPtr, complexVal.val[0]);
        complexVectorPtr += 16;
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_real_16i_neonv8(int16_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    int16x8x2_t complexVal0, complexVal1;

    for (; number < sixteenthPoints; number++) {
        complexVal0 = vld2q_s16(complexVectorPtr);
        complexVal1 = vld2q_s16(complexVectorPtr + 16);
        __VOLK_PREFETCH(complexVectorPtr + 32);

        vst1q_s16(iBufferPtr, complexVal0.val[0]);
        vst1q_s16(iBufferPtr + 8, complexVal1.val[0]);

        complexVectorPtr += 32;
        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_volk_16ic_deinterleave_real_16i_a_H */


#ifndef INCLUDED_volk_16ic_deinterleave_real_16i_u_H
#define INCLUDED_volk_16ic_deinterleave_real_16i_u_H

#include <inttypes.h>
#include <stdio.h>


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_real_16i_u_avx2(int16_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (int16_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    __m256i iMoveMask1 = _mm256_set_epi8(0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0);
    __m256i iMoveMask2 = _mm256_set_epi8(13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         13,
                                         12,
                                         9,
                                         8,
                                         5,
                                         4,
                                         1,
                                         0,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80,
                                         0x80);

    __m256i complexVal1, complexVal2, iOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm256_loadu_si256((__m256i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal1 = _mm256_shuffle_epi8(complexVal1, iMoveMask1);
        complexVal2 = _mm256_shuffle_epi8(complexVal2, iMoveMask2);

        iOutputVal = _mm256_or_si256(complexVal1, complexVal2);
        iOutputVal = _mm256_permute4x64_epi64(iOutputVal, 0xd8);

        _mm256_storeu_si256((__m256i*)iBufferPtr, iOutputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_deinterleave_real_16i_rvv(int16_t* iBuffer,
                                                       const lv_16sc_t* complexVector,
                                                       unsigned int num_points)
{
    const uint32_t* in = (const uint32_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vuint32m8_t vc = __riscv_vle32_v_u32m8(in, vl);
        __riscv_vse16((uint16_t*)iBuffer, __riscv_vnsrl(vc, 0, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_16ic_deinterleave_real_16i_u_H */
