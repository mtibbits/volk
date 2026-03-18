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
 * component and converts it to 16-bit shorts. Each 8-bit value is
 * sign-extended to 16 bits and scaled by 128 (left-shifted by 7).
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8ic_deinterleave_real_16i(int16_t* iBuffer, const lv_8sc_t* complexVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of interleaved 8-bit I/Q pairs (lv_8sc_t).
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The real (I) output vector of 16-bit shorts (int16_t).
 *
 * \b Example
 * Extract the real component from complex 8-bit samples into a 16-bit vector.
 * \code
 * #include <volk/volk.h>
 * #include <stdio.h>
 *
 * int main() {
 *     unsigned int N = 4;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     lv_8sc_t* complexVector =
 *         (lv_8sc_t*)volk_malloc(sizeof(lv_8sc_t) * N, alignment);
 *     int16_t* iBuffer = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 *
 *     // Simulate complex 8-bit samples: (I, Q) pairs
 *     complexVector[0] = lv_cmake((int8_t)127, (int8_t)-128);
 *     complexVector[1] = lv_cmake((int8_t)50, (int8_t)-50);
 *     complexVector[2] = lv_cmake((int8_t)0, (int8_t)100);
 *     complexVector[3] = lv_cmake((int8_t)-30, (int8_t)30);
 *
 *     // Extract real (I) component, scaled by 128
 *     volk_8ic_deinterleave_real_16i(iBuffer, complexVector, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *         printf("complex[%u] = (%4d, %4d)  ->  I = %6d\n",
 *                i, lv_creal(complexVector[i]), lv_cimag(complexVector[i]),
 *                iBuffer[i]);
 *     }
 *
 *     volk_free(complexVector);
 *     volk_free(iBuffer);
 *     return 0;
 * }
 * \endcode
 */

#ifndef INCLUDED_volk_8ic_deinterleave_real_16i_u_H
#define INCLUDED_volk_8ic_deinterleave_real_16i_u_H

#include <inttypes.h>
#include <stdio.h>


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


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_sse2(int16_t* iBuffer,
                                                          const lv_8sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    for (; number < eighthPoints; number++) {
        __m128i raw = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        /* Each int16 = (Q << 8) | I. Shift left 8 puts I in high byte (I*256),
         * arithmetic shift right 1 gives I*128 with sign extension. */
        __m128i result = _mm_srai_epi16(_mm_slli_epi16(raw, 8), 1);

        _mm_storeu_si128((__m128i*)iBufferPtr, result);
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_sse4_1(int16_t* iBuffer,
                                                            const lv_8sc_t* complexVector,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i complexVal, outputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal = _mm_shuffle_epi8(complexVal, moveMask);

        outputVal = _mm_cvtepi8_epi16(complexVal);
        outputVal = _mm_slli_epi16(outputVal, 7);

        _mm_storeu_si128((__m128i*)iBufferPtr, outputVal);
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_avx(int16_t* iBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i complexVal, outputVal;
    __m128i complexVal1, complexVal0, outputVal1, outputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
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
        _mm256_storeu_si256((__m256i*)iBufferPtr, outputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_avx2(int16_t* iBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
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
        complexVal = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
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

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_avx512bw(
    int16_t* iBuffer,
    const lv_8sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    /* Shuffle within each 128-bit lane: even bytes (I) to low qword,
     * odd bytes (Q) to high qword. */
    const __m512i moveMask = _mm512_set_epi8(
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);

    /* Pack the 4 I qwords (indices 0,2,4,6) into the lower 256 bits. */
    const __m512i packIdx = _mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        __m512i shuffled = _mm512_shuffle_epi8(complexVal, moveMask);
        __m512i packed = _mm512_permutexvar_epi64(packIdx, shuffled);
        __m256i iBytes = _mm512_castsi512_si256(packed);

        /* Sign-extend int8 -> int16, shift left by 7 */
        __m512i iWide = _mm512_slli_epi16(_mm512_cvtepi8_epi16(iBytes), 7);
        _mm512_storeu_si512((__m512i*)iBufferPtr, iWide);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_AVX512VBMI
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_avx512vbmi(
    int16_t* iBuffer,
    const lv_8sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    /* Extract 32 even bytes (I) from a single 512-bit register */
    const __m512i iIdx = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        62, 60, 58, 56, 54, 52, 50, 48,
        46, 44, 42, 40, 38, 36, 34, 32,
        30, 28, 26, 24, 22, 20, 18, 16,
        14, 12, 10, 8, 6, 4, 2, 0);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        /* Extract 32 I bytes into lower 256 bits */
        __m256i iBytes = _mm512_castsi512_si256(
            _mm512_permutexvar_epi8(iIdx, complexVal));

        /* Sign-extend int8 -> int16, shift left by 7 */
        __m512i iWide = _mm512_slli_epi16(_mm512_cvtepi8_epi16(iBytes), 7);
        _mm512_storeu_si512((__m512i*)iBufferPtr, iWide);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI */


#ifdef LV_HAVE_AVX512VBMI2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_u_avx512vbmi2(
    int16_t* iBuffer,
    const lv_8sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    const __mmask64 iMask = 0x5555555555555555ULL;

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i data = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        __m512i iBytes = _mm512_maskz_compress_epi8(iMask, data);
        __m256i iB = _mm512_castsi512_si256(iBytes);

        __m512i iWide = _mm512_slli_epi16(_mm512_cvtepi8_epi16(iB), 7);
        _mm512_storeu_si512((__m512i*)iBufferPtr, iWide);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI2 */


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
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_8ic_deinterleave_real_16i_rvvseg(int16_t* iBuffer,
                                                          const lv_8sc_t* complexVector,
                                                          unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e8m2(n);
        vint8m2x2_t vc =
            __riscv_vlseg2e8_v_i8m2x2((const int8_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vsll(__riscv_vsext_vf2(__riscv_vget_i8m2(vc, 0), vl), 7, vl);
        __riscv_vse16(iBuffer, vr, vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_8ic_deinterleave_real_16i_u_H */

#ifndef INCLUDED_volk_8ic_deinterleave_real_16i_a_H
#define INCLUDED_volk_8ic_deinterleave_real_16i_a_H

#include <inttypes.h>
#include <stdio.h>


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_sse2(int16_t* iBuffer,
                                                          const lv_8sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    for (; number < eighthPoints; number++) {
        __m128i raw = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        /* Each int16 = (Q << 8) | I. Shift left 8 puts I in high byte (I*256),
         * arithmetic shift right 1 gives I*128 with sign extension. */
        __m128i result = _mm_srai_epi16(_mm_slli_epi16(raw, 8), 1);

        _mm_store_si128((__m128i*)iBufferPtr, result);
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_sse4_1(int16_t* iBuffer,
                                                           const lv_8sc_t* complexVector,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i complexVal, outputVal;

    unsigned int eighthPoints = num_points / 8;

    for (number = 0; number < eighthPoints; number++) {
        complexVal = _mm_load_si128((const __m128i*)complexVectorPtr);
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
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;
    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m256i complexVal, outputVal;
    __m128i complexVal1, complexVal0, outputVal1, outputVal0;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal = _mm256_load_si256((const __m256i*)complexVectorPtr);
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


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_avx2(int16_t* iBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
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
        complexVal = _mm256_load_si256((const __m256i*)complexVectorPtr);
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

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_avx512bw(
    int16_t* iBuffer,
    const lv_8sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    /* Shuffle within each 128-bit lane: even bytes (I) to low qword,
     * odd bytes (Q) to high qword. */
    const __m512i moveMask = _mm512_set_epi8(
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
        15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);

    /* Pack the 4 I qwords (indices 0,2,4,6) into the lower 256 bits. */
    const __m512i packIdx = _mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal = _mm512_load_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        __m512i shuffled = _mm512_shuffle_epi8(complexVal, moveMask);
        __m512i packed = _mm512_permutexvar_epi64(packIdx, shuffled);
        __m256i iBytes = _mm512_castsi512_si256(packed);

        /* Sign-extend int8 -> int16, shift left by 7 */
        __m512i iWide = _mm512_slli_epi16(_mm512_cvtepi8_epi16(iBytes), 7);
        _mm512_store_si512((__m512i*)iBufferPtr, iWide);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_AVX512VBMI2
#include <immintrin.h>

static inline void volk_8ic_deinterleave_real_16i_a_avx512vbmi2(
    int16_t* iBuffer,
    const lv_8sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int16_t* iBufferPtr = iBuffer;

    const __mmask64 iMask = 0x5555555555555555ULL;

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i data = _mm512_load_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        __m512i iBytes = _mm512_maskz_compress_epi8(iMask, data);
        __m256i iB = _mm512_castsi512_si256(iBytes);

        __m512i iWide = _mm512_slli_epi16(_mm512_cvtepi8_epi16(iB), 7);
        _mm512_store_si512((__m512i*)iBufferPtr, iWide);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_deinterleave_real_16i_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI2 */

#endif /* INCLUDED_volk_8ic_deinterleave_real_16i_a_H */
