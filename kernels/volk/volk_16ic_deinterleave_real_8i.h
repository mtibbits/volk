/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_deinterleave_real_8i
 *
 * \b Overview
 *
 * Deinterleaves a complex 16-bit integer vector and returns only the real
 * (in-phase) component of each sample, right-shifted by 8 bits to produce
 * an 8-bit result.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_deinterleave_real_8i(int8_t* iBuffer, const lv_16sc_t* complexVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector (lv_16sc_t).
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The in-phase (real) output vector, right-shifted by 8 (int8_t).
 *
 * \b Example
 * Extract the real part of a complex signal, reduced to 8-bit precision.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main() {
 *     unsigned int N = 8;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     // Allocate input complex vector and output I buffer
 *     lv_16sc_t* complexVector =
 *         (lv_16sc_t*)volk_malloc(N * sizeof(lv_16sc_t), alignment);
 *     int8_t* iBuffer = (int8_t*)volk_malloc(N * sizeof(int8_t), alignment);
 *
 *     // Fill with complex samples: (real, imag)
 *     // Real parts are chosen as multiples of 256 so the >> 8 result is clear
 *     complexVector[0] = lv_cmake((int16_t)256, (int16_t)100);
 *     complexVector[1] = lv_cmake((int16_t)512, (int16_t)-200);
 *     complexVector[2] = lv_cmake((int16_t)-768, (int16_t)300);
 *     complexVector[3] = lv_cmake((int16_t)1024, (int16_t)-400);
 *     complexVector[4] = lv_cmake((int16_t)-1280, (int16_t)500);
 *     complexVector[5] = lv_cmake((int16_t)1536, (int16_t)-600);
 *     complexVector[6] = lv_cmake((int16_t)-1792, (int16_t)700);
 *     complexVector[7] = lv_cmake((int16_t)2048, (int16_t)-800);
 *
 *     // Extract real parts, right-shifted by 8 to produce 8-bit output
 *     volk_16ic_deinterleave_real_8i(iBuffer, complexVector, N);
 *
 *     // Expected: 1, 2, -3, 4, -5, 6, -7, 8
 *     for (unsigned int i = 0; i < N; i++) {
 *       printf("Complex[%u] real=%d -> iBuffer=%d\n",
 *              i, lv_creal(complexVector[i]), iBuffer[i]);
 *     }
 *
 *     volk_free(complexVector);
 *     volk_free(iBuffer);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_deinterleave_real_8i_u_H
#define INCLUDED_volk_16ic_deinterleave_real_8i_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_deinterleave_real_8i_generic(int8_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = ((int8_t)(*complexVectorPtr++ >> 8));
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_deinterleave_real_8i_u_sse2(int8_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    const __m128i realExtract = _mm_set1_epi32(1);

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        __m128i c0 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        __m128i c1 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        __m128i c2 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        __m128i c3 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        c0 = _mm_madd_epi16(c0, realExtract);
        c1 = _mm_madd_epi16(c1, realExtract);
        c2 = _mm_madd_epi16(c2, realExtract);
        c3 = _mm_madd_epi16(c3, realExtract);

        __m128i p01 = _mm_packs_epi32(c0, c1);
        __m128i p23 = _mm_packs_epi32(c2, c3);

        p01 = _mm_srai_epi16(p01, 8);
        p23 = _mm_srai_epi16(p23, 8);

        _mm_storeu_si128((__m128i*)iBufferPtr, _mm_packs_epi16(p01, p23));
        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_16ic_deinterleave_real_8i_generic(
        iBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_16ic_deinterleave_real_8i_u_ssse3(int8_t* iBuffer,
                                                           const lv_16sc_t* complexVector,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    __m128i iMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);
    __m128i iMoveMask2 = _mm_set_epi8(
        13, 12, 9, 8, 5, 4, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    __m128i complexVal1, complexVal2, complexVal3, complexVal4, iOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal3 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal4 = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal1 = _mm_shuffle_epi8(complexVal1, iMoveMask1);
        complexVal2 = _mm_shuffle_epi8(complexVal2, iMoveMask2);

        complexVal1 = _mm_or_si128(complexVal1, complexVal2);

        complexVal3 = _mm_shuffle_epi8(complexVal3, iMoveMask1);
        complexVal4 = _mm_shuffle_epi8(complexVal4, iMoveMask2);

        complexVal3 = _mm_or_si128(complexVal3, complexVal4);

        complexVal1 = _mm_srai_epi16(complexVal1, 8);
        complexVal3 = _mm_srai_epi16(complexVal3, 8);

        iOutputVal = _mm_packs_epi16(complexVal1, complexVal3);

        _mm_storeu_si128((__m128i*)iBufferPtr, iOutputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_16ic_deinterleave_real_8i_generic(
        iBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_real_8i_u_avx2(int8_t* iBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
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
    __m256i complexVal1, complexVal2, complexVal3, complexVal4, iOutputVal;

    unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        complexVal1 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal3 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal4 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal1 = _mm256_shuffle_epi8(complexVal1, iMoveMask1);
        complexVal2 = _mm256_shuffle_epi8(complexVal2, iMoveMask2);

        complexVal1 = _mm256_or_si256(complexVal1, complexVal2);
        complexVal1 = _mm256_permute4x64_epi64(complexVal1, 0xd8);

        complexVal3 = _mm256_shuffle_epi8(complexVal3, iMoveMask1);
        complexVal4 = _mm256_shuffle_epi8(complexVal4, iMoveMask2);

        complexVal3 = _mm256_or_si256(complexVal3, complexVal4);
        complexVal3 = _mm256_permute4x64_epi64(complexVal3, 0xd8);

        complexVal1 = _mm256_srai_epi16(complexVal1, 8);
        complexVal3 = _mm256_srai_epi16(complexVal3, 8);

        iOutputVal = _mm256_packs_epi16(complexVal1, complexVal3);
        iOutputVal = _mm256_permute4x64_epi64(iOutputVal, 0xd8);

        _mm256_storeu_si256((__m256i*)iBufferPtr, iOutputVal);

        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    const int16_t* int16ComplexVectorPtr = (const int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int8_t)(*int16ComplexVectorPtr++ >> 8));
        int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16ic_deinterleave_real_8i_u_avx512bw(
    int8_t* iBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    const __m512i realExtract = _mm512_set1_epi32(1);
    const __m512i perm64 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    const unsigned int sixtyfourthPoints = num_points / 64;

    for (number = 0; number < sixtyfourthPoints; number++) {
        __m512i c0 = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        __m512i c1 = _mm512_loadu_si512((const __m512i*)(complexVectorPtr + 64));
        __m512i c2 = _mm512_loadu_si512((const __m512i*)(complexVectorPtr + 128));
        __m512i c3 = _mm512_loadu_si512((const __m512i*)(complexVectorPtr + 192));
        complexVectorPtr += 256;

        __m512i r0 = _mm512_madd_epi16(c0, realExtract);
        __m512i r1 = _mm512_madd_epi16(c1, realExtract);
        __m512i r2 = _mm512_madd_epi16(c2, realExtract);
        __m512i r3 = _mm512_madd_epi16(c3, realExtract);

        __m512i r16_01 = _mm512_packs_epi32(r0, r1);
        __m512i r16_23 = _mm512_packs_epi32(r2, r3);

        r16_01 = _mm512_permutexvar_epi64(perm64, r16_01);
        r16_23 = _mm512_permutexvar_epi64(perm64, r16_23);

        r16_01 = _mm512_srai_epi16(r16_01, 8);
        r16_23 = _mm512_srai_epi16(r16_23, 8);

        __m512i r8 = _mm512_packs_epi16(r16_01, r16_23);
        r8 = _mm512_permutexvar_epi64(perm64, r8);

        _mm512_storeu_si512((__m512i*)iBufferPtr, r8);
        iBufferPtr += 64;
    }

    number = sixtyfourthPoints * 64;
    volk_16ic_deinterleave_real_8i_generic(
        iBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_AVX512VBMI
#include <immintrin.h>

static inline void volk_16ic_deinterleave_real_8i_u_avx512vbmi(
    int8_t* iBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;

    /* Extract byte 1 (I_hi) from each 4-byte complex sample across two
     * 512-bit source registers. Each register holds 16 complex samples;
     * two registers hold 32 samples, producing 32 output bytes.
     * We place the result in the lower 256 bits. */
    const __m512i iHiIdx = _mm512_set_epi8(
        /* upper 32 bytes: don't care */
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* lower 32 bytes: byte 1 of each 4-byte sample from both sources */
        125, 121, 117, 113, 109, 105, 101, 97,
        93, 89, 85, 81, 77, 73, 69, 65,
        61, 57, 53, 49, 45, 41, 37, 33,
        29, 25, 21, 17, 13, 9, 5, 1);

    const unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        __m512i complexVal0 = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        __m512i complexVal1 =
            _mm512_loadu_si512((const __m512i*)(complexVectorPtr + 64));
        complexVectorPtr += 128;

        __m256i iOut = _mm512_castsi512_si256(
            _mm512_permutex2var_epi8(complexVal0, iHiIdx, complexVal1));
        _mm256_storeu_si256((__m256i*)iBufferPtr, iOut);
        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    volk_16ic_deinterleave_real_8i_generic(
        iBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_real_8i_neon(int8_t* iBuffer,
                                                       const lv_16sc_t* complexVector,
                                                       unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    unsigned int eighth_points = num_points / 8;
    unsigned int number;

    int16x8x2_t complexInput;
    int8x8_t realOutput;
    for (number = 0; number < eighth_points; number++) {
        complexInput = vld2q_s16(complexVectorPtr);
        realOutput = vshrn_n_s16(complexInput.val[0], 8);
        vst1_s8(iBufferPtr, realOutput);
        complexVectorPtr += 16;
        iBufferPtr += 8;
    }

    for (number = eighth_points * 8; number < num_points; number++) {
        *iBufferPtr++ = ((int8_t)(*complexVectorPtr++ >> 8));
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_deinterleave_real_8i_neonv8(int8_t* iBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    const unsigned int sixteenthPoints = num_points / 16;

    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        int16x8x2_t cplx0 = vld2q_s16(complexVectorPtr);
        int16x8x2_t cplx1 = vld2q_s16(complexVectorPtr + 16);
        __VOLK_PREFETCH(complexVectorPtr + 64);

        int8x8_t out0 = vshrn_n_s16(cplx0.val[0], 8);
        int8x8_t out1 = vshrn_n_s16(cplx1.val[0], 8);

        vst1_s8(iBufferPtr, out0);
        vst1_s8(iBufferPtr + 8, out1);

        complexVectorPtr += 32;
        iBufferPtr += 16;
    }

    for (unsigned int number = sixteenthPoints * 16; number < num_points; number++) {
        *iBufferPtr++ = ((int8_t)(*complexVectorPtr++ >> 8));
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_deinterleave_real_8i_rvv(int8_t* iBuffer,
                                                      const lv_16sc_t* complexVector,
                                                      unsigned int num_points)
{
    const uint32_t* in = (const uint32_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vuint32m8_t vc = __riscv_vle32_v_u32m8(in, vl);
        __riscv_vse8(
            (uint8_t*)iBuffer, __riscv_vnsrl(__riscv_vnsrl(vc, 0, vl), 8, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_ORC

extern void volk_16ic_deinterleave_real_8i_a_orc_impl(int8_t* iBuffer,
                                                      const lv_16sc_t* complexVector,
                                                      int num_points);

static inline void volk_16ic_deinterleave_real_8i_u_orc(int8_t* iBuffer,
                                                        const lv_16sc_t* complexVector,
                                                        unsigned int num_points)
{
    volk_16ic_deinterleave_real_8i_a_orc_impl(iBuffer, complexVector, num_points);
}
#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_16ic_deinterleave_real_8i_u_H */

#ifndef INCLUDED_volk_16ic_deinterleave_real_8i_a_H
#define INCLUDED_volk_16ic_deinterleave_real_8i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_deinterleave_real_8i_a_sse2(int8_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    const __m128i realExtract = _mm_set1_epi32(1);

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        __m128i c0 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        __m128i c1 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        __m128i c2 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        __m128i c3 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        c0 = _mm_madd_epi16(c0, realExtract);
        c1 = _mm_madd_epi16(c1, realExtract);
        c2 = _mm_madd_epi16(c2, realExtract);
        c3 = _mm_madd_epi16(c3, realExtract);

        __m128i p01 = _mm_packs_epi32(c0, c1);
        __m128i p23 = _mm_packs_epi32(c2, c3);

        p01 = _mm_srai_epi16(p01, 8);
        p23 = _mm_srai_epi16(p23, 8);

        _mm_store_si128((__m128i*)iBufferPtr, _mm_packs_epi16(p01, p23));
        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_16ic_deinterleave_real_8i_generic(
        iBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void volk_16ic_deinterleave_real_8i_a_ssse3(int8_t* iBuffer,
                                                          const lv_16sc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    __m128i iMoveMask1 = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);
    __m128i iMoveMask2 = _mm_set_epi8(
        13, 12, 9, 8, 5, 4, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    __m128i complexVal1, complexVal2, complexVal3, complexVal4, iOutputVal;

    unsigned int sixteenthPoints = num_points / 16;

    for (number = 0; number < sixteenthPoints; number++) {
        complexVal1 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal2 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal3 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal4 = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        complexVal1 = _mm_shuffle_epi8(complexVal1, iMoveMask1);
        complexVal2 = _mm_shuffle_epi8(complexVal2, iMoveMask2);

        complexVal1 = _mm_or_si128(complexVal1, complexVal2);

        complexVal3 = _mm_shuffle_epi8(complexVal3, iMoveMask1);
        complexVal4 = _mm_shuffle_epi8(complexVal4, iMoveMask2);

        complexVal3 = _mm_or_si128(complexVal3, complexVal4);


        complexVal1 = _mm_srai_epi16(complexVal1, 8);
        complexVal3 = _mm_srai_epi16(complexVal3, 8);

        iOutputVal = _mm_packs_epi16(complexVal1, complexVal3);

        _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);

        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    const int16_t* int16ComplexVectorPtr = (const int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int8_t)(*int16ComplexVectorPtr++ >> 8));
        int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_deinterleave_real_8i_a_avx2(int8_t* iBuffer,
                                                         const lv_16sc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
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
    __m256i complexVal1, complexVal2, complexVal3, complexVal4, iOutputVal;

    unsigned int thirtysecondPoints = num_points / 32;

    for (number = 0; number < thirtysecondPoints; number++) {
        complexVal1 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal2 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal3 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal4 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;

        complexVal1 = _mm256_shuffle_epi8(complexVal1, iMoveMask1);
        complexVal2 = _mm256_shuffle_epi8(complexVal2, iMoveMask2);

        complexVal1 = _mm256_or_si256(complexVal1, complexVal2);
        complexVal1 = _mm256_permute4x64_epi64(complexVal1, 0xd8);

        complexVal3 = _mm256_shuffle_epi8(complexVal3, iMoveMask1);
        complexVal4 = _mm256_shuffle_epi8(complexVal4, iMoveMask2);

        complexVal3 = _mm256_or_si256(complexVal3, complexVal4);
        complexVal3 = _mm256_permute4x64_epi64(complexVal3, 0xd8);

        complexVal1 = _mm256_srai_epi16(complexVal1, 8);
        complexVal3 = _mm256_srai_epi16(complexVal3, 8);

        iOutputVal = _mm256_packs_epi16(complexVal1, complexVal3);
        iOutputVal = _mm256_permute4x64_epi64(iOutputVal, 0xd8);

        _mm256_store_si256((__m256i*)iBufferPtr, iOutputVal);

        iBufferPtr += 32;
    }

    number = thirtysecondPoints * 32;
    const int16_t* int16ComplexVectorPtr = (const int16_t*)complexVectorPtr;
    for (; number < num_points; number++) {
        *iBufferPtr++ = ((int8_t)(*int16ComplexVectorPtr++ >> 8));
        int16ComplexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16ic_deinterleave_real_8i_a_avx512bw(
    int8_t* iBuffer,
    const lv_16sc_t* complexVector,
    unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    int8_t* iBufferPtr = iBuffer;
    const __m512i realExtract = _mm512_set1_epi32(1);
    const __m512i perm64 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    const unsigned int sixtyfourthPoints = num_points / 64;

    for (number = 0; number < sixtyfourthPoints; number++) {
        __m512i c0 = _mm512_load_si512((const __m512i*)complexVectorPtr);
        __m512i c1 = _mm512_load_si512((const __m512i*)(complexVectorPtr + 64));
        __m512i c2 = _mm512_load_si512((const __m512i*)(complexVectorPtr + 128));
        __m512i c3 = _mm512_load_si512((const __m512i*)(complexVectorPtr + 192));
        complexVectorPtr += 256;

        __m512i r0 = _mm512_madd_epi16(c0, realExtract);
        __m512i r1 = _mm512_madd_epi16(c1, realExtract);
        __m512i r2 = _mm512_madd_epi16(c2, realExtract);
        __m512i r3 = _mm512_madd_epi16(c3, realExtract);

        __m512i r16_01 = _mm512_packs_epi32(r0, r1);
        __m512i r16_23 = _mm512_packs_epi32(r2, r3);

        r16_01 = _mm512_permutexvar_epi64(perm64, r16_01);
        r16_23 = _mm512_permutexvar_epi64(perm64, r16_23);

        r16_01 = _mm512_srai_epi16(r16_01, 8);
        r16_23 = _mm512_srai_epi16(r16_23, 8);

        __m512i r8 = _mm512_packs_epi16(r16_01, r16_23);
        r8 = _mm512_permutexvar_epi64(perm64, r8);

        _mm512_store_si512((__m512i*)iBufferPtr, r8);
        iBufferPtr += 64;
    }

    number = sixtyfourthPoints * 64;
    volk_16ic_deinterleave_real_8i_generic(
        iBufferPtr, (const lv_16sc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_16ic_deinterleave_real_8i_a_H */
