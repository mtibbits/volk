/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_s32f_interleave_16ic
 *
 * \b Overview
 *
 * Interleaves two float input vectors — the inphase (real) part and the
 * quadrature (imaginary) part — scales each sample by a scalar value, and
 * converts to 16-bit complex integers:
 * complexVector[i] = (int16_t)rintf(iBuffer[i] * scalar) + j*(int16_t)rintf(qBuffer[i] * scalar).
 *
 * This kernel is commonly used in transmit signal paths where baseband I/Q
 * samples in floating-point must be quantized and packed into fixed-point
 * complex format for a DAC or digital upconverter. The scalar controls the
 * full-scale mapping into the 16-bit range.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_s32f_interleave_16ic(lv_16sc_t* complexVector, const float* iBuffer,
 * const float* qBuffer, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li iBuffer: Input vector of float samples for the inphase (real) part.
 * \li qBuffer: Input vector of float samples for the quadrature (imaginary) part.
 * \li scalar: The scalar value used to scale samples before converting to 16-bit integers.
 * \li num_points: The number of samples in each input vector.
 *
 * \b Outputs
 * \li complexVector: The output vector of 16-bit complex integers (lv_16sc_t).
 *
 * \b Example
 * Scale constant I/Q values and verify the interleaved complex output.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* iBuffer = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* qBuffer = (float*)volk_malloc(sizeof(float) * N, alignment);
 * lv_16sc_t* out = (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     iBuffer[i] = 3.0f;
 *     qBuffer[i] = 4.0f;
 * }
 * float scalar = 10.0f;
 *
 * // Expected: real = rintf(3.0 * 10.0) = 30, imag = rintf(4.0 * 10.0) = 40
 * volk_32f_x2_s32f_interleave_16ic(out, iBuffer, qBuffer, scalar, N);
 *
 * printf("Expected: 30 + 40j\n");
 * printf("Result:   %d + %dj\n", lv_creal(out[0]), lv_cimag(out[0]));
 *
 * volk_free(iBuffer);
 * volk_free(qBuffer);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_s32f_interleave_16ic_u_H
#define INCLUDED_volk_32f_x2_s32f_interleave_16ic_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_s32f_interleave_16ic_generic(lv_16sc_t* complexVector,
                                                            const float* iBuffer,
                                                            const float* qBuffer,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    int16_t* complexVectorPtr = (int16_t*)complexVector;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_sse(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m128 vScalar = _mm_set_ps1(scalar);

    const unsigned int quarterPoints = num_points / 4;

    __m128 iValue, qValue, cplxValue;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        iValue = _mm_loadu_ps(iBufferPtr);
        qValue = _mm_loadu_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue = _mm_unpacklo_ps(iValue, qValue);
        cplxValue = _mm_mul_ps(cplxValue, vScalar);

        _mm_store_ps(floatBuffer, cplxValue);

        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[3]);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue = _mm_unpackhi_ps(iValue, qValue);
        cplxValue = _mm_mul_ps(cplxValue, vScalar);

        _mm_store_ps(floatBuffer, cplxValue);

        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[3]);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_sse2(lv_16sc_t* complexVector,
                                                            const float* iBuffer,
                                                            const float* qBuffer,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m128 vScalar = _mm_set_ps1(scalar);

    const unsigned int quarterPoints = num_points / 4;

    __m128 iValue, qValue, cplxValue1, cplxValue2;
    __m128i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < quarterPoints; number++) {
        iValue = _mm_loadu_ps(iBufferPtr);
        qValue = _mm_loadu_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm_cvtps_epi32(cplxValue1);
        intValue2 = _mm_cvtps_epi32(cplxValue2);

        intValue1 = _mm_packs_epi32(intValue1, intValue2);

        _mm_storeu_si128((__m128i*)complexVectorPtr, intValue1);
        complexVectorPtr += 8;

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_avx(lv_16sc_t* complexVector,
                                                          const float* iBuffer,
                                                          const float* qBuffer,
                                                          const float scalar,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const __m256 vScalar = _mm256_set1_ps(scalar);
    const unsigned int eighthPoints = num_points / 8;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        __m256 iVal = _mm256_loadu_ps(iBufferPtr);
        __m256 qVal = _mm256_loadu_ps(qBufferPtr);

        iVal = _mm256_mul_ps(iVal, vScalar);
        qVal = _mm256_mul_ps(qVal, vScalar);

        /* Split to 128-bit halves for integer ops (AVX has no 256-bit integer ops) */
        __m128 iLo = _mm256_castps256_ps128(iVal);
        __m128 iHi = _mm256_extractf128_ps(iVal, 1);
        __m128 qLo = _mm256_castps256_ps128(qVal);
        __m128 qHi = _mm256_extractf128_ps(qVal, 1);

        /* Convert float to int32 */
        __m128i iIntLo = _mm_cvtps_epi32(iLo);
        __m128i iIntHi = _mm_cvtps_epi32(iHi);
        __m128i qIntLo = _mm_cvtps_epi32(qLo);
        __m128i qIntHi = _mm_cvtps_epi32(qHi);

        /* Interleave I and Q at int32 level, then pack to int16.
         * unpacklo: [I0,Q0,I1,Q1], unpackhi: [I2,Q2,I3,Q3]
         * packs_epi32: [I0,Q0,I1,Q1,I2,Q2,I3,Q3] as int16 */
        __m128i intlvLo_lo = _mm_unpacklo_epi32(iIntLo, qIntLo);
        __m128i intlvLo_hi = _mm_unpackhi_epi32(iIntLo, qIntLo);
        __m128i packedLo = _mm_packs_epi32(intlvLo_lo, intlvLo_hi);

        __m128i intlvHi_lo = _mm_unpacklo_epi32(iIntHi, qIntHi);
        __m128i intlvHi_hi = _mm_unpackhi_epi32(iIntHi, qIntHi);
        __m128i packedHi = _mm_packs_epi32(intlvHi_lo, intlvHi_hi);

        _mm_storeu_si128((__m128i*)complexVectorPtr, packedLo);
        _mm_storeu_si128((__m128i*)(complexVectorPtr + 8), packedHi);

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_avx_fma(lv_16sc_t* complexVector,
                                                               const float* iBuffer,
                                                               const float* qBuffer,
                                                               const float scalar,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const __m256 vScalar = _mm256_set1_ps(scalar);
    const unsigned int eighthPoints = num_points / 8;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        __m256 iVal = _mm256_loadu_ps(iBufferPtr);
        __m256 qVal = _mm256_loadu_ps(qBufferPtr);

        iVal = _mm256_mul_ps(iVal, vScalar);
        qVal = _mm256_mul_ps(qVal, vScalar);

        /* Split to 128-bit halves for integer ops (AVX has no 256-bit integer ops) */
        __m128 iLo = _mm256_castps256_ps128(iVal);
        __m128 iHi = _mm256_extractf128_ps(iVal, 1);
        __m128 qLo = _mm256_castps256_ps128(qVal);
        __m128 qHi = _mm256_extractf128_ps(qVal, 1);

        /* Convert float to int32 */
        __m128i iIntLo = _mm_cvtps_epi32(iLo);
        __m128i iIntHi = _mm_cvtps_epi32(iHi);
        __m128i qIntLo = _mm_cvtps_epi32(qLo);
        __m128i qIntHi = _mm_cvtps_epi32(qHi);

        /* Interleave I and Q at int32 level, then pack to int16.
         * unpacklo: [I0,Q0,I1,Q1], unpackhi: [I2,Q2,I3,Q3]
         * packs_epi32: [I0,Q0,I1,Q1,I2,Q2,I3,Q3] as int16 */
        __m128i intlvLo_lo = _mm_unpacklo_epi32(iIntLo, qIntLo);
        __m128i intlvLo_hi = _mm_unpackhi_epi32(iIntLo, qIntLo);
        __m128i packedLo = _mm_packs_epi32(intlvLo_lo, intlvLo_hi);

        __m128i intlvHi_lo = _mm_unpacklo_epi32(iIntHi, qIntHi);
        __m128i intlvHi_hi = _mm_unpackhi_epi32(iIntHi, qIntHi);
        __m128i packedHi = _mm_packs_epi32(intlvHi_lo, intlvHi_hi);

        _mm_storeu_si128((__m128i*)complexVectorPtr, packedLo);
        _mm_storeu_si128((__m128i*)(complexVectorPtr + 8), packedHi);

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_avx2(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m256 vScalar = _mm256_set1_ps(scalar);

    const unsigned int eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2;
    __m256i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        iValue = _mm256_loadu_ps(iBufferPtr);
        qValue = _mm256_loadu_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm256_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm256_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm256_cvtps_epi32(cplxValue1);
        intValue2 = _mm256_cvtps_epi32(cplxValue2);

        intValue1 = _mm256_packs_epi32(intValue1, intValue2);

        _mm256_storeu_si256((__m256i*)complexVectorPtr, intValue1);
        complexVectorPtr += 16;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_AVX2 */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_avx2_fma(lv_16sc_t* complexVector,
                                                                const float* iBuffer,
                                                                const float* qBuffer,
                                                                const float scalar,
                                                                unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m256 vScalar = _mm256_set1_ps(scalar);

    const unsigned int eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2;
    __m256i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        iValue = _mm256_loadu_ps(iBufferPtr);
        qValue = _mm256_loadu_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm256_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm256_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm256_cvtps_epi32(cplxValue1);
        intValue2 = _mm256_cvtps_epi32(cplxValue2);

        intValue1 = _mm256_packs_epi32(intValue1, intValue2);

        _mm256_storeu_si256((__m256i*)complexVectorPtr, intValue1);
        complexVectorPtr += 16;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_avx512f(
    lv_16sc_t* complexVector,
    const float* iBuffer,
    const float* qBuffer,
    const float scalar,
    unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m512 vScalar = _mm512_set1_ps(scalar);

    const unsigned int sixteenthPoints = num_points / 16;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < sixteenthPoints; number++) {
        __m512 iValue = _mm512_loadu_ps(iBufferPtr);
        __m512 qValue = _mm512_loadu_ps(qBufferPtr);

        iValue = _mm512_mul_ps(iValue, vScalar);
        qValue = _mm512_mul_ps(qValue, vScalar);

        __m512i iInt = _mm512_cvtps_epi32(iValue);
        __m512i qInt = _mm512_cvtps_epi32(qValue);

        // Saturating narrow int32 -> int16 (AVX-512F: vpmovsdw)
        __m256i iShort = _mm512_cvtsepi32_epi16(iInt);
        __m256i qShort = _mm512_cvtsepi32_epi16(qInt);

        // Interleave I and Q int16 values (per 128-bit lane)
        __m256i lo = _mm256_unpacklo_epi16(iShort, qShort);
        __m256i hi = _mm256_unpackhi_epi16(iShort, qShort);

        // Fix cross-lane ordering
        __m256i out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(lo, hi, 0x31);

        _mm256_storeu_si256((__m256i*)complexVectorPtr, out0);
        _mm256_storeu_si256((__m256i*)(complexVectorPtr + 16), out1);

        complexVectorPtr += 32;
        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_avx512bw(
    lv_16sc_t* complexVector,
    const float* iBuffer,
    const float* qBuffer,
    const float scalar,
    unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const __m512 vScalar = _mm512_set1_ps(scalar);
    const unsigned int thirtySecondPoints = num_points / 32;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    // Permute indices to interleave 32 I and 32 Q int16 values into two __m512i.
    // I_packed holds I[0..31], Q_packed holds Q[0..31].
    // In _mm512_permutex2var_epi16: indices 0..31 select from first src, 32..63 from second.
    // out_lo = [I0,Q0,I1,Q1,...,I15,Q15], out_hi = [I16,Q16,...,I31,Q31]
    const __m512i interleave_lo = _mm512_set_epi16(
        47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41,  9, 40,  8,
        39,  7, 38,  6, 37,  5, 36,  4, 35,  3, 34,  2, 33,  1, 32,  0);
    const __m512i interleave_hi = _mm512_set_epi16(
        63, 31, 62, 30, 61, 29, 60, 28, 59, 27, 58, 26, 57, 25, 56, 24,
        55, 23, 54, 22, 53, 21, 52, 20, 51, 19, 50, 18, 49, 17, 48, 16);

    for (; number < thirtySecondPoints; number++) {
        __m512 iVal0 = _mm512_loadu_ps(iBufferPtr);
        __m512 iVal1 = _mm512_loadu_ps(iBufferPtr + 16);
        __m512 qVal0 = _mm512_loadu_ps(qBufferPtr);
        __m512 qVal1 = _mm512_loadu_ps(qBufferPtr + 16);

        iVal0 = _mm512_mul_ps(iVal0, vScalar);
        iVal1 = _mm512_mul_ps(iVal1, vScalar);
        qVal0 = _mm512_mul_ps(qVal0, vScalar);
        qVal1 = _mm512_mul_ps(qVal1, vScalar);

        // Convert float to int32
        __m512i iInt0 = _mm512_cvtps_epi32(iVal0);
        __m512i iInt1 = _mm512_cvtps_epi32(iVal1);
        __m512i qInt0 = _mm512_cvtps_epi32(qVal0);
        __m512i qInt1 = _mm512_cvtps_epi32(qVal1);

        // Saturating narrow int32 -> int16, then fix lane interleaving
        // _mm512_packs_epi32(a, b) packs within 128-bit lanes:
        // result = [a[0..3]->i16, b[0..3]->i16 | a[4..7]->i16, b[4..7]->i16 | ...]
        // fix_idx reorders so result is [a[0..15]->i16, b[0..15]->i16]
        const __m512i fix_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
        __m512i I_packed = _mm512_permutexvar_epi64(
            fix_idx, _mm512_packs_epi32(iInt0, iInt1));
        __m512i Q_packed = _mm512_permutexvar_epi64(
            fix_idx, _mm512_packs_epi32(qInt0, qInt1));

        // Interleave I and Q at int16 granularity
        __m512i out_lo = _mm512_permutex2var_epi16(I_packed, interleave_lo, Q_packed);
        __m512i out_hi = _mm512_permutex2var_epi16(I_packed, interleave_hi, Q_packed);

        _mm512_storeu_si512((__m512i*)complexVectorPtr, out_lo);
        _mm512_storeu_si512((__m512i*)(complexVectorPtr + 32), out_hi);

        complexVectorPtr += 64;
        iBufferPtr += 32;
        qBufferPtr += 32;
    }

    number = thirtySecondPoints * 32;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_s32f_interleave_16ic_neon(lv_16sc_t* complexVector,
                                                         const float* iBuffer,
                                                         const float* qBuffer,
                                                         const float scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;
    int16_t* complexVectorPtr = (int16_t*)complexVector;

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t magic = vdupq_n_f32(8388608.0f); // 2^23 for round-to-nearest-even
    float32x4_t zero = vdupq_n_f32(0.0f);

    for (; number < quarter_points; number++) {
        float32x4_t iValue = vld1q_f32(iBufferPtr);
        float32x4_t qValue = vld1q_f32(qBufferPtr);

        iValue = vmulq_f32(iValue, vScalar);
        qValue = vmulq_f32(qValue, vScalar);

        // Round to nearest even using magic number trick (matches rintf behavior)
        // For |x| < 2^23: adding then subtracting 2^23 forces IEEE 754 round-to-nearest-even
        uint32x4_t iNeg = vcltq_f32(iValue, zero);
        uint32x4_t qNeg = vcltq_f32(qValue, zero);
        float32x4_t iPosRound = vsubq_f32(vaddq_f32(iValue, magic), magic);
        float32x4_t iNegRound = vaddq_f32(vsubq_f32(iValue, magic), magic);
        float32x4_t qPosRound = vsubq_f32(vaddq_f32(qValue, magic), magic);
        float32x4_t qNegRound = vaddq_f32(vsubq_f32(qValue, magic), magic);
        iValue = vbslq_f32(iNeg, iNegRound, iPosRound);
        qValue = vbslq_f32(qNeg, qNegRound, qPosRound);

        int32x4_t iInt = vcvtq_s32_f32(iValue);
        int32x4_t qInt = vcvtq_s32_f32(qValue);

        int16x4_t iShort = vqmovn_s32(iInt);
        int16x4_t qShort = vqmovn_s32(qInt);

        int16x4x2_t interleaved;
        interleaved.val[0] = iShort;
        interleaved.val[1] = qShort;
        vst2_s16(complexVectorPtr, interleaved);

        complexVectorPtr += 8;
        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarter_points * 4;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_s32f_interleave_16ic_neonv8(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;
    int16_t* complexVectorPtr = (int16_t*)complexVector;

    float32x4_t vScalar = vdupq_n_f32(scalar);

    for (; number < eighth_points; number++) {
        float32x4_t iValue0 = vld1q_f32(iBufferPtr);
        float32x4_t iValue1 = vld1q_f32(iBufferPtr + 4);
        float32x4_t qValue0 = vld1q_f32(qBufferPtr);
        float32x4_t qValue1 = vld1q_f32(qBufferPtr + 4);
        __VOLK_PREFETCH(iBufferPtr + 8);
        __VOLK_PREFETCH(qBufferPtr + 8);

        iValue0 = vmulq_f32(iValue0, vScalar);
        iValue1 = vmulq_f32(iValue1, vScalar);
        qValue0 = vmulq_f32(qValue0, vScalar);
        qValue1 = vmulq_f32(qValue1, vScalar);

        int32x4_t iInt0 = vcvtnq_s32_f32(iValue0);
        int32x4_t iInt1 = vcvtnq_s32_f32(iValue1);
        int32x4_t qInt0 = vcvtnq_s32_f32(qValue0);
        int32x4_t qInt1 = vcvtnq_s32_f32(qValue1);

        int16x4_t iShort0 = vqmovn_s32(iInt0);
        int16x4_t iShort1 = vqmovn_s32(iInt1);
        int16x4_t qShort0 = vqmovn_s32(qInt0);
        int16x4_t qShort1 = vqmovn_s32(qInt1);

        int16x4x2_t interleaved0, interleaved1;
        interleaved0.val[0] = iShort0;
        interleaved0.val[1] = qShort0;
        interleaved1.val[0] = iShort1;
        interleaved1.val[1] = qShort1;

        vst2_s16(complexVectorPtr, interleaved0);
        vst2_s16(complexVectorPtr + 8, interleaved1);

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighth_points * 8;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_x2_s32f_interleave_16ic_rvv(lv_16sc_t* complexVector,
                                                        const float* iBuffer,
                                                        const float* qBuffer,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    uint32_t* out = (uint32_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, out += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vrf = __riscv_vle32_v_f32m8(iBuffer, vl);
        vfloat32m8_t vif = __riscv_vle32_v_f32m8(qBuffer, vl);
        vint16m4_t vri = __riscv_vfncvt_x(__riscv_vfmul(vrf, scalar, vl), vl);
        vint16m4_t vii = __riscv_vfncvt_x(__riscv_vfmul(vif, scalar, vl), vl);
        vuint16m4_t vr = __riscv_vreinterpret_u16m4(vri);
        vuint16m4_t vi = __riscv_vreinterpret_u16m4(vii);
        vuint32m8_t vc = __riscv_vwmaccu(__riscv_vwaddu_vv(vr, vi, vl), 0xFFFF, vi, vl);
        __riscv_vse32(out, vc, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32f_x2_s32f_interleave_16ic_rvvseg(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vrf = __riscv_vle32_v_f32m8(iBuffer, vl);
        vfloat32m8_t vif = __riscv_vle32_v_f32m8(qBuffer, vl);
        vint16m4_t vri = __riscv_vfncvt_x(__riscv_vfmul(vrf, scalar, vl), vl);
        vint16m4_t vii = __riscv_vfncvt_x(__riscv_vfmul(vif, scalar, vl), vl);
        __riscv_vsseg2e16(
            (int16_t*)complexVector, __riscv_vcreate_v_i16m4x2(vri, vii), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32f_x2_s32f_interleave_16ic_u_H */

#ifndef INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H
#define INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_sse(lv_16sc_t* complexVector,
                                                          const float* iBuffer,
                                                          const float* qBuffer,
                                                          const float scalar,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m128 vScalar = _mm_set_ps1(scalar);

    const unsigned int quarterPoints = num_points / 4;

    __m128 iValue, qValue, cplxValue;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        iValue = _mm_load_ps(iBufferPtr);
        qValue = _mm_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue = _mm_unpacklo_ps(iValue, qValue);
        cplxValue = _mm_mul_ps(cplxValue, vScalar);

        _mm_store_ps(floatBuffer, cplxValue);

        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[3]);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue = _mm_unpackhi_ps(iValue, qValue);
        cplxValue = _mm_mul_ps(cplxValue, vScalar);

        _mm_store_ps(floatBuffer, cplxValue);

        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[3]);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_sse2(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m128 vScalar = _mm_set_ps1(scalar);

    const unsigned int quarterPoints = num_points / 4;

    __m128 iValue, qValue, cplxValue1, cplxValue2;
    __m128i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < quarterPoints; number++) {
        iValue = _mm_load_ps(iBufferPtr);
        qValue = _mm_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm_cvtps_epi32(cplxValue1);
        intValue2 = _mm_cvtps_epi32(cplxValue2);

        intValue1 = _mm_packs_epi32(intValue1, intValue2);

        _mm_store_si128((__m128i*)complexVectorPtr, intValue1);
        complexVectorPtr += 8;

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_avx(lv_16sc_t* complexVector,
                                                          const float* iBuffer,
                                                          const float* qBuffer,
                                                          const float scalar,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const __m256 vScalar = _mm256_set1_ps(scalar);
    const unsigned int eighthPoints = num_points / 8;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        __m256 iVal = _mm256_load_ps(iBufferPtr);
        __m256 qVal = _mm256_load_ps(qBufferPtr);

        iVal = _mm256_mul_ps(iVal, vScalar);
        qVal = _mm256_mul_ps(qVal, vScalar);

        /* Split to 128-bit halves for integer ops (AVX has no 256-bit integer ops) */
        __m128 iLo = _mm256_castps256_ps128(iVal);
        __m128 iHi = _mm256_extractf128_ps(iVal, 1);
        __m128 qLo = _mm256_castps256_ps128(qVal);
        __m128 qHi = _mm256_extractf128_ps(qVal, 1);

        /* Convert float to int32 */
        __m128i iIntLo = _mm_cvtps_epi32(iLo);
        __m128i iIntHi = _mm_cvtps_epi32(iHi);
        __m128i qIntLo = _mm_cvtps_epi32(qLo);
        __m128i qIntHi = _mm_cvtps_epi32(qHi);

        /* Interleave I and Q at int32 level, then pack to int16.
         * unpacklo: [I0,Q0,I1,Q1], unpackhi: [I2,Q2,I3,Q3]
         * packs_epi32: [I0,Q0,I1,Q1,I2,Q2,I3,Q3] as int16 */
        __m128i intlvLo_lo = _mm_unpacklo_epi32(iIntLo, qIntLo);
        __m128i intlvLo_hi = _mm_unpackhi_epi32(iIntLo, qIntLo);
        __m128i packedLo = _mm_packs_epi32(intlvLo_lo, intlvLo_hi);

        __m128i intlvHi_lo = _mm_unpacklo_epi32(iIntHi, qIntHi);
        __m128i intlvHi_hi = _mm_unpackhi_epi32(iIntHi, qIntHi);
        __m128i packedHi = _mm_packs_epi32(intlvHi_lo, intlvHi_hi);

        _mm_store_si128((__m128i*)complexVectorPtr, packedLo);
        _mm_store_si128((__m128i*)(complexVectorPtr + 8), packedHi);

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_avx_fma(lv_16sc_t* complexVector,
                                                               const float* iBuffer,
                                                               const float* qBuffer,
                                                               const float scalar,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const __m256 vScalar = _mm256_set1_ps(scalar);
    const unsigned int eighthPoints = num_points / 8;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        __m256 iVal = _mm256_load_ps(iBufferPtr);
        __m256 qVal = _mm256_load_ps(qBufferPtr);

        iVal = _mm256_mul_ps(iVal, vScalar);
        qVal = _mm256_mul_ps(qVal, vScalar);

        /* Split to 128-bit halves for integer ops (AVX has no 256-bit integer ops) */
        __m128 iLo = _mm256_castps256_ps128(iVal);
        __m128 iHi = _mm256_extractf128_ps(iVal, 1);
        __m128 qLo = _mm256_castps256_ps128(qVal);
        __m128 qHi = _mm256_extractf128_ps(qVal, 1);

        /* Convert float to int32 */
        __m128i iIntLo = _mm_cvtps_epi32(iLo);
        __m128i iIntHi = _mm_cvtps_epi32(iHi);
        __m128i qIntLo = _mm_cvtps_epi32(qLo);
        __m128i qIntHi = _mm_cvtps_epi32(qHi);

        /* Interleave I and Q at int32 level, then pack to int16.
         * unpacklo: [I0,Q0,I1,Q1], unpackhi: [I2,Q2,I3,Q3]
         * packs_epi32: [I0,Q0,I1,Q1,I2,Q2,I3,Q3] as int16 */
        __m128i intlvLo_lo = _mm_unpacklo_epi32(iIntLo, qIntLo);
        __m128i intlvLo_hi = _mm_unpackhi_epi32(iIntLo, qIntLo);
        __m128i packedLo = _mm_packs_epi32(intlvLo_lo, intlvLo_hi);

        __m128i intlvHi_lo = _mm_unpacklo_epi32(iIntHi, qIntHi);
        __m128i intlvHi_hi = _mm_unpackhi_epi32(iIntHi, qIntHi);
        __m128i packedHi = _mm_packs_epi32(intlvHi_lo, intlvHi_hi);

        _mm_store_si128((__m128i*)complexVectorPtr, packedLo);
        _mm_store_si128((__m128i*)(complexVectorPtr + 8), packedHi);

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_avx2(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m256 vScalar = _mm256_set1_ps(scalar);

    const unsigned int eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2;
    __m256i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        iValue = _mm256_load_ps(iBufferPtr);
        qValue = _mm256_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm256_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm256_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm256_cvtps_epi32(cplxValue1);
        intValue2 = _mm256_cvtps_epi32(cplxValue2);

        intValue1 = _mm256_packs_epi32(intValue1, intValue2);

        _mm256_store_si256((__m256i*)complexVectorPtr, intValue1);
        complexVectorPtr += 16;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_AVX2 */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_avx2_fma(lv_16sc_t* complexVector,
                                                                const float* iBuffer,
                                                                const float* qBuffer,
                                                                const float scalar,
                                                                unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m256 vScalar = _mm256_set1_ps(scalar);

    const unsigned int eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2;
    __m256i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        iValue = _mm256_load_ps(iBufferPtr);
        qValue = _mm256_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm256_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm256_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm256_cvtps_epi32(cplxValue1);
        intValue2 = _mm256_cvtps_epi32(cplxValue2);

        intValue1 = _mm256_packs_epi32(intValue1, intValue2);

        _mm256_store_si256((__m256i*)complexVectorPtr, intValue1);
        complexVectorPtr += 16;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_avx512f(
    lv_16sc_t* complexVector,
    const float* iBuffer,
    const float* qBuffer,
    const float scalar,
    unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m512 vScalar = _mm512_set1_ps(scalar);

    const unsigned int sixteenthPoints = num_points / 16;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < sixteenthPoints; number++) {
        __m512 iValue = _mm512_load_ps(iBufferPtr);
        __m512 qValue = _mm512_load_ps(qBufferPtr);

        iValue = _mm512_mul_ps(iValue, vScalar);
        qValue = _mm512_mul_ps(qValue, vScalar);

        __m512i iInt = _mm512_cvtps_epi32(iValue);
        __m512i qInt = _mm512_cvtps_epi32(qValue);

        // Saturating narrow int32 -> int16 (AVX-512F: vpmovsdw)
        __m256i iShort = _mm512_cvtsepi32_epi16(iInt);
        __m256i qShort = _mm512_cvtsepi32_epi16(qInt);

        // Interleave I and Q int16 values (per 128-bit lane)
        __m256i lo = _mm256_unpacklo_epi16(iShort, qShort);
        __m256i hi = _mm256_unpackhi_epi16(iShort, qShort);

        // Fix cross-lane ordering
        __m256i out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        __m256i out1 = _mm256_permute2x128_si256(lo, hi, 0x31);

        _mm256_store_si256((__m256i*)complexVectorPtr, out0);
        _mm256_store_si256((__m256i*)(complexVectorPtr + 16), out1);

        complexVectorPtr += 32;
        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_avx512bw(
    lv_16sc_t* complexVector,
    const float* iBuffer,
    const float* qBuffer,
    const float scalar,
    unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const __m512 vScalar = _mm512_set1_ps(scalar);
    const unsigned int thirtySecondPoints = num_points / 32;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    const __m512i interleave_lo = _mm512_set_epi16(
        47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41,  9, 40,  8,
        39,  7, 38,  6, 37,  5, 36,  4, 35,  3, 34,  2, 33,  1, 32,  0);
    const __m512i interleave_hi = _mm512_set_epi16(
        63, 31, 62, 30, 61, 29, 60, 28, 59, 27, 58, 26, 57, 25, 56, 24,
        55, 23, 54, 22, 53, 21, 52, 20, 51, 19, 50, 18, 49, 17, 48, 16);

    for (; number < thirtySecondPoints; number++) {
        __m512 iVal0 = _mm512_load_ps(iBufferPtr);
        __m512 iVal1 = _mm512_load_ps(iBufferPtr + 16);
        __m512 qVal0 = _mm512_load_ps(qBufferPtr);
        __m512 qVal1 = _mm512_load_ps(qBufferPtr + 16);

        iVal0 = _mm512_mul_ps(iVal0, vScalar);
        iVal1 = _mm512_mul_ps(iVal1, vScalar);
        qVal0 = _mm512_mul_ps(qVal0, vScalar);
        qVal1 = _mm512_mul_ps(qVal1, vScalar);

        __m512i iInt0 = _mm512_cvtps_epi32(iVal0);
        __m512i iInt1 = _mm512_cvtps_epi32(iVal1);
        __m512i qInt0 = _mm512_cvtps_epi32(qVal0);
        __m512i qInt1 = _mm512_cvtps_epi32(qVal1);

        const __m512i fix_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
        __m512i I_packed = _mm512_permutexvar_epi64(
            fix_idx, _mm512_packs_epi32(iInt0, iInt1));
        __m512i Q_packed = _mm512_permutexvar_epi64(
            fix_idx, _mm512_packs_epi32(qInt0, qInt1));

        __m512i out_lo = _mm512_permutex2var_epi16(I_packed, interleave_lo, Q_packed);
        __m512i out_hi = _mm512_permutex2var_epi16(I_packed, interleave_hi, Q_packed);

        _mm512_store_si512((__m512i*)complexVectorPtr, out_lo);
        _mm512_store_si512((__m512i*)(complexVectorPtr + 32), out_hi);

        complexVectorPtr += 64;
        iBufferPtr += 32;
        qBufferPtr += 32;
    }

    number = thirtySecondPoints * 32;
    volk_32f_x2_s32f_interleave_16ic_generic(
        &complexVector[number], iBufferPtr, qBufferPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */


#endif /* INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H */
