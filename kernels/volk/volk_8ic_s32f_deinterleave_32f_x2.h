/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8ic_s32f_deinterleave_32f_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex 8-bit char vector into separate I and Q
 * float vectors, dividing each component by the scalar factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8ic_s32f_deinterleave_32f_x2(float* iBuffer, float* qBuffer, const
 * lv_8sc_t* complexVector, const float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of interleaved 8-bit I/Q pairs (lv_8sc_t).
 * \li scalar: The scalar divisor applied to each output value.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The in-phase (I) output vector of floats.
 * \li qBuffer: The quadrature (Q) output vector of floats.
 *
 * \b Example
 * Deinterleave complex 8-bit samples and scale to normalized floats.
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
 *     float* iBuffer = (float*)volk_malloc(sizeof(float) * N, alignment);
 *     float* qBuffer = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 *     // Simulate complex 8-bit samples: (I, Q) pairs
 *     complexVector[0] = lv_cmake((int8_t)127, (int8_t)-128);
 *     complexVector[1] = lv_cmake((int8_t)50, (int8_t)-50);
 *     complexVector[2] = lv_cmake((int8_t)0, (int8_t)100);
 *     complexVector[3] = lv_cmake((int8_t)-30, (int8_t)30);
 *
 *     // Divide by 128 to normalize to approximately [-1.0, 1.0]
 *     float scalar = 128.0f;
 *     volk_8ic_s32f_deinterleave_32f_x2(iBuffer, qBuffer, complexVector, scalar, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *         printf("Sample %u: I = %+.4f  Q = %+.4f\n", i, iBuffer[i], qBuffer[i]);
 *     }
 *
 *     volk_free(complexVector);
 *     volk_free(iBuffer);
 *     volk_free(qBuffer);
 * }
 * \endcode
 */

#ifndef INCLUDED_volk_8ic_s32f_deinterleave_32f_x2_u_H
#define INCLUDED_volk_8ic_s32f_deinterleave_32f_x2_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC

static inline void
volk_8ic_s32f_deinterleave_32f_x2_generic(float* iBuffer,
                                          float* qBuffer,
                                          const lv_8sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;
    unsigned int number;
    const float invScalar = 1.0f / scalar;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * invScalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) * invScalar;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_u_sse(float* iBuffer,
                                                            float* qBuffer,
                                                            const lv_8sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    __m128 cplxValue1, cplxValue2, iValue, qValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[8];

    for (; number < quarterPoints; number++) {
        floatBuffer[0] = (float)(complexVectorPtr[0]);
        floatBuffer[1] = (float)(complexVectorPtr[1]);
        floatBuffer[2] = (float)(complexVectorPtr[2]);
        floatBuffer[3] = (float)(complexVectorPtr[3]);

        floatBuffer[4] = (float)(complexVectorPtr[4]);
        floatBuffer[5] = (float)(complexVectorPtr[5]);
        floatBuffer[6] = (float)(complexVectorPtr[6]);
        floatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&floatBuffer[0]);
        cplxValue2 = _mm_load_ps(&floatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        _mm_storeu_ps(iBufferPtr, iValue);
        _mm_storeu_ps(qBufferPtr, qValue);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    volk_8ic_s32f_deinterleave_32f_x2_generic(
        iBufferPtr, qBufferPtr, &complexVector[number], scalar, num_points - number);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_32f_x2_u_sse4_1(float* iBuffer,
                                            float* qBuffer,
                                            const lv_8sc_t* complexVector,
                                            const float scalar,
                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    __m128 iFloatValue, qFloatValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    __m128i complexVal, iIntVal, qIntVal, iComplexVal, qComplexVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m128i iMoveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i qMoveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < eighthPoints; number++) {
        complexVal = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        iComplexVal = _mm_shuffle_epi8(complexVal, iMoveMask);
        qComplexVal = _mm_shuffle_epi8(complexVal, qMoveMask);

        iIntVal = _mm_cvtepi8_epi32(iComplexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);
        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);
        _mm_storeu_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 4;

        iComplexVal = _mm_srli_si128(iComplexVal, 4);

        iIntVal = _mm_cvtepi8_epi32(iComplexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);
        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);
        _mm_storeu_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 4;

        qIntVal = _mm_cvtepi8_epi32(qComplexVal);
        qFloatValue = _mm_cvtepi32_ps(qIntVal);
        qFloatValue = _mm_mul_ps(qFloatValue, invScalar);
        _mm_storeu_ps(qBufferPtr, qFloatValue);
        qBufferPtr += 4;

        qComplexVal = _mm_srli_si128(qComplexVal, 4);

        qIntVal = _mm_cvtepi8_epi32(qComplexVal);
        qFloatValue = _mm_cvtepi32_ps(qIntVal);
        qFloatValue = _mm_mul_ps(qFloatValue, invScalar);
        _mm_storeu_ps(qBufferPtr, qFloatValue);

        qBufferPtr += 4;
    }

    number = eighthPoints * 8;
    volk_8ic_s32f_deinterleave_32f_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_u_avx2(float* iBuffer,
                                                            float* qBuffer,
                                                            const lv_8sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    __m256 iFloatValue, qFloatValue;

    const float iScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    __m256i complexVal, iIntVal, qIntVal;
    __m128i iComplexVal, qComplexVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m256i MoveMask = _mm256_set_epi8(15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0,
                                       15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0);

    for (; number < sixteenthPoints; number++) {
        complexVal = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal = _mm256_shuffle_epi8(complexVal, MoveMask);
        complexVal = _mm256_permute4x64_epi64(complexVal, 0xd8);
        iComplexVal = _mm256_extractf128_si256(complexVal, 0);
        qComplexVal = _mm256_extractf128_si256(complexVal, 1);

        iIntVal = _mm256_cvtepi8_epi32(iComplexVal);
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);
        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);
        _mm256_storeu_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 8;

        qIntVal = _mm256_cvtepi8_epi32(qComplexVal);
        qFloatValue = _mm256_cvtepi32_ps(qIntVal);
        qFloatValue = _mm256_mul_ps(qFloatValue, invScalar);
        _mm256_storeu_ps(qBufferPtr, qFloatValue);
        qBufferPtr += 8;

        complexVal = _mm256_srli_si256(complexVal, 8);
        iComplexVal = _mm256_extractf128_si256(complexVal, 0);
        qComplexVal = _mm256_extractf128_si256(complexVal, 1);

        iIntVal = _mm256_cvtepi8_epi32(iComplexVal);
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);
        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);
        _mm256_storeu_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 8;

        qIntVal = _mm256_cvtepi8_epi32(qComplexVal);
        qFloatValue = _mm256_cvtepi32_ps(qIntVal);
        qFloatValue = _mm256_mul_ps(qFloatValue, invScalar);
        _mm256_storeu_ps(qBufferPtr, qFloatValue);
        qBufferPtr += 8;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_u_avx512f(
    float* iBuffer,
    float* qBuffer,
    const lv_8sc_t* complexVector,
    const float scalar,
    unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    const float iScalar = 1.0f / scalar;
    const __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m256i MoveMask = _mm256_set_epi8(15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0,
                                       15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0);

    for (; number < thirtysecondPoints; number++) {
        /* Load 32 complex int8 samples (64 bytes) in two 256-bit chunks */
        __m256i complexVal0 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        __m256i complexVal1 =
            _mm256_loadu_si256((const __m256i*)(complexVectorPtr + 32));
        complexVectorPtr += 64;

        /* Separate I and Q bytes within each 128-bit lane */
        complexVal0 = _mm256_shuffle_epi8(complexVal0, MoveMask);
        complexVal0 = _mm256_permute4x64_epi64(complexVal0, 0xd8);
        __m128i iBytes0 = _mm256_extracti128_si256(complexVal0, 0);
        __m128i qBytes0 = _mm256_extracti128_si256(complexVal0, 1);

        complexVal1 = _mm256_shuffle_epi8(complexVal1, MoveMask);
        complexVal1 = _mm256_permute4x64_epi64(complexVal1, 0xd8);
        __m128i iBytes1 = _mm256_extracti128_si256(complexVal1, 0);
        __m128i qBytes1 = _mm256_extracti128_si256(complexVal1, 1);

        /* Convert first 16 I values: sign-extend int8 -> int32 -> float */
        __m512i iInt0 = _mm512_cvtepi8_epi32(iBytes0);
        __m512 iFloat0 = _mm512_cvtepi32_ps(iInt0);
        iFloat0 = _mm512_mul_ps(iFloat0, invScalar);
        _mm512_storeu_ps(iBufferPtr, iFloat0);
        iBufferPtr += 16;

        /* Convert next 16 I values */
        __m512i iInt1 = _mm512_cvtepi8_epi32(iBytes1);
        __m512 iFloat1 = _mm512_cvtepi32_ps(iInt1);
        iFloat1 = _mm512_mul_ps(iFloat1, invScalar);
        _mm512_storeu_ps(iBufferPtr, iFloat1);
        iBufferPtr += 16;

        /* Convert first 16 Q values */
        __m512i qInt0 = _mm512_cvtepi8_epi32(qBytes0);
        __m512 qFloat0 = _mm512_cvtepi32_ps(qInt0);
        qFloat0 = _mm512_mul_ps(qFloat0, invScalar);
        _mm512_storeu_ps(qBufferPtr, qFloat0);
        qBufferPtr += 16;

        /* Convert next 16 Q values */
        __m512i qInt1 = _mm512_cvtepi8_epi32(qBytes1);
        __m512 qFloat1 = _mm512_cvtepi32_ps(qInt1);
        qFloat1 = _mm512_mul_ps(qFloat1, invScalar);
        _mm512_storeu_ps(qBufferPtr, qFloat1);
        qBufferPtr += 16;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_s32f_deinterleave_32f_x2_generic(
        iBufferPtr,
        qBufferPtr,
        (const lv_8sc_t*)complexVectorPtr,
        scalar,
        num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX512VBMI
#include <immintrin.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_u_avx512vbmi(
    float* iBuffer,
    float* qBuffer,
    const lv_8sc_t* complexVector,
    const float scalar,
    unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    const float iScalar = 1.0f / scalar;
    const __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    /* Index vectors to extract even bytes (I) and odd bytes (Q) from
     * a single 512-bit register containing 32 complex int8 samples.
     * _mm512_permutexvar_epi8 uses bits [5:0] of each index to select
     * a byte from the source. We place 32 results in the lower 256 bits. */
    const __m512i iIdx = _mm512_set_epi8(
        /* upper 32 bytes: don't care */
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* lower 32 bytes: even byte positions 0,2,4,...,62 */
        62, 60, 58, 56, 54, 52, 50, 48,
        46, 44, 42, 40, 38, 36, 34, 32,
        30, 28, 26, 24, 22, 20, 18, 16,
        14, 12, 10, 8, 6, 4, 2, 0);

    const __m512i qIdx = _mm512_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        63, 61, 59, 57, 55, 53, 51, 49,
        47, 45, 43, 41, 39, 37, 35, 33,
        31, 29, 27, 25, 23, 21, 19, 17,
        15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < thirtysecondPoints; number++) {
        __m512i complexVal = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        /* Extract 32 I bytes and 32 Q bytes via byte-granularity permute */
        __m256i iBytes = _mm512_castsi512_si256(
            _mm512_permutexvar_epi8(iIdx, complexVal));
        __m256i qBytes = _mm512_castsi512_si256(
            _mm512_permutexvar_epi8(qIdx, complexVal));

        /* Convert lower 16 I bytes: int8 -> int32 -> float */
        __m128i iLo = _mm256_castsi256_si128(iBytes);
        __m512i iInt0 = _mm512_cvtepi8_epi32(iLo);
        _mm512_storeu_ps(iBufferPtr,
                         _mm512_mul_ps(_mm512_cvtepi32_ps(iInt0), invScalar));
        iBufferPtr += 16;

        /* Convert upper 16 I bytes */
        __m128i iHi = _mm256_extracti128_si256(iBytes, 1);
        __m512i iInt1 = _mm512_cvtepi8_epi32(iHi);
        _mm512_storeu_ps(iBufferPtr,
                         _mm512_mul_ps(_mm512_cvtepi32_ps(iInt1), invScalar));
        iBufferPtr += 16;

        /* Convert lower 16 Q bytes */
        __m128i qLo = _mm256_castsi256_si128(qBytes);
        __m512i qInt0 = _mm512_cvtepi8_epi32(qLo);
        _mm512_storeu_ps(qBufferPtr,
                         _mm512_mul_ps(_mm512_cvtepi32_ps(qInt0), invScalar));
        qBufferPtr += 16;

        /* Convert upper 16 Q bytes */
        __m128i qHi = _mm256_extracti128_si256(qBytes, 1);
        __m512i qInt1 = _mm512_cvtepi8_epi32(qHi);
        _mm512_storeu_ps(qBufferPtr,
                         _mm512_mul_ps(_mm512_cvtepi32_ps(qInt1), invScalar));
        qBufferPtr += 16;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_s32f_deinterleave_32f_x2_generic(
        iBufferPtr,
        qBufferPtr,
        (const lv_8sc_t*)complexVectorPtr,
        scalar,
        num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_neon(float* iBuffer,
                                                          float* qBuffer,
                                                          const lv_8sc_t* complexVector,
                                                          const float scalar,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < eighth_points; number++) {
        int8x8x2_t input = vld2_s8(complexVectorPtr);
        complexVectorPtr += 16;

        int16x8_t iShort = vmovl_s8(input.val[0]);
        int16x8_t qShort = vmovl_s8(input.val[1]);

        int32x4_t iInt0 = vmovl_s16(vget_low_s16(iShort));
        int32x4_t iInt1 = vmovl_s16(vget_high_s16(iShort));
        int32x4_t qInt0 = vmovl_s16(vget_low_s16(qShort));
        int32x4_t qInt1 = vmovl_s16(vget_high_s16(qShort));

        float32x4_t iFloat0 = vcvtq_f32_s32(iInt0);
        float32x4_t iFloat1 = vcvtq_f32_s32(iInt1);
        float32x4_t qFloat0 = vcvtq_f32_s32(qInt0);
        float32x4_t qFloat1 = vcvtq_f32_s32(qInt1);

        iFloat0 = vmulq_f32(iFloat0, vInvScalar);
        iFloat1 = vmulq_f32(iFloat1, vInvScalar);
        qFloat0 = vmulq_f32(qFloat0, vInvScalar);
        qFloat1 = vmulq_f32(qFloat1, vInvScalar);

        vst1q_f32(iBufferPtr, iFloat0);
        vst1q_f32(iBufferPtr + 4, iFloat1);
        vst1q_f32(qBufferPtr, qFloat0);
        vst1q_f32(qBufferPtr + 4, qFloat1);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * invScalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) * invScalar;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_rvv(float* iBuffer,
                                                         float* qBuffer,
                                                         const lv_8sc_t* complexVector,
                                                         const float scalar,
                                                         unsigned int num_points)
{
    const uint16_t* in = (const uint16_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vuint16m4_t vc = __riscv_vle16_v_u16m4(in, vl);
        vint8m2_t vr = __riscv_vreinterpret_i8m2(__riscv_vnsrl(vc, 0, vl));
        vint8m2_t vi = __riscv_vreinterpret_i8m2(__riscv_vnsrl(vc, 8, vl));
        vfloat32m8_t vrf = __riscv_vfwcvt_f(__riscv_vsext_vf2(vr, vl), vl);
        vfloat32m8_t vif = __riscv_vfwcvt_f(__riscv_vsext_vf2(vi, vl), vl);
        __riscv_vse32(iBuffer, __riscv_vfmul(vrf, 1.0f / scalar, vl), vl);
        __riscv_vse32(qBuffer, __riscv_vfmul(vif, 1.0f / scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_8ic_s32f_deinterleave_32f_x2_u_H */


#ifndef INCLUDED_volk_8ic_s32f_deinterleave_32f_x2_a_H
#define INCLUDED_volk_8ic_s32f_deinterleave_32f_x2_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_a_sse(float* iBuffer,
                                                           float* qBuffer,
                                                           const lv_8sc_t* complexVector,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    __m128 cplxValue1, cplxValue2, iValue, qValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[8];

    for (; number < quarterPoints; number++) {
        floatBuffer[0] = (float)(complexVectorPtr[0]);
        floatBuffer[1] = (float)(complexVectorPtr[1]);
        floatBuffer[2] = (float)(complexVectorPtr[2]);
        floatBuffer[3] = (float)(complexVectorPtr[3]);

        floatBuffer[4] = (float)(complexVectorPtr[4]);
        floatBuffer[5] = (float)(complexVectorPtr[5]);
        floatBuffer[6] = (float)(complexVectorPtr[6]);
        floatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&floatBuffer[0]);
        cplxValue2 = _mm_load_ps(&floatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        _mm_store_ps(iBufferPtr, iValue);
        _mm_store_ps(qBufferPtr, qValue);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    complexVectorPtr = (const int8_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_32f_x2_a_sse4_1(float* iBuffer,
                                           float* qBuffer,
                                           const lv_8sc_t* complexVector,
                                           const float scalar,
                                           unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    __m128 iFloatValue, qFloatValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    __m128i complexVal, iIntVal, qIntVal, iComplexVal, qComplexVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m128i iMoveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i qMoveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < eighthPoints; number++) {
        complexVal = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        iComplexVal = _mm_shuffle_epi8(complexVal, iMoveMask);
        qComplexVal = _mm_shuffle_epi8(complexVal, qMoveMask);

        iIntVal = _mm_cvtepi8_epi32(iComplexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);
        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);
        _mm_store_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 4;

        iComplexVal = _mm_srli_si128(iComplexVal, 4);

        iIntVal = _mm_cvtepi8_epi32(iComplexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);
        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);
        _mm_store_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 4;

        qIntVal = _mm_cvtepi8_epi32(qComplexVal);
        qFloatValue = _mm_cvtepi32_ps(qIntVal);
        qFloatValue = _mm_mul_ps(qFloatValue, invScalar);
        _mm_store_ps(qBufferPtr, qFloatValue);
        qBufferPtr += 4;

        qComplexVal = _mm_srli_si128(qComplexVal, 4);

        qIntVal = _mm_cvtepi8_epi32(qComplexVal);
        qFloatValue = _mm_cvtepi32_ps(qIntVal);
        qFloatValue = _mm_mul_ps(qFloatValue, invScalar);
        _mm_store_ps(qBufferPtr, qFloatValue);

        qBufferPtr += 4;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
    }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_a_avx2(float* iBuffer,
                                                            float* qBuffer,
                                                            const lv_8sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    __m256 iFloatValue, qFloatValue;

    const float iScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    __m256i complexVal, iIntVal, qIntVal, iComplexVal, qComplexVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m256i iMoveMask = _mm256_set_epi8(0x80,
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
    __m256i qMoveMask = _mm256_set_epi8(0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        15,
                                        13,
                                        11,
                                        9,
                                        7,
                                        5,
                                        3,
                                        1,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        0x80,
                                        15,
                                        13,
                                        11,
                                        9,
                                        7,
                                        5,
                                        3,
                                        1);

    for (; number < sixteenthPoints; number++) {
        complexVal = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        iComplexVal = _mm256_shuffle_epi8(complexVal, iMoveMask);
        qComplexVal = _mm256_shuffle_epi8(complexVal, qMoveMask);

        iIntVal = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(iComplexVal));
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);
        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);
        _mm256_store_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 8;

        iComplexVal = _mm256_permute4x64_epi64(iComplexVal, 0b11000110);
        iIntVal = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(iComplexVal));
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);
        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);
        _mm256_store_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 8;

        qIntVal = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(qComplexVal));
        qFloatValue = _mm256_cvtepi32_ps(qIntVal);
        qFloatValue = _mm256_mul_ps(qFloatValue, invScalar);
        _mm256_store_ps(qBufferPtr, qFloatValue);
        qBufferPtr += 8;

        qComplexVal = _mm256_permute4x64_epi64(qComplexVal, 0b11000110);
        qIntVal = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(qComplexVal));
        qFloatValue = _mm256_cvtepi32_ps(qIntVal);
        qFloatValue = _mm256_mul_ps(qFloatValue, invScalar);
        _mm256_store_ps(qBufferPtr, qFloatValue);
        qBufferPtr += 8;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        *qBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_8ic_s32f_deinterleave_32f_x2_a_avx512f(
    float* iBuffer,
    float* qBuffer,
    const lv_8sc_t* complexVector,
    const float scalar,
    unsigned int num_points)
{
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int thirtysecondPoints = num_points / 32;

    const float iScalar = 1.0f / scalar;
    const __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m256i MoveMask = _mm256_set_epi8(15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0,
                                       15,
                                       13,
                                       11,
                                       9,
                                       7,
                                       5,
                                       3,
                                       1,
                                       14,
                                       12,
                                       10,
                                       8,
                                       6,
                                       4,
                                       2,
                                       0);

    for (; number < thirtysecondPoints; number++) {
        /* Load 32 complex int8 samples (64 bytes) in two 256-bit chunks */
        __m256i complexVal0 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        __m256i complexVal1 =
            _mm256_load_si256((const __m256i*)(complexVectorPtr + 32));
        complexVectorPtr += 64;

        /* Separate I and Q bytes within each 128-bit lane */
        complexVal0 = _mm256_shuffle_epi8(complexVal0, MoveMask);
        complexVal0 = _mm256_permute4x64_epi64(complexVal0, 0xd8);
        __m128i iBytes0 = _mm256_extracti128_si256(complexVal0, 0);
        __m128i qBytes0 = _mm256_extracti128_si256(complexVal0, 1);

        complexVal1 = _mm256_shuffle_epi8(complexVal1, MoveMask);
        complexVal1 = _mm256_permute4x64_epi64(complexVal1, 0xd8);
        __m128i iBytes1 = _mm256_extracti128_si256(complexVal1, 0);
        __m128i qBytes1 = _mm256_extracti128_si256(complexVal1, 1);

        /* Convert first 16 I values: sign-extend int8 -> int32 -> float */
        __m512i iInt0 = _mm512_cvtepi8_epi32(iBytes0);
        __m512 iFloat0 = _mm512_cvtepi32_ps(iInt0);
        iFloat0 = _mm512_mul_ps(iFloat0, invScalar);
        _mm512_store_ps(iBufferPtr, iFloat0);
        iBufferPtr += 16;

        /* Convert next 16 I values */
        __m512i iInt1 = _mm512_cvtepi8_epi32(iBytes1);
        __m512 iFloat1 = _mm512_cvtepi32_ps(iInt1);
        iFloat1 = _mm512_mul_ps(iFloat1, invScalar);
        _mm512_store_ps(iBufferPtr, iFloat1);
        iBufferPtr += 16;

        /* Convert first 16 Q values */
        __m512i qInt0 = _mm512_cvtepi8_epi32(qBytes0);
        __m512 qFloat0 = _mm512_cvtepi32_ps(qInt0);
        qFloat0 = _mm512_mul_ps(qFloat0, invScalar);
        _mm512_store_ps(qBufferPtr, qFloat0);
        qBufferPtr += 16;

        /* Convert next 16 Q values */
        __m512i qInt1 = _mm512_cvtepi8_epi32(qBytes1);
        __m512 qFloat1 = _mm512_cvtepi32_ps(qInt1);
        qFloat1 = _mm512_mul_ps(qFloat1, invScalar);
        _mm512_store_ps(qBufferPtr, qFloat1);
        qBufferPtr += 16;
    }

    number = thirtysecondPoints * 32;
    volk_8ic_s32f_deinterleave_32f_x2_generic(
        iBufferPtr,
        qBufferPtr,
        (const lv_8sc_t*)complexVectorPtr,
        scalar,
        num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#endif /* INCLUDED_volk_8ic_s32f_deinterleave_32f_x2_a_H */
