/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8ic_s32f_deinterleave_real_32f
 *
 * \b Overview
 *
 * Deinterleaves the complex 8-bit char vector into just the real (I)
 * vector, converts the samples to floats, and divides the results by
 * the scalar factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8ic_s32f_deinterleave_real_32f(float* iBuffer, const lv_8sc_t*
 * complexVector, const float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li scalar: The scalar value used to divide the floating point results.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 *
 * \b Example
 * Extract and scale the real (I) component from 8-bit complex samples.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main(){
 *     unsigned int N = 8;
 *     float scalar = 127.0f; // normalize to [-1.0, 1.0] range
 *     unsigned int alignment = volk_get_alignment();
 *
 *     // Allocate aligned memory
 *     lv_8sc_t* complexVector = (lv_8sc_t*)volk_malloc(sizeof(lv_8sc_t) * N, alignment);
 *     float* iBuffer = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 *     // Fill with sample complex data (I, Q pairs)
 *     complexVector[0] = lv_cmake((int8_t)127,  (int8_t)0);
 *     complexVector[1] = lv_cmake((int8_t)90,   (int8_t)90);
 *     complexVector[2] = lv_cmake((int8_t)0,    (int8_t)127);
 *     complexVector[3] = lv_cmake((int8_t)-90,  (int8_t)90);
 *     complexVector[4] = lv_cmake((int8_t)-127, (int8_t)0);
 *     complexVector[5] = lv_cmake((int8_t)-90,  (int8_t)-90);
 *     complexVector[6] = lv_cmake((int8_t)0,    (int8_t)-127);
 *     complexVector[7] = lv_cmake((int8_t)90,   (int8_t)-90);
 *
 *     // Deinterleave real part and divide by scalar
 *     volk_8ic_s32f_deinterleave_real_32f(iBuffer, complexVector, scalar, N);
 *
 *     for(unsigned int i = 0; i < N; i++){
 *       printf("I[%u] = %1.4f\n", i, iBuffer[i]);
 *     }
 *
 *     volk_free(complexVector);
 *     volk_free(iBuffer);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_8ic_s32f_deinterleave_real_32f_u_H
#define INCLUDED_volk_8ic_s32f_deinterleave_real_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_8ic_s32f_deinterleave_real_32f_generic(float* iBuffer,
                                            const lv_8sc_t* complexVector,
                                            const float scalar,
                                            unsigned int num_points)
{
    unsigned int number = 0;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    float* iBufferPtr = iBuffer;
    const float invScalar = 1.0f / scalar;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = ((float)(*complexVectorPtr++)) * invScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_u_sse(float* iBuffer,
                                          const lv_8sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    __m128 iValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        floatBuffer[0] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[1] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[2] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[3] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;

        iValue = _mm_load_ps(floatBuffer);

        iValue = _mm_mul_ps(iValue, invScalar);

        _mm_storeu_ps(iBufferPtr, iValue);

        iBufferPtr += 4;
    }

    number = quarterPoints * 4;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_u_sse2(float* iBuffer,
                                            const lv_8sc_t* complexVector,
                                            const float scalar,
                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    for (; number < eighthPoints; number++) {
        __m128i complexVal = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        // Sign-extend low byte of each 16-bit lane (the I component)
        __m128i iVals16 = _mm_srai_epi16(_mm_slli_epi16(complexVal, 8), 8);

        // Sign-extend int16 to int32
        __m128i sign = _mm_srai_epi16(iVals16, 15);
        __m128i iInt0 = _mm_unpacklo_epi16(iVals16, sign);
        __m128i iInt1 = _mm_unpackhi_epi16(iVals16, sign);

        // Convert to float and multiply by inverse scalar
        __m128 iFloat0 = _mm_mul_ps(_mm_cvtepi32_ps(iInt0), invScalar);
        __m128 iFloat1 = _mm_mul_ps(_mm_cvtepi32_ps(iInt1), invScalar);

        _mm_storeu_ps(iBufferPtr, iFloat0);
        _mm_storeu_ps(iBufferPtr + 4, iFloat1);
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_u_sse4_1(float* iBuffer,
                                              const lv_8sc_t* complexVector,
                                              const float scalar,
                                              unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    __m128 iFloatValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    __m128i complexVal, iIntVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);

    for (; number < eighthPoints; number++) {
        complexVal = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal = _mm_shuffle_epi8(complexVal, moveMask);

        iIntVal = _mm_cvtepi8_epi32(complexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);

        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);

        _mm_storeu_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 4;

        complexVal = _mm_srli_si128(complexVal, 4);
        iIntVal = _mm_cvtepi8_epi32(complexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);

        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);

        _mm_storeu_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 4;
    }

    number = eighthPoints * 8;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_u_avx(float* iBuffer,
                                           const lv_8sc_t* complexVector,
                                           const float scalar,
                                           unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float iScalar = 1.0f / scalar;
    const __m256 invScalar = _mm256_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    /* Shuffle bytes: move real (even) bytes of 8 complex pairs to positions 0-7,
     * zero-fill the upper 8 bytes. Real bytes are at positions 0,2,4,6,8,10,12,14. */
    const __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);

    for (; number < eighthPoints; number++) {
        __m128i complexVal = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        /* Gather the 8 real bytes into the low 8 positions */
        complexVal = _mm_shuffle_epi8(complexVal, moveMask);

        /* Sign-extend low 4 bytes (int8) to int32, convert to float */
        __m128i iInt0 = _mm_cvtepi8_epi32(complexVal);
        __m128 iFloat0 = _mm_mul_ps(_mm_cvtepi32_ps(iInt0), _mm256_castps256_ps128(invScalar));

        /* Sign-extend next 4 bytes to int32, convert to float */
        complexVal = _mm_srli_si128(complexVal, 4);
        __m128i iInt1 = _mm_cvtepi8_epi32(complexVal);
        __m128 iFloat1 = _mm_mul_ps(_mm_cvtepi32_ps(iInt1), _mm256_castps256_ps128(invScalar));

        /* Combine two __m128 results into one __m256 and store */
        _mm256_storeu_ps(iBufferPtr, _mm256_set_m128(iFloat1, iFloat0));
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_u_avx2(float* iBuffer,
                                           const lv_8sc_t* complexVector,
                                           const float scalar,
                                           unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    __m256 iFloatValue;

    const float iScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    __m256i complexVal, iIntVal;
    __m128i hcomplexVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

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

    for (; number < sixteenthPoints; number++) {
        complexVal = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);

        hcomplexVal = _mm256_extracti128_si256(complexVal, 0);
        iIntVal = _mm256_cvtepi8_epi32(hcomplexVal);
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);

        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);

        _mm256_storeu_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 8;

        hcomplexVal = _mm256_extracti128_si256(complexVal, 1);
        iIntVal = _mm256_cvtepi8_epi32(hcomplexVal);
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);

        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);

        _mm256_storeu_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 8;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_u_avx512f(float* iBuffer,
                                               const lv_8sc_t* complexVector,
                                               const float scalar,
                                               unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float iScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

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

    for (; number < sixteenthPoints; number++) {
        __m256i complexVal = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);

        // Combine 8 real bytes from each 128-bit lane into one __m128i
        __m128i lo = _mm256_castsi256_si128(complexVal);
        __m128i hi = _mm256_extracti128_si256(complexVal, 1);
        __m128i combined = _mm_unpacklo_epi64(lo, hi);

        // Sign-extend 16 x int8 to 16 x int32, convert to float
        __m512i iIntVal = _mm512_cvtepi8_epi32(combined);
        __m512 iFloatValue = _mm512_cvtepi32_ps(iIntVal);
        iFloatValue = _mm512_mul_ps(iFloatValue, invScalar);

        _mm512_storeu_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512VBMI
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_u_avx512vbmi(float* iBuffer,
                                                   const lv_8sc_t* complexVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int sixtyFourthPoints = num_points / 64;

    const float iScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    /* Index vector: pick byte 0 (I) from each 2-byte complex sample.
     * Indices 0-63 select from src1, 64-127 select from src2. */
    const __m512i iIdx = _mm512_set_epi8(
        126, 124, 122, 120, 118, 116, 114, 112,
        110, 108, 106, 104, 102, 100, 98, 96,
        94, 92, 90, 88, 86, 84, 82, 80,
        78, 76, 74, 72, 70, 68, 66, 64,
        62, 60, 58, 56, 54, 52, 50, 48,
        46, 44, 42, 40, 38, 36, 34, 32,
        30, 28, 26, 24, 22, 20, 18, 16,
        14, 12, 10, 8, 6, 4, 2, 0);

    for (; number < sixtyFourthPoints; number++) {
        __m512i src1 = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;
        __m512i src2 = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        /* Extract 64 I bytes into one register */
        __m512i iBytes = _mm512_permutex2var_epi8(src1, iIdx, src2);

        /* Process in 4 batches of 16: int8 -> int32 -> float */
        __m128i chunk0 = _mm512_castsi512_si128(iBytes);
        __m512i iInt0 = _mm512_cvtepi8_epi32(chunk0);
        __m512 iFloat0 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt0), invScalar);
        _mm512_storeu_ps(iBufferPtr, iFloat0);
        iBufferPtr += 16;

        __m128i chunk1 = _mm512_extracti32x4_epi32(iBytes, 1);
        __m512i iInt1 = _mm512_cvtepi8_epi32(chunk1);
        __m512 iFloat1 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt1), invScalar);
        _mm512_storeu_ps(iBufferPtr, iFloat1);
        iBufferPtr += 16;

        __m128i chunk2 = _mm512_extracti32x4_epi32(iBytes, 2);
        __m512i iInt2 = _mm512_cvtepi8_epi32(chunk2);
        __m512 iFloat2 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt2), invScalar);
        _mm512_storeu_ps(iBufferPtr, iFloat2);
        iBufferPtr += 16;

        __m128i chunk3 = _mm512_extracti32x4_epi32(iBytes, 3);
        __m512i iInt3 = _mm512_cvtepi8_epi32(chunk3);
        __m512 iFloat3 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt3), invScalar);
        _mm512_storeu_ps(iBufferPtr, iFloat3);
        iBufferPtr += 16;
    }

    number = sixtyFourthPoints * 64;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8ic_s32f_deinterleave_real_32f_neon(float* iBuffer,
                                                            const lv_8sc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    float* iBufferPtr = iBuffer;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < eighth_points; number++) {
        int8x8x2_t input = vld2_s8(complexVectorPtr);
        complexVectorPtr += 16;

        int16x8_t iShort = vmovl_s8(input.val[0]);
        int32x4_t iInt0 = vmovl_s16(vget_low_s16(iShort));
        int32x4_t iInt1 = vmovl_s16(vget_high_s16(iShort));

        float32x4_t iFloat0 = vcvtq_f32_s32(iInt0);
        float32x4_t iFloat1 = vcvtq_f32_s32(iInt1);

        iFloat0 = vmulq_f32(iFloat0, vInvScalar);
        iFloat1 = vmulq_f32(iFloat1, vInvScalar);

        vst1q_f32(iBufferPtr, iFloat0);
        vst1q_f32(iBufferPtr + 4, iFloat1);
        iBufferPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * invScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_neonv8(float* iBuffer,
                                            const lv_8sc_t* complexVector,
                                            const float scalar,
                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const int8_t* complexVectorPtr = (const int8_t*)complexVector;
    float* iBufferPtr = iBuffer;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    for (; number < eighth_points; number++) {
        int8x8x2_t input = vld2_s8(complexVectorPtr);
        complexVectorPtr += 16;

        int16x8_t iWide = vmovl_s8(input.val[0]);
        int32x4_t iLo = vmovl_s16(vget_low_s16(iWide));
        int32x4_t iHi = vmovl_s16(vget_high_s16(iWide));
        float32x4_t iFltLo = vmulq_f32(vcvtq_f32_s32(iLo), vInvScalar);
        float32x4_t iFltHi = vmulq_f32(vcvtq_f32_s32(iHi), vInvScalar);

        vst1q_f32(iBufferPtr, iFltLo);
        iBufferPtr += 4;
        vst1q_f32(iBufferPtr, iFltHi);
        iBufferPtr += 4;
    }

    number = eighth_points * 8;
    complexVectorPtr = (const int8_t*)&complexVector[number];
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * invScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8ic_s32f_deinterleave_real_32f_rvv(float* iBuffer,
                                                           const lv_8sc_t* complexVector,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    const uint16_t* in = (const uint16_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vuint16m4_t vc = __riscv_vle16_v_u16m4(in, vl);
        vint8m2_t vr = __riscv_vreinterpret_i8m2(__riscv_vnsrl(vc, 0, vl));
        vfloat32m8_t vrf = __riscv_vfwcvt_f(__riscv_vsext_vf2(vr, vl), vl);
        __riscv_vse32(iBuffer, __riscv_vfmul(vrf, 1.0f / scalar, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_8ic_s32f_deinterleave_real_32f_rvvseg(float* iBuffer,
                                                               const lv_8sc_t* complexVector,
                                                               const float scalar,
                                                               unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e8m2(n);
        vint8m2x2_t vc =
            __riscv_vlseg2e8_v_i8m2x2((const int8_t*)complexVector, vl);
        vfloat32m8_t vrf =
            __riscv_vfwcvt_f(__riscv_vsext_vf2(__riscv_vget_i8m2(vc, 0), vl), vl);
        __riscv_vse32(iBuffer, __riscv_vfmul(vrf, 1.0f / scalar, vl), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_8ic_s32f_deinterleave_real_32f_u_H */

#ifndef INCLUDED_volk_8ic_s32f_deinterleave_real_32f_a_H
#define INCLUDED_volk_8ic_s32f_deinterleave_real_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_a_sse(float* iBuffer,
                                          const lv_8sc_t* complexVector,
                                          const float scalar,
                                          unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    __m128 iValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        floatBuffer[0] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[1] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[2] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;
        floatBuffer[3] = (float)(*complexVectorPtr);
        complexVectorPtr += 2;

        iValue = _mm_load_ps(floatBuffer);

        iValue = _mm_mul_ps(iValue, invScalar);

        _mm_store_ps(iBufferPtr, iValue);

        iBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_a_sse2(float* iBuffer,
                                            const lv_8sc_t* complexVector,
                                            const float scalar,
                                            unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    for (; number < eighthPoints; number++) {
        __m128i complexVal = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        // Sign-extend low byte of each 16-bit lane (the I component)
        __m128i iVals16 = _mm_srai_epi16(_mm_slli_epi16(complexVal, 8), 8);

        // Sign-extend int16 to int32
        __m128i sign = _mm_srai_epi16(iVals16, 15);
        __m128i iInt0 = _mm_unpacklo_epi16(iVals16, sign);
        __m128i iInt1 = _mm_unpackhi_epi16(iVals16, sign);

        // Convert to float and multiply by inverse scalar
        __m128 iFloat0 = _mm_mul_ps(_mm_cvtepi32_ps(iInt0), invScalar);
        __m128 iFloat1 = _mm_mul_ps(_mm_cvtepi32_ps(iInt1), invScalar);

        _mm_store_ps(iBufferPtr, iFloat0);
        _mm_store_ps(iBufferPtr + 4, iFloat1);
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_a_sse4_1(float* iBuffer,
                                             const lv_8sc_t* complexVector,
                                             const float scalar,
                                             unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    __m128 iFloatValue;

    const float iScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(iScalar);
    __m128i complexVal, iIntVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);

    for (; number < eighthPoints; number++) {
        complexVal = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;
        complexVal = _mm_shuffle_epi8(complexVal, moveMask);

        iIntVal = _mm_cvtepi8_epi32(complexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);

        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);

        _mm_store_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 4;

        complexVal = _mm_srli_si128(complexVal, 4);
        iIntVal = _mm_cvtepi8_epi32(complexVal);
        iFloatValue = _mm_cvtepi32_ps(iIntVal);

        iFloatValue = _mm_mul_ps(iFloatValue, invScalar);

        _mm_store_ps(iBufferPtr, iFloatValue);

        iBufferPtr += 4;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_a_avx(float* iBuffer,
                                           const lv_8sc_t* complexVector,
                                           const float scalar,
                                           unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float iScalar = 1.0f / scalar;
    const __m256 invScalar = _mm256_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    /* Shuffle bytes: move real (even) bytes of 8 complex pairs to positions 0-7,
     * zero-fill the upper 8 bytes. Real bytes are at positions 0,2,4,6,8,10,12,14. */
    const __m128i moveMask = _mm_set_epi8(
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);

    for (; number < eighthPoints; number++) {
        __m128i complexVal = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 16;

        /* Gather the 8 real bytes into the low 8 positions */
        complexVal = _mm_shuffle_epi8(complexVal, moveMask);

        /* Sign-extend low 4 bytes (int8) to int32, convert to float */
        __m128i iInt0 = _mm_cvtepi8_epi32(complexVal);
        __m128 iFloat0 = _mm_mul_ps(_mm_cvtepi32_ps(iInt0), _mm256_castps256_ps128(invScalar));

        /* Sign-extend next 4 bytes to int32, convert to float */
        complexVal = _mm_srli_si128(complexVal, 4);
        __m128i iInt1 = _mm_cvtepi8_epi32(complexVal);
        __m128 iFloat1 = _mm_mul_ps(_mm_cvtepi32_ps(iInt1), _mm256_castps256_ps128(invScalar));

        /* Combine two __m128 results into one __m256 and store */
        _mm256_store_ps(iBufferPtr, _mm256_set_m128(iFloat1, iFloat0));
        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_a_avx2(float* iBuffer,
                                           const lv_8sc_t* complexVector,
                                           const float scalar,
                                           unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    __m256 iFloatValue;

    const float iScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(iScalar);
    __m256i complexVal, iIntVal;
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

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
    for (; number < sixteenthPoints; number++) {
        complexVal = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);

        iIntVal = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(complexVal));
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);
        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);
        _mm256_store_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 8;

        complexVal = _mm256_permute4x64_epi64(complexVal, 0b11000110);
        iIntVal = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(complexVal));
        iFloatValue = _mm256_cvtepi32_ps(iIntVal);
        iFloatValue = _mm256_mul_ps(iFloatValue, invScalar);
        _mm256_store_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 8;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (float)(*complexVectorPtr++) * iScalar;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_a_avx512f(float* iBuffer,
                                               const lv_8sc_t* complexVector,
                                               const float scalar,
                                               unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float iScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

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

    for (; number < sixteenthPoints; number++) {
        __m256i complexVal = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 32;
        complexVal = _mm256_shuffle_epi8(complexVal, moveMask);

        // Combine 8 real bytes from each 128-bit lane into one __m128i
        __m128i lo = _mm256_castsi256_si128(complexVal);
        __m128i hi = _mm256_extracti128_si256(complexVal, 1);
        __m128i combined = _mm_unpacklo_epi64(lo, hi);

        // Sign-extend 16 x int8 to 16 x int32, convert to float
        __m512i iIntVal = _mm512_cvtepi8_epi32(combined);
        __m512 iFloatValue = _mm512_cvtepi32_ps(iIntVal);
        iFloatValue = _mm512_mul_ps(iFloatValue, invScalar);

        _mm512_store_ps(iBufferPtr, iFloatValue);
        iBufferPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512VBMI
#include <immintrin.h>

static inline void
volk_8ic_s32f_deinterleave_real_32f_a_avx512vbmi(float* iBuffer,
                                                   const lv_8sc_t* complexVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    float* iBufferPtr = iBuffer;

    unsigned int number = 0;
    const unsigned int sixtyFourthPoints = num_points / 64;

    const float iScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(iScalar);
    const int8_t* complexVectorPtr = (const int8_t*)complexVector;

    /* Index vector: pick byte 0 (I) from each 2-byte complex sample.
     * Indices 0-63 select from src1, 64-127 select from src2. */
    const __m512i iIdx = _mm512_set_epi8(
        126, 124, 122, 120, 118, 116, 114, 112,
        110, 108, 106, 104, 102, 100, 98, 96,
        94, 92, 90, 88, 86, 84, 82, 80,
        78, 76, 74, 72, 70, 68, 66, 64,
        62, 60, 58, 56, 54, 52, 50, 48,
        46, 44, 42, 40, 38, 36, 34, 32,
        30, 28, 26, 24, 22, 20, 18, 16,
        14, 12, 10, 8, 6, 4, 2, 0);

    for (; number < sixtyFourthPoints; number++) {
        __m512i src1 = _mm512_load_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;
        __m512i src2 = _mm512_load_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 64;

        /* Extract 64 I bytes into one register */
        __m512i iBytes = _mm512_permutex2var_epi8(src1, iIdx, src2);

        /* Process in 4 batches of 16: int8 -> int32 -> float */
        __m128i chunk0 = _mm512_castsi512_si128(iBytes);
        __m512i iInt0 = _mm512_cvtepi8_epi32(chunk0);
        __m512 iFloat0 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt0), invScalar);
        _mm512_store_ps(iBufferPtr, iFloat0);
        iBufferPtr += 16;

        __m128i chunk1 = _mm512_extracti32x4_epi32(iBytes, 1);
        __m512i iInt1 = _mm512_cvtepi8_epi32(chunk1);
        __m512 iFloat1 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt1), invScalar);
        _mm512_store_ps(iBufferPtr, iFloat1);
        iBufferPtr += 16;

        __m128i chunk2 = _mm512_extracti32x4_epi32(iBytes, 2);
        __m512i iInt2 = _mm512_cvtepi8_epi32(chunk2);
        __m512 iFloat2 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt2), invScalar);
        _mm512_store_ps(iBufferPtr, iFloat2);
        iBufferPtr += 16;

        __m128i chunk3 = _mm512_extracti32x4_epi32(iBytes, 3);
        __m512i iInt3 = _mm512_cvtepi8_epi32(chunk3);
        __m512 iFloat3 = _mm512_mul_ps(_mm512_cvtepi32_ps(iInt3), invScalar);
        _mm512_store_ps(iBufferPtr, iFloat3);
        iBufferPtr += 16;
    }

    number = sixtyFourthPoints * 64;
    volk_8ic_s32f_deinterleave_real_32f_generic(
        iBufferPtr, (const lv_8sc_t*)complexVectorPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512VBMI */


#endif /* INCLUDED_volk_8ic_s32f_deinterleave_real_32f_a_H */
