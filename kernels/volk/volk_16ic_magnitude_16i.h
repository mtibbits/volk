/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_magnitude_16i
 *
 * \b Overview
 *
 * Computes the magnitude of each complex 16-bit integer sample and stores
 * the results as 16-bit integers. Values are internally normalized by
 * SHRT_MAX before computing sqrt(I^2 + Q^2), then scaled back.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_magnitude_16i(int16_t* magnitudeVector, const lv_16sc_t* complexVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector (lv_16sc_t).
 * \li num_points: The number of complex samples.
 *
 * \b Outputs
 * \li magnitudeVector: The magnitude of each complex value (int16_t).
 *
 * \b Example
 * Compute the magnitude of several complex samples with known I/Q values.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main() {
 *     unsigned int N = 4;
 *     unsigned int alignment = volk_get_alignment();
 *
 *     // Allocate input and output vectors
 *     lv_16sc_t* complexVector =
 *         (lv_16sc_t*)volk_malloc(N * sizeof(lv_16sc_t), alignment);
 *     int16_t* magnitudeVector =
 *         (int16_t*)volk_malloc(N * sizeof(int16_t), alignment);
 *
 *     // Fill with complex samples whose magnitudes are easy to verify
 *     complexVector[0] = lv_cmake((int16_t)3000, (int16_t)4000);   // mag ~ 5000
 *     complexVector[1] = lv_cmake((int16_t)0, (int16_t)10000);     // mag ~ 10000
 *     complexVector[2] = lv_cmake((int16_t)-7000, (int16_t)0);     // mag ~ 7000
 *     complexVector[3] = lv_cmake((int16_t)20000, (int16_t)20000); // mag ~ 28284
 *
 *     // Compute magnitudes
 *     volk_16ic_magnitude_16i(magnitudeVector, complexVector, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *       printf("mag[%u] = %d\n", i, magnitudeVector[i]);
 *     }
 *
 *     volk_free(magnitudeVector);
 *     volk_free(complexVector);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_magnitude_16i_u_H
#define INCLUDED_volk_16ic_magnitude_16i_u_H

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_magnitude_16i_generic(int16_t* magnitudeVector,
                                                   const lv_16sc_t* complexVector,
                                                   unsigned int num_points)
{
    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    const float scalar = SHRT_MAX;
    const float invScalar = 1.0f / scalar;
    for (number = 0; number < num_points; number++) {
        float real = ((float)(*complexVectorPtr++)) * invScalar;
        float imag = ((float)(*complexVectorPtr++)) * invScalar;
        *magnitudeVectorPtr++ =
            (int16_t)rintf(sqrtf((real * real) + (imag * imag)) * scalar);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16ic_magnitude_16i_u_sse(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    __m128 cplxValue1, cplxValue2, iValue, qValue, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[4];
    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        cplxValue1 = _mm_load_ps(inputFloatBuffer);
        complexVectorPtr += 4;

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        cplxValue2 = _mm_load_ps(inputFloatBuffer);
        complexVectorPtr += 4;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        // Arrange in q1q2q3q4 format
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        iValue = _mm_mul_ps(iValue, iValue); // Square the I values
        qValue = _mm_mul_ps(qValue, qValue); // Square the Q Values

        result = _mm_add_ps(iValue, qValue); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        result = _mm_mul_ps(result, vScalar); // Scale the results

        _mm_store_ps(outputFloatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[3]);
    }

    volk_16ic_magnitude_16i_generic(
        magnitudeVector + quarterPoints * 4,
        complexVector + quarterPoints * 4,
        num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_magnitude_16i_u_sse2(int16_t* magnitudeVector,
                                                    const lv_16sc_t* complexVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    for (; number < quarterPoints; number++) {
        /* Load 4 complex int16 samples = 8 int16 values */
        __m128i raw = _mm_loadu_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        /* Sign-extend int16 -> int32 */
        __m128i lo32 = _mm_srai_epi32(_mm_unpacklo_epi16(raw, raw), 16);
        __m128i hi32 = _mm_srai_epi32(_mm_unpackhi_epi16(raw, raw), 16);

        /* Convert int32 -> float and scale */
        __m128 cplxValue1 = _mm_mul_ps(_mm_cvtepi32_ps(lo32), invScalar);
        __m128 cplxValue2 = _mm_mul_ps(_mm_cvtepi32_ps(hi32), invScalar);

        /* Deinterleave I and Q */
        __m128 iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag = sqrt(I^2 + Q^2) * scalar */
        iValue = _mm_mul_ps(iValue, iValue);
        qValue = _mm_mul_ps(qValue, qValue);
        __m128 result = _mm_sqrt_ps(_mm_add_ps(iValue, qValue));
        result = _mm_mul_ps(result, vScalar);

        /* Convert float -> int32 -> int16 (saturating) and store */
        __m128i resultInt = _mm_cvtps_epi32(result);
        _mm_storel_epi64((__m128i*)magnitudeVectorPtr,
                         _mm_packs_epi32(resultInt, resultInt));
        magnitudeVectorPtr += 4;
    }

    volk_16ic_magnitude_16i_generic(
        magnitudeVector + quarterPoints * 4,
        complexVector + quarterPoints * 4,
        num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_16ic_magnitude_16i_u_sse3(int16_t* magnitudeVector,
                                                    const lv_16sc_t* complexVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];
    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        result = _mm_mul_ps(result, vScalar); // Scale the results

        _mm_store_ps(outputFloatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[3]);
    }

    volk_16ic_magnitude_16i_generic(
        magnitudeVector + quarterPoints * 4,
        complexVector + quarterPoints * 4,
        num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_magnitude_16i_u_avx2(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m256 vScalar = _mm256_set1_ps(SHRT_MAX);
    __m256 invScalar = _mm256_set1_ps(fInvScalar);
    __m256i int1, int2;
    __m128i short1, short2;
    __m256 cplxValue1, cplxValue2, result;
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0);

    for (; number < eighthPoints; number++) {

        int1 = _mm256_loadu_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        short1 = _mm256_extracti128_si256(int1, 0);
        short2 = _mm256_extracti128_si256(int1, 1);

        int1 = _mm256_cvtepi16_epi32(short1);
        int2 = _mm256_cvtepi16_epi32(short2);
        cplxValue1 = _mm256_cvtepi32_ps(int1);
        cplxValue2 = _mm256_cvtepi32_ps(int2);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm256_sqrt_ps(result); // Square root the values

        result = _mm256_mul_ps(result, vScalar); // Scale the results

        int1 = _mm256_cvtps_epi32(result);
        int1 = _mm256_packs_epi32(int1, int1);
        int1 = _mm256_permutevar8x32_epi32(
            int1, idx); // permute to compensate for shuffling in hadd and packs
        short1 = _mm256_extracti128_si256(int1, 0);
        _mm_storeu_si128((__m128i*)magnitudeVectorPtr, short1);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_magnitude_16i_u_avx512f(int16_t* magnitudeVector,
                                                     const lv_16sc_t* complexVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m512 vInvScalar = _mm512_set1_ps(fInvScalar);
    __m512 vScalar = _mm512_set1_ps(SHRT_MAX);

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        /* Load 16 complex int16 samples = 512 bits */
        __m512i raw = _mm512_loadu_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 32;

        /* Split into two 256-bit halves and widen int16 -> int32 */
        __m256i lo_half = _mm512_castsi512_si256(raw);
        __m256i hi_half = _mm512_extracti64x4_epi64(raw, 1);
        __m512i wide_lo = _mm512_cvtepi16_epi32(lo_half);
        __m512i wide_hi = _mm512_cvtepi16_epi32(hi_half);

        /* Convert int32 -> float */
        __m512 flt_lo = _mm512_cvtepi32_ps(wide_lo);
        __m512 flt_hi = _mm512_cvtepi32_ps(wide_hi);

        /* Scale by 1/SHRT_MAX */
        flt_lo = _mm512_mul_ps(flt_lo, vInvScalar);
        flt_hi = _mm512_mul_ps(flt_hi, vInvScalar);

        /* Deinterleave re/im from each 512-bit float vector */
        __m512 re = _mm512_permutex2var_ps(flt_lo, idx_re, flt_hi);
        __m512 im = _mm512_permutex2var_ps(flt_lo, idx_im, flt_hi);

        /* mag = sqrt(re^2 + im^2) */
        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 magSquared = _mm512_add_ps(reSquared, imSquared);
        __m512 mag = _mm512_sqrt_ps(magSquared);

        /* Scale back */
        mag = _mm512_mul_ps(mag, vScalar);

        /* Convert float -> int32 -> int16 (saturating) */
        __m512i magInt = _mm512_cvtps_epi32(mag);
        __m256i magShort = _mm512_cvtsepi32_epi16(magInt);

        _mm256_storeu_si256((__m256i*)magnitudeVectorPtr, magShort);
        magnitudeVectorPtr += 16;
    }

    volk_16ic_magnitude_16i_generic(
        magnitudeVector + sixteenthPoints * 16,
        complexVector + sixteenthPoints * 16,
        num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEONV7
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void volk_16ic_magnitude_16i_neonv7(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    const float scalar = SHRT_MAX;
    const float inv_scalar = 1.0f / scalar;

    int16_t* magnitudeVectorPtr = magnitudeVector;
    const lv_16sc_t* complexVectorPtr = complexVector;

    float32x4_t mag_vec;
    float32x4x2_t c_vec;

    for (number = 0; number < quarter_points; number++) {
        const int16x4x2_t c16_vec = vld2_s16((const int16_t*)complexVectorPtr);
        __VOLK_PREFETCH(complexVectorPtr + 4);
        c_vec.val[0] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[0]));
        c_vec.val[1] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[1]));
        // Scale to close to 0-1
        c_vec.val[0] = vmulq_n_f32(c_vec.val[0], inv_scalar);
        c_vec.val[1] = vmulq_n_f32(c_vec.val[1], inv_scalar);
        // vsqrtq_f32 is armv8
        const float32x4_t mag_vec_squared = _vmagnitudesquaredq_f32(c_vec);
        const float32x4_t mag_nonzero =
            vmulq_f32(mag_vec_squared, _vinvsqrtq_f32(mag_vec_squared));
        const uint32x4_t nonzero_mask = vcgtq_f32(mag_vec_squared, vdupq_n_f32(0.0f));
        mag_vec = vbslq_f32(nonzero_mask, mag_nonzero, vdupq_n_f32(0.0f));
        // Reconstruct
        mag_vec = vmulq_n_f32(mag_vec, scalar);
        // Add 0.5 for correct rounding because vcvtq_s32_f32 truncates.
        // This works because the magnitude is always positive.
        mag_vec = vaddq_f32(mag_vec, vdupq_n_f32(0.5));
        const int16x4_t mag16_vec = vmovn_s32(vcvtq_s32_f32(mag_vec));
        vst1_s16(magnitudeVectorPtr, mag16_vec);
        // Advance pointers
        magnitudeVectorPtr += 4;
        complexVectorPtr += 4;
    }

    // Deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        const float real = lv_creal(*complexVectorPtr) * inv_scalar;
        const float imag = lv_cimag(*complexVectorPtr) * inv_scalar;
        *magnitudeVectorPtr =
            (int16_t)rintf(sqrtf((real * real) + (imag * imag)) * scalar);
        complexVectorPtr++;
        magnitudeVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV7 */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_magnitude_16i_neonv8(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    const float scalar = SHRT_MAX;
    const float inv_scalar = 1.0f / scalar;

    int16_t* magnitudeVectorPtr = magnitudeVector;
    const lv_16sc_t* complexVectorPtr = complexVector;

    float32x4_t mag_vec, mag_sq;
    float32x4x2_t c_vec;

    for (number = 0; number < quarter_points; number++) {
        const int16x4x2_t c16_vec = vld2_s16((const int16_t*)complexVectorPtr);
        __VOLK_PREFETCH(complexVectorPtr + 4);
        c_vec.val[0] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[0]));
        c_vec.val[1] = vcvtq_f32_s32(vmovl_s16(c16_vec.val[1]));
        // Scale to close to 0-1
        c_vec.val[0] = vmulq_n_f32(c_vec.val[0], inv_scalar);
        c_vec.val[1] = vmulq_n_f32(c_vec.val[1], inv_scalar);
        // ARMv8: Use FMA for magnitude squared and native sqrt
        mag_sq =
            vfmaq_f32(vmulq_f32(c_vec.val[0], c_vec.val[0]), c_vec.val[1], c_vec.val[1]);
        mag_vec = vsqrtq_f32(mag_sq);
        // Reconstruct
        mag_vec = vmulq_n_f32(mag_vec, scalar);
        // Add 0.5 for correct rounding because vcvtq_s32_f32 truncates.
        mag_vec = vaddq_f32(mag_vec, vdupq_n_f32(0.5f));
        const int16x4_t mag16_vec = vmovn_s32(vcvtq_s32_f32(mag_vec));
        vst1_s16(magnitudeVectorPtr, mag16_vec);
        // Advance pointers
        magnitudeVectorPtr += 4;
        complexVectorPtr += 4;
    }

    // Deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        const float real = lv_creal(*complexVectorPtr) * inv_scalar;
        const float imag = lv_cimag(*complexVectorPtr) * inv_scalar;
        *magnitudeVectorPtr =
            (int16_t)rintf(sqrtf((real * real) + (imag * imag)) * scalar);
        complexVectorPtr++;
        magnitudeVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16ic_magnitude_16i_rvv(int16_t* magnitudeVector,
                                               const lv_16sc_t* complexVector,
                                               unsigned int num_points)
{
    const float scale = SHRT_MAX, iscale = 1.0f / scale;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint32m8_t vc = __riscv_vle32_v_i32m8((const int32_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vnsra(vc, 0, vl);
        vint16m4_t vi = __riscv_vnsra(vc, 16, vl);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), iscale, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), iscale, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        vf = __riscv_vfmul(__riscv_vfsqrt(vf, vl), scale, vl);
        __riscv_vse16(magnitudeVector, __riscv_vfncvt_x(vf, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_16ic_magnitude_16i_rvvseg(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    const float scale = SHRT_MAX, iscale = 1.0f / scale;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4x2_t vc = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)complexVector, vl);
        vint16m4_t vr = __riscv_vget_i16m4(vc, 0);
        vint16m4_t vi = __riscv_vget_i16m4(vc, 1);
        vfloat32m8_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), iscale, vl);
        vfloat32m8_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), iscale, vl);
        vfloat32m8_t vf = __riscv_vfmacc(__riscv_vfmul(vif, vif, vl), vrf, vrf, vl);
        vf = __riscv_vfmul(__riscv_vfsqrt(vf, vl), scale, vl);
        __riscv_vse16(magnitudeVector, __riscv_vfncvt_x(vf, vl), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_16ic_magnitude_16i_u_H */


#ifndef INCLUDED_volk_16ic_magnitude_16i_a_H
#define INCLUDED_volk_16ic_magnitude_16i_a_H

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_16ic_magnitude_16i_a_sse(int16_t* magnitudeVector,
                                                 const lv_16sc_t* complexVector,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    __m128 cplxValue1, cplxValue2, iValue, qValue, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[4];
    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        cplxValue1 = _mm_load_ps(inputFloatBuffer);
        complexVectorPtr += 4;

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        cplxValue2 = _mm_load_ps(inputFloatBuffer);
        complexVectorPtr += 4;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        // Arrange in q1q2q3q4 format
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        iValue = _mm_mul_ps(iValue, iValue); // Square the I values
        qValue = _mm_mul_ps(qValue, qValue); // Square the Q Values

        result = _mm_add_ps(iValue, qValue); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        result = _mm_mul_ps(result, vScalar); // Scale the results

        _mm_store_ps(outputFloatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_magnitude_16i_a_sse2(int16_t* magnitudeVector,
                                                    const lv_16sc_t* complexVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    for (; number < quarterPoints; number++) {
        /* Load 4 complex int16 samples = 8 int16 values */
        __m128i raw = _mm_load_si128((const __m128i*)complexVectorPtr);
        complexVectorPtr += 8;

        /* Sign-extend int16 -> int32 */
        __m128i lo32 = _mm_srai_epi32(_mm_unpacklo_epi16(raw, raw), 16);
        __m128i hi32 = _mm_srai_epi32(_mm_unpackhi_epi16(raw, raw), 16);

        /* Convert int32 -> float and scale */
        __m128 cplxValue1 = _mm_mul_ps(_mm_cvtepi32_ps(lo32), invScalar);
        __m128 cplxValue2 = _mm_mul_ps(_mm_cvtepi32_ps(hi32), invScalar);

        /* Deinterleave I and Q */
        __m128 iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag = sqrt(I^2 + Q^2) * scalar */
        iValue = _mm_mul_ps(iValue, iValue);
        qValue = _mm_mul_ps(qValue, qValue);
        __m128 result = _mm_sqrt_ps(_mm_add_ps(iValue, qValue));
        result = _mm_mul_ps(result, vScalar);

        /* Convert float -> int32 -> int16 (saturating) and store */
        __m128i resultInt = _mm_cvtps_epi32(result);
        _mm_storel_epi64((__m128i*)magnitudeVectorPtr,
                         _mm_packs_epi32(resultInt, resultInt));
        magnitudeVectorPtr += 4;
    }

    volk_16ic_magnitude_16i_generic(
        magnitudeVector + quarterPoints * 4,
        complexVector + quarterPoints * 4,
        num_points - quarterPoints * 4);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_16ic_magnitude_16i_a_sse3(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m128 vScalar = _mm_set_ps1(SHRT_MAX);
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];
    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {

        inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
        inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
        inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
        inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

        inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
        inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
        inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
        inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

        cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
        cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

        complexVectorPtr += 8;

        cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result); // Square root the values

        result = _mm_mul_ps(result, vScalar); // Scale the results

        _mm_store_ps(outputFloatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_magnitude_16i_a_avx2(int16_t* magnitudeVector,
                                                  const lv_16sc_t* complexVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m256 vScalar = _mm256_set1_ps(SHRT_MAX);
    __m256 invScalar = _mm256_set1_ps(fInvScalar);
    __m256i int1, int2;
    __m128i short1, short2;
    __m256 cplxValue1, cplxValue2, result;
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0);

    for (; number < eighthPoints; number++) {

        int1 = _mm256_load_si256((const __m256i*)complexVectorPtr);
        complexVectorPtr += 16;
        short1 = _mm256_extracti128_si256(int1, 0);
        short2 = _mm256_extracti128_si256(int1, 1);

        int1 = _mm256_cvtepi16_epi32(short1);
        int2 = _mm256_cvtepi16_epi32(short2);
        cplxValue1 = _mm256_cvtepi32_ps(int1);
        cplxValue2 = _mm256_cvtepi32_ps(int2);

        cplxValue1 = _mm256_mul_ps(cplxValue1, invScalar);
        cplxValue2 = _mm256_mul_ps(cplxValue2, invScalar);

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm256_sqrt_ps(result); // Square root the values

        result = _mm256_mul_ps(result, vScalar); // Scale the results

        int1 = _mm256_cvtps_epi32(result);
        int1 = _mm256_packs_epi32(int1, int1);
        int1 = _mm256_permutevar8x32_epi32(
            int1, idx); // permute to compensate for shuffling in hadd and packs
        short1 = _mm256_extracti128_si256(int1, 0);
        _mm_store_si128((__m128i*)magnitudeVectorPtr, short1);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    magnitudeVectorPtr = &magnitudeVector[number];
    complexVectorPtr = (const int16_t*)&complexVector[number];
    for (; number < num_points; number++) {
        const float val1Real = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Imag = (float)(*complexVectorPtr++) * fInvScalar;
        const float val1Result =
            sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * SHRT_MAX;
        *magnitudeVectorPtr++ = (int16_t)rintf(val1Result);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_16ic_magnitude_16i_a_avx512f(int16_t* magnitudeVector,
                                                     const lv_16sc_t* complexVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const int16_t* complexVectorPtr = (const int16_t*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const float fInvScalar = 1.0f / SHRT_MAX;
    __m512 vInvScalar = _mm512_set1_ps(fInvScalar);
    __m512 vScalar = _mm512_set1_ps(SHRT_MAX);

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        /* Load 16 complex int16 samples = 512 bits */
        __m512i raw = _mm512_load_si512((const __m512i*)complexVectorPtr);
        complexVectorPtr += 32;

        /* Split into two 256-bit halves and widen int16 -> int32 */
        __m256i lo_half = _mm512_castsi512_si256(raw);
        __m256i hi_half = _mm512_extracti64x4_epi64(raw, 1);
        __m512i wide_lo = _mm512_cvtepi16_epi32(lo_half);
        __m512i wide_hi = _mm512_cvtepi16_epi32(hi_half);

        /* Convert int32 -> float */
        __m512 flt_lo = _mm512_cvtepi32_ps(wide_lo);
        __m512 flt_hi = _mm512_cvtepi32_ps(wide_hi);

        /* Scale by 1/SHRT_MAX */
        flt_lo = _mm512_mul_ps(flt_lo, vInvScalar);
        flt_hi = _mm512_mul_ps(flt_hi, vInvScalar);

        /* Deinterleave re/im from each 512-bit float vector */
        __m512 re = _mm512_permutex2var_ps(flt_lo, idx_re, flt_hi);
        __m512 im = _mm512_permutex2var_ps(flt_lo, idx_im, flt_hi);

        /* mag = sqrt(re^2 + im^2) */
        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 magSquared = _mm512_add_ps(reSquared, imSquared);
        __m512 mag = _mm512_sqrt_ps(magSquared);

        /* Scale back */
        mag = _mm512_mul_ps(mag, vScalar);

        /* Convert float -> int32 -> int16 (saturating) */
        __m512i magInt = _mm512_cvtps_epi32(mag);
        __m256i magShort = _mm512_cvtsepi32_epi16(magInt);

        _mm256_store_si256((__m256i*)magnitudeVectorPtr, magShort);
        magnitudeVectorPtr += 16;
    }

    volk_16ic_magnitude_16i_generic(
        magnitudeVector + sixteenthPoints * 16,
        complexVector + sixteenthPoints * 16,
        num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_16ic_magnitude_16i_a_H */
