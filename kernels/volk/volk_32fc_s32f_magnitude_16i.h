/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_magnitude_16i
 *
 * \b Overview
 *
 * Computes the magnitude of each complex sample, scales by a scalar factor,
 * and converts the result to a 16-bit integer:
 * out[i] = (int16_t)round(scalar * sqrt(real[i]^2 + imag[i]^2)).
 *
 * This kernel is useful in receiver chains where magnitude detection must feed
 * into fixed-point processing stages, such as RSSI estimation, signal strength
 * indication, or envelope detection. The scalar factor allows mapping
 * floating-point magnitudes into a meaningful integer range for downstream
 * quantized processing.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_magnitude_16i(int16_t* magnitudeVector, const lv_32fc_t*
 * complexVector, const float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector (lv_32fc_t).
 * \li scalar: Scale factor applied to each magnitude before conversion to int16.
 * \li num_points: The number of complex samples.
 *
 * \b Outputs
 * \li magnitudeVector: The scaled magnitudes as 16-bit integers (int16_t).
 *
 * \b Example
 * Use a 3-4-5 Pythagorean triple so the expected magnitude is exact.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* in = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * int16_t* out = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 * float scalar = 10.0f;
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     in[i] = lv_cmake(3.0f, 4.0f); // magnitude = 5.0
 * }
 *
 * // Expected: round(10.0 * 5.0) = 50
 * int16_t expected = 50;
 *
 * volk_32fc_s32f_magnitude_16i(out, in, scalar, N);
 *
 * printf("Expected: %d\n", expected);
 * printf("Result:   %d\n", out[0]);
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_magnitude_16i_u_H
#define INCLUDED_volk_32fc_s32f_magnitude_16i_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32f_magnitude_16i_generic(int16_t* magnitudeVector,
                                                        const lv_32fc_t* complexVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    for (number = 0; number < num_points; number++) {
        float real = *complexVectorPtr++;
        float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ =
            (int16_t)rintf(scalar * sqrtf((real * real) + (imag * imag)));
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_u_sse(int16_t* magnitudeVector,
                                                      const lv_32fc_t* complexVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m128 vScalar = _mm_set_ps1(scalar);

    __m128 cplxValue1, cplxValue2, result;
    __m128 iValue, qValue;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        // Arrange in q1q2q3q4 format
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        __m128 iValue2 = _mm_mul_ps(iValue, iValue); // Square the I values
        __m128 qValue2 = _mm_mul_ps(qValue, qValue); // Square the Q Values

        result = _mm_add_ps(iValue2, qValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result);

        result = _mm_mul_ps(result, vScalar);

        _mm_store_ps(floatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[3]);
    }

    number = quarterPoints * 4;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_u_sse3(int16_t* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m128 vScalar = _mm_set_ps1(scalar);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result);

        result = _mm_mul_ps(result, vScalar);

        _mm_store_ps(floatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[3]);
    }

    number = quarterPoints * 4;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_u_avx(int16_t* magnitudeVector,
                                                      const lv_32fc_t* complexVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        __m256 cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        /* Deinterleave real and imaginary within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag = sqrt(re^2 + im^2) */
        __m256 mag = _mm256_sqrt_ps(
            _mm256_add_ps(_mm256_mul_ps(re, re), _mm256_mul_ps(im, im)));

        /* Scale */
        mag = _mm256_mul_ps(mag, vScalar);

        /* Fix cross-lane ordering from in-lane shuffle:
         * mag = [m0,m1,m4,m5 | m2,m3,m6,m7]
         * need  [m0,m1,m2,m3 | m4,m5,m6,m7] */
        __m128 mag_lo = _mm256_castps256_ps128(mag);
        __m128 mag_hi = _mm256_extractf128_ps(mag, 1);
        __m128 out_lo = _mm_shuffle_ps(mag_lo, mag_hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(mag_lo, mag_hi, _MM_SHUFFLE(3, 2, 3, 2));

        /* Convert float -> int32 -> int16 (saturating) */
        __m128i int_lo = _mm_cvtps_epi32(out_lo);
        __m128i int_hi = _mm_cvtps_epi32(out_hi);
        __m128i resultShort = _mm_packs_epi32(int_lo, int_hi);

        _mm_storeu_si128((__m128i*)magnitudeVectorPtr, resultShort);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_u_avx_fma(int16_t* magnitudeVector,
                                                            const lv_32fc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        __m256 cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        /* Deinterleave real and imaginary within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));

        /* Fix cross-lane ordering: mag_sq = [m0,m1,m4,m5 | m2,m3,m6,m7]
         * need [m0,m1,m2,m3 | m4,m5,m6,m7] */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));

        /* sqrt and scale */
        __m128 mag_lo = _mm_sqrt_ps(out_lo);
        __m128 mag_hi = _mm_sqrt_ps(out_hi);
        mag_lo = _mm_mul_ps(mag_lo, _mm256_castps256_ps128(vScalar));
        mag_hi = _mm_mul_ps(mag_hi, _mm256_castps256_ps128(vScalar));

        /* Convert float -> int32 -> int16 (saturating) */
        __m128i int_lo = _mm_cvtps_epi32(mag_lo);
        __m128i int_hi = _mm_cvtps_epi32(mag_hi);
        __m128i resultShort = _mm_packs_epi32(int_lo, int_hi);

        _mm_storeu_si128((__m128i*)magnitudeVectorPtr, resultShort);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_u_avx2(int16_t* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0);
    __m256 cplxValue1, cplxValue2, result;
    __m256i resultInt;
    __m128i resultShort;

    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm256_sqrt_ps(result);

        result = _mm256_mul_ps(result, vScalar);

        resultInt = _mm256_cvtps_epi32(result);
        resultInt = _mm256_packs_epi32(resultInt, resultInt);
        resultInt = _mm256_permutevar8x32_epi32(
            resultInt, idx); // permute to compensate for shuffling in hadd and packs
        resultShort = _mm256_extracti128_si256(resultInt, 0);
        _mm_storeu_si128((__m128i*)magnitudeVectorPtr, resultShort);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32fc_s32f_magnitude_16i_u_avx2_fma(int16_t* magnitudeVector,
                                                            const lv_32fc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;
        __m256 cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        __m256 result = _mm256_magnitudesquared_ps_avx2_fma(cplxValue1, cplxValue2);

        result = _mm256_sqrt_ps(result);
        result = _mm256_mul_ps(result, vScalar);

        __m256i resultInt = _mm256_cvtps_epi32(result);
        resultInt = _mm256_packs_epi32(resultInt, resultInt);
        resultInt = _mm256_permute4x64_epi64(resultInt, 0x08);
        _mm_storeu_si128((__m128i*)magnitudeVectorPtr,
                         _mm256_extracti128_si256(resultInt, 0));
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_u_avx512f(int16_t* magnitudeVector,
                                                           const lv_32fc_t* complexVector,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m512 vScalar = _mm512_set1_ps(scalar);

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        __m512 cplxValue1 = _mm512_loadu_ps(complexVectorPtr);
        __m512 cplxValue2 = _mm512_loadu_ps(complexVectorPtr + 16);
        complexVectorPtr += 32;

        /* Deinterleave re/im */
        __m512 re = _mm512_permutex2var_ps(cplxValue1, idx_re, cplxValue2);
        __m512 im = _mm512_permutex2var_ps(cplxValue1, idx_im, cplxValue2);

        /* mag = sqrt(re^2 + im^2) */
        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 magSquared = _mm512_add_ps(reSquared, imSquared);
        __m512 mag = _mm512_sqrt_ps(magSquared);

        /* Scale */
        mag = _mm512_mul_ps(mag, vScalar);

        /* Convert float -> int32 -> int16 (saturating) */
        __m512i magInt = _mm512_cvtps_epi32(mag);
        __m256i magShort = _mm512_cvtsepi32_epi16(magInt);

        _mm256_storeu_si256((__m256i*)magnitudeVectorPtr, magShort);
        magnitudeVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_u_avx512bw(
    int16_t* magnitudeVector,
    const lv_32fc_t* complexVector,
    const float scalar,
    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtySecondPoints = num_points / 32;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const __m512 vScalar = _mm512_set1_ps(scalar);

    /* Indices to deinterleave re/im from two consecutive __m512 (32 floats = 16 complex) */
    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);
    /* Fix lane interleaving introduced by _mm512_packs_epi32 */
    const __m512i pack_fix = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    for (; number < thirtySecondPoints; number++) {
        __m512 cplx1 = _mm512_loadu_ps(complexVectorPtr);
        __m512 cplx2 = _mm512_loadu_ps(complexVectorPtr + 16);
        __m512 cplx3 = _mm512_loadu_ps(complexVectorPtr + 32);
        __m512 cplx4 = _mm512_loadu_ps(complexVectorPtr + 48);
        complexVectorPtr += 64;

        /* Deinterleave: pair 1 → 16 re + 16 im */
        __m512 re1 = _mm512_permutex2var_ps(cplx1, idx_re, cplx2);
        __m512 im1 = _mm512_permutex2var_ps(cplx1, idx_im, cplx2);

        /* Deinterleave: pair 2 → 16 re + 16 im */
        __m512 re2 = _mm512_permutex2var_ps(cplx3, idx_re, cplx4);
        __m512 im2 = _mm512_permutex2var_ps(cplx3, idx_im, cplx4);

        /* mag = sqrt(re^2 + im^2) * scalar */
        __m512 mag1 = _mm512_mul_ps(
            _mm512_sqrt_ps(
                _mm512_fmadd_ps(re1, re1, _mm512_mul_ps(im1, im1))),
            vScalar);
        __m512 mag2 = _mm512_mul_ps(
            _mm512_sqrt_ps(
                _mm512_fmadd_ps(re2, re2, _mm512_mul_ps(im2, im2))),
            vScalar);

        /* Convert float -> int32, pack int32 -> int16 (saturating), fix lane order */
        __m512i int1 = _mm512_cvtps_epi32(mag1);
        __m512i int2 = _mm512_cvtps_epi32(mag2);
        __m512i packed = _mm512_permutexvar_epi64(
            pack_fix, _mm512_packs_epi32(int1, int2));

        _mm512_storeu_si512((__m512i*)magnitudeVectorPtr, packed);
        magnitudeVectorPtr += 32;
    }

    number = thirtySecondPoints * 32;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar,
        num_points - number);
}

#endif /* LV_HAVE_AVX512BW */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_s32f_magnitude_16i_neon(int16_t* magnitudeVector,
                                                     const lv_32fc_t* complexVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;
    float32x4_t vScalar = vdupq_n_f32(scalar);

    /* Magic number for round-to-nearest-even on ARMv7 NEON (1.5 * 2^23) */
    float32x4_t magic = vdupq_n_f32(12582912.0f);

    for (; number < quarter_points; number++) {
        float32x4x2_t input = vld2q_f32(complexVectorPtr);
        complexVectorPtr += 8;

        float32x4_t realSquared = vmulq_f32(input.val[0], input.val[0]);
        float32x4_t imagSquared = vmulq_f32(input.val[1], input.val[1]);
        float32x4_t sumSquared = vaddq_f32(realSquared, imagSquared);

        /* Use reciprocal square root estimate with Newton-Raphson refinement */
        float32x4_t rsqrt = vrsqrteq_f32(sumSquared);
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(sumSquared, rsqrt), rsqrt));
        rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(vmulq_f32(sumSquared, rsqrt), rsqrt));
        float32x4_t magnitude = vmulq_f32(sumSquared, rsqrt);

        /* Handle zero case */
        uint32x4_t zero_mask = vceqq_f32(sumSquared, vdupq_n_f32(0.0f));
        magnitude = vbslq_f32(zero_mask, sumSquared, magnitude);

        float32x4_t scaled = vmulq_f32(magnitude, vScalar);
        float32x4_t rounded = vsubq_f32(vaddq_f32(scaled, magic), magic);
        int32x4_t intVal = vcvtq_s32_f32(rounded);
        int16x4_t shortVal = vqmovn_s32(intVal);

        vst1_s16(magnitudeVectorPtr, shortVal);
        magnitudeVectorPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        float real = *complexVectorPtr++;
        float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ =
            (int16_t)rintf(scalar * sqrtf((real * real) + (imag * imag)));
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_s32f_magnitude_16i_neonv8(int16_t* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;
    float32x4_t vScalar = vdupq_n_f32(scalar);

    for (; number < eighth_points; number++) {
        float32x4x2_t input0 = vld2q_f32(complexVectorPtr);
        float32x4x2_t input1 = vld2q_f32(complexVectorPtr + 8);
        complexVectorPtr += 16;
        __VOLK_PREFETCH(complexVectorPtr + 16);

        float32x4_t sumSquared0 = vfmaq_f32(
            vmulq_f32(input0.val[1], input0.val[1]), input0.val[0], input0.val[0]);
        float32x4_t sumSquared1 = vfmaq_f32(
            vmulq_f32(input1.val[1], input1.val[1]), input1.val[0], input1.val[0]);

        float32x4_t magnitude0 = vsqrtq_f32(sumSquared0);
        float32x4_t magnitude1 = vsqrtq_f32(sumSquared1);

        float32x4_t scaled0 = vmulq_f32(magnitude0, vScalar);
        float32x4_t scaled1 = vmulq_f32(magnitude1, vScalar);

        int32x4_t intVal0 = vcvtnq_s32_f32(scaled0);
        int32x4_t intVal1 = vcvtnq_s32_f32(scaled1);

        int16x4_t shortVal0 = vqmovn_s32(intVal0);
        int16x4_t shortVal1 = vqmovn_s32(intVal1);

        vst1_s16(magnitudeVectorPtr, shortVal0);
        vst1_s16(magnitudeVectorPtr + 4, shortVal1);
        magnitudeVectorPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        float real = *complexVectorPtr++;
        float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ =
            (int16_t)rintf(scalar * sqrtf((real * real) + (imag * imag)));
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_s32f_magnitude_16i_rvv(int16_t* magnitudeVector,
                                                    const lv_32fc_t* complexVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 0, vl));
        vfloat32m4_t vi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 32, vl));
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        v = __riscv_vfmul(__riscv_vfsqrt(v, vl), scalar, vl);
        __riscv_vse16(magnitudeVector, __riscv_vfncvt_x(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_s32f_magnitude_16i_rvvseg(int16_t* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0);
        vfloat32m4_t vi = __riscv_vget_f32m4(vc, 1);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        v = __riscv_vfmul(__riscv_vfsqrt(v, vl), scalar, vl);
        __riscv_vse16(magnitudeVector, __riscv_vfncvt_x(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_s32f_magnitude_16i_u_H */

#ifndef INCLUDED_volk_32fc_s32f_magnitude_16i_a_H
#define INCLUDED_volk_32fc_s32f_magnitude_16i_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_a_sse(int16_t* magnitudeVector,
                                                      const lv_32fc_t* complexVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m128 vScalar = _mm_set_ps1(scalar);

    __m128 cplxValue1, cplxValue2, result;
    __m128 iValue, qValue;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        // Arrange in q1q2q3q4 format
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        __m128 iValue2 = _mm_mul_ps(iValue, iValue); // Square the I values
        __m128 qValue2 = _mm_mul_ps(qValue, qValue); // Square the Q Values

        result = _mm_add_ps(iValue2, qValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result);

        result = _mm_mul_ps(result, vScalar);

        _mm_store_ps(floatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[3]);
    }

    number = quarterPoints * 4;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_a_sse3(int16_t* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m128 vScalar = _mm_set_ps1(scalar);

    __m128 cplxValue1, cplxValue2, result;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm_sqrt_ps(result);

        result = _mm_mul_ps(result, vScalar);

        _mm_store_ps(floatBuffer, result);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[3]);
    }

    number = quarterPoints * 4;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_a_avx(int16_t* magnitudeVector,
                                                      const lv_32fc_t* complexVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        __m256 cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        /* Deinterleave real and imaginary within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag = sqrt(re^2 + im^2) */
        __m256 mag = _mm256_sqrt_ps(
            _mm256_add_ps(_mm256_mul_ps(re, re), _mm256_mul_ps(im, im)));

        /* Scale */
        mag = _mm256_mul_ps(mag, vScalar);

        /* Fix cross-lane ordering from in-lane shuffle:
         * mag = [m0,m1,m4,m5 | m2,m3,m6,m7]
         * need  [m0,m1,m2,m3 | m4,m5,m6,m7] */
        __m128 mag_lo = _mm256_castps256_ps128(mag);
        __m128 mag_hi = _mm256_extractf128_ps(mag, 1);
        __m128 out_lo = _mm_shuffle_ps(mag_lo, mag_hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(mag_lo, mag_hi, _MM_SHUFFLE(3, 2, 3, 2));

        /* Convert float -> int32 -> int16 (saturating) */
        __m128i int_lo = _mm_cvtps_epi32(out_lo);
        __m128i int_hi = _mm_cvtps_epi32(out_hi);
        __m128i resultShort = _mm_packs_epi32(int_lo, int_hi);

        _mm_store_si128((__m128i*)magnitudeVectorPtr, resultShort);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_a_avx_fma(int16_t* magnitudeVector,
                                                            const lv_32fc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        __m256 cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        /* Deinterleave real and imaginary within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));

        /* Fix cross-lane ordering: mag_sq = [m0,m1,m4,m5 | m2,m3,m6,m7]
         * need [m0,m1,m2,m3 | m4,m5,m6,m7] */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));

        /* sqrt and scale */
        __m128 mag_lo = _mm_sqrt_ps(out_lo);
        __m128 mag_hi = _mm_sqrt_ps(out_hi);
        mag_lo = _mm_mul_ps(mag_lo, _mm256_castps256_ps128(vScalar));
        mag_hi = _mm_mul_ps(mag_hi, _mm256_castps256_ps128(vScalar));

        /* Convert float -> int32 -> int16 (saturating) */
        __m128i int_lo = _mm_cvtps_epi32(mag_lo);
        __m128i int_hi = _mm_cvtps_epi32(mag_hi);
        __m128i resultShort = _mm_packs_epi32(int_lo, int_hi);

        _mm_store_si128((__m128i*)magnitudeVectorPtr, resultShort);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_a_avx2(int16_t* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       const float scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 5, 1, 4, 0);
    __m256 cplxValue1, cplxValue2, result;
    __m256i resultInt;
    __m128i resultShort;

    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
        cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

        result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

        result = _mm256_sqrt_ps(result);

        result = _mm256_mul_ps(result, vScalar);

        resultInt = _mm256_cvtps_epi32(result);
        resultInt = _mm256_packs_epi32(resultInt, resultInt);
        resultInt = _mm256_permutevar8x32_epi32(
            resultInt, idx); // permute to compensate for shuffling in hadd and packs
        resultShort = _mm256_extracti128_si256(resultInt, 0);
        _mm_store_si128((__m128i*)magnitudeVectorPtr, resultShort);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32fc_s32f_magnitude_16i_a_avx2_fma(int16_t* magnitudeVector,
                                                            const lv_32fc_t* complexVector,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m256 vScalar = _mm256_set1_ps(scalar);

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;
        __m256 cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        __m256 result = _mm256_magnitudesquared_ps_avx2_fma(cplxValue1, cplxValue2);

        result = _mm256_sqrt_ps(result);
        result = _mm256_mul_ps(result, vScalar);

        __m256i resultInt = _mm256_cvtps_epi32(result);
        resultInt = _mm256_packs_epi32(resultInt, resultInt);
        resultInt = _mm256_permute4x64_epi64(resultInt, 0x08);
        _mm_store_si128((__m128i*)magnitudeVectorPtr,
                        _mm256_extracti128_si256(resultInt, 0));
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_a_avx512f(int16_t* magnitudeVector,
                                                           const lv_32fc_t* complexVector,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    __m512 vScalar = _mm512_set1_ps(scalar);

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        __m512 cplxValue1 = _mm512_load_ps(complexVectorPtr);
        __m512 cplxValue2 = _mm512_load_ps(complexVectorPtr + 16);
        complexVectorPtr += 32;

        /* Deinterleave re/im */
        __m512 re = _mm512_permutex2var_ps(cplxValue1, idx_re, cplxValue2);
        __m512 im = _mm512_permutex2var_ps(cplxValue1, idx_im, cplxValue2);

        /* mag = sqrt(re^2 + im^2) */
        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 magSquared = _mm512_add_ps(reSquared, imSquared);
        __m512 mag = _mm512_sqrt_ps(magSquared);

        /* Scale */
        mag = _mm512_mul_ps(mag, vScalar);

        /* Convert float -> int32 -> int16 (saturating) */
        __m512i magInt = _mm512_cvtps_epi32(mag);
        __m256i magShort = _mm512_cvtsepi32_epi16(magInt);

        _mm256_store_si256((__m256i*)magnitudeVectorPtr, magShort);
        magnitudeVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_32fc_s32f_magnitude_16i_a_avx512bw(
    int16_t* magnitudeVector,
    const lv_32fc_t* complexVector,
    const float scalar,
    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int thirtySecondPoints = num_points / 32;

    const float* complexVectorPtr = (const float*)complexVector;
    int16_t* magnitudeVectorPtr = magnitudeVector;

    const __m512 vScalar = _mm512_set1_ps(scalar);

    /* Indices to deinterleave re/im from two consecutive __m512 (32 floats = 16 complex) */
    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);
    /* Fix lane interleaving introduced by _mm512_packs_epi32 */
    const __m512i pack_fix = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);

    for (; number < thirtySecondPoints; number++) {
        __m512 cplx1 = _mm512_load_ps(complexVectorPtr);
        __m512 cplx2 = _mm512_load_ps(complexVectorPtr + 16);
        __m512 cplx3 = _mm512_load_ps(complexVectorPtr + 32);
        __m512 cplx4 = _mm512_load_ps(complexVectorPtr + 48);
        complexVectorPtr += 64;

        /* Deinterleave: pair 1 → 16 re + 16 im */
        __m512 re1 = _mm512_permutex2var_ps(cplx1, idx_re, cplx2);
        __m512 im1 = _mm512_permutex2var_ps(cplx1, idx_im, cplx2);

        /* Deinterleave: pair 2 → 16 re + 16 im */
        __m512 re2 = _mm512_permutex2var_ps(cplx3, idx_re, cplx4);
        __m512 im2 = _mm512_permutex2var_ps(cplx3, idx_im, cplx4);

        /* mag = sqrt(re^2 + im^2) * scalar */
        __m512 mag1 = _mm512_mul_ps(
            _mm512_sqrt_ps(
                _mm512_fmadd_ps(re1, re1, _mm512_mul_ps(im1, im1))),
            vScalar);
        __m512 mag2 = _mm512_mul_ps(
            _mm512_sqrt_ps(
                _mm512_fmadd_ps(re2, re2, _mm512_mul_ps(im2, im2))),
            vScalar);

        /* Convert float -> int32, pack int32 -> int16 (saturating), fix lane order */
        __m512i int1 = _mm512_cvtps_epi32(mag1);
        __m512i int2 = _mm512_cvtps_epi32(mag2);
        __m512i packed = _mm512_permutexvar_epi64(
            pack_fix, _mm512_packs_epi32(int1, int2));

        _mm512_store_si512((__m512i*)magnitudeVectorPtr, packed);
        magnitudeVectorPtr += 32;
    }

    number = thirtySecondPoints * 32;
    volk_32fc_s32f_magnitude_16i_generic(
        magnitudeVector + number, complexVector + number, scalar,
        num_points - number);
}

#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_32fc_s32f_magnitude_16i_a_H */
