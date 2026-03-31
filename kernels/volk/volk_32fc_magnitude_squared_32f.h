/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_magnitude_squared_32f
 *
 * \b Overview
 *
 * Computes the squared magnitude of each complex sample:
 * magnitudeVector[i] = real(complexVector[i])^2 + imag(complexVector[i])^2.
 *
 * Squared magnitude is widely used in signal processing for power estimation,
 * energy detection, and automatic gain control (AGC), where the true magnitude
 * is not needed and avoiding the square-root operation saves computation. It is
 * also commonly used in spectral analysis to compute power spectral density
 * from FFT output.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_magnitude_squared_32f(float* magnitudeVector, const lv_32fc_t*
 * complexVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of samples (lv_32fc_t).
 * \li num_points: The number of complex samples.
 *
 * \b Outputs
 * \li magnitudeVector: The squared magnitude of each sample (float).
 *
 * \b Example
 * Compute the squared magnitude of a 3-4-5 Pythagorean triple: |3+4j|^2 = 25.
 * \code
 *   unsigned int N = 4;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 *   for (unsigned int i = 0; i < N; ++i) {
 *       in[i] = lv_cmake(3.0f, 4.0f);
 *   }
 *
 *   // Expected: 3^2 + 4^2 = 25 for each sample
 *   float expected = 25.0f;
 *
 *   volk_32fc_magnitude_squared_32f(out, in, N);
 *
 *   printf("Expected: %f\n", expected);
 *   for (unsigned int i = 0; i < N; ++i) {
 *       printf("Result[%u]: %f\n", i, out[i]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_magnitude_squared_32f_u_H
#define INCLUDED_volk_32fc_magnitude_squared_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_magnitude_squared_32f_generic(float* magnitudeVector,
                                                           const lv_32fc_t* complexVector,
                                                           unsigned int num_points)
{
    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    for (number = 0; number < num_points; number++) {
        const float real = *complexVectorPtr++;
        const float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (real * real) + (imag * imag);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void volk_32fc_magnitude_squared_32f_u_sse(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;

    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_u_sse3(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;
    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps_sse3(cplxValue1, cplxValue2);
        _mm_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_u_avx(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m256 cplxValue1, cplxValue2, result;

    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        cplxValue2 = _mm256_loadu_ps(complexVectorPtr + 8);
        result = _mm256_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm256_storeu_ps(magnitudeVectorPtr, result);

        complexVectorPtr += 16;
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_u_avx_fma(float* magnitudeVector,
                                                               const lv_32fc_t* complexVector,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        __m256 cplxValue2 = _mm256_loadu_ps(complexVectorPtr + 8);
        complexVectorPtr += 16;

        /* Deinterleave re/im within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));

        /* Fix cross-lane order using permute2f128 + SSE shuffles (AVX only, no AVX2) */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_castps128_ps256(out_lo);
        result = _mm256_insertf128_ps(result, out_hi, 1);

        _mm256_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_u_avx2(float* magnitudeVector,
                                                            const lv_32fc_t* complexVector,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        __m256 cplxValue2 = _mm256_loadu_ps(complexVectorPtr + 8);
        complexVectorPtr += 16;

        /* Deinterleave re/im within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_add_ps(_mm256_mul_ps(re, re), _mm256_mul_ps(im, im));

        /* Fix cross-lane order using permute2f128 + SSE shuffles */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_castps128_ps256(out_lo);
        result = _mm256_insertf128_ps(result, out_hi, 1);

        _mm256_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX2 */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_u_avx2_fma(float* magnitudeVector,
                                                               const lv_32fc_t* complexVector,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        __m256 cplxValue2 = _mm256_loadu_ps(complexVectorPtr + 8);
        complexVectorPtr += 16;

        /* Deinterleave re/im within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));

        /* Fix cross-lane order using permute2f128 + SSE shuffles */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_castps128_ps256(out_lo);
        result = _mm256_insertf128_ps(result, out_hi, 1);

        _mm256_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_u_avx512f(float* magnitudeVector,
                                                              const lv_32fc_t* complexVector,
                                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        __m512 cplxValue1 = _mm512_loadu_ps(complexVectorPtr);
        __m512 cplxValue2 = _mm512_loadu_ps(complexVectorPtr + 16);

        __m512 re = _mm512_permutex2var_ps(cplxValue1, idx_re, cplxValue2);
        __m512 im = _mm512_permutex2var_ps(cplxValue1, idx_im, cplxValue2);

        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 result = _mm512_add_ps(reSquared, imSquared);

        _mm512_storeu_ps(magnitudeVectorPtr, result);

        complexVectorPtr += 32;
        magnitudeVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32fc_magnitude_squared_32f_generic(
        magnitudeVectorPtr, (const lv_32fc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_magnitude_squared_32f_neon(float* magnitudeVector,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    float32x4x2_t cmplx_val;
    float32x4_t result;
    for (; number < quarterPoints; number++) {
        cmplx_val = vld2q_f32(complexVectorPtr);
        complexVectorPtr += 8;

        cmplx_val.val[0] =
            vmulq_f32(cmplx_val.val[0], cmplx_val.val[0]); // Square the values
        cmplx_val.val[1] =
            vmulq_f32(cmplx_val.val[1], cmplx_val.val[1]); // Square the values

        result =
            vaddq_f32(cmplx_val.val[0], cmplx_val.val[1]); // Add the I2 and Q2 values

        vst1q_f32(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_magnitude_squared_32f_neonv8(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int n = num_points;
    const float* in = (const float*)complexVector;
    float* out = magnitudeVector;

    /* Process 4 complex numbers per iteration using interleaved loads + pairwise add
     * Load: [r0,i0,r1,i1] [r2,i2,r3,i3]
     * Square: [r0²,i0²,r1²,i1²] [r2²,i2²,r3²,i3²]
     * Pairwise add: [r0²+i0²,r1²+i1²,r2²+i2²,r3²+i3²]
     */
    while (n >= 4) {
        float32x4_t v0 = vld1q_f32(in);     /* r0,i0,r1,i1 */
        float32x4_t v1 = vld1q_f32(in + 4); /* r2,i2,r3,i3 */
        __VOLK_PREFETCH(in + 16);

        /* Square all elements */
        v0 = vmulq_f32(v0, v0); /* r0²,i0²,r1²,i1² */
        v1 = vmulq_f32(v1, v1); /* r2²,i2²,r3²,i3² */

        /* Pairwise add: vpaddq adds adjacent pairs */
        float32x4_t mag = vpaddq_f32(v0, v1); /* r0²+i0²,r1²+i1²,r2²+i2²,r3²+i3² */

        vst1q_f32(out, mag);

        in += 8;
        out += 4;
        n -= 4;
    }

    /* Process remaining 2 complex numbers */
    if (n >= 2) {
        float32x4_t v0 = vld1q_f32(in); /* r0,i0,r1,i1 */
        v0 = vmulq_f32(v0, v0);
        float32x2_t mag = vpadd_f32(vget_low_f32(v0), vget_high_f32(v0));
        vst1_f32(out, mag);
        in += 4;
        out += 2;
        n -= 2;
    }

    /* Scalar tail */
    if (n > 0) {
        float re = *in++;
        float im = *in++;
        *out++ = (re * re) + (im * im);
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_magnitude_squared_32f_rvv(float* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 0, vl));
        vfloat32m4_t vi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 32, vl));
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(magnitudeVector, v, vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_magnitude_squared_32f_rvvseg(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0);
        vfloat32m4_t vi = __riscv_vget_f32m4(vc, 1);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(magnitudeVector, v, vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#ifdef LV_HAVE_ORC

extern void volk_32fc_magnitude_squared_32f_a_orc_impl(float* magnitudeVector,
                                                        const lv_32fc_t* complexVector,
                                                        int num_points);

static inline void volk_32fc_magnitude_squared_32f_u_orc(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    volk_32fc_magnitude_squared_32f_a_orc_impl(magnitudeVector, complexVector, num_points);
}

#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_32fc_magnitude_squared_32f_u_H */

#ifndef INCLUDED_volk_32fc_magnitude_squared_32f_a_H
#define INCLUDED_volk_32fc_magnitude_squared_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void volk_32fc_magnitude_squared_32f_a_sse(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;
    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_a_sse3(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;
    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps_sse3(cplxValue1, cplxValue2);
        _mm_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_a_avx(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m256 cplxValue1, cplxValue2, result;
    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        result = _mm256_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_a_avx_fma(float* magnitudeVector,
                                                               const lv_32fc_t* complexVector,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_load_ps(complexVectorPtr);
        __m256 cplxValue2 = _mm256_load_ps(complexVectorPtr + 8);
        complexVectorPtr += 16;

        /* Deinterleave re/im within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));

        /* Fix cross-lane order using permute2f128 + SSE shuffles (AVX only, no AVX2) */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_castps128_ps256(out_lo);
        result = _mm256_insertf128_ps(result, out_hi, 1);

        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_a_avx2(float* magnitudeVector,
                                                            const lv_32fc_t* complexVector,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_load_ps(complexVectorPtr);
        __m256 cplxValue2 = _mm256_load_ps(complexVectorPtr + 8);
        complexVectorPtr += 16;

        /* Deinterleave re/im within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_add_ps(_mm256_mul_ps(re, re), _mm256_mul_ps(im, im));

        /* Fix cross-lane order using permute2f128 + SSE shuffles */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_castps128_ps256(out_lo);
        result = _mm256_insertf128_ps(result, out_hi, 1);

        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX2 */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_a_avx2_fma(float* magnitudeVector,
                                                               const lv_32fc_t* complexVector,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    for (; number < eighthPoints; number++) {
        __m256 cplxValue1 = _mm256_load_ps(complexVectorPtr);
        __m256 cplxValue2 = _mm256_load_ps(complexVectorPtr + 8);
        complexVectorPtr += 16;

        /* Deinterleave re/im within 128-bit lanes */
        __m256 re = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        /* mag² = re*re + im*im using FMA (lane-interleaved: [0,1,4,5 | 2,3,6,7]) */
        __m256 mag_sq = _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));

        /* Fix cross-lane order using permute2f128 + SSE shuffles */
        __m128 lo = _mm256_castps256_ps128(mag_sq);
        __m128 hi = _mm256_extractf128_ps(mag_sq, 1);
        __m128 out_lo = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 out_hi = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
        __m256 result = _mm256_castps128_ps256(out_lo);
        result = _mm256_insertf128_ps(result, out_hi, 1);

        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_magnitude_squared_32f_a_avx512f(float* magnitudeVector,
                                                              const lv_32fc_t* complexVector,
                                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    const float* complexVectorPtr = (const float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {
        __m512 cplxValue1 = _mm512_load_ps(complexVectorPtr);
        __m512 cplxValue2 = _mm512_load_ps(complexVectorPtr + 16);

        __m512 re = _mm512_permutex2var_ps(cplxValue1, idx_re, cplxValue2);
        __m512 im = _mm512_permutex2var_ps(cplxValue1, idx_im, cplxValue2);

        __m512 reSquared = _mm512_mul_ps(re, re);
        __m512 imSquared = _mm512_mul_ps(im, im);
        __m512 result = _mm512_add_ps(reSquared, imSquared);

        _mm512_store_ps(magnitudeVectorPtr, result);

        complexVectorPtr += 32;
        magnitudeVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32fc_magnitude_squared_32f_generic(
        magnitudeVectorPtr, (const lv_32fc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#endif /* INCLUDED_volk_32fc_magnitude_squared_32f_a_H */
