/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_deinterleave_64f_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex floating point vector into I & Q vector
 * data. The output vectors are converted to doubles.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_deinterleave_64f_x2(double* iBuffer, double* qBuffer, const
 * lv_32fc_t* complexVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 * \li qBuffer: The Q buffer output data.
 *
 * \b Example
 * Generate complex numbers around the top half of the unit circle and
 * deinterleave into real and imaginary double buffers.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   double* re = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   double* im = (double*)volk_malloc(sizeof(double)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *
 *   volk_32fc_deinterleave_64f_x2(re, im, in, N);
 *
 *   printf("          re  | im\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %+.1g | %+.1g\n", ii, re[ii], im[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(re);
 *   volk_free(im);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_deinterleave_64f_x2_u_H
#define INCLUDED_volk_32fc_deinterleave_64f_x2_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_deinterleave_64f_x2_generic(double* iBuffer,
                                                          double* qBuffer,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = (double)*complexVectorPtr++;
        *qBufferPtr++ = (double)*complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_u_sse2(double* iBuffer,
                                                         double* qBuffer,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int halfPoints = num_points / 2;
    __m128 cplxValue, fVal;
    __m128d dVal;

    for (; number < halfPoints; number++) {

        cplxValue = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        // Arrange in i1i2i1i2 format
        fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(2, 0, 2, 0));
        dVal = _mm_cvtps_pd(fVal);
        _mm_storeu_pd(iBufferPtr, dVal);

        // Arrange in q1q2q1q2 format
        fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(3, 1, 3, 1));
        dVal = _mm_cvtps_pd(fVal);
        _mm_storeu_pd(qBufferPtr, dVal);

        iBufferPtr += 2;
        qBufferPtr += 2;
    }

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_u_avx(double* iBuffer,
                                                        double* qBuffer,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int quarterPoints = num_points / 4;
    __m256 cplxValue;
    __m128 complexH, complexL, fVal;
    __m256d dVal;

    for (; number < quarterPoints; number++) {

        cplxValue = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        complexH = _mm256_extractf128_ps(cplxValue, 1);
        complexL = _mm256_extractf128_ps(cplxValue, 0);

        // Arrange in i1i2i1i2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(2, 0, 2, 0));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_storeu_pd(iBufferPtr, dVal);

        // Arrange in q1q2q1q2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(3, 1, 3, 1));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_storeu_pd(qBufferPtr, dVal);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_u_avx2(double* iBuffer,
                                                        double* qBuffer,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int quarterPoints = num_points / 4;
    __m256 cplxValue;
    __m128 complexH, complexL, fVal;
    __m256d dVal;

    for (; number < quarterPoints; number++) {

        cplxValue = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        complexH = _mm256_extractf128_ps(cplxValue, 1);
        complexL = _mm256_extractf128_ps(cplxValue, 0);

        // Arrange in i1i2i1i2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(2, 0, 2, 0));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_storeu_pd(iBufferPtr, dVal);

        // Arrange in q1q2q1q2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(3, 1, 3, 1));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_storeu_pd(qBufferPtr, dVal);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_u_avx512f(double* iBuffer,
                                                             double* qBuffer,
                                                             const lv_32fc_t* complexVector,
                                                             unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int sixteenthPoints = num_points / 16;

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {

        __m512 cplxValue1 = _mm512_loadu_ps(complexVectorPtr);
        __m512 cplxValue2 = _mm512_loadu_ps(complexVectorPtr + 16);
        complexVectorPtr += 32;

        // Deinterleave: extract real and imaginary parts
        __m512 reFloat = _mm512_permutex2var_ps(cplxValue1, idx_re, cplxValue2);
        __m512 imFloat = _mm512_permutex2var_ps(cplxValue1, idx_im, cplxValue2);

        // Convert lower 8 floats to doubles and store
        __m256 reLo = _mm512_castps512_ps256(reFloat);
        __m256 reHi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(reFloat), 1));
        _mm512_storeu_pd(iBufferPtr, _mm512_cvtps_pd(reLo));
        _mm512_storeu_pd(iBufferPtr + 8, _mm512_cvtps_pd(reHi));

        __m256 imLo = _mm512_castps512_ps256(imFloat);
        __m256 imHi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(imFloat), 1));
        _mm512_storeu_pd(qBufferPtr, _mm512_cvtps_pd(imLo));
        _mm512_storeu_pd(qBufferPtr + 8, _mm512_cvtps_pd(imHi));

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    volk_32fc_deinterleave_64f_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_32fc_t*)complexVectorPtr, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_deinterleave_64f_x2_neonv8(double* iBuffer,
                                                       double* qBuffer,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int half_points = num_points / 2;
    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;
    float32x2x2_t complexInput;
    float64x2_t iVal, qVal;

    for (number = 0; number < half_points; number++) {
        complexInput = vld2_f32(complexVectorPtr);

        iVal = vcvt_f64_f32(complexInput.val[0]);
        qVal = vcvt_f64_f32(complexInput.val[1]);

        vst1q_f64(iBufferPtr, iVal);
        vst1q_f64(qBufferPtr, qVal);

        complexVectorPtr += 4;
        iBufferPtr += 2;
        qBufferPtr += 2;
    }

    for (number = half_points * 2; number < num_points; number++) {
        *iBufferPtr++ = (double)*complexVectorPtr++;
        *qBufferPtr++ = (double)*complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_deinterleave_64f_x2_rvv(double* iBuffer,
                                                      double* qBuffer,
                                                      const lv_32fc_t* complexVector,
                                                      unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 0, vl));
        vfloat32m4_t vi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 32, vl));
        __riscv_vse64(iBuffer, __riscv_vfwcvt_f(vr, vl), vl);
        __riscv_vse64(qBuffer, __riscv_vfwcvt_f(vi, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_deinterleave_64f_x2_rvvseg(double* iBuffer,
                                                         double* qBuffer,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0);
        vfloat32m4_t vi = __riscv_vget_f32m4(vc, 1);
        __riscv_vse64(iBuffer, __riscv_vfwcvt_f(vr, vl), vl);
        __riscv_vse64(qBuffer, __riscv_vfwcvt_f(vi, vl), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_32fc_deinterleave_64f_x2_u_H */
#ifndef INCLUDED_volk_32fc_deinterleave_64f_x2_a_H
#define INCLUDED_volk_32fc_deinterleave_64f_x2_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_a_sse2(double* iBuffer,
                                                         double* qBuffer,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int halfPoints = num_points / 2;
    __m128 cplxValue, fVal;
    __m128d dVal;

    for (; number < halfPoints; number++) {

        cplxValue = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        // Arrange in i1i2i1i2 format
        fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(2, 0, 2, 0));
        dVal = _mm_cvtps_pd(fVal);
        _mm_store_pd(iBufferPtr, dVal);

        // Arrange in q1q2q1q2 format
        fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(3, 1, 3, 1));
        dVal = _mm_cvtps_pd(fVal);
        _mm_store_pd(qBufferPtr, dVal);

        iBufferPtr += 2;
        qBufferPtr += 2;
    }

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_a_avx(double* iBuffer,
                                                        double* qBuffer,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int quarterPoints = num_points / 4;
    __m256 cplxValue;
    __m128 complexH, complexL, fVal;
    __m256d dVal;

    for (; number < quarterPoints; number++) {

        cplxValue = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        complexH = _mm256_extractf128_ps(cplxValue, 1);
        complexL = _mm256_extractf128_ps(cplxValue, 0);

        // Arrange in i1i2i1i2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(2, 0, 2, 0));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_store_pd(iBufferPtr, dVal);

        // Arrange in q1q2q1q2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(3, 1, 3, 1));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_store_pd(qBufferPtr, dVal);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_a_avx2(double* iBuffer,
                                                        double* qBuffer,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int quarterPoints = num_points / 4;
    __m256 cplxValue;
    __m128 complexH, complexL, fVal;
    __m256d dVal;

    for (; number < quarterPoints; number++) {

        cplxValue = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        complexH = _mm256_extractf128_ps(cplxValue, 1);
        complexL = _mm256_extractf128_ps(cplxValue, 0);

        // Arrange in i1i2i1i2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(2, 0, 2, 0));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_store_pd(iBufferPtr, dVal);

        // Arrange in q1q2q1q2 format
        fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(3, 1, 3, 1));
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_store_pd(qBufferPtr, dVal);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_deinterleave_64f_x2_a_avx512f(double* iBuffer,
                                                             double* qBuffer,
                                                             const lv_32fc_t* complexVector,
                                                             unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (const float*)complexVector;
    double* iBufferPtr = iBuffer;
    double* qBufferPtr = qBuffer;

    const unsigned int sixteenthPoints = num_points / 16;

    const __m512i idx_re = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                            14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i idx_im = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                            15, 13, 11, 9, 7, 5, 3, 1);

    for (; number < sixteenthPoints; number++) {

        __m512 cplxValue1 = _mm512_load_ps(complexVectorPtr);
        __m512 cplxValue2 = _mm512_load_ps(complexVectorPtr + 16);
        complexVectorPtr += 32;

        // Deinterleave: extract real and imaginary parts
        __m512 reFloat = _mm512_permutex2var_ps(cplxValue1, idx_re, cplxValue2);
        __m512 imFloat = _mm512_permutex2var_ps(cplxValue1, idx_im, cplxValue2);

        // Convert lower 8 floats to doubles and store
        __m256 reLo = _mm512_castps512_ps256(reFloat);
        __m256 reHi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(reFloat), 1));
        _mm512_store_pd(iBufferPtr, _mm512_cvtps_pd(reLo));
        _mm512_store_pd(iBufferPtr + 8, _mm512_cvtps_pd(reHi));

        __m256 imLo = _mm512_castps512_ps256(imFloat);
        __m256 imHi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(imFloat), 1));
        _mm512_store_pd(qBufferPtr, _mm512_cvtps_pd(imLo));
        _mm512_store_pd(qBufferPtr + 8, _mm512_cvtps_pd(imHi));

        iBufferPtr += 16;
        qBufferPtr += 16;
    }

    volk_32fc_deinterleave_64f_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_32fc_t*)complexVectorPtr, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32fc_deinterleave_64f_x2_a_H */
