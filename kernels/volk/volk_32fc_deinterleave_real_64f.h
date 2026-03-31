/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_deinterleave_real_64f
 *
 * \b Overview
 *
 * Deinterleaves the complex floating point vector and returns the real
 * (in-phase) component of each sample, converted to double precision.
 *
 * Extracting the in-phase component is a common step in IQ signal processing
 * pipelines where downstream operations — such as spectral analysis,
 * synchronization, or demodulation — require only the real part of the
 * baseband signal at higher numeric precision.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_deinterleave_real_64f(double* iBuffer, const lv_32fc_t*
 * complexVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of baseband samples (lv_32fc_t).
 * \li num_points: The number of complex samples to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The double-precision output buffer containing the in-phase
 * (real) component of each sample (double).
 *
 * \b Example
 * Extract real parts from constant complex values and verify the result.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* in = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * double* re = (double*)volk_malloc(sizeof(double) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     in[i] = lv_cmake(3.0f, 4.0f);
 * }
 *
 * // Expected: each output element equals 3.0
 * double expected = 3.0;
 *
 * volk_32fc_deinterleave_real_64f(re, in, N);
 *
 * printf("Expected: %f\n", expected);
 * printf("Result:   %f\n", re[0]);
 *
 * volk_free(in);
 * volk_free(re);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_deinterleave_real_64f_a_H
#define INCLUDED_volk_32fc_deinterleave_real_64f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_deinterleave_real_64f_a_avx2(double* iBuffer,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (float*)complexVector;
    double* iBufferPtr = iBuffer;

    const unsigned int quarterPoints = num_points / 4;
    __m256 cplxValue;
    __m128 fVal;
    __m256d dVal;
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
    for (; number < quarterPoints; number++) {

        cplxValue = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        // Arrange in i1i2i1i2 format
        cplxValue = _mm256_permutevar8x32_ps(cplxValue, idx);
        fVal = _mm256_extractf128_ps(cplxValue, 0);
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_store_pd(iBufferPtr, dVal);

        iBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (double)*complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_deinterleave_real_64f_a_sse2(double* iBuffer,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (float*)complexVector;
    double* iBufferPtr = iBuffer;

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

        iBufferPtr += 2;
    }

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (double)*complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_deinterleave_real_64f_generic(double* iBuffer,
                                                           const lv_32fc_t* complexVector,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* complexVectorPtr = (float*)complexVector;
    double* iBufferPtr = iBuffer;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = (double)*complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_deinterleave_real_64f_neon(double* iBuffer,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    const float* complexVectorPtr = (float*)complexVector;
    double* iBufferPtr = iBuffer;
    float32x2x4_t complexInput;
    float64x2_t iVal1;
    float64x2_t iVal2;
    float64x2x2_t iVal;

    for (number = 0; number < quarter_points; number++) {
        // Load data into register
        complexInput = vld4_f32(complexVectorPtr);

        // Perform single to double precision conversion
        iVal1 = vcvt_f64_f32(complexInput.val[0]);
        iVal2 = vcvt_f64_f32(complexInput.val[2]);
        iVal.val[0] = iVal1;
        iVal.val[1] = iVal2;

        // Store results into memory buffer
        vst2q_f64(iBufferPtr, iVal);

        // Update pointers
        iBufferPtr += 4;
        complexVectorPtr += 8;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *iBufferPtr++ = (double)*complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32fc_deinterleave_real_64f_a_H */

#ifndef INCLUDED_volk_32fc_deinterleave_real_64f_u_H
#define INCLUDED_volk_32fc_deinterleave_real_64f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_deinterleave_real_64f_u_avx2(double* iBuffer,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;

    const float* complexVectorPtr = (float*)complexVector;
    double* iBufferPtr = iBuffer;

    const unsigned int quarterPoints = num_points / 4;
    __m256 cplxValue;
    __m128 fVal;
    __m256d dVal;
    __m256i idx = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
    for (; number < quarterPoints; number++) {

        cplxValue = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        // Arrange in i1i2i1i2 format
        cplxValue = _mm256_permutevar8x32_ps(cplxValue, idx);
        fVal = _mm256_extractf128_ps(cplxValue, 0);
        dVal = _mm256_cvtps_pd(fVal);
        _mm256_storeu_pd(iBufferPtr, dVal);

        iBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = (double)*complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_deinterleave_real_64f_rvv(double* iBuffer,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    const uint64_t* in = (const uint64_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint32m4_t vi = __riscv_vnsrl(__riscv_vle64_v_u64m8(in, vl), 0, vl);
        __riscv_vse64(iBuffer, __riscv_vfwcvt_f(__riscv_vreinterpret_f32m4(vi), vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_deinterleave_real_64f_u_H */
