/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_deinterleave_real_32f
 *
 * \b Overview
 *
 * Deinterleaves the complex floating point vector and returns the real
 * (in-phase) component of each sample. For a complex input vector
 * \f$z[n] = I[n] + jQ[n]\f$, the output is \f$I[n]\f$.
 *
 * Extracting the in-phase component is a common step in demodulation,
 * synchronization, and baseband processing pipelines where downstream
 * blocks operate on real-valued samples rather than full complex data.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_deinterleave_real_32f(float* iBuffer, const lv_32fc_t* complexVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of interleaved I/Q samples (lv_32fc_t).
 * \li num_points: The number of complex samples to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The real-part (in-phase) output buffer (float).
 *
 * \b Example
 * Extract the real parts from four complex samples and verify the output.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* in = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * float* out = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * in[0] = lv_cmake(1.0f, 2.0f);
 * in[1] = lv_cmake(3.0f, 4.0f);
 * in[2] = lv_cmake(5.0f, 6.0f);
 * in[3] = lv_cmake(7.0f, 8.0f);
 *
 * // Expected real parts: 1.0, 3.0, 5.0, 7.0
 *
 * volk_32fc_deinterleave_real_32f(out, in, N);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     printf("Expected: %+.1f  Result: %+.1f\n", lv_creal(in[i]), out[i]);
 * }
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_deinterleave_real_32f_a_H
#define INCLUDED_volk_32fc_deinterleave_real_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_deinterleave_real_32f_a_avx2(float* iBuffer,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* iBufferPtr = iBuffer;

    __m256 cplxValue1, cplxValue2;
    __m256 iValue;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    for (; number < eighthPoints; number++) {

        cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        // Arrange in i1i2i3i4 format
        iValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        iValue = _mm256_permutevar8x32_ps(iValue, idx);

        _mm256_store_ps(iBufferPtr, iValue);

        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_deinterleave_real_32f_a_sse(float* iBuffer,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (const float*)complexVector;
    float* iBufferPtr = iBuffer;

    __m128 cplxValue1, cplxValue2, iValue;
    for (; number < quarterPoints; number++) {

        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));

        _mm_store_ps(iBufferPtr, iValue);

        iBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_deinterleave_real_32f_generic(float* iBuffer,
                                                           const lv_32fc_t* complexVector,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_deinterleave_real_32f_neon(float* iBuffer,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float32x4x2_t complexInput;

    for (number = 0; number < quarter_points; number++) {
        complexInput = vld2q_f32(complexVectorPtr);
        vst1q_f32(iBufferPtr, complexInput.val[0]);
        complexVectorPtr += 8;
        iBufferPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_deinterleave_real_32f_neonv8(float* iBuffer,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;

    for (unsigned int number = 0; number < eighthPoints; number++) {
        float32x4x2_t cplx0 = vld2q_f32(complexVectorPtr);
        float32x4x2_t cplx1 = vld2q_f32(complexVectorPtr + 8);
        __VOLK_PREFETCH(complexVectorPtr + 32);

        vst1q_f32(iBufferPtr, cplx0.val[0]);
        vst1q_f32(iBufferPtr + 4, cplx1.val[0]);

        complexVectorPtr += 16;
        iBufferPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32fc_deinterleave_real_32f_a_H */


#ifndef INCLUDED_volk_32fc_deinterleave_real_32f_u_H
#define INCLUDED_volk_32fc_deinterleave_real_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_deinterleave_real_32f_u_avx2(float* iBuffer,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (const float*)complexVector;
    float* iBufferPtr = iBuffer;

    __m256 cplxValue1, cplxValue2;
    __m256 iValue;
    __m256i idx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    for (; number < eighthPoints; number++) {

        cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        // Arrange in i1i2i3i4 format
        iValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        iValue = _mm256_permutevar8x32_ps(iValue, idx);

        _mm256_storeu_ps(iBufferPtr, iValue);

        iBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_deinterleave_real_32f_rvv(float* iBuffer,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    const uint64_t* in = (const uint64_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, iBuffer += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8(in, vl);
        __riscv_vse32((uint32_t*)iBuffer, __riscv_vnsrl(vc, 0, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_deinterleave_real_32f_u_H */
