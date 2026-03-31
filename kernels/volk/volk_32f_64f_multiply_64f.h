/* -*- c++ -*- */
/*
 * Copyright 2018 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_64f_multiply_64f
 *
 * \b Overview
 *
 * Multiplies a single-precision float vector by a double-precision vector,
 * producing a double-precision result. Each float sample is widened to double
 * before multiplication: c[i] = (double)a[i] * b[i].
 *
 * This mixed-precision element-wise multiply is useful in signal processing
 * pipelines where double-precision correction factors (e.g. gain coefficients,
 * frequency-offset ramps, or calibration weights) must be applied to
 * single-precision sample streams without accumulating rounding error.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_64f_multiply_64f(double* cVector, const float* aVector, const double*
 * bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: Input sample vector (float).
 * \li bVector: Input coefficient vector (double).
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li cVector: The output product vector (double).
 *
 * \b Example
 * Multiply a constant float vector by a constant double vector.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* aVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 * double* bVector = (double*)volk_malloc(sizeof(double) * N, alignment);
 * double* out = (double*)volk_malloc(sizeof(double) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = 2.5f;
 *     bVector[i] = 4.0;
 * }
 *
 * // Expected: each element = 2.5 * 4.0 = 10.0
 * double expected = 2.5 * 4.0;
 *
 * volk_32f_64f_multiply_64f(out, aVector, bVector, N);
 *
 * printf("Expected: %f\n", expected);
 * printf("Result:   %f\n", out[0]);
 *
 * volk_free(aVector);
 * volk_free(bVector);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_64f_multiply_64f_H
#define INCLUDED_volk_32f_64f_multiply_64f_H

#include <inttypes.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_64f_multiply_64f_generic(double* cVector,
                                                     const float* aVector,
                                                     const double* bVector,
                                                     unsigned int num_points)
{
    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) * (*bPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */

/*
 * Unaligned versions
 */


#ifdef LV_HAVE_AVX

#include <immintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_64f_multiply_64f_u_avx(double* cVector,
                                                   const float* aVector,
                                                   const double* bVector,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;

    __m256 aVal;
    __m128 aVal1, aVal2;
    __m256d aDbl1, aDbl2, bVal1, bVal2, cVal1, cVal2;
    for (; number < eighth_points; number++) {

        aVal = _mm256_loadu_ps(aPtr);
        bVal1 = _mm256_loadu_pd(bPtr);
        bVal2 = _mm256_loadu_pd(bPtr + 4);

        aVal1 = _mm256_extractf128_ps(aVal, 0);
        aVal2 = _mm256_extractf128_ps(aVal, 1);

        aDbl1 = _mm256_cvtps_pd(aVal1);
        aDbl2 = _mm256_cvtps_pd(aVal2);

        cVal1 = _mm256_mul_pd(aDbl1, bVal1);
        cVal2 = _mm256_mul_pd(aDbl2, bVal2);

        _mm256_storeu_pd(cPtr, cVal1);     // Store the results back into the C container
        _mm256_storeu_pd(cPtr + 4, cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) * (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX

#include <immintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_64f_multiply_64f_a_avx(double* cVector,
                                                   const float* aVector,
                                                   const double* bVector,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;

    __m256 aVal;
    __m128 aVal1, aVal2;
    __m256d aDbl1, aDbl2, bVal1, bVal2, cVal1, cVal2;
    for (; number < eighth_points; number++) {

        aVal = _mm256_load_ps(aPtr);
        bVal1 = _mm256_load_pd(bPtr);
        bVal2 = _mm256_load_pd(bPtr + 4);

        aVal1 = _mm256_extractf128_ps(aVal, 0);
        aVal2 = _mm256_extractf128_ps(aVal, 1);

        aDbl1 = _mm256_cvtps_pd(aVal1);
        aDbl2 = _mm256_cvtps_pd(aVal2);

        cVal1 = _mm256_mul_pd(aDbl1, bVal1);
        cVal2 = _mm256_mul_pd(aDbl2, bVal2);

        _mm256_store_pd(cPtr, cVal1);     // Store the results back into the C container
        _mm256_store_pd(cPtr + 4, cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) * (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_64f_multiply_64f_neonv8(double* cVector,
                                                    const float* aVector,
                                                    const double* bVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;

    for (; number < eighth_points; number++) {
        float32x4_t aVal0 = vld1q_f32(aPtr);
        float32x4_t aVal1 = vld1q_f32(aPtr + 4);
        __VOLK_PREFETCH(aPtr + 8);

        float64x2_t bVal0 = vld1q_f64(bPtr);
        float64x2_t bVal1 = vld1q_f64(bPtr + 2);
        float64x2_t bVal2 = vld1q_f64(bPtr + 4);
        float64x2_t bVal3 = vld1q_f64(bPtr + 6);
        __VOLK_PREFETCH(bPtr + 8);

        float64x2_t aDbl0 = vcvt_f64_f32(vget_low_f32(aVal0));
        float64x2_t aDbl1 = vcvt_f64_f32(vget_high_f32(aVal0));
        float64x2_t aDbl2 = vcvt_f64_f32(vget_low_f32(aVal1));
        float64x2_t aDbl3 = vcvt_f64_f32(vget_high_f32(aVal1));

        float64x2_t cVal0 = vmulq_f64(aDbl0, bVal0);
        float64x2_t cVal1 = vmulq_f64(aDbl1, bVal1);
        float64x2_t cVal2 = vmulq_f64(aDbl2, bVal2);
        float64x2_t cVal3 = vmulq_f64(aDbl3, bVal3);

        vst1q_f64(cPtr, cVal0);
        vst1q_f64(cPtr + 2, cVal1);
        vst1q_f64(cPtr + 4, cVal2);
        vst1q_f64(cPtr + 6, cVal3);

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) * (*bPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_64f_multiply_64f_rvv(double* cVector,
                                                 const float* aVector,
                                                 const double* bVector,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vfloat64m8_t va = __riscv_vfwcvt_f(__riscv_vle32_v_f32m4(aVector, vl), vl);
        vfloat64m8_t vb = __riscv_vle64_v_f64m8(bVector, vl);
        __riscv_vse64(cVector, __riscv_vfmul(va, vb, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_64f_multiply_64f_u_H */
