/* -*- c++ -*- */
/*
 * Copyright 2018 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_64f_add_64f
 *
 * \b Overview
 *
 * Adds a single-precision float vector to a double-precision vector element by
 * element, upcasting the float input to double before the addition. The result
 * is stored as double-precision.
 *
 * c[i] = (double)a[i] + b[i]
 *
 * Mixed-precision addition is common in signal processing pipelines where
 * samples arrive as 32-bit floats but must be accumulated or combined with
 * higher-precision quantities — for example, accumulating signal energy over
 * long integration windows or adding correction terms in adaptive filtering.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_64f_add_64f(double* cVector, const float* aVector, const
 * double* bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: First input vector of samples (float).
 * \li bVector: Second input vector (double).
 * \li num_points: The number of elements in each input vector.
 *
 * \b Outputs
 * \li cVector: The output vector containing the element-wise sums (double).
 *
 * \b Example
 * Add a constant float vector to a constant double vector and verify the sum.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* aVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 * double* bVector = (double*)volk_malloc(sizeof(double) * N, alignment);
 * double* cVector = (double*)volk_malloc(sizeof(double) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = 1.5f;
 *     bVector[i] = 2.5;
 * }
 *
 * double expected = 1.5 + 2.5;  // 4.0
 *
 * volk_32f_64f_add_64f(cVector, aVector, bVector, N);
 *
 * printf("Expected: %1.2f\n", expected);
 * printf("Result:   %1.2f\n", cVector[0]);
 *
 * volk_free(aVector);
 * volk_free(bVector);
 * volk_free(cVector);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_64f_add_64f_H
#define INCLUDED_volk_32f_64f_add_64f_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_64f_add_64f_generic(double* cVector,
                                                const float* aVector,
                                                const double* bVector,
                                                unsigned int num_points)
{
    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_64f_add_64f_neonv8(double* cVector,
                                               const float* aVector,
                                               const double* bVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    double* cPtr = cVector;
    const float* aPtr = aVector;
    const double* bPtr = bVector;

    for (; number < quarter_points; number++) {
        // Load 4 floats
        float32x4_t aVal_f32 = vld1q_f32(aPtr);
        // Load 4 doubles (2x2)
        float64x2_t bVal0 = vld1q_f64(bPtr);
        float64x2_t bVal1 = vld1q_f64(bPtr + 2);
        __VOLK_PREFETCH(aPtr + 4);
        __VOLK_PREFETCH(bPtr + 4);

        // Convert float to double (low and high halves)
        float64x2_t aVal0 = vcvt_f64_f32(vget_low_f32(aVal_f32));
        float64x2_t aVal1 = vcvt_f64_f32(vget_high_f32(aVal_f32));

        // Add
        float64x2_t cVal0 = vaddq_f64(aVal0, bVal0);
        float64x2_t cVal1 = vaddq_f64(aVal1, bVal1);

        // Store
        vst1q_f64(cPtr, cVal0);
        vst1q_f64(cPtr + 2, cVal1);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_AVX

#include <immintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_64f_add_64f_u_avx(double* cVector,
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

        cVal1 = _mm256_add_pd(aDbl1, bVal1);
        cVal2 = _mm256_add_pd(aDbl2, bVal2);

        _mm256_storeu_pd(cPtr,
                         cVal1); // Store the results back into the C container
        _mm256_storeu_pd(cPtr + 4,
                         cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX

#include <immintrin.h>
#include <xmmintrin.h>

static inline void volk_32f_64f_add_64f_a_avx(double* cVector,
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

        cVal1 = _mm256_add_pd(aDbl1, bVal1);
        cVal2 = _mm256_add_pd(aDbl2, bVal2);

        _mm256_store_pd(cPtr, cVal1); // Store the results back into the C container
        _mm256_store_pd(cPtr + 4,
                        cVal2); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        *cPtr++ = ((double)(*aPtr++)) + (*bPtr++);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_64f_add_64f_rvv(double* cVector,
                                            const float* aVector,
                                            const double* bVector,
                                            unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vfloat64m8_t va = __riscv_vfwcvt_f(__riscv_vle32_v_f32m4(aVector, vl), vl);
        vfloat64m8_t vb = __riscv_vle64_v_f64m8(bVector, vl);
        __riscv_vse64(cVector, __riscv_vfadd(va, vb, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_64f_add_64f_u_H */
