/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_divide_32f
 *
 * \b Overview
 *
 * Performs element-wise division of two floating-point vectors:
 *
 * c[i] = a[i] / b[i]
 *
 * Element-wise division is used in signal processing for operations such as
 * normalization, computing signal-to-noise ratios, applying per-bin gain
 * corrections in spectral analysis, and implementing adaptive gain control
 * where each sample is scaled by a per-element divisor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_divide_32f(float* cVector, const float* aVector, const float* bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The numerator vector of input samples (float).
 * \li bVector: The denominator vector of input samples (float).
 * \li num_points: The number of float values in each input vector.
 *
 * \b Outputs
 * \li cVector: The output vector of quotients (float).
 *
 * \b Example
 * Divide a constant vector by another and verify the result.
 * \code
 *   unsigned int N = 4;
 *   unsigned int alignment = volk_get_alignment();
 *   float* aVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 *   float* bVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 *   float* cVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 *   for (unsigned int i = 0; i < N; ++i) {
 *       aVector[i] = 10.0f;
 *       bVector[i] = 2.0f;
 *   }
 *
 *   // Expected: 10.0 / 2.0 = 5.0 for every element
 *   float expected = 5.0f;
 *
 *   volk_32f_x2_divide_32f(cVector, aVector, bVector, N);
 *
 *   printf("Expected: %1.2f\n", expected);
 *   printf("Result:   %1.2f\n", cVector[0]);
 *
 *   volk_free(aVector);
 *   volk_free(bVector);
 *   volk_free(cVector);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_divide_32f_a_H
#define INCLUDED_volk_32f_x2_divide_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_divide_32f_a_avx512f(float* cVector,
                                                    const float* aVector,
                                                    const float* bVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;

    __m512 aVal, bVal, cVal;
    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_load_ps(aPtr);
        bVal = _mm512_load_ps(bPtr);

        cVal = _mm512_div_ps(aVal, bVal);

        _mm512_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_divide_32f_a_avx(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;

    __m256 aVal, bVal, cVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        bVal = _mm256_load_ps(bPtr);

        cVal = _mm256_div_ps(aVal, bVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_divide_32f_a_sse(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;

    __m128 aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        bVal = _mm_load_ps(bPtr);

        cVal = _mm_div_ps(aVal, bVal);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_divide_32f_neon(float* cVector,
                                               const float* aVector,
                                               const float* bVector,
                                               unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;

    float32x4x4_t aVal, bVal, bInv, cVal;

    const unsigned int eighthPoints = num_points / 16;
    unsigned int number = 0;
    for (; number < eighthPoints; number++) {
        aVal = vld4q_f32(aPtr);
        aPtr += 16;
        bVal = vld4q_f32(bPtr);
        bPtr += 16;

        __VOLK_PREFETCH(aPtr + 16);
        __VOLK_PREFETCH(bPtr + 16);

        bInv.val[0] = vrecpeq_f32(bVal.val[0]);
        bInv.val[0] = vmulq_f32(bInv.val[0], vrecpsq_f32(bInv.val[0], bVal.val[0]));
        bInv.val[0] = vmulq_f32(bInv.val[0], vrecpsq_f32(bInv.val[0], bVal.val[0]));
        cVal.val[0] = vmulq_f32(aVal.val[0], bInv.val[0]);

        bInv.val[1] = vrecpeq_f32(bVal.val[1]);
        bInv.val[1] = vmulq_f32(bInv.val[1], vrecpsq_f32(bInv.val[1], bVal.val[1]));
        bInv.val[1] = vmulq_f32(bInv.val[1], vrecpsq_f32(bInv.val[1], bVal.val[1]));
        cVal.val[1] = vmulq_f32(aVal.val[1], bInv.val[1]);

        bInv.val[2] = vrecpeq_f32(bVal.val[2]);
        bInv.val[2] = vmulq_f32(bInv.val[2], vrecpsq_f32(bInv.val[2], bVal.val[2]));
        bInv.val[2] = vmulq_f32(bInv.val[2], vrecpsq_f32(bInv.val[2], bVal.val[2]));
        cVal.val[2] = vmulq_f32(aVal.val[2], bInv.val[2]);

        bInv.val[3] = vrecpeq_f32(bVal.val[3]);
        bInv.val[3] = vmulq_f32(bInv.val[3], vrecpsq_f32(bInv.val[3], bVal.val[3]));
        bInv.val[3] = vmulq_f32(bInv.val[3], vrecpsq_f32(bInv.val[3], bVal.val[3]));
        cVal.val[3] = vmulq_f32(aVal.val[3], bInv.val[3]);

        vst4q_f32(cPtr, cVal);
        cPtr += 16;
    }

    for (number = eighthPoints * 16; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_divide_32f_neonv8(float* cVector,
                                                 const float* aVector,
                                                 const float* bVector,
                                                 unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = aVector;
    const float* bPtr = bVector;
    float* cPtr = cVector;

    for (unsigned int number = 0; number < eighthPoints; number++) {
        float32x4_t a0 = vld1q_f32(aPtr);
        float32x4_t a1 = vld1q_f32(aPtr + 4);
        float32x4_t b0 = vld1q_f32(bPtr);
        float32x4_t b1 = vld1q_f32(bPtr + 4);
        __VOLK_PREFETCH(aPtr + 16);
        __VOLK_PREFETCH(bPtr + 16);

        /* ARMv8 has native divide instruction */
        vst1q_f32(cPtr, vdivq_f32(a0, b0));
        vst1q_f32(cPtr + 4, vdivq_f32(a1, b1));

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_divide_32f_generic(float* cVector,
                                                  const float* aVector,
                                                  const float* bVector,
                                                  unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC

extern void volk_32f_x2_divide_32f_a_orc_impl(float* cVector,
                                              const float* aVector,
                                              const float* bVector,
                                              int num_points);

static inline void volk_32f_x2_divide_32f_u_orc(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    volk_32f_x2_divide_32f_a_orc_impl(cVector, aVector, bVector, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_32f_x2_divide_32f_a_H */


#ifndef INCLUDED_volk_32f_x2_divide_32f_u_H
#define INCLUDED_volk_32f_x2_divide_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_divide_32f_u_avx512f(float* cVector,
                                                    const float* aVector,
                                                    const float* bVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;

    __m512 aVal, bVal, cVal;
    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_loadu_ps(aPtr);
        bVal = _mm512_loadu_ps(bPtr);

        cVal = _mm512_div_ps(aVal, bVal);

        _mm512_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_divide_32f_u_avx(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;

    __m256 aVal, bVal, cVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        bVal = _mm256_loadu_ps(bPtr);

        cVal = _mm256_div_ps(aVal, bVal);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) / (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_x2_divide_32f_rvv(float* cVector,
                                              const float* aVector,
                                              const float* bVector,
                                              unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(aVector, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(bVector, vl);
        __riscv_vse32(cVector, __riscv_vfdiv(va, vb, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_x2_divide_32f_u_H */
