/* -*- c++ -*- */
/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_add_32f
 *
 * \b Overview
 *
 * Adds a scalar floating-point value to each element of a float vector:
 * <tt>cVector[i] = aVector[i] + scalar</tt>.
 *
 * This operation is commonly used in signal processing for DC offset correction,
 * level shifting, or bias adjustment. For example, adding a constant to baseband
 * samples compensates for DC bias introduced by an ADC or mixer stage.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_add_32f(float* cVector, const float* aVector, const float scalar,
 * unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of samples (float).
 * \li scalar: The scalar offset to add to each element (float).
 * \li num_points: The number of samples to process.
 *
 * \b Outputs
 * \li cVector: The output vector of shifted samples (float).
 *
 * \b Example
 * Add a DC offset of 10.0 to a vector of constant-valued samples.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* aVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* cVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = 3.0f;
 * }
 * float scalar = 10.0f;
 *
 * // Expected: 3.0 + 10.0 = 13.0 for each element
 * float expected = 13.0f;
 *
 * volk_32f_s32f_add_32f(cVector, aVector, scalar, N);
 *
 * printf("Expected: %f\n", expected);
 * printf("Result:   %f\n", cVector[0]);
 *
 * volk_free(aVector);
 * volk_free(cVector);
 * \endcode
 */

#include <inttypes.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_s32f_add_32f_u_H
#define INCLUDED_volk_32f_s32f_add_32f_u_H

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_add_32f_generic(float* cVector,
                                                 const float* aVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    const float* inputPtr = aVector;
    float* outputPtr = cVector;
    for (number = 0; number < num_points; number++) {
        *outputPtr = (*inputPtr) + scalar;
        inputPtr++;
        outputPtr++;
    }
}

#endif /* LV_HAVE_GENERIC */
#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_add_32f_u_sse(float* cVector,
                                               const float* aVector,
                                               const float scalar,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m128 aVal, bVal, cVal;
    bVal = _mm_set_ps1(scalar);
    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);

        cVal = _mm_add_ps(aVal, bVal);

        _mm_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_s32f_add_32f_generic(cPtr, aPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_add_32f_u_avx(float* cVector,
                                               const float* aVector,
                                               const float scalar,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, bVal, cVal;
    bVal = _mm256_set1_ps(scalar);
    for (; number < eighthPoints; number++) {

        aVal = _mm256_loadu_ps(aPtr);

        cVal = _mm256_add_ps(aVal, bVal);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_add_32f_generic(cPtr, aPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_add_32f_u_neon(float* cVector,
                                                const float* aVector,
                                                const float scalar,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const float* inputPtr = aVector;
    float* outputPtr = cVector;
    const unsigned int quarterPoints = num_points / 4;

    float32x4_t aVal, cVal, scalarvec;

    scalarvec = vdupq_n_f32(scalar);

    for (number = 0; number < quarterPoints; number++) {
        aVal = vld1q_f32(inputPtr);        // Load into NEON regs
        cVal = vaddq_f32(aVal, scalarvec); // Do the add
        vst1q_f32(outputPtr, cVal);        // Store results back to output
        inputPtr += 4;
        outputPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_s32f_add_32f_generic(outputPtr, inputPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_add_32f_neonv8(float* cVector,
                                                const float* aVector,
                                                const float scalar,
                                                unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = aVector;
    float* cPtr = cVector;
    const float32x4_t scalarVec = vdupq_n_f32(scalar);

    for (unsigned int number = 0; number < eighthPoints; number++) {
        float32x4_t a0 = vld1q_f32(aPtr);
        float32x4_t a1 = vld1q_f32(aPtr + 4);
        __VOLK_PREFETCH(aPtr + 16);

        vst1q_f32(cPtr, vaddq_f32(a0, scalarVec));
        vst1q_f32(cPtr + 4, vaddq_f32(a1, scalarVec));

        aPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + scalar;
    }
}
#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_volk_32f_s32f_add_32f_u_H */


#ifndef INCLUDED_volk_32f_s32f_add_32f_a_H
#define INCLUDED_volk_32f_s32f_add_32f_a_H

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_add_32f_a_sse(float* cVector,
                                               const float* aVector,
                                               const float scalar,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m128 aVal, bVal, cVal;
    bVal = _mm_set_ps1(scalar);
    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);

        cVal = _mm_add_ps(aVal, bVal);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_s32f_add_32f_generic(cPtr, aPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_add_32f_a_avx(float* cVector,
                                               const float* aVector,
                                               const float scalar,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, bVal, cVal;
    bVal = _mm256_set1_ps(scalar);
    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);

        cVal = _mm256_add_ps(aVal, bVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_s32f_add_32f_generic(cPtr, aPtr, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_ORC

extern void volk_32f_s32f_add_32f_a_orc_impl(float* dst,
                                             const float* src,
                                             const float scalar,
                                             int num_points);

static inline void volk_32f_s32f_add_32f_u_orc(float* cVector,
                                               const float* aVector,
                                               const float scalar,
                                               unsigned int num_points)
{
    volk_32f_s32f_add_32f_a_orc_impl(cVector, aVector, scalar, num_points);
}
#endif /* LV_HAVE_ORC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_add_32f_rvv(float* cVector,
                                             const float* aVector,
                                             const float scalar,
                                             unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(aVector, vl);
        __riscv_vse32(cVector, __riscv_vfadd(v, scalar, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_add_32f_a_H */
