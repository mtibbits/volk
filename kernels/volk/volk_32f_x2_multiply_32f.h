/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_multiply_32f
 *
 * \b Overview
 *
 * Performs element-wise multiplication of two floating-point vectors:
 *
 * c[i] = a[i] * b[i]
 *
 * Element-wise multiplication is a fundamental building block in many DSP
 * operations. It is used for applying window functions to time-domain samples
 * before an FFT, for scaling signal vectors by per-element gain in AGC loops,
 * and for frequency-domain filtering where multiplication replaces convolution.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_multiply_32f(float* cVector, const float* aVector, const float* bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: First input vector (float).
 * \li bVector: Second input vector (float).
 * \li num_points: The number of float samples in each input vector.
 *
 * \b Outputs
 * \li cVector: The element-wise product output vector (float).
 *
 * \b Example
 * Multiply a constant vector by a ramp and verify one element by hand.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* aVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* bVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* cVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = 3.0f;
 *     bVector[i] = (float)(i + 1);  // {1, 2, 3, 4}
 * }
 *
 * // Expected: c[i] = 3.0 * (i+1) => {3, 6, 9, 12}
 * float expected = 3.0f * 4.0f;  // c[3]
 *
 * volk_32f_x2_multiply_32f(cVector, aVector, bVector, N);
 *
 * printf("Expected c[3]: %1.1f\n", expected);
 * printf("Result   c[3]: %1.1f\n", cVector[3]);
 *
 * volk_free(aVector);
 * volk_free(bVector);
 * volk_free(cVector);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_multiply_32f_u_H
#define INCLUDED_volk_32f_x2_multiply_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_multiply_32f_generic(float* cVector,
                                                    const float* aVector,
                                                    const float* bVector,
                                                    unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_multiply_32f_u_sse(float* cVector,
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

        aVal = _mm_loadu_ps(aPtr);
        bVal = _mm_loadu_ps(bPtr);

        cVal = _mm_mul_ps(aVal, bVal);

        _mm_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_multiply_32f_u_avx(float* cVector,
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

        cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x2_multiply_32f_u_avx2(float* cVector,
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

        cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_multiply_32f_u_avx512f(float* cVector,
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

        cVal = _mm512_mul_ps(aVal, bVal);

        _mm512_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_multiply_32f_neon(float* cVector,
                                                 const float* aVector,
                                                 const float* bVector,
                                                 unsigned int num_points)
{
    const unsigned int quarter_points = num_points / 4;
    unsigned int number;
    float32x4_t avec, bvec, cvec;
    for (number = 0; number < quarter_points; ++number) {
        avec = vld1q_f32(aVector);
        bvec = vld1q_f32(bVector);
        cvec = vmulq_f32(avec, bvec);
        vst1q_f32(cVector, cvec);
        aVector += 4;
        bVector += 4;
        cVector += 4;
    }
    for (number = quarter_points * 4; number < num_points; ++number) {
        *cVector++ = *aVector++ * *bVector++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_multiply_32f_neonv8(float* cVector,
                                                   const float* aVector,
                                                   const float* bVector,
                                                   unsigned int num_points)
{
    unsigned int n = num_points;
    float* c = cVector;
    const float* a = aVector;
    const float* b = bVector;

    /* Process 8 floats per iteration (2x unroll) */
    while (n >= 8) {
        float32x4_t a0 = vld1q_f32(a);
        float32x4_t a1 = vld1q_f32(a + 4);
        float32x4_t b0 = vld1q_f32(b);
        float32x4_t b1 = vld1q_f32(b + 4);
        __VOLK_PREFETCH(a + 16);
        __VOLK_PREFETCH(b + 16);

        vst1q_f32(c, vmulq_f32(a0, b0));
        vst1q_f32(c + 4, vmulq_f32(a1, b1));

        a += 8;
        b += 8;
        c += 8;
        n -= 8;
    }

    /* Process remaining 4 floats */
    if (n >= 4) {
        vst1q_f32(c, vmulq_f32(vld1q_f32(a), vld1q_f32(b)));
        a += 4;
        b += 4;
        c += 4;
        n -= 4;
    }

    /* Scalar tail */
    while (n > 0) {
        *c++ = *a++ * *b++;
        n--;
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_x2_multiply_32f_rvv(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(aVector, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(bVector, vl);
        __riscv_vse32(cVector, __riscv_vfmul(va, vb, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_ORC
extern void volk_32f_x2_multiply_32f_a_orc_impl(float* cVector,
                                                const float* aVector,
                                                const float* bVector,
                                                int num_points);

static inline void volk_32f_x2_multiply_32f_u_orc(float* cVector,
                                                  const float* aVector,
                                                  const float* bVector,
                                                  unsigned int num_points)
{
    volk_32f_x2_multiply_32f_a_orc_impl(cVector, aVector, bVector, num_points);
}
#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_32f_x2_multiply_32f_u_H */


#ifndef INCLUDED_volk_32f_x2_multiply_32f_a_H
#define INCLUDED_volk_32f_x2_multiply_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_multiply_32f_a_sse(float* cVector,
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

        cVal = _mm_mul_ps(aVal, bVal);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_multiply_32f_a_avx(float* cVector,
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

        cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x2_multiply_32f_a_avx2(float* cVector,
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

        cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_x2_multiply_32f_a_avx512f(float* cVector,
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

        cVal = _mm512_mul_ps(aVal, bVal);

        _mm512_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32f_x2_multiply_32f_a_H */
