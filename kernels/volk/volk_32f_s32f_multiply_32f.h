/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_multiply_32f
 *
 * \b Overview
 *
 * Multiplies a floating point vector by a floating point scalar:
 * cVector[i] = aVector[i] * scalar.
 *
 * Scalar-vector multiplication is a fundamental building block in signal
 * processing, commonly used for gain control, amplitude scaling, and
 * normalization. In an AGC loop, for example, this kernel applies the
 * computed gain correction factor uniformly across a block of samples.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_multiply_32f(float* cVector, const float* aVector, const float
 * scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of samples (float).
 * \li scalar: The scalar gain factor to multiply against \p aVector (float).
 * \li num_points: The number of samples to process.
 *
 * \b Outputs
 * \li cVector: The output vector of scaled samples (float).
 *
 * \b Example
 * Scale a constant vector by 3.0 and verify one element.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* aVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* cVector = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = 2.0f;
 * }
 * float scalar = 3.0f;
 *
 * // Expected: each output element = 2.0 * 3.0 = 6.0
 * float expected = 2.0f * 3.0f;
 *
 * volk_32f_s32f_multiply_32f(cVector, aVector, scalar, N);
 *
 * printf("Expected: %f\n", expected);
 * printf("Result:   %f\n", cVector[0]);
 *
 * volk_free(aVector);
 * volk_free(cVector);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_multiply_32f_u_H
#define INCLUDED_volk_32f_s32f_multiply_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_s32f_multiply_32f_generic(float* cVector,
                                                      const float* aVector,
                                                      const float scalar,
                                                      unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; number++) {
        *cVector++ = (*aVector++) * scalar;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_multiply_32f_u_sse(float* cVector,
                                                    const float* aVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m128 bVal = _mm_set_ps1(scalar);
    for (unsigned int number = 0; number < quarterPoints; number++) {
        __m128 aVal = _mm_loadu_ps(aPtr);

        __m128 cVal = _mm_mul_ps(aVal, bVal);

        _mm_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        cPtr += 4;
    }

    for (unsigned int number = quarterPoints * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * scalar;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_multiply_32f_u_avx(float* cVector,
                                                    const float* aVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m256 bVal = _mm256_set1_ps(scalar);
    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        __m256 cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * scalar;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_multiply_32f_u_avx2(float* cVector,
                                                    const float* aVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m256 bVal = _mm256_set1_ps(scalar);
    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aPtr);

        __m256 cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * scalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_multiply_32f_u_avx512f(float* cVector,
                                                         const float* aVector,
                                                         const float scalar,
                                                         unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m512 bVal = _mm512_set1_ps(scalar);
    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aPtr);

        __m512 cVal = _mm512_mul_ps(aVal, bVal);

        _mm512_storeu_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    volk_32f_s32f_multiply_32f_generic(
        cPtr, aPtr, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_multiply_32f_neon(float* cVector,
                                                     const float* aVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float* inputPtr = aVector;
    float* outputPtr = cVector;

    for (unsigned int number = 0; number < quarterPoints; number++) {
        float32x4_t aVal = vld1q_f32(inputPtr);       // Load into NEON regs
        float32x4_t cVal = vmulq_n_f32(aVal, scalar); // Do the multiply
        vst1q_f32(outputPtr, cVal);                   // Store results back to output
        inputPtr += 4;
        outputPtr += 4;
    }

    for (unsigned int number = quarterPoints * 4; number < num_points; number++) {
        *outputPtr++ = (*inputPtr++) * scalar;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_multiply_32f_neonv8(float* cVector,
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

        vst1q_f32(cPtr, vmulq_f32(a0, scalarVec));
        vst1q_f32(cPtr + 4, vmulq_f32(a1, scalarVec));

        aPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * scalar;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_multiply_32f_rvv(float* cVector,
                                                  const float* aVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(aVector, vl);
        __riscv_vse32(cVector, __riscv_vfmul(v, scalar, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RISCV64
extern void volk_32f_s32f_multiply_32f_sifive_u74(float* cVector,
                                                  const float* aVector,
                                                  const float scalar,
                                                  unsigned int num_points);
#endif /* LV_HAVE_RISCV64 */

#ifdef LV_HAVE_ORC

extern void volk_32f_s32f_multiply_32f_a_orc_impl(float* dst,
                                                  const float* src,
                                                  const float scalar,
                                                  int num_points);

static inline void volk_32f_s32f_multiply_32f_u_orc(float* cVector,
                                                    const float* aVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    volk_32f_s32f_multiply_32f_a_orc_impl(cVector, aVector, scalar, num_points);
}

#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_32f_s32f_multiply_32f_u_H */


#ifndef INCLUDED_volk_32f_s32f_multiply_32f_a_H
#define INCLUDED_volk_32f_s32f_multiply_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_multiply_32f_a_sse(float* cVector,
                                                    const float* aVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m128 bVal = _mm_set_ps1(scalar);
    for (unsigned int number = 0; number < quarterPoints; number++) {
        __m128 aVal = _mm_load_ps(aPtr);

        __m128 cVal = _mm_mul_ps(aVal, bVal);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        cPtr += 4;
    }

    for (unsigned int number = quarterPoints * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * scalar;
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_multiply_32f_a_avx(float* cVector,
                                                    const float* aVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m256 bVal = _mm256_set1_ps(scalar);
    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        __m256 cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * scalar;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_multiply_32f_a_avx2(float* cVector,
                                                    const float* aVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m256 bVal = _mm256_set1_ps(scalar);
    for (unsigned int number = 0; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aPtr);

        __m256 cVal = _mm256_mul_ps(aVal, bVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * scalar;
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_multiply_32f_a_avx512f(float* cVector,
                                                         const float* aVector,
                                                         const float scalar,
                                                         unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    const __m512 bVal = _mm512_set1_ps(scalar);
    for (unsigned int number = 0; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aPtr);

        __m512 cVal = _mm512_mul_ps(aVal, bVal);

        _mm512_store_ps(cPtr, cVal);

        aPtr += 16;
        cPtr += 16;
    }

    volk_32f_s32f_multiply_32f_generic(
        cPtr, aPtr, scalar, num_points - sixteenthPoints * 16);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32f_s32f_multiply_32f_a_H */
