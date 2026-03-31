/* -*- c++ -*- */
/*
 * Copyright 2018 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_add_32fc
 *
 * \b Overview
 *
 * Adds two complex vectors together element by element:
 *
 * c[i] = a[i] + b[i]
 *
 * Complex addition is a fundamental building block in many DSP operations.
 * It is used for signal combining, diversity combining, beamforming, and
 * accumulating partial results in channelized receivers. For example, in a
 * multi-antenna system the baseband samples from each antenna element may be
 * summed to form a composite signal.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_add_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector,
 * const lv_32fc_t* bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: First vector of complex input samples (lv_32fc_t).
 * \li bVector: Second vector of complex input samples (lv_32fc_t).
 * \li num_points: The number of complex samples in each input vector.
 *
 * \b Outputs
 * \li cVector: The output vector of complex sums (lv_32fc_t).
 *
 * \b Example
 *
 * Adds two constant complex vectors so the expected result is easy to verify:
 * (1+2j) + (3+4j) = (4+6j) for every element.
 *
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* aVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* bVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = lv_cmake(1.0f, 2.0f);
 *     bVector[i] = lv_cmake(3.0f, 4.0f);
 * }
 *
 * lv_32fc_t expected = lv_cmake(4.0f, 6.0f);
 *
 * volk_32fc_x2_add_32fc(out, aVector, bVector, N);
 *
 * printf("Expected: (%1.1f, %1.1f)\n", lv_creal(expected), lv_cimag(expected));
 * printf("Result:   (%1.1f, %1.1f)\n", lv_creal(out[0]), lv_cimag(out[0]));
 *
 * volk_free(aVector);
 * volk_free(bVector);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_add_32fc_u_H
#define INCLUDED_volk_32fc_x2_add_32fc_u_H

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_x2_add_32fc_generic(lv_32fc_t* cVector,
                                                 const lv_32fc_t* aVector,
                                                 const lv_32fc_t* bVector,
                                                 unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_x2_add_32fc_u_sse(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;

    __m128 aVal, bVal, cVal;
    for (; number < halfPoints; number++) {

        aVal = _mm_loadu_ps((const float*)aPtr);
        bVal = _mm_loadu_ps((const float*)bPtr);

        cVal = _mm_add_ps(aVal, bVal);

        _mm_storeu_ps((float*)cPtr, cVal); // Store the results back into the C container

        aPtr += 2;
        bPtr += 2;
        cPtr += 2;
    }

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_x2_add_32fc_u_avx(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;

    __m256 aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_loadu_ps((const float*)aPtr);
        bVal = _mm256_loadu_ps((const float*)bPtr);

        cVal = _mm256_add_ps(aVal, bVal);

        _mm256_storeu_ps((float*)cPtr,
                         cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_x2_add_32fc_u_avx2(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;

    __m256 aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_loadu_ps((const float*)aPtr);
        bVal = _mm256_loadu_ps((const float*)bPtr);

        cVal = _mm256_add_ps(aVal, bVal);

        _mm256_storeu_ps((float*)cPtr,
                         cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_x2_add_32fc_u_avx512f(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = (const float*)aVector;
    const float* bPtr = (const float*)bVector;
    float* cPtr = (float*)cVector;

    __m512 aVal, bVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm512_loadu_ps(aPtr);
        bVal = _mm512_loadu_ps(bPtr);

        _mm512_storeu_ps(cPtr, _mm512_add_ps(aVal, bVal));

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    volk_32fc_x2_add_32fc_generic(
        (lv_32fc_t*)cPtr, (const lv_32fc_t*)aPtr, (const lv_32fc_t*)bPtr,
        num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_x2_add_32fc_neon(lv_32fc_t* cVector,
                                                const lv_32fc_t* aVector,
                                                const lv_32fc_t* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;
    float32x4_t aVal, bVal, cVal;
    for (number = 0; number < halfPoints; number++) {
        // Load in to NEON registers
        aVal = vld1q_f32((const float32_t*)(aPtr));
        bVal = vld1q_f32((const float32_t*)(bPtr));
        __VOLK_PREFETCH(aPtr + 2);
        __VOLK_PREFETCH(bPtr + 2);

        // vector add
        cVal = vaddq_f32(aVal, bVal);
        // Store the results back into the C container
        vst1q_f32((float*)(cPtr), cVal);

        aPtr += 2;
        bPtr += 2;
        cPtr += 2;
    }

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_x2_add_32fc_neonv8(lv_32fc_t* cVector,
                                                const lv_32fc_t* aVector,
                                                const lv_32fc_t* bVector,
                                                unsigned int num_points)
{
    const unsigned int quarterPoints = num_points / 4;

    const float* aPtr = (const float*)aVector;
    const float* bPtr = (const float*)bVector;
    float* cPtr = (float*)cVector;

    for (unsigned int number = 0; number < quarterPoints; number++) {
        float32x4_t a0 = vld1q_f32(aPtr);
        float32x4_t a1 = vld1q_f32(aPtr + 4);
        float32x4_t b0 = vld1q_f32(bPtr);
        float32x4_t b1 = vld1q_f32(bPtr + 4);
        __VOLK_PREFETCH(aPtr + 16);
        __VOLK_PREFETCH(bPtr + 16);

        vst1q_f32(cPtr, vaddq_f32(a0, b0));
        vst1q_f32(cPtr + 4, vaddq_f32(a1, b1));

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = quarterPoints * 4; number < num_points; number++) {
        cVector[number] = aVector[number] + bVector[number];
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_x2_add_32fc_rvv(lv_32fc_t* cVector,
                                             const lv_32fc_t* aVector,
                                             const lv_32fc_t* bVector,
                                             unsigned int num_points)
{
    const float* ina = (const float*)aVector;
    const float* inb = (const float*)bVector;
    float* out = (float*)cVector;
    size_t n = (size_t)num_points * 2;
    for (size_t vl; n > 0; n -= vl, ina += vl, inb += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(ina, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(inb, vl);
        __riscv_vse32(out, __riscv_vfadd(va, vb, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_x2_add_32fc_rvvseg(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  const lv_32fc_t* bVector,
                                                  unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t va =
            __riscv_vlseg2e32_v_f32m4x2((const float*)aVector, vl);
        vfloat32m4x2_t vb =
            __riscv_vlseg2e32_v_f32m4x2((const float*)bVector, vl);
        vfloat32m4_t vr =
            __riscv_vfadd(__riscv_vget_f32m4(va, 0), __riscv_vget_f32m4(vb, 0), vl);
        vfloat32m4_t vi =
            __riscv_vfadd(__riscv_vget_f32m4(va, 1), __riscv_vget_f32m4(vb, 1), vl);
        __riscv_vsseg2e32_v_f32m4x2(
            (float*)cVector, __riscv_vcreate_v_f32m4x2(vr, vi), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#ifdef LV_HAVE_ORC

extern void volk_32fc_x2_add_32fc_a_orc_impl(lv_32fc_t* cVector,
                                              const lv_32fc_t* aVector,
                                              const lv_32fc_t* bVector,
                                              int num_points);

static inline void volk_32fc_x2_add_32fc_u_orc(lv_32fc_t* cVector,
                                                const lv_32fc_t* aVector,
                                                const lv_32fc_t* bVector,
                                                unsigned int num_points)
{
    volk_32fc_x2_add_32fc_a_orc_impl(cVector, aVector, bVector, num_points);
}

#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_32fc_x2_add_32fc_u_H */

#ifndef INCLUDED_volk_32fc_x2_add_32fc_a_H
#define INCLUDED_volk_32fc_x2_add_32fc_a_H

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_x2_add_32fc_a_sse(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;

    __m128 aVal, bVal, cVal;
    for (; number < halfPoints; number++) {
        aVal = _mm_load_ps((const float*)aPtr);
        bVal = _mm_load_ps((const float*)bPtr);

        cVal = _mm_add_ps(aVal, bVal);

        _mm_store_ps((float*)cPtr, cVal); // Store the results back into the C container

        aPtr += 2;
        bPtr += 2;
        cPtr += 2;
    }

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_x2_add_32fc_a_avx(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;

    __m256 aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_load_ps((const float*)aPtr);
        bVal = _mm256_load_ps((const float*)bPtr);

        cVal = _mm256_add_ps(aVal, bVal);

        _mm256_store_ps((float*)cPtr,
                        cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_x2_add_32fc_a_avx2(lv_32fc_t* cVector,
                                               const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector,
                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;

    __m256 aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_load_ps((const float*)aPtr);
        bVal = _mm256_load_ps((const float*)bPtr);

        cVal = _mm256_add_ps(aVal, bVal);

        _mm256_store_ps((float*)cPtr,
                        cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_x2_add_32fc_a_avx512f(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* aPtr = (const float*)aVector;
    const float* bPtr = (const float*)bVector;
    float* cPtr = (float*)cVector;

    __m512 aVal, bVal;
    for (; number < eighthPoints; number++) {
        aVal = _mm512_load_ps(aPtr);
        bVal = _mm512_load_ps(bPtr);

        _mm512_store_ps(cPtr, _mm512_add_ps(aVal, bVal));

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    volk_32fc_x2_add_32fc_generic(
        (lv_32fc_t*)cPtr, (const lv_32fc_t*)aPtr, (const lv_32fc_t*)bPtr,
        num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32fc_x2_add_32fc_a_H */
