/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_64f_x2_max_64f
 *
 * \b Overview
 *
 * Computes the element-wise maximum of two double-precision vectors:
 *
 * c[i] = max(a[i], b[i])
 *
 * In signal processing, element-wise maximum operations are used for envelope
 * detection, spectral peak tracking, combining diversity receiver paths, and
 * implementing clipping or limiting stages in AGC loops.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64f_x2_max_64f(double* cVector, const double* aVector, const double* bVector,
 * unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: First input vector (double).
 * \li bVector: Second input vector (double).
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li cVector: The output vector (double).
 *
 * \b Example
 * Maximum of a constant vector (3.0) and a counting vector (0, 1, 2, 3, 4).
 * \code
 * unsigned int N = 5;
 * unsigned int alignment = volk_get_alignment();
 * double* aVector = (double*)volk_malloc(sizeof(double) * N, alignment);
 * double* bVector = (double*)volk_malloc(sizeof(double) * N, alignment);
 * double* out = (double*)volk_malloc(sizeof(double) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = 3.0;
 *     bVector[i] = (double)i;
 * }
 *
 * // Expected: max(3,0)=3, max(3,1)=3, max(3,2)=3, max(3,3)=3, max(3,4)=4
 *
 * volk_64f_x2_max_64f(out, aVector, bVector, N);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     double expected = (aVector[i] > bVector[i]) ? aVector[i] : bVector[i];
 *     printf("Expected: %1.2g  Result: %1.2g\n", expected, out[i]);
 * }
 *
 * volk_free(aVector);
 * volk_free(bVector);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_64f_x2_max_64f_u_H
#define INCLUDED_volk_64f_x2_max_64f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_64f_x2_max_64f_generic(double* cVector,
                                               const double* aVector,
                                               const double* bVector,
                                               unsigned int num_points)
{
    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_x2_max_64f_u_avx(double* cVector,
                                             const double* aVector,
                                             const double* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m256d aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_loadu_pd(aPtr);
        bVal = _mm256_loadu_pd(bPtr);

        cVal = _mm256_max_pd(aVal, bVal);

        _mm256_storeu_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_64f_x2_max_64f_u_avx2(double* cVector,
                                             const double* aVector,
                                             const double* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m256d aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_loadu_pd(aPtr);
        bVal = _mm256_loadu_pd(bPtr);

        cVal = _mm256_max_pd(aVal, bVal);

        _mm256_storeu_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_x2_max_64f_u_avx512f(double* cVector,
                                                 const double* aVector,
                                                 const double* bVector,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eigthPoints = num_points / 8;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m512d aVal, bVal, cVal;
    for (; number < eigthPoints; number++) {

        aVal = _mm512_loadu_pd(aPtr);
        bVal = _mm512_loadu_pd(bPtr);

        cVal = _mm512_max_pd(aVal, bVal);

        _mm512_storeu_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eigthPoints * 8;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_64f_x2_max_64f_neonv8(double* cVector,
                                              const double* aVector,
                                              const double* bVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    for (; number < quarter_points; number++) {
        float64x2_t aVal0 = vld1q_f64(aPtr);
        float64x2_t aVal1 = vld1q_f64(aPtr + 2);
        float64x2_t bVal0 = vld1q_f64(bPtr);
        float64x2_t bVal1 = vld1q_f64(bPtr + 2);
        __VOLK_PREFETCH(aPtr + 4);
        __VOLK_PREFETCH(bPtr + 4);

        float64x2_t cVal0 = vmaxq_f64(aVal0, bVal0);
        float64x2_t cVal1 = vmaxq_f64(aVal1, bVal1);

        vst1q_f64(cPtr, cVal0);
        vst1q_f64(cPtr + 2, cVal1);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_64f_x2_max_64f_rvv(double* cVector,
                                           const double* aVector,
                                           const double* bVector,
                                           unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vfloat64m8_t va = __riscv_vle64_v_f64m8(aVector, vl);
        vfloat64m8_t vb = __riscv_vle64_v_f64m8(bVector, vl);
        __riscv_vse64(cVector, __riscv_vfmax(va, vb, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_ORC

extern void volk_64f_x2_max_64f_a_orc_impl(double* cVector,
                                             const double* aVector,
                                             const double* bVector,
                                             int num_points);

static inline void volk_64f_x2_max_64f_u_orc(double* cVector,
                                               const double* aVector,
                                               const double* bVector,
                                               unsigned int num_points)
{
    volk_64f_x2_max_64f_a_orc_impl(cVector, aVector, bVector, num_points);
}

#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_64f_x2_max_64f_u_H */


#ifndef INCLUDED_volk_64f_x2_max_64f_a_H
#define INCLUDED_volk_64f_x2_max_64f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_x2_max_64f_a_sse2(double* cVector,
                                              const double* aVector,
                                              const double* bVector,
                                              unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m128d aVal, bVal, cVal;
    for (; number < halfPoints; number++) {

        aVal = _mm_load_pd(aPtr);
        bVal = _mm_load_pd(bPtr);

        cVal = _mm_max_pd(aVal, bVal);

        _mm_store_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 2;
        bPtr += 2;
        cPtr += 2;
    }

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_x2_max_64f_a_avx(double* cVector,
                                             const double* aVector,
                                             const double* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m256d aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_load_pd(aPtr);
        bVal = _mm256_load_pd(bPtr);

        cVal = _mm256_max_pd(aVal, bVal);

        _mm256_store_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_64f_x2_max_64f_a_avx2(double* cVector,
                                             const double* aVector,
                                             const double* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m256d aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {

        aVal = _mm256_load_pd(aPtr);
        bVal = _mm256_load_pd(bPtr);

        cVal = _mm256_max_pd(aVal, bVal);

        _mm256_store_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_x2_max_64f_a_avx512f(double* cVector,
                                                 const double* aVector,
                                                 const double* bVector,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eigthPoints = num_points / 8;

    double* cPtr = cVector;
    const double* aPtr = aVector;
    const double* bPtr = bVector;

    __m512d aVal, bVal, cVal;
    for (; number < eigthPoints; number++) {

        aVal = _mm512_load_pd(aPtr);
        bVal = _mm512_load_pd(bPtr);

        cVal = _mm512_max_pd(aVal, bVal);

        _mm512_store_pd(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eigthPoints * 8;
    for (; number < num_points; number++) {
        const double a = *aPtr++;
        const double b = *bPtr++;
        *cPtr++ = (a > b ? a : b);
    }
}
#endif /* LV_HAVE_AVX512F */


#endif /* INCLUDED_volk_64f_x2_max_64f_a_H */
