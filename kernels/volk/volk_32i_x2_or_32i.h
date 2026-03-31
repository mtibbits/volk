/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32i_x2_or_32i
 *
 * \b Overview
 *
 * Computes the element-wise bitwise OR of two 32-bit integer vectors:
 * cVector[i] = aVector[i] | bVector[i]. This operation combines bit fields
 * from two sources, preserving all set bits from both inputs.
 *
 * Bitwise OR is used in digital signal processing for tasks such as combining
 * channel masks, merging flag vectors from parallel processing stages, or
 * assembling control words for hardware registers. It is also useful in
 * codec and protocol implementations where packed bit fields must be merged.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32i_x2_or_32i(int32_t* cVector, const int32_t* aVector, const int32_t* bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: First input vector of 32-bit integers (int32_t).
 * \li bVector: Second input vector of 32-bit integers (int32_t).
 * \li num_points: The number of 32-bit integer values to process.
 *
 * \b Outputs
 * \li cVector: The output vector of bitwise OR results (int32_t).
 *
 * \b Example
 * OR two constant vectors and verify against the expected result.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * int32_t* aVector = (int32_t*)volk_malloc(sizeof(int32_t) * N, alignment);
 * int32_t* bVector = (int32_t*)volk_malloc(sizeof(int32_t) * N, alignment);
 * int32_t* cVector = (int32_t*)volk_malloc(sizeof(int32_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     aVector[i] = 0x0F;
 *     bVector[i] = 0xF0;
 * }
 *
 * // Expected: 0x0F | 0xF0 = 0xFF for every element
 * int32_t expected = 0xFF;
 *
 * volk_32i_x2_or_32i(cVector, aVector, bVector, N);
 *
 * printf("Expected: 0x%02x\n", expected);
 * printf("Result:   0x%02x\n", cVector[0]);
 *
 * volk_free(aVector);
 * volk_free(bVector);
 * volk_free(cVector);
 * \endcode
 */

#ifndef INCLUDED_volk_32i_x2_or_32i_a_H
#define INCLUDED_volk_32i_x2_or_32i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_a_avx512f(int32_t* cVector,
                                                const int32_t* aVector,
                                                const int32_t* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    int32_t* cPtr = (int32_t*)cVector;
    const int32_t* aPtr = (int32_t*)aVector;
    const int32_t* bPtr = (int32_t*)bVector;

    __m512i aVal, bVal, cVal;
    for (; number < sixteenthPoints; number++) {

        aVal = _mm512_load_si512(aPtr);
        bVal = _mm512_load_si512(bPtr);

        cVal = _mm512_or_si512(aVal, bVal);

        _mm512_store_si512(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_a_avx2(int32_t* cVector,
                                             const int32_t* aVector,
                                             const int32_t* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEightPoints = num_points / 8;

    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;

    __m256i aVal, bVal, cVal;
    for (; number < oneEightPoints; number++) {

        aVal = _mm256_load_si256((__m256i*)aPtr);
        bVal = _mm256_load_si256((__m256i*)bPtr);

        cVal = _mm256_or_si256(aVal, bVal);

        _mm256_store_si256((__m256i*)cPtr,
                           cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32i_x2_or_32i_a_sse(int32_t* cVector,
                                            const int32_t* aVector,
                                            const int32_t* bVector,
                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = (float*)cVector;
    const float* aPtr = (float*)aVector;
    const float* bPtr = (float*)bVector;

    __m128 aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        bVal = _mm_load_ps(bPtr);

        cVal = _mm_or_ps(aVal, bVal);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32i_x2_or_32i_neon(int32_t* cVector,
                                           const int32_t* aVector,
                                           const int32_t* bVector,
                                           unsigned int num_points)
{
    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    int32x4_t a_val, b_val, c_val;

    for (number = 0; number < quarter_points; number++) {
        a_val = vld1q_s32(aPtr);
        b_val = vld1q_s32(bPtr);
        c_val = vorrq_s32(a_val, b_val);
        vst1q_s32(cPtr, c_val);
        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++) | (*bPtr++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32i_x2_or_32i_neonv8(int32_t* cVector,
                                             const int32_t* aVector,
                                             const int32_t* bVector,
                                             unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;
    int32_t* cPtr = cVector;

    for (unsigned int number = 0; number < eighthPoints; number++) {
        int32x4_t a0 = vld1q_s32(aPtr);
        int32x4_t a1 = vld1q_s32(aPtr + 4);
        int32x4_t b0 = vld1q_s32(bPtr);
        int32x4_t b1 = vld1q_s32(bPtr + 4);
        __VOLK_PREFETCH(aPtr + 16);
        __VOLK_PREFETCH(bPtr + 16);

        vst1q_s32(cPtr, vorrq_s32(a0, b0));
        vst1q_s32(cPtr + 4, vorrq_s32(a1, b1));

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++) | (*bPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_GENERIC

static inline void volk_32i_x2_or_32i_generic(int32_t* cVector,
                                              const int32_t* aVector,
                                              const int32_t* bVector,
                                              unsigned int num_points)
{
    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) | (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC
extern void volk_32i_x2_or_32i_a_orc_impl(int32_t* cVector,
                                          const int32_t* aVector,
                                          const int32_t* bVector,
                                          int num_points);

static inline void volk_32i_x2_or_32i_u_orc(int32_t* cVector,
                                            const int32_t* aVector,
                                            const int32_t* bVector,
                                            unsigned int num_points)
{
    volk_32i_x2_or_32i_a_orc_impl(cVector, aVector, bVector, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_32i_x2_or_32i_a_H */


#ifndef INCLUDED_volk_32i_x2_or_32i_u_H
#define INCLUDED_volk_32i_x2_or_32i_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_u_avx512f(int32_t* cVector,
                                                const int32_t* aVector,
                                                const int32_t* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    int32_t* cPtr = (int32_t*)cVector;
    const int32_t* aPtr = (int32_t*)aVector;
    const int32_t* bPtr = (int32_t*)bVector;

    __m512i aVal, bVal, cVal;
    for (; number < sixteenthPoints; number++) {

        aVal = _mm512_loadu_si512(aPtr);
        bVal = _mm512_loadu_si512(bPtr);

        cVal = _mm512_or_si512(aVal, bVal);

        _mm512_storeu_si512(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_u_avx2(int32_t* cVector,
                                             const int32_t* aVector,
                                             const int32_t* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEightPoints = num_points / 8;

    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;

    __m256i aVal, bVal, cVal;
    for (; number < oneEightPoints; number++) {

        aVal = _mm256_loadu_si256((__m256i*)aPtr);
        bVal = _mm256_loadu_si256((__m256i*)bPtr);

        cVal = _mm256_or_si256(aVal, bVal);

        _mm256_storeu_si256((__m256i*)cPtr,
                            cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32i_x2_or_32i_rvv(int32_t* cVector,
                                          const int32_t* aVector,
                                          const int32_t* bVector,
                                          unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(aVector, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8(bVector, vl);
        __riscv_vse32(cVector, __riscv_vor(va, vb, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32i_x2_or_32i_u_H */
