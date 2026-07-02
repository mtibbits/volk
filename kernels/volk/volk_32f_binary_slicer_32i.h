/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_binary_slicer_32i
 *
 * \b Overview
 *
 * Performs binary slicing on a vector of float samples, outputting 1 when the
 * input sample is >= 0 and 0 when < 0. Results are returned as 32-bit ints.
 *
 * Binary slicing (hard decision) is a fundamental operation in digital
 * demodulation and decoding. It converts soft-decision values (e.g. matched
 * filter output or log-likelihood ratios) into hard-decision bits, as needed
 * by downstream symbol detectors, Viterbi decoders, or bit-error-rate
 * measurement blocks.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_binary_slicer_32i(int* cVector, const float* aVector, unsigned int
 * num_points) \endcode
 *
 * \b Inputs
 * \li aVector: Input vector of soft-decision float samples.
 * \li num_points: The number of samples to slice.
 *
 * \b Outputs
 * \li cVector: Output vector of hard-decision bits (int, 1 or 0).
 *
 * \b Example
 * Slice four soft-decision samples and verify the binary output.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* in = (float*)volk_malloc(sizeof(float) * N, alignment);
 * int* out = (int*)volk_malloc(sizeof(int) * N, alignment);
 *
 * in[0] =  1.5f;
 * in[1] = -0.3f;
 * in[2] =  0.0f;
 * in[3] = -2.0f;
 *
 * // Expected: {1, 0, 1, 0} (>= 0 yields 1, < 0 yields 0)
 *
 * volk_32f_binary_slicer_32i(out, in, N);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     printf("Expected: %d  Result: %d\n", (in[i] >= 0) ? 1 : 0, out[i]);
 * }
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_binary_slicer_32i_H
#define INCLUDED_volk_32f_binary_slicer_32i_H


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_binary_slicer_32i_generic(int* cVector,
                                                      const float* aVector,
                                                      unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        if (*aPtr++ >= 0) {
            *cPtr++ = 1;
        } else {
            *cPtr++ = 0;
        }
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_binary_slicer_32i_generic_branchless(int* cVector,
                                                                 const float* aVector,
                                                                 unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++ >= 0);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_binary_slicer_32i_a_sse2(int* cVector,
                                                     const float* aVector,
                                                     unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    unsigned int quarter_points = num_points / 4;
    __m128 a_val, res_f;
    __m128i res_i, binary_i;
    __m128 zero_val;
    zero_val = _mm_set1_ps(0.0f);

    for (number = 0; number < quarter_points; number++) {
        a_val = _mm_load_ps(aPtr);

        res_f = _mm_cmpge_ps(a_val, zero_val);
        res_i = _mm_cvtps_epi32(res_f);
        binary_i = _mm_srli_epi32(res_i, 31);

        _mm_store_si128((__m128i*)cPtr, binary_i);

        cPtr += 4;
        aPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        if (*aPtr++ >= 0) {
            *cPtr++ = 1;
        } else {
            *cPtr++ = 0;
        }
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_binary_slicer_32i_a_avx(int* cVector,
                                                    const float* aVector,
                                                    unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    unsigned int quarter_points = num_points / 8;
    __m256 a_val, res_f, binary_f;
    __m256i binary_i;
    __m256 zero_val, one_val;
    zero_val = _mm256_set1_ps(0.0f);
    one_val = _mm256_set1_ps(1.0f);

    for (number = 0; number < quarter_points; number++) {
        a_val = _mm256_load_ps(aPtr);

        res_f = _mm256_cmp_ps(a_val, zero_val, _CMP_GE_OS);
        binary_f = _mm256_and_ps(res_f, one_val);
        binary_i = _mm256_cvtps_epi32(binary_f);

        _mm256_store_si256((__m256i*)cPtr, binary_i);

        cPtr += 8;
        aPtr += 8;
    }

    for (number = quarter_points * 8; number < num_points; number++) {
        if (*aPtr++ >= 0) {
            *cPtr++ = 1;
        } else {
            *cPtr++ = 0;
        }
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_binary_slicer_32i_u_sse2(int* cVector,
                                                     const float* aVector,
                                                     unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    unsigned int quarter_points = num_points / 4;
    __m128 a_val, res_f;
    __m128i res_i, binary_i;
    __m128 zero_val;
    zero_val = _mm_set1_ps(0.0f);

    for (number = 0; number < quarter_points; number++) {
        a_val = _mm_loadu_ps(aPtr);

        res_f = _mm_cmpge_ps(a_val, zero_val);
        res_i = _mm_cvtps_epi32(res_f);
        binary_i = _mm_srli_epi32(res_i, 31);

        _mm_storeu_si128((__m128i*)cPtr, binary_i);

        cPtr += 4;
        aPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        if (*aPtr++ >= 0) {
            *cPtr++ = 1;
        } else {
            *cPtr++ = 0;
        }
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_binary_slicer_32i_u_avx(int* cVector,
                                                    const float* aVector,
                                                    unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    unsigned int quarter_points = num_points / 8;
    __m256 a_val, res_f, binary_f;
    __m256i binary_i;
    __m256 zero_val, one_val;
    zero_val = _mm256_set1_ps(0.0f);
    one_val = _mm256_set1_ps(1.0f);

    for (number = 0; number < quarter_points; number++) {
        a_val = _mm256_loadu_ps(aPtr);

        res_f = _mm256_cmp_ps(a_val, zero_val, _CMP_GE_OS);
        binary_f = _mm256_and_ps(res_f, one_val);
        binary_i = _mm256_cvtps_epi32(binary_f);

        _mm256_storeu_si256((__m256i*)cPtr, binary_i);

        cPtr += 8;
        aPtr += 8;
    }

    for (number = quarter_points * 8; number < num_points; number++) {
        if (*aPtr++ >= 0) {
            *cPtr++ = 1;
        } else {
            *cPtr++ = 0;
        }
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_binary_slicer_32i_neon(int* cVector,
                                                   const float* aVector,
                                                   unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    float32x4_t zero_val = vdupq_n_f32(0.0f);

    for (; number < quarter_points; number++) {
        float32x4_t a_val = vld1q_f32(aPtr);
        uint32x4_t cmp = vcgeq_f32(a_val, zero_val);
        uint32x4_t result = vshrq_n_u32(cmp, 31);
        vst1q_s32(cPtr, vreinterpretq_s32_u32(result));
        aPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++ >= 0) ? 1 : 0;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_binary_slicer_32i_neonv8(int* cVector,
                                                     const float* aVector,
                                                     unsigned int num_points)
{
    int* cPtr = cVector;
    const float* aPtr = aVector;
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    float32x4_t zero_val = vdupq_n_f32(0.0f);

    for (; number < eighth_points; number++) {
        float32x4_t a_val0 = vld1q_f32(aPtr);
        float32x4_t a_val1 = vld1q_f32(aPtr + 4);
        __VOLK_PREFETCH(aPtr + 8);

        uint32x4_t cmp0 = vcgeq_f32(a_val0, zero_val);
        uint32x4_t cmp1 = vcgeq_f32(a_val1, zero_val);
        uint32x4_t result0 = vshrq_n_u32(cmp0, 31);
        uint32x4_t result1 = vshrq_n_u32(cmp1, 31);

        vst1q_s32(cPtr, vreinterpretq_s32_u32(result0));
        vst1q_s32(cPtr + 4, vreinterpretq_s32_u32(result1));
        aPtr += 8;
        cPtr += 8;
    }

    for (number = eighth_points * 8; number < num_points; number++) {
        *cPtr++ = (*aPtr++ >= 0) ? 1 : 0;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_binary_slicer_32i_rvv(int* cVector,
                                                  const float* aVector,
                                                  unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vuint32m8_t v = __riscv_vle32_v_u32m8((uint32_t*)aVector, vl);
        v = __riscv_vsrl(__riscv_vnot(v, vl), 31, vl);
        __riscv_vse32((uint32_t*)cVector, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_binary_slicer_32i_H */
