/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16i_x2_add_saturated_16i
 *
 * \b Overview
 *
 * Adds two vectors of 16-bit signed integer samples element-wise with
 * saturation arithmetic. Results are clamped to the representable range
 * [-32768, 32767], preventing overflow wraparound that would corrupt
 * signal data.
 *
 * Saturated addition is essential in fixed-point DSP pipelines where
 * intermediate sums can exceed the dynamic range of 16-bit samples.
 * Common use cases include mixing audio signals, combining baseband
 * I/Q streams before quantization, and accumulating correlator outputs
 * in synchronization or beamforming stages.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16i_x2_add_saturated_16i(int16_t* outVector, const int16_t* inVectorA, const int16_t* inVectorB, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inVectorA: First input vector of samples (int16_t).
 * \li inVectorB: Second input vector of samples (int16_t).
 * \li num_points: The number of samples to process.
 *
 * \b Outputs
 * \li outVector: Saturated element-wise sum of the two input vectors (int16_t).
 *
 * \b Example
 * Demonstrate saturated addition with values that overflow in both directions.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * int16_t* a = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 * int16_t* b = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 * int16_t* result = (int16_t*)volk_malloc(sizeof(int16_t) * N, alignment);
 *
 * a[0] = 30000;  b[0] = 10000;   // 40000 saturates to 32767
 * a[1] = -30000; b[1] = -10000;  // -40000 saturates to -32768
 * a[2] = 10000;  b[2] = 5000;    // 15000 fits, no saturation
 * a[3] = -100;   b[3] = 50;      // -50 fits, no saturation
 *
 * volk_16i_x2_add_saturated_16i(result, a, b, N);
 *
 * printf("Expected: 32767, -32768, 15000, -50\n");
 * printf("Result:   %d, %d, %d, %d\n", result[0], result[1], result[2], result[3]);
 *
 * volk_free(a);
 * volk_free(b);
 * volk_free(result);
 * \endcode
 */

#ifndef INCLUDED_volk_16i_x2_add_saturated_16i_u_H
#define INCLUDED_volk_16i_x2_add_saturated_16i_u_H

#include <inttypes.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16i_x2_add_saturated_16i_generic(int16_t* outVector,
                                                         const int16_t* inVectorA,
                                                         const int16_t* inVectorB,
                                                         unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        int16_t a = inVectorA[i];
        int16_t b = inVectorB[i];
        int16_t sum = a + b;
        // Overflow if a and b have same sign but sum has different sign
        int16_t overflow = ((a ^ sum) & (b ^ sum)) >> 15;
        // Saturation value: 32767 if a >= 0, -32768 if a < 0
        int16_t sat_val = (a >> 15) ^ 0x7FFF;
        outVector[i] = (overflow & sat_val) | (~overflow & sum);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_x2_add_saturated_16i_u_sse2(int16_t* outVector,
                                                        const int16_t* inVectorA,
                                                        const int16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    for (; number < eighthPoints; number++) {
        __m128i a = _mm_loadu_si128((const __m128i*)(inVectorA + 8 * number));
        __m128i b = _mm_loadu_si128((const __m128i*)(inVectorB + 8 * number));
        __m128i result = _mm_adds_epi16(a, b);
        _mm_storeu_si128((__m128i*)(outVector + 8 * number), result);
    }

    for (number = eighthPoints * 8; number < num_points; number++) {
        int32_t sum = (int32_t)inVectorA[number] + (int32_t)inVectorB[number];
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        outVector[number] = (int16_t)sum;
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_x2_add_saturated_16i_u_avx2(int16_t* outVector,
                                                        const int16_t* inVectorA,
                                                        const int16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(inVectorA + 16 * number));
        __m256i b = _mm256_loadu_si256((const __m256i*)(inVectorB + 16 * number));
        __m256i result = _mm256_adds_epi16(a, b);
        _mm256_storeu_si256((__m256i*)(outVector + 16 * number), result);
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        int32_t sum = (int32_t)inVectorA[number] + (int32_t)inVectorB[number];
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        outVector[number] = (int16_t)sum;
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_x2_add_saturated_16i_u_avx512bw(int16_t* outVector,
                                                            const int16_t* inVectorA,
                                                            const int16_t* inVectorB,
                                                            unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    unsigned int number = 0;

    for (; number < thirtysecondPoints; number++) {
        __m512i a = _mm512_loadu_si512((const __m512i*)(inVectorA + 32 * number));
        __m512i b = _mm512_loadu_si512((const __m512i*)(inVectorB + 32 * number));
        __m512i result = _mm512_adds_epi16(a, b);
        _mm512_storeu_si512((__m512i*)(outVector + 32 * number), result);
    }

    for (number = thirtysecondPoints * 32; number < num_points; number++) {
        int32_t sum = (int32_t)inVectorA[number] + (int32_t)inVectorB[number];
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        outVector[number] = (int16_t)sum;
    }
}

#endif /* LV_HAVE_AVX512BW */


#endif /* INCLUDED_volk_16i_x2_add_saturated_16i_u_H */


#ifndef INCLUDED_volk_16i_x2_add_saturated_16i_a_H
#define INCLUDED_volk_16i_x2_add_saturated_16i_a_H

#include <inttypes.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16i_x2_add_saturated_16i_a_sse2(int16_t* outVector,
                                                        const int16_t* inVectorA,
                                                        const int16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    for (; number < eighthPoints; number++) {
        __m128i a = _mm_load_si128((const __m128i*)(inVectorA + 8 * number));
        __m128i b = _mm_load_si128((const __m128i*)(inVectorB + 8 * number));
        __m128i result = _mm_adds_epi16(a, b);
        _mm_store_si128((__m128i*)(outVector + 8 * number), result);
    }

    for (number = eighthPoints * 8; number < num_points; number++) {
        int32_t sum = (int32_t)inVectorA[number] + (int32_t)inVectorB[number];
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        outVector[number] = (int16_t)sum;
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16i_x2_add_saturated_16i_a_avx2(int16_t* outVector,
                                                        const int16_t* inVectorA,
                                                        const int16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __m256i a = _mm256_load_si256((const __m256i*)(inVectorA + 16 * number));
        __m256i b = _mm256_load_si256((const __m256i*)(inVectorB + 16 * number));
        __m256i result = _mm256_adds_epi16(a, b);
        _mm256_store_si256((__m256i*)(outVector + 16 * number), result);
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        int32_t sum = (int32_t)inVectorA[number] + (int32_t)inVectorB[number];
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        outVector[number] = (int16_t)sum;
    }
}

#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_16i_x2_add_saturated_16i_a_avx512bw(int16_t* outVector,
                                                            const int16_t* inVectorA,
                                                            const int16_t* inVectorB,
                                                            unsigned int num_points)
{
    const unsigned int thirtysecondPoints = num_points / 32;
    unsigned int number = 0;

    for (; number < thirtysecondPoints; number++) {
        __m512i a = _mm512_load_si512((const __m512i*)(inVectorA + 32 * number));
        __m512i b = _mm512_load_si512((const __m512i*)(inVectorB + 32 * number));
        __m512i result = _mm512_adds_epi16(a, b);
        _mm512_store_si512((__m512i*)(outVector + 32 * number), result);
    }

    for (number = thirtysecondPoints * 32; number < num_points; number++) {
        int32_t sum = (int32_t)inVectorA[number] + (int32_t)inVectorB[number];
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        outVector[number] = (int16_t)sum;
    }
}

#endif /* LV_HAVE_AVX512BW */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16i_x2_add_saturated_16i_neon(int16_t* outVector,
                                                      const int16_t* inVectorA,
                                                      const int16_t* inVectorB,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    for (; number < eighthPoints; number++) {
        int16x8_t a = vld1q_s16(inVectorA + 8 * number);
        int16x8_t b = vld1q_s16(inVectorB + 8 * number);
        vst1q_s16(outVector + 8 * number, vqaddq_s16(a, b));
    }

    for (number = eighthPoints * 8; number < num_points; number++) {
        int32_t sum = (int32_t)inVectorA[number] + (int32_t)inVectorB[number];
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        outVector[number] = (int16_t)sum;
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_common.h>

static inline void volk_16i_x2_add_saturated_16i_neonv8(int16_t* outVector,
                                                        const int16_t* inVectorA,
                                                        const int16_t* inVectorB,
                                                        unsigned int num_points)
{
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int number = 0;

    for (; number < sixteenthPoints; number++) {
        __VOLK_PREFETCH(inVectorA + 32);
        __VOLK_PREFETCH(inVectorB + 32);
        int16x8_t a0 = vld1q_s16(inVectorA);
        int16x8_t b0 = vld1q_s16(inVectorB);
        int16x8_t a1 = vld1q_s16(inVectorA + 8);
        int16x8_t b1 = vld1q_s16(inVectorB + 8);
        vst1q_s16(outVector, vqaddq_s16(a0, b0));
        vst1q_s16(outVector + 8, vqaddq_s16(a1, b1));
        inVectorA += 16;
        inVectorB += 16;
        outVector += 16;
    }

    for (number = sixteenthPoints * 16; number < num_points; number++) {
        int32_t sum = (int32_t)(*inVectorA++) + (int32_t)(*inVectorB++);
        if (sum > 32767)
            sum = 32767;
        else if (sum < -32768)
            sum = -32768;
        *outVector++ = (int16_t)sum;
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_16i_x2_add_saturated_16i_rvv(int16_t* outVector,
                                                     const int16_t* inVectorA,
                                                     const int16_t* inVectorB,
                                                     unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inVectorA += vl, inVectorB += vl, outVector += vl) {
        vl = __riscv_vsetvl_e16m8(n);
        vint16m8_t a = __riscv_vle16_v_i16m8(inVectorA, vl);
        vint16m8_t b = __riscv_vle16_v_i16m8(inVectorB, vl);
        __riscv_vse16(outVector, __riscv_vsadd(a, b, vl), vl);
    }
}

#endif /* LV_HAVE_RVV */


#endif /* INCLUDED_volk_16i_x2_add_saturated_16i_a_H */
