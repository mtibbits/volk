/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_convert_32i
 *
 * \b Overview
 *
 * Scales each floating-point sample by a scalar factor and converts the
 * result to a 32-bit integer:
 * <tt>outputVector[i] = (int32_t)(inputVector[i] * scalar)</tt>.
 * Values are clamped to the 32-bit integer range to prevent overflow.
 *
 * This quantization operation is common when preparing baseband samples for
 * a DAC or when converting normalized floating-point signal data to
 * fixed-point representation for hardware interfaces or integer-based DSP
 * processing stages.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_convert_32i(int32_t* outputVector, const float* inputVector, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of floating-point samples (float).
 * \li scalar: The scaling factor applied to each sample before conversion (float).
 * \li num_points: The number of samples to convert.
 *
 * \b Outputs
 * \li outputVector: The scaled and quantized output vector (int32_t).
 *
 * \b Example
 * Scale floats by 100 and convert to 32-bit integers.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * float* in = (float*)volk_malloc(sizeof(float) * N, alignment);
 * int32_t* out = (int32_t*)volk_malloc(sizeof(int32_t) * N, alignment);
 *
 * in[0] = 1.0f; in[1] = -1.0f; in[2] = 0.5f; in[3] = -0.5f;
 * float scalar = 100.0f;
 *
 * // Expected: 1.0*100=100, -1.0*100=-100, 0.5*100=50, -0.5*100=-50
 *
 * volk_32f_s32f_convert_32i(out, in, scalar, N);
 *
 * printf("Expected: 100, -100, 50, -50\n");
 * printf("Result:   %d, %d, %d, %d\n", out[0], out[1], out[2], out[3]);
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_convert_32i_u_H
#define INCLUDED_volk_32f_s32f_convert_32i_u_H

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_convert_32i_generic(int32_t* outputVector,
                                                     const float* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    int32_t* outputVectorPtr = outputVector;
    const float* inputVectorPtr = inputVector;
    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    for (unsigned int number = 0; number < num_points; number++) {
        const float r = *inputVectorPtr++ * scalar;
        int s;
        if (r >= max_val)
            s = INT_MAX;
        else if (r < min_val)
            s = INT_MIN;
        else
            s = (int32_t)rintf(r);
        *outputVectorPtr++ = s;
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_32i_u_sse(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 ret;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {
        ret = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_max_ps(_mm_min_ps(_mm_mul_ps(ret, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        *outputVectorPtr++ = (outputFloatBuffer[0] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[0]);
        *outputVectorPtr++ = (outputFloatBuffer[1] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[1]);
        *outputVectorPtr++ = (outputFloatBuffer[2] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[2]);
        *outputVectorPtr++ = (outputFloatBuffer[3] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_32i_u_sse2(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 inputVal1;
    __m128i intInputVal1;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    for (; number < quarterPoints; number++) {
        inputVal1 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;

        inputVal1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm_cvtps_epi32(inputVal1);
        // Fix positive overflow: CVTPS2DQ returns 0x80000000 for values >= 2^31
        intInputVal1 = _mm_xor_si128(
            intInputVal1,
            _mm_castps_si128(_mm_cmpge_ps(inputVal1, vmax_val)));

        _mm_storeu_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_u_avx(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int eighthPoints = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256 inputVal1;
    __m256i intInputVal1;
    __m256 vmin_val = _mm256_set1_ps(min_val);
    __m256 vmax_val = _mm256_set1_ps(max_val);

    for (; number < eighthPoints; number++) {
        inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        // Fix positive overflow: VCVTPS2DQ returns 0x80000000 for values >= 2^31
        intInputVal1 = _mm256_castps_si256(_mm256_xor_ps(
            _mm256_castsi256_ps(intInputVal1),
            _mm256_cmp_ps(inputVal1, vmax_val, _CMP_GE_OS)));

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_u_avx2(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int eighthPoints = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256 inputVal1;
    __m256i intInputVal1;
    __m256 vmin_val = _mm256_set1_ps(min_val);
    __m256 vmax_val = _mm256_set1_ps(max_val);

    for (; number < eighthPoints; number++) {
        inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        // Fix positive overflow: VCVTPS2DQ returns 0x80000000 for values >= 2^31
        intInputVal1 = _mm256_castps_si256(_mm256_xor_ps(
            _mm256_castsi256_ps(intInputVal1),
            _mm256_cmp_ps(inputVal1, vmax_val, _CMP_GE_OS)));

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_u_avx512f(int32_t* outputVector,
                                                        const float* inputVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int sixteenthPoints = num_points / 16;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);

    for (; number < sixteenthPoints; number++) {
        __m512 inputVal = _mm512_loadu_ps(inputVectorPtr);
        inputVectorPtr += 16;

        inputVal = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal, vScalar), vmax_val), vmin_val);
        __m512i intInputVal = _mm512_cvtps_epi32(inputVal);
        // Fix positive overflow: VCVTPS2DQ returns 0x80000000 for values >= 2^31
        __mmask16 overflow = _mm512_cmp_ps_mask(inputVal, vmax_val, _CMP_GE_OS);
        intInputVal = _mm512_mask_xor_epi32(
            intInputVal, overflow, intInputVal, _mm512_set1_epi32(-1));

        _mm512_storeu_si512((__m512i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_s32f_convert_32i_generic(
        outputVector + number, inputVector + number, scalar, num_points - number);
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_32i_neon(int32_t* outputVector,
                                                  const float* inputVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const float* inputPtr = inputVector;
    int32_t* outputPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);
    for (; number < quarter_points; number++) {
        float32x4_t inputVal = vld1q_f32(inputPtr);
        inputVal = vmulq_f32(inputVal, vScalar);
        inputVal = vmaxq_f32(vminq_f32(inputVal, vmax_val), vmin_val);
        float buf[4];
        vst1q_f32(buf, inputVal);
        *outputPtr++ = (buf[0] >= max_val) ? INT_MAX : (int32_t)rintf(buf[0]);
        *outputPtr++ = (buf[1] >= max_val) ? INT_MAX : (int32_t)rintf(buf[1]);
        *outputPtr++ = (buf[2] >= max_val) ? INT_MAX : (int32_t)rintf(buf[2]);
        *outputPtr++ = (buf[3] >= max_val) ? INT_MAX : (int32_t)rintf(buf[3]);
        inputPtr += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        float r = *inputPtr++ * scalar;
        if (r >= max_val)
            *outputPtr++ = INT_MAX;
        else if (r < min_val)
            *outputPtr++ = INT_MIN;
        else
            *outputPtr++ = (int32_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_convert_32i_neonv8(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const float* inputPtr = inputVector;
    int32_t* outputPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    float32x4_t vScalar = vdupq_n_f32(scalar);
    float32x4_t vmin_val = vdupq_n_f32(min_val);
    float32x4_t vmax_val = vdupq_n_f32(max_val);

    for (; number < eighth_points; number++) {
        float32x4_t inputVal0 = vld1q_f32(inputPtr);
        float32x4_t inputVal1 = vld1q_f32(inputPtr + 4);
        __VOLK_PREFETCH(inputPtr + 8);

        inputVal0 = vmulq_f32(inputVal0, vScalar);
        inputVal1 = vmulq_f32(inputVal1, vScalar);
        inputVal0 = vmaxq_f32(vminq_f32(inputVal0, vmax_val), vmin_val);
        inputVal1 = vmaxq_f32(vminq_f32(inputVal1, vmax_val), vmin_val);

        int32x4_t intVal0 = vcvtnq_s32_f32(inputVal0);
        int32x4_t intVal1 = vcvtnq_s32_f32(inputVal1);

        vst1q_s32(outputPtr, intVal0);
        vst1q_s32(outputPtr + 4, intVal1);
        inputPtr += 8;
        outputPtr += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        float r = *inputPtr++ * scalar;
        if (r >= max_val)
            *outputPtr++ = INT_MAX;
        else if (r < min_val)
            *outputPtr++ = INT_MIN;
        else
            *outputPtr++ = (int32_t)rintf(r);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_convert_32i_rvv(int32_t* outputVector,
                                                 const float* inputVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(inputVector, vl);
        v = __riscv_vfmul(v, scalar, vl);
        __riscv_vse32(outputVector, __riscv_vfcvt_x(v, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_ORC

extern void volk_32f_s32f_convert_32i_a_orc_impl(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   int num_points);

static inline void volk_32f_s32f_convert_32i_u_orc(int32_t* outputVector,
                                                     const float* inputVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    volk_32f_s32f_convert_32i_a_orc_impl(outputVector, inputVector, scalar, num_points);
}

#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_32f_s32f_convert_32i_u_H */
#ifndef INCLUDED_volk_32f_s32f_convert_32i_a_H
#define INCLUDED_volk_32f_s32f_convert_32i_a_H

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_convert_32i_a_sse(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 ret;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    __VOLK_ATTR_ALIGNED(16) float outputFloatBuffer[4];

    for (; number < quarterPoints; number++) {
        ret = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

        ret = _mm_max_ps(_mm_min_ps(_mm_mul_ps(ret, vScalar), vmax_val), vmin_val);

        _mm_store_ps(outputFloatBuffer, ret);
        *outputVectorPtr++ = (outputFloatBuffer[0] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[0]);
        *outputVectorPtr++ = (outputFloatBuffer[1] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[1]);
        *outputVectorPtr++ = (outputFloatBuffer[2] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[2]);
        *outputVectorPtr++ = (outputFloatBuffer[3] >= max_val) ? INT_MAX : (int32_t)rintf(outputFloatBuffer[3]);
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_s32f_convert_32i_a_sse2(int32_t* outputVector,
                                                    const float* inputVector,
                                                    const float scalar,
                                                    unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int quarterPoints = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m128 vScalar = _mm_set_ps1(scalar);
    __m128 inputVal1;
    __m128i intInputVal1;
    __m128 vmin_val = _mm_set_ps1(min_val);
    __m128 vmax_val = _mm_set_ps1(max_val);

    for (; number < quarterPoints; number++) {
        inputVal1 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;

        inputVal1 =
            _mm_max_ps(_mm_min_ps(_mm_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm_cvtps_epi32(inputVal1);
        // Fix positive overflow: CVTPS2DQ returns 0x80000000 for values >= 2^31
        intInputVal1 = _mm_xor_si128(
            intInputVal1,
            _mm_castps_si128(_mm_cmpge_ps(inputVal1, vmax_val)));

        _mm_store_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_a_avx(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int eighthPoints = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256 inputVal1;
    __m256i intInputVal1;
    __m256 vmin_val = _mm256_set1_ps(min_val);
    __m256 vmax_val = _mm256_set1_ps(max_val);

    for (; number < eighthPoints; number++) {
        inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        // Fix positive overflow: VCVTPS2DQ returns 0x80000000 for values >= 2^31
        intInputVal1 = _mm256_castps_si256(_mm256_xor_ps(
            _mm256_castsi256_ps(intInputVal1),
            _mm256_cmp_ps(inputVal1, vmax_val, _CMP_GE_OS)));

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_a_avx2(int32_t* outputVector,
                                                   const float* inputVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int eighthPoints = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    float min_val = INT_MIN;
    float max_val = (uint32_t)INT_MAX + 1;
    float r;

    __m256 vScalar = _mm256_set1_ps(scalar);
    __m256 inputVal1;
    __m256i intInputVal1;
    __m256 vmin_val = _mm256_set1_ps(min_val);
    __m256 vmax_val = _mm256_set1_ps(max_val);

    for (; number < eighthPoints; number++) {
        inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;

        inputVal1 = _mm256_max_ps(
            _mm256_min_ps(_mm256_mul_ps(inputVal1, vScalar), vmax_val), vmin_val);
        intInputVal1 = _mm256_cvtps_epi32(inputVal1);
        // Fix positive overflow: VCVTPS2DQ returns 0x80000000 for values >= 2^31
        intInputVal1 = _mm256_castps_si256(_mm256_xor_ps(
            _mm256_castsi256_ps(intInputVal1),
            _mm256_cmp_ps(inputVal1, vmax_val, _CMP_GE_OS)));

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        r = inputVector[number] * scalar;
        if (r >= max_val)
            outputVector[number] = INT_MAX;
        else if (r < min_val)
            outputVector[number] = INT_MIN;
        else
            outputVector[number] = (int32_t)rintf(r);
    }
}

#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32f_s32f_convert_32i_a_avx512f(int32_t* outputVector,
                                                        const float* inputVector,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;

    const unsigned int sixteenthPoints = num_points / 16;

    const float* inputVectorPtr = (const float*)inputVector;
    int32_t* outputVectorPtr = outputVector;

    const float min_val = (float)INT_MIN;
    const float max_val = (float)((uint32_t)INT_MAX + 1);

    const __m512 vScalar = _mm512_set1_ps(scalar);
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);

    for (; number < sixteenthPoints; number++) {
        __m512 inputVal = _mm512_load_ps(inputVectorPtr);
        inputVectorPtr += 16;

        inputVal = _mm512_max_ps(
            _mm512_min_ps(_mm512_mul_ps(inputVal, vScalar), vmax_val), vmin_val);
        __m512i intInputVal = _mm512_cvtps_epi32(inputVal);
        // Fix positive overflow: VCVTPS2DQ returns 0x80000000 for values >= 2^31
        __mmask16 overflow = _mm512_cmp_ps_mask(inputVal, vmax_val, _CMP_GE_OS);
        intInputVal = _mm512_mask_xor_epi32(
            intInputVal, overflow, intInputVal, _mm512_set1_epi32(-1));

        _mm512_store_si512((__m512i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_s32f_convert_32i_generic(
        outputVector + number, inputVector + number, scalar, num_points - number);
}

#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32f_s32f_convert_32i_a_H */
