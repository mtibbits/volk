/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_convert_16ic
 *
 * \b Overview
 *
 * Converts a complex vector of 32-bit float samples into a complex vector of
 * 16-bit integer samples. Each component (real and imaginary) is rounded to the
 * nearest integer and saturated to the range [SHRT_MIN, SHRT_MAX].
 *
 * This kernel is commonly used when quantizing complex baseband samples from
 * floating-point processing chains into fixed-point representation for storage,
 * transmission over digital interfaces, or feeding into hardware accelerators
 * that operate on 16-bit I/Q data.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_convert_16ic(lv_16sc_t* outputVector, const lv_32fc_t* inputVector,
 * unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The complex 32-bit float input data buffer (lv_32fc_t).
 * \li num_points: The number of complex samples to be converted.
 *
 * \b Outputs
 * \li outputVector: The complex 16-bit integer output data buffer (lv_16sc_t).
 *
 * \b Example
 * Convert four complex float samples to complex 16-bit integers with saturation.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_32fc_t* input = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_16sc_t* output = (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t) * N, alignment);
 *
 * // Use values within int16 range and one that saturates
 * input[0] = lv_cmake(100.0f, -200.0f);
 * input[1] = lv_cmake(0.0f, 32767.0f);
 * input[2] = lv_cmake(-32768.0f, 50000.0f);  // 50000 exceeds SHRT_MAX, will saturate
 * input[3] = lv_cmake(-1.5f, 1.5f);
 *
 * volk_32fc_convert_16ic(output, input, N);
 *
 * // Expected: (100,-200), (0,32767), (-32768,32767), (-2,2)
 * // Note: 50000 saturates to 32767; -1.5 and 1.5 round to -2 and 2
 * printf("Expected: (100,-200) (0,32767) (-32768,32767) (-2,2)\n");
 * printf("Result:   (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
 *        lv_creal(output[0]), lv_cimag(output[0]),
 *        lv_creal(output[1]), lv_cimag(output[1]),
 *        lv_creal(output[2]), lv_cimag(output[2]),
 *        lv_creal(output[3]), lv_cimag(output[3]));
 *
 * volk_free(input);
 * volk_free(output);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_convert_16ic_u_H
#define INCLUDED_volk_32fc_convert_16ic_u_H

#include "volk/volk_complex.h"
#include <limits.h>
#include <math.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_convert_16ic_generic(lv_16sc_t* outputVector,
                                                  const lv_32fc_t* inputVector,
                                                  unsigned int num_points)
{
    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;
    float aux;
    unsigned int i;
    for (i = 0; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val)
            aux = max_val;
        else if (aux < min_val)
            aux = min_val;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_convert_16ic_u_sse2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m128 inputVal1, inputVal2;
    __m128i intInputVal1, intInputVal2;
    __m128 ret1, ret2;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    unsigned int i;
    for (i = 0; i < sse_iters; i++) {
        inputVal1 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;
        inputVal2 = _mm_loadu_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __VOLK_PREFETCH(inputVectorPtr + 8);

        // Clip
        ret1 = _mm_max_ps(_mm_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm_max_ps(_mm_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm_cvtps_epi32(ret1);
        intInputVal2 = _mm_cvtps_epi32(ret2);

        intInputVal1 = _mm_packs_epi32(intInputVal1, intInputVal2);

        _mm_storeu_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    for (i = sse_iters * 8; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val)
            aux = max_val;
        else if (aux < min_val)
            aux = min_val;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_u_avx(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m256 inputVal1, inputVal2;
    __m256i intInputVal1, intInputVal2;
    __m256 ret1, ret2;
    __m128i lo1, hi1, lo2, hi2;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm256_cvtps_epi32(ret1);
        intInputVal2 = _mm256_cvtps_epi32(ret2);

        // Extract 128-bit halves and pack with SSE2 (AVX lacks 256-bit integer pack)
        lo1 = _mm256_castsi256_si128(intInputVal1);
        hi1 = _mm256_extractf128_si256(intInputVal1, 1);
        lo2 = _mm256_castsi256_si128(intInputVal2);
        hi2 = _mm256_extractf128_si256(intInputVal2, 1);

        _mm_storeu_si128((__m128i*)outputVectorPtr, _mm_packs_epi32(lo1, hi1));
        outputVectorPtr += 8;
        _mm_storeu_si128((__m128i*)outputVectorPtr, _mm_packs_epi32(lo2, hi2));
        outputVectorPtr += 8;
    }

    volk_32fc_convert_16ic_generic(
        (lv_16sc_t*)outputVectorPtr, (const lv_32fc_t*)inputVectorPtr, num_points - avx_iters * 8);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_u_avx2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m256 inputVal1, inputVal2;
    __m256i intInputVal1, intInputVal2;
    __m256 ret1, ret2;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm256_cvtps_epi32(ret1);
        intInputVal2 = _mm256_cvtps_epi32(ret2);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0xd8);

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 16;
    }

    for (i = avx_iters * 16; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val)
            aux = max_val;
        else if (aux < min_val)
            aux = min_val;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_u_avx2_fma(lv_16sc_t* outputVector,
                                                      const lv_32fc_t* inputVector,
                                                      unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m256 inputVal1, inputVal2;
    __m256i intInputVal1, intInputVal2;
    __m256 ret1, ret2;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        inputVal1 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm256_loadu_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm256_cvtps_epi32(ret1);
        intInputVal2 = _mm256_cvtps_epi32(ret2);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0xd8);

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 16;
    }

    volk_32fc_convert_16ic_generic(
        (lv_16sc_t*)outputVectorPtr,
        (const lv_32fc_t*)inputVectorPtr,
        num_points - avx_iters * 8);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_u_avx512(lv_16sc_t* outputVector,
                                                   const lv_32fc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m512 inputVal1;
    __m256i intInputVal;
    __m512 ret1;
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx512_iters; i++) {
        inputVal1 = _mm512_loadu_ps(inputVectorPtr);
        inputVectorPtr += 16;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm512_max_ps(_mm512_min_ps(inputVal1, vmax_val), vmin_val);

        // Convert float to int32, then pack to int16 with saturation
        intInputVal = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(ret1));

        _mm256_storeu_si256((__m256i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 16;
    }

    for (i = avx512_iters * 16; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val)
            aux = max_val;
        else if (aux < min_val)
            aux = min_val;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_u_avx512bw(lv_16sc_t* outputVector,
                                                      const lv_32fc_t* inputVector,
                                                      unsigned int num_points)
{
    const unsigned int avx512bw_iters = num_points / 16;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);
    // After _mm512_packs_epi32(a, b), elements are interleaved within 128-bit lanes.
    // This permute reorders the 64-bit chunks so the result is [a[0..15], b[0..15]].
    const __m512i fix_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    unsigned int i;

    for (i = 0; i < avx512bw_iters; i++) {
        __m512 inputVal1 = _mm512_loadu_ps(inputVectorPtr);
        inputVectorPtr += 16;
        __m512 inputVal2 = _mm512_loadu_ps(inputVectorPtr);
        inputVectorPtr += 16;

        // Clip to [SHRT_MIN, SHRT_MAX]
        __m512 ret1 = _mm512_max_ps(_mm512_min_ps(inputVal1, vmax_val), vmin_val);
        __m512 ret2 = _mm512_max_ps(_mm512_min_ps(inputVal2, vmax_val), vmin_val);

        // Convert float to int32
        __m512i intVal1 = _mm512_cvtps_epi32(ret1);
        __m512i intVal2 = _mm512_cvtps_epi32(ret2);

        // Saturating narrow int32 -> int16, fix lane interleaving
        __m512i packed = _mm512_permutexvar_epi64(fix_idx, _mm512_packs_epi32(intVal1, intVal2));

        _mm512_storeu_si512((__m512i*)outputVectorPtr, packed);
        outputVectorPtr += 32;
    }

    volk_32fc_convert_16ic_generic(
        (lv_16sc_t*)outputVectorPtr,
        (const lv_32fc_t*)inputVectorPtr,
        num_points - avx512bw_iters * 16);
}
#endif /* LV_HAVE_AVX512BW */

#if LV_HAVE_NEONV7
#include <arm_neon.h>

static inline void volk_32fc_convert_16ic_neon(lv_16sc_t* outputVector,
                                               const lv_32fc_t* inputVector,
                                               unsigned int num_points)
{

    const unsigned int neon_iters = num_points / 4;

    const float32_t* inputVectorPtr = (const float32_t*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val_f = (float)SHRT_MIN;
    const float max_val_f = (float)SHRT_MAX;
    float32_t aux;
    unsigned int i;

    const float32x4_t min_val = vmovq_n_f32(min_val_f);
    const float32x4_t max_val = vmovq_n_f32(max_val_f);
    const float32x4_t magic = vdupq_n_f32(8388608.0f); /* 2^23: round-to-nearest-even */
    float32x4_t ret1, ret2, a, b;

    int32x4_t toint_a = { 0, 0, 0, 0 };
    int32x4_t toint_b = { 0, 0, 0, 0 };
    int16x4_t intInputVal1, intInputVal2;
    int16x8_t res;

    for (i = 0; i < neon_iters; i++) {
        a = vld1q_f32((const float32_t*)(inputVectorPtr));
        inputVectorPtr += 4;
        b = vld1q_f32((const float32_t*)(inputVectorPtr));
        inputVectorPtr += 4;
        __VOLK_PREFETCH(inputVectorPtr + 8);

        ret1 = vmaxq_f32(vminq_f32(a, max_val), min_val);
        ret2 = vmaxq_f32(vminq_f32(b, max_val), min_val);

        /* round-to-nearest-even via magic-number trick (|val| < 2^23) */
        toint_a = vcvtq_s32_f32(vsubq_f32(vaddq_f32(ret1, magic), magic));
        toint_b = vcvtq_s32_f32(vsubq_f32(vaddq_f32(ret2, magic), magic));

        intInputVal1 = vqmovn_s32(toint_a);
        intInputVal2 = vqmovn_s32(toint_b);

        res = vcombine_s16(intInputVal1, intInputVal2);
        vst1q_s16((int16_t*)outputVectorPtr, res);
        outputVectorPtr += 8;
    }

    for (i = neon_iters * 8; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val_f)
            aux = max_val_f;
        else if (aux < min_val_f)
            aux = min_val_f;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}

#endif /* LV_HAVE_NEONV7 */

#if LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_convert_16ic_neonv8(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int neon_iters = num_points / 4;

    const float32_t* inputVectorPtr = (const float32_t*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val_f = (float)SHRT_MIN;
    const float max_val_f = (float)SHRT_MAX;
    float32_t aux;
    unsigned int i;

    const float32x4_t min_val = vmovq_n_f32(min_val_f);
    const float32x4_t max_val = vmovq_n_f32(max_val_f);
    float32x4_t ret1, ret2, a, b;

    int32x4_t toint_a = { 0, 0, 0, 0 }, toint_b = { 0, 0, 0, 0 };
    int16x4_t intInputVal1, intInputVal2;
    int16x8_t res;

    for (i = 0; i < neon_iters; i++) {
        a = vld1q_f32((const float32_t*)(inputVectorPtr));
        inputVectorPtr += 4;
        b = vld1q_f32((const float32_t*)(inputVectorPtr));
        inputVectorPtr += 4;
        __VOLK_PREFETCH(inputVectorPtr + 8);

        ret1 = vmaxq_f32(vminq_f32(a, max_val), min_val);
        ret2 = vmaxq_f32(vminq_f32(b, max_val), min_val);

        // vrndiq takes into account the current rounding mode (as does rintf)
        toint_a = vcvtq_s32_f32(vrndiq_f32(ret1));
        toint_b = vcvtq_s32_f32(vrndiq_f32(ret2));

        intInputVal1 = vqmovn_s32(toint_a);
        intInputVal2 = vqmovn_s32(toint_b);

        res = vcombine_s16(intInputVal1, intInputVal2);
        vst1q_s16((int16_t*)outputVectorPtr, res);
        outputVectorPtr += 8;
    }

    for (i = neon_iters * 8; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val_f)
            aux = max_val_f;
        else if (aux < min_val_f)
            aux = min_val_f;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_convert_16ic_rvv(lv_16sc_t* outputVector,
                                              const lv_32fc_t* inputVector,
                                              unsigned int num_points)
{
    int16_t* out = (int16_t*)outputVector;
    const float* in = (const float*)inputVector;
    size_t n = (size_t)num_points * 2;
    for (size_t vl; n > 0; n -= vl, in += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(in, vl);
        __riscv_vse16(out, __riscv_vfncvt_x(v, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_convert_16ic_rvvseg(lv_16sc_t* outputVector,
                                                  const lv_32fc_t* inputVector,
                                                  unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc =
            __riscv_vlseg2e32_v_f32m4x2((const float*)inputVector, vl);
        vint16m2_t vr = __riscv_vfncvt_x(__riscv_vget_f32m4(vc, 0), vl);
        vint16m2_t vi = __riscv_vfncvt_x(__riscv_vget_f32m4(vc, 1), vl);
        __riscv_vsseg2e16_v_i16m2x2(
            (int16_t*)outputVector, __riscv_vcreate_v_i16m2x2(vr, vi), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_32fc_convert_16ic_u_H */

#ifndef INCLUDED_volk_32fc_convert_16ic_a_H
#define INCLUDED_volk_32fc_convert_16ic_a_H

#include "volk/volk_complex.h"
#include <limits.h>
#include <math.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_convert_16ic_a_sse2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m128 inputVal1, inputVal2;
    __m128i intInputVal1, intInputVal2;
    __m128 ret1, ret2;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);
    unsigned int i;

    for (i = 0; i < sse_iters; i++) {
        inputVal1 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;
        inputVal2 = _mm_load_ps(inputVectorPtr);
        inputVectorPtr += 4;
        __VOLK_PREFETCH(inputVectorPtr + 8);

        // Clip
        ret1 = _mm_max_ps(_mm_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm_max_ps(_mm_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm_cvtps_epi32(ret1);
        intInputVal2 = _mm_cvtps_epi32(ret2);

        intInputVal1 = _mm_packs_epi32(intInputVal1, intInputVal2);

        _mm_store_si128((__m128i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 8;
    }

    for (i = sse_iters * 8; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val)
            aux = max_val;
        else if (aux < min_val)
            aux = min_val;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_a_avx(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m256 inputVal1, inputVal2;
    __m256i intInputVal1, intInputVal2;
    __m256 ret1, ret2;
    __m128i lo1, hi1, lo2, hi2;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm256_cvtps_epi32(ret1);
        intInputVal2 = _mm256_cvtps_epi32(ret2);

        // Extract 128-bit halves and pack with SSE2 (AVX lacks 256-bit integer pack)
        lo1 = _mm256_castsi256_si128(intInputVal1);
        hi1 = _mm256_extractf128_si256(intInputVal1, 1);
        lo2 = _mm256_castsi256_si128(intInputVal2);
        hi2 = _mm256_extractf128_si256(intInputVal2, 1);

        _mm_store_si128((__m128i*)outputVectorPtr, _mm_packs_epi32(lo1, hi1));
        outputVectorPtr += 8;
        _mm_store_si128((__m128i*)outputVectorPtr, _mm_packs_epi32(lo2, hi2));
        outputVectorPtr += 8;
    }

    volk_32fc_convert_16ic_generic(
        (lv_16sc_t*)outputVectorPtr, (const lv_32fc_t*)inputVectorPtr, num_points - avx_iters * 8);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_a_avx2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m256 inputVal1, inputVal2;
    __m256i intInputVal1, intInputVal2;
    __m256 ret1, ret2;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm256_cvtps_epi32(ret1);
        intInputVal2 = _mm256_cvtps_epi32(ret2);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0xd8);

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 16;
    }

    for (i = avx_iters * 16; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val)
            aux = max_val;
        else if (aux < min_val)
            aux = min_val;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_a_avx2_fma(lv_16sc_t* outputVector,
                                                      const lv_32fc_t* inputVector,
                                                      unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m256 inputVal1, inputVal2;
    __m256i intInputVal1, intInputVal2;
    __m256 ret1, ret2;
    const __m256 vmin_val = _mm256_set1_ps(min_val);
    const __m256 vmax_val = _mm256_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx_iters; i++) {
        inputVal1 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        inputVal2 = _mm256_load_ps(inputVectorPtr);
        inputVectorPtr += 8;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);

        intInputVal1 = _mm256_cvtps_epi32(ret1);
        intInputVal2 = _mm256_cvtps_epi32(ret2);

        intInputVal1 = _mm256_packs_epi32(intInputVal1, intInputVal2);
        intInputVal1 = _mm256_permute4x64_epi64(intInputVal1, 0xd8);

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal1);
        outputVectorPtr += 16;
    }

    volk_32fc_convert_16ic_generic(
        (lv_16sc_t*)outputVectorPtr,
        (const lv_32fc_t*)inputVectorPtr,
        num_points - avx_iters * 8);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_a_avx512(lv_16sc_t* outputVector,
                                                   const lv_32fc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m512 inputVal1;
    __m256i intInputVal;
    __m512 ret1;
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);
    unsigned int i;

    for (i = 0; i < avx512_iters; i++) {
        inputVal1 = _mm512_load_ps(inputVectorPtr);
        inputVectorPtr += 16;
        __VOLK_PREFETCH(inputVectorPtr + 16);

        // Clip
        ret1 = _mm512_max_ps(_mm512_min_ps(inputVal1, vmax_val), vmin_val);

        // Convert float to int32, then pack to int16 with saturation
        intInputVal = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(ret1));

        _mm256_store_si256((__m256i*)outputVectorPtr, intInputVal);
        outputVectorPtr += 16;
    }

    for (i = avx512_iters * 16; i < num_points * 2; i++) {
        aux = *inputVectorPtr++;
        if (aux > max_val)
            aux = max_val;
        else if (aux < min_val)
            aux = min_val;
        *outputVectorPtr++ = (int16_t)rintf(aux);
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_a_avx512bw(lv_16sc_t* outputVector,
                                                      const lv_32fc_t* inputVector,
                                                      unsigned int num_points)
{
    const unsigned int avx512bw_iters = num_points / 16;

    const float* inputVectorPtr = (const float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;
    const __m512 vmin_val = _mm512_set1_ps(min_val);
    const __m512 vmax_val = _mm512_set1_ps(max_val);
    const __m512i fix_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    unsigned int i;

    for (i = 0; i < avx512bw_iters; i++) {
        __m512 inputVal1 = _mm512_load_ps(inputVectorPtr);
        inputVectorPtr += 16;
        __m512 inputVal2 = _mm512_load_ps(inputVectorPtr);
        inputVectorPtr += 16;

        // Clip to [SHRT_MIN, SHRT_MAX]
        __m512 ret1 = _mm512_max_ps(_mm512_min_ps(inputVal1, vmax_val), vmin_val);
        __m512 ret2 = _mm512_max_ps(_mm512_min_ps(inputVal2, vmax_val), vmin_val);

        // Convert float to int32
        __m512i intVal1 = _mm512_cvtps_epi32(ret1);
        __m512i intVal2 = _mm512_cvtps_epi32(ret2);

        // Saturating narrow int32 -> int16, fix lane interleaving
        __m512i packed = _mm512_permutexvar_epi64(fix_idx, _mm512_packs_epi32(intVal1, intVal2));

        _mm512_store_si512((__m512i*)outputVectorPtr, packed);
        outputVectorPtr += 32;
    }

    volk_32fc_convert_16ic_generic(
        (lv_16sc_t*)outputVectorPtr,
        (const lv_32fc_t*)inputVectorPtr,
        num_points - avx512bw_iters * 16);
}
#endif /* LV_HAVE_AVX512BW */

#endif /* INCLUDED_volk_32fc_convert_16ic_a_H */
