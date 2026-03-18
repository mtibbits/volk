/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_asin_32f
 *
 * \b Overview
 *
 * Computes arcsine of input vector and stores results in output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_asin_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The vector where results will be stored.
 *
 * \b Example
 * Calculate common angles around the top half of the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0;
 *   in[1] = 0.5;
 *   in[2] = std::sqrt(2.f)/2.f;
 *   in[3] = std::sqrt(3.f)/2.f;
 *   in[4] = in[5] = 1;
 *   for(unsigned int ii = 6; ii < N; ++ii){
 *       in[ii] = - in[N-ii-1];
 *   }
 *
 *   volk_32f_asin_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("asin(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_asin_32f_u_H
#define INCLUDED_volk_32f_asin_32f_u_H

#include <volk/volk_mathematical_functions.h>

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <volk/volk_common.h>


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_asin_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    for (unsigned int i = 0; i < num_points; i++) {
        bVector[i] = volk_arcsin(aVector[i]);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void
volk_32f_asin_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m128 pi_2 = _mm_set1_ps(0x1.921fb6p0f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 two = _mm_set1_ps(2.0f);
    const __m128 sign_mask = _mm_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_loadu_ps(aVector);

        __m128 sign = _mm_and_ps(aVal, sign_mask);
        __m128 ax = _mm_andnot_ps(sign_mask, aVal);

        __m128 t = _mm_mul_ps(_mm_sub_ps(one, ax), half);
        __m128 s = _mm_sqrt_ps(t);

        __m128 poly_small = _mm_arcsin_poly_sse(ax);
        __m128 poly_large = _mm_arcsin_poly_sse(s);

        __m128 result_large = _mm_sub_ps(pi_2, _mm_mul_ps(two, poly_large));

        __m128 mask = _mm_cmpgt_ps(ax, half);
        __m128 result = _mm_blendv_ps(poly_small, result_large, mask);

        result = _mm_or_ps(result, sign);

        _mm_storeu_ps(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32f_asin_32f_u_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aVector);

        __m256 sign = _mm256_and_ps(aVal, sign_mask);
        __m256 ax = _mm256_andnot_ps(sign_mask, aVal);

        __m256 t = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
        __m256 s = _mm256_sqrt_ps(t);

        __m256 poly_small = _mm256_arcsin_poly_avx(ax);
        __m256 poly_large = _mm256_arcsin_poly_avx(s);

        __m256 result_large = _mm256_sub_ps(pi_2, _mm256_mul_ps(two, poly_large));

        __m256 mask = _mm256_cmp_ps(ax, half, _CMP_GT_OS);
        __m256 result = _mm256_blendv_ps(poly_small, result_large, mask);

        result = _mm256_or_ps(result, sign);

        _mm256_storeu_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_asin_32f_u_avx_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const __m256 c0 = _mm256_set1_ps(0x1.ffffcep-1f);
    const __m256 c1 = _mm256_set1_ps(0x1.55b648p-3f);
    const __m256 c2 = _mm256_set1_ps(0x1.24d192p-4f);
    const __m256 c3 = _mm256_set1_ps(0x1.0a788p-4f);

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aVector);

        __m256 sign = _mm256_and_ps(aVal, sign_mask);
        __m256 ax = _mm256_andnot_ps(sign_mask, aVal);

        __m256 t = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
        __m256 s = _mm256_sqrt_ps(t);

        /* arcsin poly (small range): Horner with FMA on ax */
        __m256 u_small = _mm256_mul_ps(ax, ax);
        __m256 p_small = c3;
        p_small = _mm256_fmadd_ps(u_small, p_small, c2);
        p_small = _mm256_fmadd_ps(u_small, p_small, c1);
        p_small = _mm256_fmadd_ps(u_small, p_small, c0);
        __m256 poly_small = _mm256_mul_ps(ax, p_small);

        /* arcsin poly (large range): Horner with FMA on s */
        __m256 u_large = _mm256_mul_ps(s, s);
        __m256 p_large = c3;
        p_large = _mm256_fmadd_ps(u_large, p_large, c2);
        p_large = _mm256_fmadd_ps(u_large, p_large, c1);
        p_large = _mm256_fmadd_ps(u_large, p_large, c0);
        __m256 poly_large = _mm256_mul_ps(s, p_large);

        __m256 result_large = _mm256_fnmadd_ps(two, poly_large, pi_2);

        __m256 mask = _mm256_cmp_ps(ax, half, _CMP_GT_OS);
        __m256 result = _mm256_blendv_ps(poly_small, result_large, mask);

        result = _mm256_or_ps(result, sign);

        _mm256_storeu_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32f_asin_32f_u_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_loadu_ps(aVector);

        __m256 sign = _mm256_and_ps(aVal, sign_mask);
        __m256 ax = _mm256_andnot_ps(sign_mask, aVal);

        __m256 t = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
        __m256 s = _mm256_sqrt_ps(t);

        __m256 poly_small = _mm256_arcsin_poly_avx2_fma(ax);
        __m256 poly_large = _mm256_arcsin_poly_avx2_fma(s);

        __m256 result_large = _mm256_fnmadd_ps(two, poly_large, pi_2);

        __m256 mask = _mm256_cmp_ps(ax, half, _CMP_GT_OS);
        __m256 result = _mm256_blendv_ps(poly_small, result_large, mask);

        result = _mm256_or_ps(result, sign);

        _mm256_storeu_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void
volk_32f_asin_32f_u_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m512 pi_2 = _mm512_set1_ps(0x1.921fb6p0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aVector);

        __m512i sign = _mm512_and_epi32(_mm512_castps_si512(aVal), _mm512_castps_si512(sign_mask));
        __m512 ax = _mm512_castsi512_ps(_mm512_andnot_epi32(_mm512_castps_si512(sign_mask), _mm512_castps_si512(aVal)));

        __m512 t = _mm512_mul_ps(_mm512_sub_ps(one, ax), half);
        __m512 s = _mm512_sqrt_ps(t);

        __m512 poly_small = _mm512_arcsin_poly_avx512(ax);
        __m512 poly_large = _mm512_arcsin_poly_avx512(s);

        __m512 result_large = _mm512_fnmadd_ps(two, poly_large, pi_2);

        __mmask16 mask = _mm512_cmp_ps_mask(ax, half, _CMP_GT_OS);
        __m512 result = _mm512_mask_blend_ps(mask, poly_small, result_large);

        result = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(result), sign));

        _mm512_storeu_ps(bVector, result);

        aVector += 16;
        bVector += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512DQ
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void
volk_32f_asin_32f_u_avx512dq(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m512 pi_2 = _mm512_set1_ps(0x1.921fb6p0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_loadu_ps(aVector);

        __m512 sign = _mm512_and_ps(aVal, sign_mask);
        __m512 ax = _mm512_andnot_ps(sign_mask, aVal);

        __m512 t = _mm512_mul_ps(_mm512_sub_ps(one, ax), half);
        __m512 s = _mm512_sqrt_ps(t);

        __m512 poly_small = _mm512_arcsin_poly_avx512(ax);
        __m512 poly_large = _mm512_arcsin_poly_avx512(s);

        __m512 result_large = _mm512_fnmadd_ps(two, poly_large, pi_2);

        __mmask16 mask = _mm512_cmp_ps_mask(ax, half, _CMP_GT_OS);
        __m512 result = _mm512_mask_blend_ps(mask, poly_small, result_large);

        result = _mm512_or_ps(result, sign);

        _mm512_storeu_ps(bVector, result);

        aVector += 16;
        bVector += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX512DQ */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_asin_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    const float32x4_t pi_2 = vdupq_n_f32(0x1.921fb6p0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        float32x4_t aVal = vld1q_f32(aVector);

        // Get absolute value and sign
        float32x4_t ax = vabsq_f32(aVal);
        uint32x4_t sign_bits =
            vandq_u32(vreinterpretq_u32_f32(aVal), vdupq_n_u32(0x80000000));

        // Two-range computation
        float32x4_t t = vmulq_f32(vsubq_f32(one, ax), half);
        float32x4_t s = _vsqrtq_f32(t);

        // Compute polynomial for both ranges
        float32x4_t poly_small = _varcsinq_f32(ax);
        float32x4_t poly_large = _varcsinq_f32(s);

        // Large range: pi/2 - 2*poly_large
        float32x4_t result_large = vmlsq_f32(pi_2, two, poly_large);

        // Blend based on |x| > 0.5
        uint32x4_t mask = vcgtq_f32(ax, half);
        float32x4_t result = vbslq_f32(mask, result_large, poly_small);

        // Apply sign
        result =
            vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(result), sign_bits));

        vst1q_f32(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_asin_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    const float32x4_t pi_2 = vdupq_n_f32(0x1.921fb6p0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        float32x4_t aVal = vld1q_f32(aVector);

        float32x4_t ax = vabsq_f32(aVal);
        uint32x4_t sign_bits =
            vandq_u32(vreinterpretq_u32_f32(aVal), vdupq_n_u32(0x80000000));

        float32x4_t t = vmulq_f32(vsubq_f32(one, ax), half);
        float32x4_t s = vsqrtq_f32(t);

        float32x4_t poly_small = _varcsinq_f32_neonv8(ax);
        float32x4_t poly_large = _varcsinq_f32_neonv8(s);

        float32x4_t result_large = vfmsq_f32(pi_2, two, poly_large);

        uint32x4_t mask = vcgtq_f32(ax, half);
        float32x4_t result = vbslq_f32(mask, result_large, poly_small);

        result =
            vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(result), sign_bits));

        vst1q_f32(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_asin_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    /* arcsin polynomial coefficients: asin(x) = x * P(x^2) on |x| <= 0.5 */
    const float c0 = 0x1.ffffcep-1f;
    const float c1 = 0x1.55b648p-3f;
    const float c2 = 0x1.24d192p-4f;
    const float c3 = 0x1.0a788p-4f;
    const float pi_2 = 0x1.921fb6p0f;

    size_t n = (size_t)num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t x = __riscv_vle32_v_f32m4(aVector, vl);

        /* Extract sign and take absolute value */
        vfloat32m4_t ax = __riscv_vfabs(x, vl);
        vuint32m4_t sign_bits = __riscv_vand(
            __riscv_vreinterpret_u32m4(__riscv_vreinterpret_v_f32m4_i32m4(x)),
            0x80000000u, vl);

        /* Two-range approximation */
        vbool8_t large = __riscv_vmfgt(ax, 0.5f, vl);

        /* Large range: t = (1 - |x|) * 0.5, s = sqrt(t) */
        vfloat32m4_t t = __riscv_vfmul(
            __riscv_vfrsub(ax, 1.0f, vl), 0.5f, vl);
        vfloat32m4_t s = __riscv_vfsqrt(t, vl);

        /* Select input for polynomial: |x| when small, sqrt(t) when large */
        vfloat32m4_t poly_in = __riscv_vmerge(ax, s, large, vl);

        /* Evaluate arcsin polynomial via Horner: x * (c0 + u*(c1 + u*(c2 + u*c3))) */
        vfloat32m4_t u = __riscv_vfmul(poly_in, poly_in, vl);
        vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(c3, vl);
        p = __riscv_vfmadd(p, u, __riscv_vfmv_v_f_f32m4(c2, vl), vl);
        p = __riscv_vfmadd(p, u, __riscv_vfmv_v_f_f32m4(c1, vl), vl);
        p = __riscv_vfmadd(p, u, __riscv_vfmv_v_f_f32m4(c0, vl), vl);
        vfloat32m4_t poly_val = __riscv_vfmul(poly_in, p, vl);

        /* Large range result: pi/2 - 2 * poly_val */
        vfloat32m4_t result_large = __riscv_vfnmsac(
            __riscv_vfmv_v_f_f32m4(pi_2, vl), 2.0f, poly_val, vl);

        /* Select result */
        vfloat32m4_t result = __riscv_vmerge(poly_val, result_large, large, vl);

        /* Apply sign */
        vuint32m4_t result_bits = __riscv_vreinterpret_u32m4(
            __riscv_vreinterpret_v_f32m4_i32m4(result));
        result_bits = __riscv_vor(result_bits, sign_bits, vl);
        result = __riscv_vreinterpret_f32m4(
            __riscv_vreinterpret_v_u32m4_i32m4(result_bits));

        __riscv_vse32(bVector, result, vl);
    }
}
#endif /* LV_HAVE_RVV */

#endif /* INCLUDED_volk_32f_asin_32f_u_H */

#ifndef INCLUDED_volk_32f_asin_32f_a_H
#define INCLUDED_volk_32f_asin_32f_a_H

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void
volk_32f_asin_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m128 pi_2 = _mm_set1_ps(0x1.921fb6p0f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 two = _mm_set1_ps(2.0f);
    const __m128 sign_mask = _mm_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    for (; number < quarterPoints; number++) {
        __m128 aVal = _mm_load_ps(aVector);

        // Get absolute value and sign
        __m128 sign = _mm_and_ps(aVal, sign_mask);
        __m128 ax = _mm_andnot_ps(sign_mask, aVal);

        // Two-range computation
        // Small: result = arcsin_poly(x)
        // Large: result = pi/2 - 2*arcsin_poly(sqrt((1-|x|)/2))

        __m128 t = _mm_mul_ps(_mm_sub_ps(one, ax), half);
        __m128 s = _mm_sqrt_ps(t);

        // Compute polynomial for both ranges
        __m128 poly_small = _mm_arcsin_poly_sse(ax);
        __m128 poly_large = _mm_arcsin_poly_sse(s);

        // Large range: pi/2 - 2*poly_large
        __m128 result_large = _mm_sub_ps(pi_2, _mm_mul_ps(two, poly_large));

        // Blend based on |x| > 0.5
        __m128 mask = _mm_cmpgt_ps(ax, half);
        __m128 result = _mm_blendv_ps(poly_small, result_large, mask);

        // Apply sign
        result = _mm_or_ps(result, sign);

        _mm_store_ps(bVector, result);

        aVector += 4;
        bVector += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32f_asin_32f_a_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aVector);

        // Get absolute value and sign
        __m256 sign = _mm256_and_ps(aVal, sign_mask);
        __m256 ax = _mm256_andnot_ps(sign_mask, aVal);

        // Two-range computation
        __m256 t = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
        __m256 s = _mm256_sqrt_ps(t);

        // Compute polynomial for both ranges
        __m256 poly_small = _mm256_arcsin_poly_avx(ax);
        __m256 poly_large = _mm256_arcsin_poly_avx(s);

        // Large range: pi/2 - 2*poly_large
        __m256 result_large = _mm256_sub_ps(pi_2, _mm256_mul_ps(two, poly_large));

        // Blend based on |x| > 0.5
        __m256 mask = _mm256_cmp_ps(ax, half, _CMP_GT_OS);
        __m256 result = _mm256_blendv_ps(poly_small, result_large, mask);

        // Apply sign
        result = _mm256_or_ps(result, sign);

        _mm256_store_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32f_asin_32f_a_avx_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    const __m256 c0 = _mm256_set1_ps(0x1.ffffcep-1f);
    const __m256 c1 = _mm256_set1_ps(0x1.55b648p-3f);
    const __m256 c2 = _mm256_set1_ps(0x1.24d192p-4f);
    const __m256 c3 = _mm256_set1_ps(0x1.0a788p-4f);

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aVector);

        __m256 sign = _mm256_and_ps(aVal, sign_mask);
        __m256 ax = _mm256_andnot_ps(sign_mask, aVal);

        __m256 t = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
        __m256 s = _mm256_sqrt_ps(t);

        /* arcsin poly (small range): Horner with FMA on ax */
        __m256 u_small = _mm256_mul_ps(ax, ax);
        __m256 p_small = c3;
        p_small = _mm256_fmadd_ps(u_small, p_small, c2);
        p_small = _mm256_fmadd_ps(u_small, p_small, c1);
        p_small = _mm256_fmadd_ps(u_small, p_small, c0);
        __m256 poly_small = _mm256_mul_ps(ax, p_small);

        /* arcsin poly (large range): Horner with FMA on s */
        __m256 u_large = _mm256_mul_ps(s, s);
        __m256 p_large = c3;
        p_large = _mm256_fmadd_ps(u_large, p_large, c2);
        p_large = _mm256_fmadd_ps(u_large, p_large, c1);
        p_large = _mm256_fmadd_ps(u_large, p_large, c0);
        __m256 poly_large = _mm256_mul_ps(s, p_large);

        __m256 result_large = _mm256_fnmadd_ps(two, poly_large, pi_2);

        __m256 mask = _mm256_cmp_ps(ax, half, _CMP_GT_OS);
        __m256 result = _mm256_blendv_ps(poly_small, result_large, mask);

        result = _mm256_or_ps(result, sign);

        _mm256_store_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
#include <volk/volk_avx2_fma_intrinsics.h>

static inline void volk_32f_asin_32f_a_avx2_fma(float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    const __m256 pi_2 = _mm256_set1_ps(0x1.921fb6p0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    for (; number < eighthPoints; number++) {
        __m256 aVal = _mm256_load_ps(aVector);

        // Get absolute value and sign
        __m256 sign = _mm256_and_ps(aVal, sign_mask);
        __m256 ax = _mm256_andnot_ps(sign_mask, aVal);

        // Two-range computation
        __m256 t = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
        __m256 s = _mm256_sqrt_ps(t);

        // Compute polynomial for both ranges
        __m256 poly_small = _mm256_arcsin_poly_avx2_fma(ax);
        __m256 poly_large = _mm256_arcsin_poly_avx2_fma(s);

        // Large range: pi/2 - 2*poly_large
        __m256 result_large = _mm256_fnmadd_ps(two, poly_large, pi_2);

        // Blend based on |x| > 0.5
        __m256 mask = _mm256_cmp_ps(ax, half, _CMP_GT_OS);
        __m256 result = _mm256_blendv_ps(poly_small, result_large, mask);

        // Apply sign
        result = _mm256_or_ps(result, sign);

        _mm256_store_ps(bVector, result);

        aVector += 8;
        bVector += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void
volk_32f_asin_32f_a_avx512(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m512 pi_2 = _mm512_set1_ps(0x1.921fb6p0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aVector);

        // Get absolute value and sign using integer ops (AVX512F compatible)
        __m512i sign = _mm512_and_epi32(_mm512_castps_si512(aVal), _mm512_castps_si512(sign_mask));
        __m512 ax = _mm512_castsi512_ps(_mm512_andnot_epi32(_mm512_castps_si512(sign_mask), _mm512_castps_si512(aVal)));

        // Two-range computation
        __m512 t = _mm512_mul_ps(_mm512_sub_ps(one, ax), half);
        __m512 s = _mm512_sqrt_ps(t);

        // Compute polynomial for both ranges
        __m512 poly_small = _mm512_arcsin_poly_avx512(ax);
        __m512 poly_large = _mm512_arcsin_poly_avx512(s);

        // Large range: pi/2 - 2*poly_large
        __m512 result_large = _mm512_fnmadd_ps(two, poly_large, pi_2);

        // Blend based on |x| > 0.5
        __mmask16 mask = _mm512_cmp_ps_mask(ax, half, _CMP_GT_OS);
        __m512 result = _mm512_mask_blend_ps(mask, poly_small, result_large);

        // Apply sign
        result = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(result), sign));

        _mm512_store_ps(bVector, result);

        aVector += 16;
        bVector += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX512DQ
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

static inline void
volk_32f_asin_32f_a_avx512dq(float* bVector, const float* aVector, unsigned int num_points)
{
    const __m512 pi_2 = _mm512_set1_ps(0x1.921fb6p0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    for (; number < sixteenthPoints; number++) {
        __m512 aVal = _mm512_load_ps(aVector);

        __m512 sign = _mm512_and_ps(aVal, sign_mask);
        __m512 ax = _mm512_andnot_ps(sign_mask, aVal);

        __m512 t = _mm512_mul_ps(_mm512_sub_ps(one, ax), half);
        __m512 s = _mm512_sqrt_ps(t);

        __m512 poly_small = _mm512_arcsin_poly_avx512(ax);
        __m512 poly_large = _mm512_arcsin_poly_avx512(s);

        __m512 result_large = _mm512_fnmadd_ps(two, poly_large, pi_2);

        __mmask16 mask = _mm512_cmp_ps_mask(ax, half, _CMP_GT_OS);
        __m512 result = _mm512_mask_blend_ps(mask, poly_small, result_large);

        result = _mm512_or_ps(result, sign);

        _mm512_store_ps(bVector, result);

        aVector += 16;
        bVector += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        *bVector++ = volk_arcsin(*aVector++);
    }
}

#endif /* LV_HAVE_AVX512DQ */

#endif /* INCLUDED_volk_32f_asin_32f_a_H */
