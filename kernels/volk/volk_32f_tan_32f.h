/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_tan_32f
 *
 * \b Overview
 *
 * Computes the tangent of each element in the input buffer:
 * bVector[i] = tan(aVector[i]).
 *
 * The tangent function appears in digital filter design, particularly in the
 * bilinear transform where tan(omega/2) maps analog prototype frequencies to
 * the digital domain. It is also used in coordinate transformations and
 * phase-related computations in signal processing pipelines.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_tan_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: Input buffer of angles in radians (float).
 * \li num_points: The number of values to process.
 *
 * \b Outputs
 * \li bVector: Output buffer for tangent values (float).
 *
 * \b Example
 * Compute tan(theta) for common angles and verify against known values.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * float* in = (float*)volk_malloc(sizeof(float) * N, alignment);
 * float* out = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 * // tan(0)=0, tan(pi/4)=1, tan(-pi/4)=-1, tan(pi/6)=1/sqrt(3)
 * in[0] = 0.0f;
 * in[1] = 0.7853981f;  // pi/4
 * in[2] = -0.7853981f; // -pi/4
 * in[3] = 0.5235988f;  // pi/6
 *
 * float expected3 = 0.5773503f; // tan(pi/6) = 1/sqrt(3)
 *
 * volk_32f_tan_32f(out, in, N);
 *
 * printf("Expected: tan(0)=0, tan(pi/4)=1, tan(-pi/4)=-1, tan(pi/6)=%.7f\n", expected3);
 * printf("Result:   tan(0)=%.7f, tan(pi/4)=%.7f, tan(-pi/4)=%.7f, tan(pi/6)=%.7f\n",
 *        out[0], out[1], out[2], out[3]);
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifndef INCLUDED_volk_32f_tan_32f_u_H
#define INCLUDED_volk_32f_tan_32f_u_H

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_tan_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_tan_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    unsigned int i = 0;

    __m128 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m128 sine, cosine, tangent, condition1, condition2, condition3;
    __m128i q, r, ones, twos, fours;

    m4pi = _mm_set1_ps(1.273239545);
    pio4A = _mm_set1_ps(0.78515625);
    pio4B = _mm_set1_ps(0.241876e-3);
    ffours = _mm_set1_ps(4.0);
    ftwos = _mm_set1_ps(2.0);
    fones = _mm_set1_ps(1.0);
    fzeroes = _mm_setzero_ps();
    ones = _mm_set1_epi32(1);
    twos = _mm_set1_epi32(2);
    fours = _mm_set1_epi32(4);

    cp1 = _mm_set1_ps(1.0);
    cp2 = _mm_set1_ps(0.83333333e-1);
    cp3 = _mm_set1_ps(0.2777778e-2);
    cp4 = _mm_set1_ps(0.49603e-4);
    cp5 = _mm_set1_ps(0.551e-6);

    for (; number < quarterPoints; number++) {
        aVal = _mm_loadu_ps(aPtr);
        s = _mm_sub_ps(aVal,
                       _mm_and_ps(_mm_mul_ps(aVal, ftwos), _mm_cmplt_ps(aVal, fzeroes)));
        q = _mm_cvtps_epi32(_mm_floor_ps(_mm_mul_ps(s, m4pi)));
        r = _mm_add_epi32(q, _mm_and_si128(q, ones));

        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4A));
        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4B));

        s = _mm_div_ps(
            s, _mm_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm_mul_ps(
            _mm_add_ps(
                _mm_mul_ps(
                    _mm_sub_ps(
                        _mm_mul_ps(
                            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(s, cp5), cp4), s),
                                       cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm_mul_ps(s, _mm_sub_ps(ffours, s));
        }
        s = _mm_div_ps(s, ftwos);

        sine = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(ftwos, s), s));
        cosine = _mm_sub_ps(fones, s);

        condition1 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, ones), twos)), fzeroes);
        condition2 = _mm_cmpneq_ps(
            _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(q, fours)), fzeroes),
            _mm_cmplt_ps(aVal, fzeroes));
        condition3 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, twos), fours)), fzeroes);

        __m128 temp = cosine;
        cosine = _mm_add_ps(cosine, _mm_and_ps(_mm_sub_ps(sine, cosine), condition1));
        sine = _mm_add_ps(sine, _mm_and_ps(_mm_sub_ps(temp, sine), condition1));
        sine =
            _mm_sub_ps(sine, _mm_and_ps(_mm_mul_ps(sine, _mm_set1_ps(2.0f)), condition2));
        cosine = _mm_sub_ps(
            cosine, _mm_and_ps(_mm_mul_ps(cosine, _mm_set1_ps(2.0f)), condition3));
        tangent = _mm_div_ps(sine, cosine);
        _mm_storeu_ps(bPtr, tangent);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    /* 128-bit integer constants for range reduction (plain AVX has no 256-bit integer ops) */
    __m128i ones_128, twos_128, fours_128;

    m4pi = _mm256_set1_ps(1.273239545f);
    pio4A = _mm256_set1_ps(0.78515625f);
    pio4B = _mm256_set1_ps(0.241876e-3f);
    ffours = _mm256_set1_ps(4.0f);
    ftwos = _mm256_set1_ps(2.0f);
    fones = _mm256_set1_ps(1.0f);
    fzeroes = _mm256_setzero_ps();
    ones_128 = _mm_set1_epi32(1);
    twos_128 = _mm_set1_epi32(2);
    fours_128 = _mm_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0f);
    cp2 = _mm256_set1_ps(0.83333333e-1f);
    cp3 = _mm256_set1_ps(0.2777778e-2f);
    cp4 = _mm256_set1_ps(0.49603e-4f);
    cp5 = _mm256_set1_ps(0.551e-6f);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));

        /* Compute q = floor(s * m4pi): _mm256_cvtps_epi32 + floor IS AVX */
        __m256 s_m4pi_floor = _mm256_floor_ps(_mm256_mul_ps(s, m4pi));
        /* Split q to 128-bit halves for integer operations */
        __m128i q_lo = _mm_cvtps_epi32(_mm256_castps256_ps128(s_m4pi_floor));
        __m128i q_hi = _mm_cvtps_epi32(_mm256_extractf128_ps(s_m4pi_floor, 1));

        /* r = q + (q & 1) */
        __m128i r_lo = _mm_add_epi32(q_lo, _mm_and_si128(q_lo, ones_128));
        __m128i r_hi = _mm_add_epi32(q_hi, _mm_and_si128(q_hi, ones_128));

        /* Convert r back to float for range reduction */
        __m256 r_f = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(r_lo)),
                                          _mm_cvtepi32_ps(r_hi), 1);
        s = _mm256_sub_ps(s, _mm256_mul_ps(r_f, pio4A));
        s = _mm256_sub_ps(s, _mm256_mul_ps(r_f, pio4B));

        s = _mm256_div_ps(s, _mm256_set1_ps(8.0f)); /* 2^N for 3× argument reduction */
        s = _mm256_mul_ps(s, s);
        /* Evaluate Taylor series */
        s = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(s, cp5), cp4), s),
                                cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        /* Compute conditions using 128-bit integer ops then recombine */
        /* condition1: ((q+1) & 2) != 0 */
        __m128i q1_lo = _mm_add_epi32(q_lo, ones_128);
        __m128i q1_hi = _mm_add_epi32(q_hi, ones_128);
        __m256 cond1_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q1_lo, twos_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q1_hi, twos_128)), 1);
        condition1 = _mm256_cmp_ps(cond1_f, fzeroes, _CMP_NEQ_UQ);

        /* condition2: ((q & 4) != 0) XOR (aVal < 0) */
        __m256 q_and4_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q_hi, fours_128)), 1);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(q_and4_f, fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);

        /* condition3: ((q+2) & 4) != 0 */
        __m128i q2_lo = _mm_add_epi32(q_lo, twos_128);
        __m128i q2_hi = _mm_add_epi32(q_hi, twos_128);
        __m256 cond3_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q2_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q2_hi, fours_128)), 1);
        condition3 = _mm256_cmp_ps(cond3_f, fzeroes, _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_storeu_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for unaligned */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    /* 128-bit integer constants for range reduction (plain AVX has no 256-bit integer ops) */
    __m128i ones_128, twos_128, fours_128;

    m4pi = _mm256_set1_ps(1.273239545f);
    pio4A = _mm256_set1_ps(0.78515625f);
    pio4B = _mm256_set1_ps(0.241876e-3f);
    ffours = _mm256_set1_ps(4.0f);
    ftwos = _mm256_set1_ps(2.0f);
    fones = _mm256_set1_ps(1.0f);
    fzeroes = _mm256_setzero_ps();
    ones_128 = _mm_set1_epi32(1);
    twos_128 = _mm_set1_epi32(2);
    fours_128 = _mm_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0f);
    cp2 = _mm256_set1_ps(0.83333333e-1f);
    cp3 = _mm256_set1_ps(0.2777778e-2f);
    cp4 = _mm256_set1_ps(0.49603e-4f);
    cp5 = _mm256_set1_ps(0.551e-6f);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));

        /* Compute q = floor(s * m4pi): _mm256_cvtps_epi32 + floor IS AVX */
        __m256 s_m4pi_floor = _mm256_floor_ps(_mm256_mul_ps(s, m4pi));
        /* Split q to 128-bit halves for integer operations */
        __m128i q_lo = _mm_cvtps_epi32(_mm256_castps256_ps128(s_m4pi_floor));
        __m128i q_hi = _mm_cvtps_epi32(_mm256_extractf128_ps(s_m4pi_floor, 1));

        /* r = q + (q & 1) */
        __m128i r_lo = _mm_add_epi32(q_lo, _mm_and_si128(q_lo, ones_128));
        __m128i r_hi = _mm_add_epi32(q_hi, _mm_and_si128(q_hi, ones_128));

        /* Convert r back to float for range reduction */
        __m256 r_f = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(r_lo)),
                                          _mm_cvtepi32_ps(r_hi), 1);
        s = _mm256_fnmadd_ps(r_f, pio4A, s);
        s = _mm256_fnmadd_ps(r_f, pio4B, s);

        s = _mm256_div_ps(s, _mm256_set1_ps(8.0f)); /* 2^N for 3x argument reduction */
        s = _mm256_mul_ps(s, s);
        /* Evaluate Taylor series using FMA */
        __m256 poly = _mm256_fmsub_ps(s, cp5, cp4);
        poly = _mm256_fmadd_ps(poly, s, cp3);
        poly = _mm256_fmsub_ps(poly, s, cp2);
        poly = _mm256_fmadd_ps(poly, s, cp1);
        s = _mm256_mul_ps(poly, s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        /* Compute conditions using 128-bit integer ops then recombine */
        /* condition1: ((q+1) & 2) != 0 */
        __m128i q1_lo = _mm_add_epi32(q_lo, ones_128);
        __m128i q1_hi = _mm_add_epi32(q_hi, ones_128);
        __m256 cond1_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q1_lo, twos_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q1_hi, twos_128)), 1);
        condition1 = _mm256_cmp_ps(cond1_f, fzeroes, _CMP_NEQ_UQ);

        /* condition2: ((q & 4) != 0) XOR (aVal < 0) */
        __m256 q_and4_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q_hi, fours_128)), 1);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(q_and4_f, fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);

        /* condition3: ((q+2) & 4) != 0 */
        __m128i q2_lo = _mm_add_epi32(q_lo, twos_128);
        __m128i q2_hi = _mm_add_epi32(q_hi, twos_128);
        __m256 cond3_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q2_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q2_hi, fours_128)), 1);
        condition3 = _mm256_cmp_ps(cond3_f, fzeroes, _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_storeu_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4A));
        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4B));

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(s, cp5), cp4),
                                              s),
                                cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_storeu_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for unaligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_loadu_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4A, s);
        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4B, s);

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_fmadd_ps(
                _mm256_fmsub_ps(
                    _mm256_fmadd_ps(_mm256_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_storeu_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx512f(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int i = 0;

    __m512 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones;
    __m512 sine, cosine, tangent;
    __m512i q, r, ones, twos, fours;

    m4pi = _mm512_set1_ps(1.273239545);
    pio4A = _mm512_set1_ps(0.78515625);
    pio4B = _mm512_set1_ps(0.241876e-3);
    ffours = _mm512_set1_ps(4.0);
    ftwos = _mm512_set1_ps(2.0);
    fones = _mm512_set1_ps(1.0);
    ones = _mm512_set1_epi32(1);
    twos = _mm512_set1_epi32(2);
    fours = _mm512_set1_epi32(4);

    cp1 = _mm512_set1_ps(1.0);
    cp2 = _mm512_set1_ps(0.83333333e-1);
    cp3 = _mm512_set1_ps(0.2777778e-2);
    cp4 = _mm512_set1_ps(0.49603e-4);
    cp5 = _mm512_set1_ps(0.551e-6);

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_loadu_ps(aPtr);

        /* s = abs(aVal) */
        s = _mm512_abs_ps(aVal);

        q = _mm512_cvtps_epi32(_mm512_floor_ps(_mm512_mul_ps(s, m4pi)));
        r = _mm512_add_epi32(q, _mm512_and_si512(q, ones));

        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4A, s);
        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4B, s);

        s = _mm512_div_ps(s, _mm512_set1_ps(8.0));
        s = _mm512_mul_ps(s, s);

        /* Evaluate Taylor series */
        s = _mm512_mul_ps(
            _mm512_fmadd_ps(
                _mm512_fmsub_ps(
                    _mm512_fmadd_ps(_mm512_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm512_mul_ps(s, _mm512_sub_ps(ffours, s));
        }
        s = _mm512_div_ps(s, ftwos);

        sine = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_sub_ps(ftwos, s), s));
        cosine = _mm512_sub_ps(fones, s);

        /* Conditionally swap sine/cosine */
        __mmask16 cond1 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, ones), twos),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        __m512 temp_sine = sine;
        sine = _mm512_mask_blend_ps(cond1, sine, cosine);
        cosine = _mm512_mask_blend_ps(cond1, cosine, temp_sine);

        /* Negate sine where needed based on sign of input XOR octant */
        __mmask16 neg_input = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OS);
        __mmask16 q_and_4 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(q, fours), _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask16 cond2 = neg_input ^ q_and_4;
        sine = _mm512_mask_sub_ps(sine, cond2, _mm512_setzero_ps(), sine);

        /* Negate cosine where needed */
        __mmask16 cond3 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, twos), fours),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        cosine = _mm512_mask_sub_ps(cosine, cond3, _mm512_setzero_ps(), cosine);

        tangent = _mm512_div_ps(sine, cosine);
        _mm512_storeu_ps(bPtr, tangent);
        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_tan_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512F for unaligned */

#ifdef LV_HAVE_AVX512DQ
#include <immintrin.h>

static inline void
volk_32f_tan_32f_u_avx512dq(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int i = 0;

    __m512 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones;
    __m512 sine, cosine, tangent;
    __m512i q, r, ones, twos, fours;

    m4pi = _mm512_set1_ps(1.273239545);
    pio4A = _mm512_set1_ps(0.78515625);
    pio4B = _mm512_set1_ps(0.241876e-3);
    ffours = _mm512_set1_ps(4.0);
    ftwos = _mm512_set1_ps(2.0);
    fones = _mm512_set1_ps(1.0);
    ones = _mm512_set1_epi32(1);
    twos = _mm512_set1_epi32(2);
    fours = _mm512_set1_epi32(4);

    cp1 = _mm512_set1_ps(1.0);
    cp2 = _mm512_set1_ps(0.83333333e-1);
    cp3 = _mm512_set1_ps(0.2777778e-2);
    cp4 = _mm512_set1_ps(0.49603e-4);
    cp5 = _mm512_set1_ps(0.551e-6);

    /* DQ: abs_mask for float-domain absolute value via _mm512_and_ps */
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_loadu_ps(aPtr);

        /* s = abs(aVal) — DQ: native float-domain AND, no integer bypass */
        s = _mm512_and_ps(aVal, abs_mask);

        q = _mm512_cvtps_epi32(_mm512_floor_ps(_mm512_mul_ps(s, m4pi)));
        r = _mm512_add_epi32(q, _mm512_and_si512(q, ones));

        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4A, s);
        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4B, s);

        s = _mm512_div_ps(s, _mm512_set1_ps(8.0));
        s = _mm512_mul_ps(s, s);

        /* Evaluate Taylor series */
        s = _mm512_mul_ps(
            _mm512_fmadd_ps(
                _mm512_fmsub_ps(
                    _mm512_fmadd_ps(_mm512_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm512_mul_ps(s, _mm512_sub_ps(ffours, s));
        }
        s = _mm512_div_ps(s, ftwos);

        sine = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_sub_ps(ftwos, s), s));
        cosine = _mm512_sub_ps(fones, s);

        /* Conditionally swap sine/cosine */
        __mmask16 cond1 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, ones), twos),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        __m512 temp_sine = sine;
        sine = _mm512_mask_blend_ps(cond1, sine, cosine);
        cosine = _mm512_mask_blend_ps(cond1, cosine, temp_sine);

        /* Negate sine where needed based on sign of input XOR octant */
        __mmask16 neg_input = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OS);
        __mmask16 q_and_4 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(q, fours), _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask16 cond2 = neg_input ^ q_and_4;
        sine = _mm512_mask_sub_ps(sine, cond2, _mm512_setzero_ps(), sine);

        /* Negate cosine where needed */
        __mmask16 cond3 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, twos), fours),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        cosine = _mm512_mask_sub_ps(cosine, cond3, _mm512_setzero_ps(), cosine);

        tangent = _mm512_div_ps(sine, cosine);
        _mm512_storeu_ps(bPtr, tangent);
        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_tan_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512DQ for unaligned */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_tan_32f_neon(float* bVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    float* bVectorPtr = bVector;
    const float* aVectorPtr = aVector;

    float32x4_t b_vec;
    float32x4_t a_vec;

    for (number = 0; number < quarter_points; number++) {
        a_vec = vld1q_f32(aVectorPtr);
        // Prefetch next one, speeds things up
        __VOLK_PREFETCH(aVectorPtr + 4);
        b_vec = _vtanq_f32(a_vec);
        vst1q_f32(bVectorPtr, b_vec);
        // move pointers ahead
        bVectorPtr += 4;
        aVectorPtr += 4;
    }

    // Deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        *bVectorPtr++ = tanf(*aVectorPtr++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32f_tan_32f_neonv8(float* bVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    float* bVectorPtr = bVector;
    const float* aVectorPtr = aVector;

    for (number = 0; number < quarter_points; number++) {
        float32x4_t a_vec = vld1q_f32(aVectorPtr);
        // Use sincos, then native division for tan = sin/cos
        const float32x4x2_t sincos = _vsincosq_f32(a_vec);
        float32x4_t b_vec = vdivq_f32(sincos.val[0], sincos.val[1]);
        vst1q_f32(bVectorPtr, b_vec);
        bVectorPtr += 4;
        aVectorPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *bVectorPtr++ = tanf(*aVectorPtr++);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_tan_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t c4oPi = __riscv_vfmv_v_f_f32m2(1.2732395f, vlmax);
    const vfloat32m2_t cPio4a = __riscv_vfmv_v_f_f32m2(0.7853982f, vlmax);
    const vfloat32m2_t cPio4b = __riscv_vfmv_v_f_f32m2(7.946627e-09f, vlmax);
    const vfloat32m2_t cPio4c = __riscv_vfmv_v_f_f32m2(3.061617e-17f, vlmax);

    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vfloat32m2_t cf4 = __riscv_vfmv_v_f_f32m2(4.0f, vlmax);

    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(0.0833333333f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(0.0027777778f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(4.9603175e-05f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(5.5114638e-07f, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t v = __riscv_vle32_v_f32m2(aVector, vl);
        vfloat32m2_t s = __riscv_vfabs(v, vl);
        vint32m2_t q = __riscv_vfcvt_x(__riscv_vfmul(s, c4oPi, vl), vl);
        vfloat32m2_t r = __riscv_vfcvt_f(__riscv_vadd(q, __riscv_vand(q, 1, vl), vl), vl);

        s = __riscv_vfnmsac(s, cPio4a, r, vl);
        s = __riscv_vfnmsac(s, cPio4b, r, vl);
        s = __riscv_vfnmsac(s, cPio4c, r, vl);

        s = __riscv_vfmul(s, 1 / 8.0f, vl);
        s = __riscv_vfmul(s, s, vl);
        vfloat32m2_t t = s;
        s = __riscv_vfmsub(s, c5, c4, vl);
        s = __riscv_vfmadd(s, t, c3, vl);
        s = __riscv_vfmsub(s, t, c2, vl);
        s = __riscv_vfmadd(s, t, cf1, vl);
        s = __riscv_vfmul(s, t, vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, __riscv_vfsub(cf4, s, vl), vl);
        s = __riscv_vfmul(s, 1 / 2.0f, vl);

        vfloat32m2_t sine =
            __riscv_vfsqrt(__riscv_vfmul(__riscv_vfrsub(s, 2.0f, vl), s, vl), vl);
        vfloat32m2_t cosine = __riscv_vfsub(cf1, s, vl);

        vbool16_t m1 = __riscv_vmsne(__riscv_vand(__riscv_vadd(q, 1, vl), 2, vl), 0, vl);
        vbool16_t m2 = __riscv_vmsne(__riscv_vand(__riscv_vadd(q, 2, vl), 4, vl), 0, vl);
        vbool16_t m3 = __riscv_vmxor(__riscv_vmslt(__riscv_vreinterpret_i32m2(v), 0, vl),
                                     __riscv_vmsne(__riscv_vand(q, 4, vl), 0, vl),
                                     vl);

        vfloat32m2_t sine0 = sine;
        sine = __riscv_vmerge(sine, cosine, m1, vl);
        sine = __riscv_vfneg_mu(m3, sine, sine, vl);

        cosine = __riscv_vmerge(cosine, sine0, m1, vl);
        cosine = __riscv_vfneg_mu(m2, cosine, cosine, vl);

        __riscv_vse32(bVector, __riscv_vfdiv(sine, cosine, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#endif /* INCLUDED_volk_32f_tan_32f_u_H */

#ifndef INCLUDED_volk_32f_tan_32f_a_H
#define INCLUDED_volk_32f_tan_32f_a_H

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_tan_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int quarterPoints = num_points / 4;
    unsigned int i = 0;

    __m128 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m128 sine, cosine, tangent, condition1, condition2, condition3;
    __m128i q, r, ones, twos, fours;

    m4pi = _mm_set1_ps(1.273239545);
    pio4A = _mm_set1_ps(0.78515625);
    pio4B = _mm_set1_ps(0.241876e-3);
    ffours = _mm_set1_ps(4.0);
    ftwos = _mm_set1_ps(2.0);
    fones = _mm_set1_ps(1.0);
    fzeroes = _mm_setzero_ps();
    ones = _mm_set1_epi32(1);
    twos = _mm_set1_epi32(2);
    fours = _mm_set1_epi32(4);

    cp1 = _mm_set1_ps(1.0);
    cp2 = _mm_set1_ps(0.83333333e-1);
    cp3 = _mm_set1_ps(0.2777778e-2);
    cp4 = _mm_set1_ps(0.49603e-4);
    cp5 = _mm_set1_ps(0.551e-6);

    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        s = _mm_sub_ps(aVal,
                       _mm_and_ps(_mm_mul_ps(aVal, ftwos), _mm_cmplt_ps(aVal, fzeroes)));
        q = _mm_cvtps_epi32(_mm_floor_ps(_mm_mul_ps(s, m4pi)));
        r = _mm_add_epi32(q, _mm_and_si128(q, ones));

        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4A));
        s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4B));

        s = _mm_div_ps(
            s, _mm_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm_mul_ps(
            _mm_add_ps(
                _mm_mul_ps(
                    _mm_sub_ps(
                        _mm_mul_ps(
                            _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(s, cp5), cp4), s),
                                       cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm_mul_ps(s, _mm_sub_ps(ffours, s));
        }
        s = _mm_div_ps(s, ftwos);

        sine = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(ftwos, s), s));
        cosine = _mm_sub_ps(fones, s);

        condition1 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, ones), twos)), fzeroes);
        condition2 = _mm_cmpneq_ps(
            _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(q, fours)), fzeroes),
            _mm_cmplt_ps(aVal, fzeroes));
        condition3 = _mm_cmpneq_ps(
            _mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, twos), fours)), fzeroes);

        __m128 temp = cosine;
        cosine = _mm_add_ps(cosine, _mm_and_ps(_mm_sub_ps(sine, cosine), condition1));
        sine = _mm_add_ps(sine, _mm_and_ps(_mm_sub_ps(temp, sine), condition1));
        sine =
            _mm_sub_ps(sine, _mm_and_ps(_mm_mul_ps(sine, _mm_set1_ps(2.0f)), condition2));
        cosine = _mm_sub_ps(
            cosine, _mm_and_ps(_mm_mul_ps(cosine, _mm_set1_ps(2.0f)), condition3));
        tangent = _mm_div_ps(sine, cosine);
        _mm_store_ps(bPtr, tangent);
        aPtr += 4;
        bPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    /* 128-bit integer constants for range reduction (plain AVX has no 256-bit integer ops) */
    __m128i ones_128, twos_128, fours_128;

    m4pi = _mm256_set1_ps(1.273239545f);
    pio4A = _mm256_set1_ps(0.78515625f);
    pio4B = _mm256_set1_ps(0.241876e-3f);
    ffours = _mm256_set1_ps(4.0f);
    ftwos = _mm256_set1_ps(2.0f);
    fones = _mm256_set1_ps(1.0f);
    fzeroes = _mm256_setzero_ps();
    ones_128 = _mm_set1_epi32(1);
    twos_128 = _mm_set1_epi32(2);
    fours_128 = _mm_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0f);
    cp2 = _mm256_set1_ps(0.83333333e-1f);
    cp3 = _mm256_set1_ps(0.2777778e-2f);
    cp4 = _mm256_set1_ps(0.49603e-4f);
    cp5 = _mm256_set1_ps(0.551e-6f);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));

        /* Compute q = floor(s * m4pi): _mm256_cvtps_epi32 + floor IS AVX */
        __m256 s_m4pi_floor = _mm256_floor_ps(_mm256_mul_ps(s, m4pi));
        /* Split q to 128-bit halves for integer operations */
        __m128i q_lo = _mm_cvtps_epi32(_mm256_castps256_ps128(s_m4pi_floor));
        __m128i q_hi = _mm_cvtps_epi32(_mm256_extractf128_ps(s_m4pi_floor, 1));

        /* r = q + (q & 1) */
        __m128i r_lo = _mm_add_epi32(q_lo, _mm_and_si128(q_lo, ones_128));
        __m128i r_hi = _mm_add_epi32(q_hi, _mm_and_si128(q_hi, ones_128));

        /* Convert r back to float for range reduction */
        __m256 r_f = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(r_lo)),
                                          _mm_cvtepi32_ps(r_hi), 1);
        s = _mm256_sub_ps(s, _mm256_mul_ps(r_f, pio4A));
        s = _mm256_sub_ps(s, _mm256_mul_ps(r_f, pio4B));

        s = _mm256_div_ps(s, _mm256_set1_ps(8.0f)); /* 2^N for 3× argument reduction */
        s = _mm256_mul_ps(s, s);
        /* Evaluate Taylor series */
        s = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(s, cp5), cp4), s),
                                cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        /* Compute conditions using 128-bit integer ops then recombine */
        /* condition1: ((q+1) & 2) != 0 */
        __m128i q1_lo = _mm_add_epi32(q_lo, ones_128);
        __m128i q1_hi = _mm_add_epi32(q_hi, ones_128);
        __m256 cond1_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q1_lo, twos_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q1_hi, twos_128)), 1);
        condition1 = _mm256_cmp_ps(cond1_f, fzeroes, _CMP_NEQ_UQ);

        /* condition2: ((q & 4) != 0) XOR (aVal < 0) */
        __m256 q_and4_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q_hi, fours_128)), 1);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(q_and4_f, fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);

        /* condition3: ((q+2) & 4) != 0 */
        __m128i q2_lo = _mm_add_epi32(q_lo, twos_128);
        __m128i q2_hi = _mm_add_epi32(q_hi, twos_128);
        __m256 cond3_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q2_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q2_hi, fours_128)), 1);
        condition3 = _mm256_cmp_ps(cond3_f, fzeroes, _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_store_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX for aligned */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    /* 128-bit integer constants for range reduction (plain AVX has no 256-bit integer ops) */
    __m128i ones_128, twos_128, fours_128;

    m4pi = _mm256_set1_ps(1.273239545f);
    pio4A = _mm256_set1_ps(0.78515625f);
    pio4B = _mm256_set1_ps(0.241876e-3f);
    ffours = _mm256_set1_ps(4.0f);
    ftwos = _mm256_set1_ps(2.0f);
    fones = _mm256_set1_ps(1.0f);
    fzeroes = _mm256_setzero_ps();
    ones_128 = _mm_set1_epi32(1);
    twos_128 = _mm_set1_epi32(2);
    fours_128 = _mm_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0f);
    cp2 = _mm256_set1_ps(0.83333333e-1f);
    cp3 = _mm256_set1_ps(0.2777778e-2f);
    cp4 = _mm256_set1_ps(0.49603e-4f);
    cp5 = _mm256_set1_ps(0.551e-6f);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));

        /* Compute q = floor(s * m4pi): _mm256_cvtps_epi32 + floor IS AVX */
        __m256 s_m4pi_floor = _mm256_floor_ps(_mm256_mul_ps(s, m4pi));
        /* Split q to 128-bit halves for integer operations */
        __m128i q_lo = _mm_cvtps_epi32(_mm256_castps256_ps128(s_m4pi_floor));
        __m128i q_hi = _mm_cvtps_epi32(_mm256_extractf128_ps(s_m4pi_floor, 1));

        /* r = q + (q & 1) */
        __m128i r_lo = _mm_add_epi32(q_lo, _mm_and_si128(q_lo, ones_128));
        __m128i r_hi = _mm_add_epi32(q_hi, _mm_and_si128(q_hi, ones_128));

        /* Convert r back to float for range reduction */
        __m256 r_f = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_cvtepi32_ps(r_lo)),
                                          _mm_cvtepi32_ps(r_hi), 1);
        s = _mm256_fnmadd_ps(r_f, pio4A, s);
        s = _mm256_fnmadd_ps(r_f, pio4B, s);

        s = _mm256_div_ps(s, _mm256_set1_ps(8.0f)); /* 2^N for 3x argument reduction */
        s = _mm256_mul_ps(s, s);
        /* Evaluate Taylor series using FMA */
        __m256 poly = _mm256_fmsub_ps(s, cp5, cp4);
        poly = _mm256_fmadd_ps(poly, s, cp3);
        poly = _mm256_fmsub_ps(poly, s, cp2);
        poly = _mm256_fmadd_ps(poly, s, cp1);
        s = _mm256_mul_ps(poly, s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        /* Compute conditions using 128-bit integer ops then recombine */
        /* condition1: ((q+1) & 2) != 0 */
        __m128i q1_lo = _mm_add_epi32(q_lo, ones_128);
        __m128i q1_hi = _mm_add_epi32(q_hi, ones_128);
        __m256 cond1_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q1_lo, twos_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q1_hi, twos_128)), 1);
        condition1 = _mm256_cmp_ps(cond1_f, fzeroes, _CMP_NEQ_UQ);

        /* condition2: ((q & 4) != 0) XOR (aVal < 0) */
        __m256 q_and4_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q_hi, fours_128)), 1);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(q_and4_f, fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);

        /* condition3: ((q+2) & 4) != 0 */
        __m128i q2_lo = _mm_add_epi32(q_lo, twos_128);
        __m128i q2_hi = _mm_add_epi32(q_hi, twos_128);
        __m256 cond3_f = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm_cvtepi32_ps(_mm_and_si128(q2_lo, fours_128))),
            _mm_cvtepi32_ps(_mm_and_si128(q2_hi, fours_128)), 1);
        condition3 = _mm256_cmp_ps(cond3_f, fzeroes, _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_store_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx2(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4A));
        s = _mm256_sub_ps(s, _mm256_mul_ps(_mm256_cvtepi32_ps(r), pio4B));

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_add_ps(
                _mm256_mul_ps(
                    _mm256_sub_ps(
                        _mm256_mul_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(s, cp5), cp4),
                                              s),
                                cp3),
                            s),
                        cp2),
                    s),
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_store_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for aligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx2_fma(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    unsigned int eighthPoints = num_points / 8;
    unsigned int i = 0;

    __m256 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones,
        fzeroes;
    __m256 sine, cosine, tangent, condition1, condition2, condition3;
    __m256i q, r, ones, twos, fours;

    m4pi = _mm256_set1_ps(1.273239545);
    pio4A = _mm256_set1_ps(0.78515625);
    pio4B = _mm256_set1_ps(0.241876e-3);
    ffours = _mm256_set1_ps(4.0);
    ftwos = _mm256_set1_ps(2.0);
    fones = _mm256_set1_ps(1.0);
    fzeroes = _mm256_setzero_ps();
    ones = _mm256_set1_epi32(1);
    twos = _mm256_set1_epi32(2);
    fours = _mm256_set1_epi32(4);

    cp1 = _mm256_set1_ps(1.0);
    cp2 = _mm256_set1_ps(0.83333333e-1);
    cp3 = _mm256_set1_ps(0.2777778e-2);
    cp4 = _mm256_set1_ps(0.49603e-4);
    cp5 = _mm256_set1_ps(0.551e-6);

    for (; number < eighthPoints; number++) {
        aVal = _mm256_load_ps(aPtr);
        s = _mm256_sub_ps(aVal,
                          _mm256_and_ps(_mm256_mul_ps(aVal, ftwos),
                                        _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS)));
        q = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(s, m4pi)));
        r = _mm256_add_epi32(q, _mm256_and_si256(q, ones));

        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4A, s);
        s = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(r), pio4B, s);

        s = _mm256_div_ps(
            s,
            _mm256_set1_ps(8.0)); // The constant is 2^N, for 3 times argument reduction
        s = _mm256_mul_ps(s, s);
        // Evaluate Taylor series
        s = _mm256_mul_ps(
            _mm256_fmadd_ps(
                _mm256_fmsub_ps(
                    _mm256_fmadd_ps(_mm256_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm256_mul_ps(s, _mm256_sub_ps(ffours, s));
        }
        s = _mm256_div_ps(s, ftwos);

        sine = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_sub_ps(ftwos, s), s));
        cosine = _mm256_sub_ps(fones, s);

        condition1 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, ones), twos)),
            fzeroes,
            _CMP_NEQ_UQ);
        condition2 = _mm256_cmp_ps(
            _mm256_cmp_ps(
                _mm256_cvtepi32_ps(_mm256_and_si256(q, fours)), fzeroes, _CMP_NEQ_UQ),
            _mm256_cmp_ps(aVal, fzeroes, _CMP_LT_OS),
            _CMP_NEQ_UQ);
        condition3 = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_add_epi32(q, twos), fours)),
            fzeroes,
            _CMP_NEQ_UQ);

        __m256 temp = cosine;
        cosine =
            _mm256_add_ps(cosine, _mm256_and_ps(_mm256_sub_ps(sine, cosine), condition1));
        sine = _mm256_add_ps(sine, _mm256_and_ps(_mm256_sub_ps(temp, sine), condition1));
        sine = _mm256_sub_ps(
            sine, _mm256_and_ps(_mm256_mul_ps(sine, _mm256_set1_ps(2.0f)), condition2));
        cosine = _mm256_sub_ps(
            cosine,
            _mm256_and_ps(_mm256_mul_ps(cosine, _mm256_set1_ps(2.0f)), condition3));
        tangent = _mm256_div_ps(sine, cosine);
        _mm256_store_ps(bPtr, tangent);
        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *bPtr++ = tanf(*aPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx512f(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int i = 0;

    __m512 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones;
    __m512 sine, cosine, tangent;
    __m512i q, r, ones, twos, fours;

    m4pi = _mm512_set1_ps(1.273239545);
    pio4A = _mm512_set1_ps(0.78515625);
    pio4B = _mm512_set1_ps(0.241876e-3);
    ffours = _mm512_set1_ps(4.0);
    ftwos = _mm512_set1_ps(2.0);
    fones = _mm512_set1_ps(1.0);
    ones = _mm512_set1_epi32(1);
    twos = _mm512_set1_epi32(2);
    fours = _mm512_set1_epi32(4);

    cp1 = _mm512_set1_ps(1.0);
    cp2 = _mm512_set1_ps(0.83333333e-1);
    cp3 = _mm512_set1_ps(0.2777778e-2);
    cp4 = _mm512_set1_ps(0.49603e-4);
    cp5 = _mm512_set1_ps(0.551e-6);

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_load_ps(aPtr);

        /* s = abs(aVal) */
        s = _mm512_abs_ps(aVal);

        q = _mm512_cvtps_epi32(_mm512_floor_ps(_mm512_mul_ps(s, m4pi)));
        r = _mm512_add_epi32(q, _mm512_and_si512(q, ones));

        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4A, s);
        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4B, s);

        s = _mm512_div_ps(s, _mm512_set1_ps(8.0));
        s = _mm512_mul_ps(s, s);

        /* Evaluate Taylor series */
        s = _mm512_mul_ps(
            _mm512_fmadd_ps(
                _mm512_fmsub_ps(
                    _mm512_fmadd_ps(_mm512_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm512_mul_ps(s, _mm512_sub_ps(ffours, s));
        }
        s = _mm512_div_ps(s, ftwos);

        sine = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_sub_ps(ftwos, s), s));
        cosine = _mm512_sub_ps(fones, s);

        /* Conditionally swap sine/cosine */
        __mmask16 cond1 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, ones), twos),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        __m512 temp_sine = sine;
        sine = _mm512_mask_blend_ps(cond1, sine, cosine);
        cosine = _mm512_mask_blend_ps(cond1, cosine, temp_sine);

        /* Negate sine where needed based on sign of input XOR octant */
        __mmask16 neg_input = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OS);
        __mmask16 q_and_4 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(q, fours), _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask16 cond2 = neg_input ^ q_and_4;
        sine = _mm512_mask_sub_ps(sine, cond2, _mm512_setzero_ps(), sine);

        /* Negate cosine where needed */
        __mmask16 cond3 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, twos), fours),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        cosine = _mm512_mask_sub_ps(cosine, cond3, _mm512_setzero_ps(), cosine);

        tangent = _mm512_div_ps(sine, cosine);
        _mm512_store_ps(bPtr, tangent);
        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_tan_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512F for aligned */

#ifdef LV_HAVE_AVX512DQ
#include <immintrin.h>

static inline void
volk_32f_tan_32f_a_avx512dq(float* bVector, const float* aVector, unsigned int num_points)
{
    float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;
    unsigned int i = 0;

    __m512 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones;
    __m512 sine, cosine, tangent;
    __m512i q, r, ones, twos, fours;

    m4pi = _mm512_set1_ps(1.273239545);
    pio4A = _mm512_set1_ps(0.78515625);
    pio4B = _mm512_set1_ps(0.241876e-3);
    ffours = _mm512_set1_ps(4.0);
    ftwos = _mm512_set1_ps(2.0);
    fones = _mm512_set1_ps(1.0);
    ones = _mm512_set1_epi32(1);
    twos = _mm512_set1_epi32(2);
    fours = _mm512_set1_epi32(4);

    cp1 = _mm512_set1_ps(1.0);
    cp2 = _mm512_set1_ps(0.83333333e-1);
    cp3 = _mm512_set1_ps(0.2777778e-2);
    cp4 = _mm512_set1_ps(0.49603e-4);
    cp5 = _mm512_set1_ps(0.551e-6);

    /* DQ: abs_mask for float-domain absolute value via _mm512_and_ps */
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    for (; number < sixteenthPoints; number++) {
        aVal = _mm512_load_ps(aPtr);

        /* s = abs(aVal) — DQ: native float-domain AND, no integer bypass */
        s = _mm512_and_ps(aVal, abs_mask);

        q = _mm512_cvtps_epi32(_mm512_floor_ps(_mm512_mul_ps(s, m4pi)));
        r = _mm512_add_epi32(q, _mm512_and_si512(q, ones));

        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4A, s);
        s = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(r), pio4B, s);

        s = _mm512_div_ps(s, _mm512_set1_ps(8.0));
        s = _mm512_mul_ps(s, s);

        /* Evaluate Taylor series */
        s = _mm512_mul_ps(
            _mm512_fmadd_ps(
                _mm512_fmsub_ps(
                    _mm512_fmadd_ps(_mm512_fmsub_ps(s, cp5, cp4), s, cp3), s, cp2),
                s,
                cp1),
            s);

        for (i = 0; i < 3; i++) {
            s = _mm512_mul_ps(s, _mm512_sub_ps(ffours, s));
        }
        s = _mm512_div_ps(s, ftwos);

        sine = _mm512_sqrt_ps(_mm512_mul_ps(_mm512_sub_ps(ftwos, s), s));
        cosine = _mm512_sub_ps(fones, s);

        /* Conditionally swap sine/cosine */
        __mmask16 cond1 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, ones), twos),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        __m512 temp_sine = sine;
        sine = _mm512_mask_blend_ps(cond1, sine, cosine);
        cosine = _mm512_mask_blend_ps(cond1, cosine, temp_sine);

        /* Negate sine where needed based on sign of input XOR octant */
        __mmask16 neg_input = _mm512_cmp_ps_mask(aVal, _mm512_setzero_ps(), _CMP_LT_OS);
        __mmask16 q_and_4 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(q, fours), _mm512_setzero_si512(), _MM_CMPINT_NE);
        __mmask16 cond2 = neg_input ^ q_and_4;
        sine = _mm512_mask_sub_ps(sine, cond2, _mm512_setzero_ps(), sine);

        /* Negate cosine where needed */
        __mmask16 cond3 = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_add_epi32(q, twos), fours),
            _mm512_setzero_si512(),
            _MM_CMPINT_NE);
        cosine = _mm512_mask_sub_ps(cosine, cond3, _mm512_setzero_ps(), cosine);

        tangent = _mm512_div_ps(sine, cosine);
        _mm512_store_ps(bPtr, tangent);
        aPtr += 16;
        bPtr += 16;
    }

    number = sixteenthPoints * 16;
    volk_32f_tan_32f_generic(bPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_AVX512DQ for aligned */


#endif /* INCLUDED_volk_32f_tan_32f_a_H */
