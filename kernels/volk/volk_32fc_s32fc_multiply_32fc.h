/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32fc_multiply_32fc
 *
 * \b Deprecation
 *
 * This kernel is deprecated, because passing in `lv_32fc_t` by value results in
 * Undefined Behaviour, causing a segmentation fault on some architectures.
 * Use `volk_32fc_s32fc_multiply2_32fc` instead.
 *
 * \b Overview
 *
 * Multiplies each element of a complex input vector by a complex scalar:
 * cVector[i] = aVector[i] * scalar. This performs element-wise complex
 * multiplication using standard complex arithmetic.
 *
 * In DSP applications, scaling a complex signal by a complex scalar is used
 * for phase rotation, frequency shifting, and applying complex gains in
 * operations such as AGC, beamforming, or channel equalization.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_multiply_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input complex signal vector (lv_32fc_t).
 * \li scalar: The complex scalar to multiply against aVector (lv_32fc_t).
 * \li num_points: The number of complex samples in aVector.
 *
 * \b Outputs
 * \li cVector: The output complex vector (lv_32fc_t).
 *
 * \b Example
 * Multiply a constant complex vector by a scalar and verify the result.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 * // Input: (3, 4) for all elements; scalar: (2, -1)
 * for (unsigned int i = 0; i < N; ++i) {
 *     in[i] = lv_cmake(3.0f, 4.0f);
 * }
 * lv_32fc_t scalar = lv_cmake(2.0f, -1.0f);
 *
 * // Expected: (3+4j)*(2-1j) = 6-3j+8j-4j^2 = (10, 5)
 * lv_32fc_t expected = lv_cmake(10.0f, 5.0f);
 *
 * volk_32fc_s32fc_multiply_32fc(out, in, scalar, N);
 *
 * printf("Expected: (%1.1f, %1.1f)\n", lv_creal(expected), lv_cimag(expected));
 * printf("Result:   (%1.1f, %1.1f)\n", lv_creal(out[0]), lv_cimag(out[0]));
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32fc_multiply_32fc_u_H
#define INCLUDED_volk_32fc_s32fc_multiply_32fc_u_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_32fc_s32fc_multiply2_32fc.h>
#include <volk/volk_complex.h>

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void volk_32fc_s32fc_multiply_32fc_u_avx_fma(lv_32fc_t* cVector,
                                                           const lv_32fc_t* aVector,
                                                           const lv_32fc_t scalar,
                                                           unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_u_avx_fma(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_multiply_32fc_u_avx(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t scalar,
                                                       unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_u_avx(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

static inline void volk_32fc_s32fc_multiply_32fc_u_sse3(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t scalar,
                                                        unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_u_sse3(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_multiply_32fc_generic(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t scalar,
                                                         unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_generic(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_x2_multiply_32fc_u_H */
#ifndef INCLUDED_volk_32fc_s32fc_multiply_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_multiply_32fc_a_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void volk_32fc_s32fc_multiply_32fc_a_avx_fma(lv_32fc_t* cVector,
                                                           const lv_32fc_t* aVector,
                                                           const lv_32fc_t scalar,
                                                           unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_a_avx_fma(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX

static inline void volk_32fc_s32fc_multiply_32fc_a_avx(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t scalar,
                                                       unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_a_avx(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

static inline void volk_32fc_s32fc_multiply_32fc_a_sse3(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t scalar,
                                                        unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_a_sse3(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_NEON

static inline void volk_32fc_s32fc_multiply_32fc_neon(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_neon(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8

static inline void volk_32fc_s32fc_multiply_32fc_neonv8(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t scalar,
                                                        unsigned int num_points)
{
    volk_32fc_s32fc_multiply2_32fc_neonv8(cVector, aVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32fc_x2_multiply_32fc_a_H */
