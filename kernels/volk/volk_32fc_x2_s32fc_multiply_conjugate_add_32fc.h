/* -*- c++ -*- */
/*
 * Copyright 2019 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_s32fc_multiply_conjugate_add_32fc
 *
 * \b Deprecation
 *
 * This kernel is deprecated, because passing in `lv_32fc_t` by value results in
 * Undefined Behaviour, causing a segmentation fault on some architectures.
 * Use `volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc` instead.
 *
 * \b Overview
 *
 * Conjugates the input complex vector, multiplies each element by a complex scalar,
 * and adds the result to another input complex vector:
 *
 * c[i] = a[i] + conj(b[i]) * scalar
 *
 * This conjugate-multiply-accumulate operation is a building block for adaptive
 * filtering algorithms such as LMS and CMA equalizers, where filter tap weights
 * are updated each sample interval using conjugated signal samples scaled by an
 * error term.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_s32fc_multiply_conjugate_add_32fc(lv_32fc_t* cVector, const
 * lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar, unsigned int
 * num_points);
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector to be added, e.g. current tap weights (lv_32fc_t).
 * \li bVector: The input vector to be conjugated and multiplied, e.g. signal samples (lv_32fc_t).
 * \li scalar: The complex scalar to multiply against conjugated bVector, e.g. step-size error term (lv_32fc_t).
 * \li num_points: The number of complex values to process.
 *
 * \b Outputs
 * \li cVector: The output vector of updated values (lv_32fc_t).
 *
 * \b Example
 * Compute c[i] = a[i] + conj(b[i]) * scalar with constant inputs.
 *
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 *
 * lv_32fc_t* aVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* bVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* cVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *   aVector[i] = lv_cmake(1.0f, 2.0f);
 *   bVector[i] = lv_cmake(3.0f, 4.0f);
 * }
 * lv_32fc_t scalar = lv_cmake(2.0f, 0.0f);
 *
 * // conj(b) = (3, -4), conj(b) * scalar = (6, -8)
 * // c = a + conj(b) * scalar = (1+6, 2-8) = (7, -6)
 * lv_32fc_t expected = lv_cmake(7.0f, -6.0f);
 *
 * volk_32fc_x2_s32fc_multiply_conjugate_add_32fc(cVector, aVector, bVector, scalar, N);
 *
 * printf("Expected: (%1.1f, %1.1f)\n", lv_creal(expected), lv_cimag(expected));
 * printf("Result:   (%1.1f, %1.1f)\n", lv_creal(cVector[0]), lv_cimag(cVector[0]));
 *
 * volk_free(aVector);
 * volk_free(bVector);
 * volk_free(cVector);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_H
#define INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_generic(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* bVector,
                                                       const lv_32fc_t scalar,
                                                       unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_generic(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_sse(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_sse(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE3

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_sse3(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_sse3(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_AVX

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_avx(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     const lv_32fc_t scalar,
                                                     unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_avx(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_avx_fma(lv_32fc_t* cVector,
                                                          const lv_32fc_t* aVector,
                                                          const lv_32fc_t* bVector,
                                                          const lv_32fc_t scalar,
                                                          unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_avx_fma(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */


#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_avx2_fma(lv_32fc_t* cVector,
                                                           const lv_32fc_t* aVector,
                                                           const lv_32fc_t* bVector,
                                                           const lv_32fc_t scalar,
                                                           unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_avx2_fma(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_avx512f(lv_32fc_t* cVector,
                                                          const lv_32fc_t* aVector,
                                                          const lv_32fc_t* bVector,
                                                          const lv_32fc_t scalar,
                                                          unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_avx512f(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_neon(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    const lv_32fc_t scalar,
                                                    unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_neon(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_neonv8(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_neonv8(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_rvv(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    const lv_32fc_t scalar,
                                                    unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_rvv(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_rvvseg(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* bVector,
                                                       const lv_32fc_t scalar,
                                                       unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_rvvseg(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_u_H */

#ifndef INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_H
#define INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_H


#ifdef LV_HAVE_SSE

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_sse(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_sse(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_SSE3

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_sse3(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t scalar,
                                                      unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_sse3(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_AVX

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_avx(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     const lv_32fc_t scalar,
                                                     unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_avx(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_avx_fma(lv_32fc_t* cVector,
                                                          const lv_32fc_t* aVector,
                                                          const lv_32fc_t* bVector,
                                                          const lv_32fc_t scalar,
                                                          unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_avx_fma(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_avx2_fma(lv_32fc_t* cVector,
                                                           const lv_32fc_t* aVector,
                                                           const lv_32fc_t* bVector,
                                                           const lv_32fc_t scalar,
                                                           unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_avx2_fma(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_avx512f(lv_32fc_t* cVector,
                                                          const lv_32fc_t* aVector,
                                                          const lv_32fc_t* bVector,
                                                          const lv_32fc_t scalar,
                                                          unsigned int num_points)
{
    volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_avx512f(
        cVector, aVector, bVector, &scalar, num_points);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add_32fc_a_H */
