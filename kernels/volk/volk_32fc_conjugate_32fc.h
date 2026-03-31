/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_conjugate_32fc
 *
 * \b Overview
 *
 * Computes the complex conjugate of each element in a vector, i.e.
 * cVector[i] = conj(aVector[i]). The real part is unchanged and the
 * imaginary part is negated.
 *
 * Complex conjugation is a fundamental building block in many DSP operations.
 * It appears in cross-correlation, matched filtering, and computing
 * cross-spectral density, where one signal must be conjugated before
 * multiplication. It is also used when reversing the sense of a frequency
 * shift or reflecting a spectrum.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_conjugate_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of complex samples (lv_32fc_t).
 * \li num_points: The number of complex samples.
 *
 * \b Outputs
 * \li cVector: The conjugated output vector of complex samples (lv_32fc_t).
 *
 * \b Example
 * Conjugate a constant vector and verify the imaginary part is negated.
 * \code
 * unsigned int N = 4;
 * unsigned int alignment = volk_get_alignment();
 * lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 * lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 * for (unsigned int i = 0; i < N; ++i) {
 *     in[i] = lv_cmake(3.0f, 4.0f);
 * }
 *
 * // Expected: conj(3 + 4j) = 3 - 4j
 * lv_32fc_t expected = lv_cmake(3.0f, -4.0f);
 *
 * volk_32fc_conjugate_32fc(out, in, N);
 *
 * printf("Expected: %+.1f %+.1fj\n", lv_creal(expected), lv_cimag(expected));
 * printf("Result:   %+.1f %+.1fj\n", lv_creal(out[0]), lv_cimag(out[0]));
 *
 * volk_free(in);
 * volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_conjugate_32fc_u_H
#define INCLUDED_volk_32fc_conjugate_32fc_u_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_conjugate_32fc_generic(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = lv_conj(*aPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_conjugate_32fc_u_sse(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m128 conjugator = _mm_setr_ps(0, -0.f, 0, -0.f);

    for (; number < halfPoints; number++) {

        x = _mm_loadu_ps((const float*)a);

        x = _mm_xor_ps(x, conjugator);

        _mm_storeu_ps((float*)c, x);

        a += 2;
        c += 2;
    }

    volk_32fc_conjugate_32fc_generic(c, a, num_points - halfPoints * 2);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_32fc_conjugate_32fc_u_sse3(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m128 conjugator = _mm_setr_ps(0, -0.f, 0, -0.f);

    for (; number < halfPoints; number++) {

        x = _mm_loadu_ps((const float*)a); // Load the complex data as ar,ai,br,bi

        x = _mm_xor_ps(x, conjugator); // conjugate register

        _mm_storeu_ps((float*)c, x); // Store the results back into the C container

        a += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = lv_conj(*a);
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_conjugate_32fc_u_avx(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);

    for (; number < quarterPoints; number++) {

        x = _mm256_loadu_ps((const float*)a); // Load the complex data as ar,ai,br,bi

        x = _mm256_xor_ps(x, conjugator); // conjugate register

        _mm256_storeu_ps((float*)c, x); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for (; number < num_points; number++) {
        *c++ = lv_conj(*a++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_conjugate_32fc_u_avx2(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);

    for (; number < quarterPoints; number++) {

        x = _mm256_loadu_ps((const float*)a); // Load the complex data as ar,ai,br,bi

        x = _mm256_xor_ps(x, conjugator); // conjugate register

        _mm256_storeu_ps((float*)c, x); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for (; number < num_points; number++) {
        *c++ = lv_conj(*a++);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_conjugate_32fc_u_avx512f(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    const __m512i conjugator = _mm512_set1_epi64(0x8000000000000000LL);

    for (; number < eighthPoints; number++) {

        __m512i x = _mm512_loadu_si512((const __m512i*)a);

        x = _mm512_xor_si512(x, conjugator);

        _mm512_storeu_si512((__m512i*)c, x);

        a += 8;
        c += 8;
    }

    volk_32fc_conjugate_32fc_generic(c, a, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_conjugate_32fc_neon(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   unsigned int num_points)
{
    unsigned int number;
    const unsigned int quarterPoints = num_points / 4;

    float32x4x2_t x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    for (number = 0; number < quarterPoints; number++) {
        __VOLK_PREFETCH(a + 4);
        x = vld2q_f32((const float*)a); // Load the complex data as ar,br,cr,dr; ai,bi,ci,di

        // xor the imaginary lane
        x.val[1] = vnegq_f32(x.val[1]);

        vst2q_f32((float*)c, x); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    for (number = quarterPoints * 4; number < num_points; number++) {
        *c++ = lv_conj(*a++);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_conjugate_32fc_neonv8(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   unsigned int num_points)
{
    unsigned int n = num_points;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    /* Sign mask to flip imaginary parts: [0, -0, 0, -0] */
    const uint32x4_t sign_mask =
        vreinterpretq_u32_f32((float32x4_t){ 0.0f, -0.0f, 0.0f, -0.0f });

    /* Process 4 complex numbers per iteration (2x unroll) */
    while (n >= 4) {
        float32x4_t v0 = vld1q_f32((const float*)a);
        float32x4_t v1 = vld1q_f32((const float*)(a + 2));
        __VOLK_PREFETCH(a + 8);

        /* XOR to flip sign of imaginary parts */
        v0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v0), sign_mask));
        v1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v1), sign_mask));

        vst1q_f32((float*)c, v0);
        vst1q_f32((float*)(c + 2), v1);

        a += 4;
        c += 4;
        n -= 4;
    }

    /* Scalar tail */
    while (n > 0) {
        *c++ = lv_conj(*a++);
        n--;
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_conjugate_32fc_rvv(lv_32fc_t* cVector,
                                                const lv_32fc_t* aVector,
                                                unsigned int num_points)
{
    size_t n = num_points;
    vuint64m8_t m = __riscv_vmv_v_x_u64m8(1ull << 63, __riscv_vsetvlmax_e64m8());
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t v = __riscv_vle64_v_u64m8((const uint64_t*)aVector, vl);
        __riscv_vse64((uint64_t*)cVector, __riscv_vxor(v, m, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_conjugate_32fc_rvvseg(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc =
            __riscv_vlseg2e32_v_f32m4x2((const float*)aVector, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0);
        vfloat32m4_t vi = __riscv_vfneg(__riscv_vget_f32m4(vc, 1), vl);
        __riscv_vsseg2e32_v_f32m4x2(
            (float*)cVector, __riscv_vcreate_v_f32m4x2(vr, vi), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#ifdef LV_HAVE_ORC

extern void volk_32fc_conjugate_32fc_a_orc_impl(lv_32fc_t* cVector,
                                                 const lv_32fc_t* aVector,
                                                 const float negone,
                                                 int num_points);

static inline void volk_32fc_conjugate_32fc_u_orc(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   unsigned int num_points)
{
    volk_32fc_conjugate_32fc_a_orc_impl(cVector, aVector, -1.0f, num_points);
}

#endif /* LV_HAVE_ORC */

#endif /* INCLUDED_volk_32fc_conjugate_32fc_u_H */
#ifndef INCLUDED_volk_32fc_conjugate_32fc_a_H
#define INCLUDED_volk_32fc_conjugate_32fc_a_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_conjugate_32fc_a_sse(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m128 conjugator = _mm_setr_ps(0, -0.f, 0, -0.f);

    for (; number < halfPoints; number++) {

        x = _mm_load_ps((const float*)a);

        x = _mm_xor_ps(x, conjugator);

        _mm_store_ps((float*)c, x);

        a += 2;
        c += 2;
    }

    volk_32fc_conjugate_32fc_generic(c, a, num_points - halfPoints * 2);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_32fc_conjugate_32fc_a_sse3(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m128 conjugator = _mm_setr_ps(0, -0.f, 0, -0.f);

    for (; number < halfPoints; number++) {

        x = _mm_load_ps((const float*)a); // Load the complex data as ar,ai,br,bi

        x = _mm_xor_ps(x, conjugator); // conjugate register

        _mm_store_ps((float*)c, x); // Store the results back into the C container

        a += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = lv_conj(*a);
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_conjugate_32fc_a_avx(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);

    for (; number < quarterPoints; number++) {

        x = _mm256_load_ps((const float*)a); // Load the complex data as ar,ai,br,bi

        x = _mm256_xor_ps(x, conjugator); // conjugate register

        _mm256_store_ps((float*)c, x); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for (; number < num_points; number++) {
        *c++ = lv_conj(*a++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_conjugate_32fc_a_avx2(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 x;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);

    for (; number < quarterPoints; number++) {

        x = _mm256_load_ps((const float*)a); // Load the complex data as ar,ai,br,bi

        x = _mm256_xor_ps(x, conjugator); // conjugate register

        _mm256_store_ps((float*)c, x); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for (; number < num_points; number++) {
        *c++ = lv_conj(*a++);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_conjugate_32fc_a_avx512f(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    const __m512i conjugator = _mm512_set1_epi64(0x8000000000000000LL);

    for (; number < eighthPoints; number++) {

        __m512i x = _mm512_load_si512((const __m512i*)a);

        x = _mm512_xor_si512(x, conjugator);

        _mm512_store_si512((__m512i*)c, x);

        a += 8;
        c += 8;
    }

    volk_32fc_conjugate_32fc_generic(c, a, num_points - eighthPoints * 8);
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32fc_conjugate_32fc_a_H */
