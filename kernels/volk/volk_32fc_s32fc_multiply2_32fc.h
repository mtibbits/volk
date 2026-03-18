/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32fc_multiply2_32fc
 *
 * \b Overview
 *
 * Multiplies the input complex vector by a complex scalar and returns
 * the results.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_multiply2_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * lv_32fc_t* scalar, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector to be multiplied.
 * \li scalar: The complex scalar to multiply against aVector.
 * \li num_points: The number of complex values in aVector.
 *
 * \b Outputs
 * \li cVector: The vector where the results will be stored.
 *
 * \b Example
 * Generate points around the unit circle and shift the phase pi/3 rad.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t scalar = lv_cmake((float)std::cos(M_PI/3.f), (float)std::sin(M_PI/3.f));
 *
 *   float delta = 2.f*M_PI / (float)N;
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       // Generate points around the unit circle
 *       float real = std::cos(delta * (float)ii);
 *       float imag = std::sin(delta * (float)ii);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii+N/2] = lv_cmake(-real, -imag);
 *   }
 *
 *   volk_32fc_s32fc_multiply2_32fc(out, in, &scalar, N);
 *
 *   printf(" mag   phase  |   mag   phase\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("%+1.2f  %+1.2f  |  %+1.2f   %+1.2f\n",
 *           std::abs(in[ii]), std::arg(in[ii]),
 *           std::abs(out[ii]), std::arg(out[ii]));
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32fc_multiply2_32fc_u_H
#define INCLUDED_volk_32fc_s32fc_multiply2_32fc_u_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_multiply2_32fc_generic(lv_32fc_t* cVector,
                                                          const lv_32fc_t* aVector,
                                                          const lv_32fc_t* scalar,
                                                          unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    unsigned int number = num_points;

    // unwrap loop
    while (number >= 8) {
        *cPtr++ = (*aPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) * (*scalar);
        number -= 8;
    }

    // clean up any remaining
    while (number-- > 0)
        *cPtr++ = *aPtr++ * (*scalar);
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_u_sse(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t* scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    const __m128 neg_mask = _mm_setr_ps(-0.f, 0.f, -0.f, 0.f);
    __m128 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    yl = _mm_set_ps1(lv_creal(*scalar));
    yh = _mm_set_ps1(lv_cimag(*scalar));

    for (; number < halfPoints; number++) {
        x = _mm_loadu_ps((const float*)a);
        tmp1 = _mm_mul_ps(x, yl);
        x = _mm_shuffle_ps(x, x, 0xB1);
        tmp2 = _mm_mul_ps(x, yh);
        z = _mm_add_ps(tmp1, _mm_xor_ps(tmp2, neg_mask));
        _mm_storeu_ps((float*)c, z);
        a += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = (*a) * (*scalar);
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_u_sse3(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t* scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm_set_ps1(lv_creal(*scalar));
    yh = _mm_set_ps1(lv_cimag(*scalar));

    for (; number < halfPoints; number++) {

        x = _mm_loadu_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = _mm_mul_ps(x, yl); // tmp1 = ar*cr,ai*cr,br*dr,bi*dr

        x = _mm_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm_addsub_ps(tmp1,
                          tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm_storeu_ps((float*)c, z); // Store the results back into the C container

        a += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = (*a) * (*scalar);
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_u_avx(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t* scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_loadu_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = _mm256_mul_ps(x, yl); // tmp1 = ar*cr,ai*cr,br*dr,bi*dr

        x = _mm256_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm256_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm256_addsub_ps(tmp1,
                             tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_storeu_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_u_avx2(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t* scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_loadu_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = _mm256_mul_ps(x, yl); // tmp1 = ar*cr,ai*cr,br*dr,bi*dr

        x = _mm256_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm256_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm256_addsub_ps(tmp1,
                             tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_storeu_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_u_avx_fma(lv_32fc_t* cVector,
                                                            const lv_32fc_t* aVector,
                                                            const lv_32fc_t* scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_loadu_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = x;

        x = _mm256_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm256_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm256_fmaddsub_ps(
            tmp1, yl, tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_storeu_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_u_avx2_fma(lv_32fc_t* cVector,
                                                               const lv_32fc_t* aVector,
                                                               const lv_32fc_t* scalar,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_loadu_ps((const float*)a);
        tmp1 = x;
        x = _mm256_shuffle_ps(x, x, 0xB1);
        tmp2 = _mm256_mul_ps(x, yh);
        z = _mm256_fmaddsub_ps(tmp1, yl, tmp2);
        _mm256_storeu_ps((float*)c, z);
        a += 4;
        c += 4;
    }

    for (unsigned int i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_u_avx512f(lv_32fc_t* cVector,
                                                             const lv_32fc_t* aVector,
                                                             const lv_32fc_t* scalar,
                                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    unsigned int remainder = num_points & 7;
    __m512 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm512_set1_ps(lv_creal(*scalar));
    yh = _mm512_set1_ps(lv_cimag(*scalar));

    for (; number < eighthPoints; number++) {
        x = _mm512_loadu_ps((const float*)a);

        tmp1 = x;

        x = _mm512_shuffle_ps(x, x, 0xB1); // Swap re/im: ai,ar,bi,br,...

        tmp2 = _mm512_mul_ps(x, yh); // tmp2 = ai*ci, ar*ci, ...

        z = _mm512_fmaddsub_ps(tmp1, yl, tmp2); // re-im, im+re, ...

        _mm512_storeu_ps((float*)c, z);

        a += 8;
        c += 8;
    }

    if (remainder) {
        volk_32fc_s32fc_multiply2_32fc_generic(c, a, scalar, remainder);
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_s32fc_multiply2_32fc_neon(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* scalar,
                                                       unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    float32x4x2_t a_val, scalar_val;
    float32x4x2_t tmp_imag;

    scalar_val.val[0] = vld1q_dup_f32((const float*)scalar);
    scalar_val.val[1] = vld1q_dup_f32(((const float*)scalar) + 1);
    for (number = 0; number < quarter_points; ++number) {
        a_val = vld2q_f32((const float*)aPtr);
        tmp_imag.val[1] = vmulq_f32(a_val.val[1], scalar_val.val[0]);
        tmp_imag.val[0] = vmulq_f32(a_val.val[0], scalar_val.val[0]);

        tmp_imag.val[1] = vmlaq_f32(tmp_imag.val[1], a_val.val[0], scalar_val.val[1]);
        tmp_imag.val[0] = vmlsq_f32(tmp_imag.val[0], a_val.val[1], scalar_val.val[1]);

        vst2q_f32((float*)cPtr, tmp_imag);
        aPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = *aPtr++ * (*scalar);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_s32fc_multiply2_32fc_neonv8(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t* scalar,
                                                         unsigned int num_points)
{
    unsigned int n = num_points;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    /* Broadcast scalar real and imag parts */
    const float32x4_t sr = vdupq_n_f32(lv_creal(*scalar));
    const float32x4_t si = vdupq_n_f32(lv_cimag(*scalar));

    /* Process 8 complex numbers per iteration (2x unroll) */
    while (n >= 8) {
        float32x4x2_t a0 = vld2q_f32((const float*)a);
        float32x4x2_t a1 = vld2q_f32((const float*)(a + 4));
        __VOLK_PREFETCH(a + 8);

        /* Complex multiply using FMA:
         * real = ar*sr - ai*si = fms(ar*sr, ai, si)
         * imag = ar*si + ai*sr = fma(ar*si, ai, sr)
         */
        float32x4x2_t c0, c1;
        c0.val[0] = vfmsq_f32(vmulq_f32(a0.val[0], sr), a0.val[1], si);
        c0.val[1] = vfmaq_f32(vmulq_f32(a0.val[0], si), a0.val[1], sr);
        c1.val[0] = vfmsq_f32(vmulq_f32(a1.val[0], sr), a1.val[1], si);
        c1.val[1] = vfmaq_f32(vmulq_f32(a1.val[0], si), a1.val[1], sr);

        vst2q_f32((float*)c, c0);
        vst2q_f32((float*)(c + 4), c1);

        a += 8;
        c += 8;
        n -= 8;
    }

    /* Process remaining 4 */
    if (n >= 4) {
        float32x4x2_t a0 = vld2q_f32((const float*)a);
        float32x4x2_t c0;
        c0.val[0] = vfmsq_f32(vmulq_f32(a0.val[0], sr), a0.val[1], si);
        c0.val[1] = vfmaq_f32(vmulq_f32(a0.val[0], si), a0.val[1], sr);
        vst2q_f32((float*)c, c0);
        a += 4;
        c += 4;
        n -= 4;
    }

    /* Scalar tail */
    while (n > 0) {
        *c++ = (*a++) * (*scalar);
        n--;
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_s32fc_multiply2_32fc_rvv(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* scalar,
                                                       unsigned int num_points)
{
    const float sr = lv_creal(*scalar);
    const float si = lv_cimag(*scalar);
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e64m4(n);
        vuint64m4_t vc = __riscv_vle64_v_u64m4((const uint64_t*)aVector, vl);
        vfloat32m2_t ar = __riscv_vreinterpret_f32m2(__riscv_vnsrl(vc, 0, vl));
        vfloat32m2_t ai = __riscv_vreinterpret_f32m2(__riscv_vnsrl(vc, 32, vl));
        vfloat32m2_t cr =
            __riscv_vfsub(__riscv_vfmul(ar, sr, vl), __riscv_vfmul(ai, si, vl), vl);
        vfloat32m2_t ci =
            __riscv_vfadd(__riscv_vfmul(ar, si, vl), __riscv_vfmul(ai, sr, vl), vl);
        vuint64m4_t lo = __riscv_vzext_vf2(__riscv_vreinterpret_u32m2(cr), vl);
        vuint64m4_t hi =
            __riscv_vsll(__riscv_vzext_vf2(__riscv_vreinterpret_u32m2(ci), vl), 32, vl);
        __riscv_vse64((uint64_t*)cVector, __riscv_vor(lo, hi, vl), vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_s32fc_multiply2_32fc_rvvseg(lv_32fc_t* cVector,
                                                          const lv_32fc_t* aVector,
                                                          const lv_32fc_t* scalar,
                                                          unsigned int num_points)
{
    const float sr = lv_creal(*scalar);
    const float si = lv_cimag(*scalar);
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t va =
            __riscv_vlseg2e32_v_f32m4x2((const float*)aVector, vl);
        vfloat32m4_t ar = __riscv_vget_f32m4(va, 0);
        vfloat32m4_t ai = __riscv_vget_f32m4(va, 1);
        vfloat32m4_t cr =
            __riscv_vfsub(__riscv_vfmul(ar, sr, vl), __riscv_vfmul(ai, si, vl), vl);
        vfloat32m4_t ci =
            __riscv_vfadd(__riscv_vfmul(ar, si, vl), __riscv_vfmul(ai, sr, vl), vl);
        __riscv_vsseg2e32_v_f32m4x2(
            (float*)cVector, __riscv_vcreate_v_f32m4x2(cr, ci), vl);
    }
}
#endif /* LV_HAVE_RVVSEG */

#endif /* INCLUDED_volk_32fc_s32fc_multiply2_32fc_u_H */
#ifndef INCLUDED_volk_32fc_s32fc_multiply2_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_multiply2_32fc_a_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_a_sse(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t* scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    const __m128 neg_mask = _mm_setr_ps(-0.f, 0.f, -0.f, 0.f);
    __m128 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    yl = _mm_set_ps1(lv_creal(*scalar));
    yh = _mm_set_ps1(lv_cimag(*scalar));

    for (; number < halfPoints; number++) {
        x = _mm_load_ps((const float*)a);
        tmp1 = _mm_mul_ps(x, yl);
        x = _mm_shuffle_ps(x, x, 0xB1);
        tmp2 = _mm_mul_ps(x, yh);
        z = _mm_add_ps(tmp1, _mm_xor_ps(tmp2, neg_mask));
        _mm_store_ps((float*)c, z);
        a += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = (*a) * (*scalar);
    }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_a_sse3(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t* scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm_set_ps1(lv_creal(*scalar));
    yh = _mm_set_ps1(lv_cimag(*scalar));

    for (; number < halfPoints; number++) {

        x = _mm_load_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = _mm_mul_ps(x, yl); // tmp1 = ar*cr,ai*cr,br*dr,bi*dr

        x = _mm_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm_addsub_ps(tmp1,
                          tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm_store_ps((float*)c, z); // Store the results back into the C container

        a += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = (*a) * (*scalar);
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_a_avx(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t* scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_load_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = _mm256_mul_ps(x, yl); // tmp1 = ar*cr,ai*cr,br*dr,bi*dr

        x = _mm256_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm256_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm256_addsub_ps(tmp1,
                             tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_store_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_a_avx2(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t* scalar,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_load_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = _mm256_mul_ps(x, yl); // tmp1 = ar*cr,ai*cr,br*dr,bi*dr

        x = _mm256_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm256_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm256_addsub_ps(tmp1,
                             tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_store_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_a_avx_fma(lv_32fc_t* cVector,
                                                            const lv_32fc_t* aVector,
                                                            const lv_32fc_t* scalar,
                                                            unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_load_ps((const float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi

        tmp1 = x;

        x = _mm256_shuffle_ps(x, x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        tmp2 = _mm256_mul_ps(x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        z = _mm256_fmaddsub_ps(
            tmp1, yl, tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_store_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_a_avx2_fma(lv_32fc_t* cVector,
                                                               const lv_32fc_t* aVector,
                                                               const lv_32fc_t* scalar,
                                                               unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;
    __m256 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    yl = _mm256_set1_ps(lv_creal(*scalar));
    yh = _mm256_set1_ps(lv_cimag(*scalar));

    for (; number < quarterPoints; number++) {
        x = _mm256_load_ps((const float*)a);
        tmp1 = x;
        x = _mm256_shuffle_ps(x, x, 0xB1);
        tmp2 = _mm256_mul_ps(x, yh);
        z = _mm256_fmaddsub_ps(tmp1, yl, tmp2);
        _mm256_store_ps((float*)c, z);
        a += 4;
        c += 4;
    }

    for (unsigned int i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_s32fc_multiply2_32fc_a_avx512f(lv_32fc_t* cVector,
                                                             const lv_32fc_t* aVector,
                                                             const lv_32fc_t* scalar,
                                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;
    unsigned int remainder = num_points & 7;
    __m512 x, yl, yh, z, tmp1, tmp2;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;

    // Set up constant scalar vector
    yl = _mm512_set1_ps(lv_creal(*scalar));
    yh = _mm512_set1_ps(lv_cimag(*scalar));

    for (; number < eighthPoints; number++) {
        x = _mm512_load_ps((const float*)a);

        tmp1 = x;

        x = _mm512_shuffle_ps(x, x, 0xB1); // Swap re/im: ai,ar,bi,br,...

        tmp2 = _mm512_mul_ps(x, yh); // tmp2 = ai*ci, ar*ci, ...

        z = _mm512_fmaddsub_ps(tmp1, yl, tmp2); // re-im, im+re, ...

        _mm512_store_ps((float*)c, z);

        a += 8;
        c += 8;
    }

    if (remainder) {
        volk_32fc_s32fc_multiply2_32fc_generic(c, a, scalar, remainder);
    }
}
#endif /* LV_HAVE_AVX512F */

#endif /* INCLUDED_volk_32fc_s32fc_multiply2_32fc_a_H */
