/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8ic_x2_s32f_multiply_conjugate_32fc
 *
 * \b Overview
 *
 * Multiplies each element of one complex 8-bit integer vector by the complex
 * conjugate of the corresponding element in a second complex 8-bit integer
 * vector, producing complex 32-bit float results scaled by 1/scalar.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8ic_x2_s32f_multiply_conjugate_32fc(lv_32fc_t* cVector, const lv_8sc_t*
 * aVector, const lv_8sc_t* bVector, const float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: The first complex 8-bit input vector.
 * \li bVector: The second complex 8-bit input vector whose conjugate is used.
 * \li scalar: Each output value is scaled by 1/scalar.
 * \li num_points: The number of complex values to process.
 *
 * \b Outputs
 * \li cVector: The complex 32-bit float output vector.
 *
 * \b Example
 * Multiply a complex signal by the conjugate of a reference and scale the result.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *
 *   int main(){
 *     unsigned int N = 4;
 *     unsigned int alignment = volk_get_alignment();
 *     float scalar = 2.0f;
 *
 *     // Allocate aligned memory
 *     lv_8sc_t* aVector = (lv_8sc_t*)volk_malloc(sizeof(lv_8sc_t) * N, alignment);
 *     lv_8sc_t* bVector = (lv_8sc_t*)volk_malloc(sizeof(lv_8sc_t) * N, alignment);
 *     lv_32fc_t* cVector = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * N, alignment);
 *
 *     // Initialize input: signal samples as complex 8-bit integers
 *     aVector[0] = lv_cmake((int8_t)3, (int8_t)4);
 *     aVector[1] = lv_cmake((int8_t)-2, (int8_t)5);
 *     aVector[2] = lv_cmake((int8_t)7, (int8_t)-1);
 *     aVector[3] = lv_cmake((int8_t)1, (int8_t)6);
 *
 *     // Initialize reference: complex 8-bit integers
 *     bVector[0] = lv_cmake((int8_t)1, (int8_t)2);
 *     bVector[1] = lv_cmake((int8_t)3, (int8_t)-1);
 *     bVector[2] = lv_cmake((int8_t)2, (int8_t)4);
 *     bVector[3] = lv_cmake((int8_t)-3, (int8_t)2);
 *
 *     // Compute: cVector[i] = aVector[i] * conj(bVector[i]) / scalar
 *     volk_8ic_x2_s32f_multiply_conjugate_32fc(cVector, aVector, bVector, scalar, N);
 *
 *     for (unsigned int i = 0; i < N; i++) {
 *       printf("result[%u] = (%f, %f)\n", i, lv_creal(cVector[i]), lv_cimag(cVector[i]));
 *     }
 *
 *     volk_free(aVector);
 *     volk_free(bVector);
 *     volk_free(cVector);
 *     return 0;
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_8ic_x2_s32f_multiply_conjugate_32fc_u_H
#define INCLUDED_volk_8ic_x2_s32f_multiply_conjugate_32fc_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_generic(lv_32fc_t* cVector,
                                                 const lv_8sc_t* aVector,
                                                 const lv_8sc_t* bVector,
                                                 const float scalar,
                                                 unsigned int num_points)
{
    unsigned int number = 0;
    float* cPtr = (float*)cVector;
    const float invScalar = 1.0f / scalar;
    const int8_t* a8Ptr = (const int8_t*)aVector;
    const int8_t* b8Ptr = (const int8_t*)bVector;
    for (number = 0; number < num_points; number++) {
        float aReal = (float)*a8Ptr++;
        float aImag = (float)*a8Ptr++;
        lv_32fc_t aVal = lv_cmake(aReal, aImag);
        float bReal = (float)*b8Ptr++;
        float bImag = (float)*b8Ptr++;
        lv_32fc_t bVal = lv_cmake(bReal, -bImag);
        lv_32fc_t temp = aVal * bVal;

        *cPtr++ = (lv_creal(temp) * invScalar);
        *cPtr++ = (lv_cimag(temp) * invScalar);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_u_sse4_1(lv_32fc_t* cVector,
                                                   const lv_8sc_t* aVector,
                                                   const lv_8sc_t* bVector,
                                                   const float scalar,
                                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m128i x, y, realz, imagz;
    __m128 ret;
    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;
    __m128i conjugateSign = _mm_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1);

    const float fInvScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    for (; number < quarterPoints; number++) {
        // Convert into 8 bit values into 16 bit values
        x = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)a));
        y = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)b));

        // Calculate the ar*cr - ai*(-ci) portions
        realz = _mm_madd_epi16(x, y);

        // Calculate the complex conjugate of the cr + ci j values
        y = _mm_sign_epi16(y, conjugateSign);

        // Shift the order of the cr and ci values
        y = _mm_shufflehi_epi16(_mm_shufflelo_epi16(y, _MM_SHUFFLE(2, 3, 0, 1)),
                                _MM_SHUFFLE(2, 3, 0, 1));

        // Calculate the ar*(-ci) + cr*(ai)
        imagz = _mm_madd_epi16(x, y);

        // Interleave real and imaginary and then convert to float values
        ret = _mm_cvtepi32_ps(_mm_unpacklo_epi32(realz, imagz));

        // Normalize the floating point values
        ret = _mm_mul_ps(ret, invScalar);

        // Store the floating point values
        _mm_storeu_ps((float*)c, ret);
        c += 2;

        // Interleave real and imaginary and then convert to float values
        ret = _mm_cvtepi32_ps(_mm_unpackhi_epi32(realz, imagz));

        // Normalize the floating point values
        ret = _mm_mul_ps(ret, invScalar);

        // Store the floating point values
        _mm_storeu_ps((float*)c, ret);
        c += 2;

        a += 4;
        b += 4;
    }

    number = quarterPoints * 4;
    volk_8ic_x2_s32f_multiply_conjugate_32fc_generic(
        c, a, b, scalar, num_points - number);
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_u_avx2(lv_32fc_t* cVector,
                                                const lv_8sc_t* aVector,
                                                const lv_8sc_t* bVector,
                                                const float scalar,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEigthPoints = num_points / 8;

    __m256i x, y, realz, imagz;
    __m256 ret, retlo, rethi;
    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;
    __m256i conjugateSign =
        _mm256_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < oneEigthPoints; number++) {
        // Convert  8 bit values into 16 bit values
        x = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
        y = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));

        // Calculate the ar*cr - ai*(-ci) portions
        realz = _mm256_madd_epi16(x, y);

        // Calculate the complex conjugate of the cr + ci j values
        y = _mm256_sign_epi16(y, conjugateSign);

        // Shift the order of the cr and ci values
        y = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(y, _MM_SHUFFLE(2, 3, 0, 1)),
                                   _MM_SHUFFLE(2, 3, 0, 1));

        // Calculate the ar*(-ci) + cr*(ai)
        imagz = _mm256_madd_epi16(x, y);

        // Interleave real and imaginary and then convert to float values
        retlo = _mm256_cvtepi32_ps(_mm256_unpacklo_epi32(realz, imagz));

        // Normalize the floating point values
        retlo = _mm256_mul_ps(retlo, invScalar);

        // Interleave real and imaginary and then convert to float values
        rethi = _mm256_cvtepi32_ps(_mm256_unpackhi_epi32(realz, imagz));

        // Normalize the floating point values
        rethi = _mm256_mul_ps(rethi, invScalar);

        ret = _mm256_permute2f128_ps(retlo, rethi, 0b00100000);
        _mm256_storeu_ps((float*)c, ret);
        c += 4;

        ret = _mm256_permute2f128_ps(retlo, rethi, 0b00110001);
        _mm256_storeu_ps((float*)c, ret);
        c += 4;

        a += 8;
        b += 8;
    }

    number = oneEigthPoints * 8;
    float* cFloatPtr = (float*)&cVector[number];
    const int8_t* a8Ptr = (const int8_t*)&aVector[number];
    const int8_t* b8Ptr = (const int8_t*)&bVector[number];
    for (; number < num_points; number++) {
        float aReal = (float)*a8Ptr++;
        float aImag = (float)*a8Ptr++;
        lv_32fc_t aVal = lv_cmake(aReal, aImag);
        float bReal = (float)*b8Ptr++;
        float bImag = (float)*b8Ptr++;
        lv_32fc_t bVal = lv_cmake(bReal, -bImag);
        lv_32fc_t temp = aVal * bVal;

        *cFloatPtr++ = lv_creal(temp) * fInvScalar;
        *cFloatPtr++ = lv_cimag(temp) * fInvScalar;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_u_avx512f(lv_32fc_t* cVector,
                                                     const lv_8sc_t* aVector,
                                                     const lv_8sc_t* bVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;

    const float fInvScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(fInvScalar);

    __m256i conjugateSign =
        _mm256_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);

    for (; number < sixteenthPoints; number++) {
        /* Process first 8 complex values using AVX2 widening */
        __m256i x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
        __m256i y0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));

        __m256i realz0 = _mm256_madd_epi16(x0, y0);
        y0 = _mm256_sign_epi16(y0, conjugateSign);
        y0 = _mm256_shufflehi_epi16(
            _mm256_shufflelo_epi16(y0, _MM_SHUFFLE(2, 3, 0, 1)),
            _MM_SHUFFLE(2, 3, 0, 1));
        __m256i imagz0 = _mm256_madd_epi16(x0, y0);

        /* Process second 8 complex values using AVX2 widening */
        __m256i x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + 8)));
        __m256i y1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b + 8)));

        __m256i realz1 = _mm256_madd_epi16(x1, y1);
        y1 = _mm256_sign_epi16(y1, conjugateSign);
        y1 = _mm256_shufflehi_epi16(
            _mm256_shufflelo_epi16(y1, _MM_SHUFFLE(2, 3, 0, 1)),
            _MM_SHUFFLE(2, 3, 0, 1));
        __m256i imagz1 = _mm256_madd_epi16(x1, y1);

        /* Combine into 512-bit vectors: [batch0 | batch1] */
        __m512i realz = _mm512_inserti64x4(_mm512_castsi256_si512(realz0), realz1, 1);
        __m512i imagz = _mm512_inserti64x4(_mm512_castsi256_si512(imagz0), imagz1, 1);

        /* Interleave real and imaginary, convert to float, and scale */
        __m512i lo = _mm512_unpacklo_epi32(realz, imagz);
        __m512i hi = _mm512_unpackhi_epi32(realz, imagz);

        const __m512i merge_lo_idx = _mm512_set_epi32(
            23, 22, 21, 20, 7, 6, 5, 4, 19, 18, 17, 16, 3, 2, 1, 0);
        const __m512i merge_hi_idx = _mm512_set_epi32(
            31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8);

        __m512 retlo = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_lo_idx, hi)),
            invScalar);
        __m512 rethi = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_hi_idx, hi)),
            invScalar);

        _mm512_storeu_ps((float*)c, retlo);
        c += 8;
        _mm512_storeu_ps((float*)c, rethi);
        c += 8;

        a += 16;
        b += 16;
    }

    number = sixteenthPoints * 16;
    volk_8ic_x2_s32f_multiply_conjugate_32fc_generic(
        c, a, b, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_8ic_x2_s32f_multiply_conjugate_32fc_neon(lv_32fc_t* cVector,
                                                                 const lv_8sc_t* aVector,
                                                                 const lv_8sc_t* bVector,
                                                                 const float scalar,
                                                                 unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    lv_32fc_t* cPtr = cVector;
    const lv_8sc_t* aPtr = aVector;
    const lv_8sc_t* bPtr = bVector;
    const float invScalar = 1.0f / scalar;
    float32x4_t vInvScalar = vdupq_n_f32(invScalar);

    int8x8x2_t aVal, bVal;
    int16x8_t aReal, aImag, bReal, bImag;
    int32x4_t realLo, realHi, imagLo, imagHi;
    float32x4_t realFloatLo, realFloatHi, imagFloatLo, imagFloatHi;

    for (; number < eighthPoints; number++) {
        // Load 8 complex 8-bit values (deinterleaved)
        aVal = vld2_s8((const int8_t*)aPtr);
        bVal = vld2_s8((const int8_t*)bPtr);

        // Widen to 16-bit
        aReal = vmovl_s8(aVal.val[0]);
        aImag = vmovl_s8(aVal.val[1]);
        bReal = vmovl_s8(bVal.val[0]);
        bImag = vmovl_s8(bVal.val[1]);

        // Complex multiply with conjugate: (ar + ai*j) * (br - bi*j)
        // real = ar*br + ai*bi
        // imag = ai*br - ar*bi

        // Low half (first 4 complex values)
        realLo = vmlal_s16(vmull_s16(vget_low_s16(aReal), vget_low_s16(bReal)),
                           vget_low_s16(aImag),
                           vget_low_s16(bImag));
        imagLo = vmlsl_s16(vmull_s16(vget_low_s16(aImag), vget_low_s16(bReal)),
                           vget_low_s16(aReal),
                           vget_low_s16(bImag));

        // High half (next 4 complex values)
        realHi = vmlal_s16(vmull_s16(vget_high_s16(aReal), vget_high_s16(bReal)),
                           vget_high_s16(aImag),
                           vget_high_s16(bImag));
        imagHi = vmlsl_s16(vmull_s16(vget_high_s16(aImag), vget_high_s16(bReal)),
                           vget_high_s16(aReal),
                           vget_high_s16(bImag));

        // Convert to float and scale
        realFloatLo = vmulq_f32(vcvtq_f32_s32(realLo), vInvScalar);
        imagFloatLo = vmulq_f32(vcvtq_f32_s32(imagLo), vInvScalar);
        realFloatHi = vmulq_f32(vcvtq_f32_s32(realHi), vInvScalar);
        imagFloatHi = vmulq_f32(vcvtq_f32_s32(imagHi), vInvScalar);

        // Store interleaved (first 4 complex values)
        float32x4x2_t resultLo;
        resultLo.val[0] = realFloatLo;
        resultLo.val[1] = imagFloatLo;
        vst2q_f32((float*)cPtr, resultLo);
        cPtr += 4;

        // Store interleaved (next 4 complex values)
        float32x4x2_t resultHi;
        resultHi.val[0] = realFloatHi;
        resultHi.val[1] = imagFloatHi;
        vst2q_f32((float*)cPtr, resultHi);
        cPtr += 4;

        aPtr += 8;
        bPtr += 8;
    }

    number = eighthPoints * 8;
    float* cFloatPtr = (float*)&cVector[number];
    const int8_t* a8Ptr = (const int8_t*)&aVector[number];
    const int8_t* b8Ptr = (const int8_t*)&bVector[number];
    for (; number < num_points; number++) {
        float aReal_f = (float)*a8Ptr++;
        float aImag_f = (float)*a8Ptr++;
        lv_32fc_t aVal_c = lv_cmake(aReal_f, aImag_f);
        float bReal_f = (float)*b8Ptr++;
        float bImag_f = (float)*b8Ptr++;
        lv_32fc_t bVal_c = lv_cmake(bReal_f, -bImag_f);
        lv_32fc_t temp = aVal_c * bVal_c;

        *cFloatPtr++ = lv_creal(temp) * invScalar;
        *cFloatPtr++ = lv_cimag(temp) * invScalar;
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_8ic_x2_s32f_multiply_conjugate_32fc_rvv(lv_32fc_t* cVector,
                                                                const lv_8sc_t* aVector,
                                                                const lv_8sc_t* bVector,
                                                                const float scalar,
                                                                unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e8m1(n);
        vint16m2_t va = __riscv_vle16_v_i16m2((const int16_t*)aVector, vl);
        vint16m2_t vb = __riscv_vle16_v_i16m2((const int16_t*)bVector, vl);
        vint8m1_t var = __riscv_vnsra(va, 0, vl), vai = __riscv_vnsra(va, 8, vl);
        vint8m1_t vbr = __riscv_vnsra(vb, 0, vl), vbi = __riscv_vnsra(vb, 8, vl);
        vint16m2_t vr = __riscv_vwmacc(__riscv_vwmul(var, vbr, vl), vai, vbi, vl);
        vint16m2_t vi =
            __riscv_vsub(__riscv_vwmul(vai, vbr, vl), __riscv_vwmul(var, vbi, vl), vl);
        vfloat32m4_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), 1.0f / scalar, vl);
        vfloat32m4_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), 1.0f / scalar, vl);
        vuint32m4_t vru = __riscv_vreinterpret_u32m4(vrf);
        vuint32m4_t viu = __riscv_vreinterpret_u32m4(vif);
        vuint64m8_t v =
            __riscv_vwmaccu(__riscv_vwaddu_vv(vru, viu, vl), 0xFFFFFFFF, viu, vl);
        __riscv_vse64((uint64_t*)cVector, v, vl);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_rvvseg(lv_32fc_t* cVector,
                                                const lv_8sc_t* aVector,
                                                const lv_8sc_t* bVector,
                                                const float scalar,
                                                unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e8m1(n);
        vint8m1x2_t va = __riscv_vlseg2e8_v_i8m1x2((const int8_t*)aVector, vl);
        vint8m1x2_t vb = __riscv_vlseg2e8_v_i8m1x2((const int8_t*)bVector, vl);
        vint8m1_t var = __riscv_vget_i8m1(va, 0), vai = __riscv_vget_i8m1(va, 1);
        vint8m1_t vbr = __riscv_vget_i8m1(vb, 0), vbi = __riscv_vget_i8m1(vb, 1);
        vint16m2_t vr = __riscv_vwmacc(__riscv_vwmul(var, vbr, vl), vai, vbi, vl);
        vint16m2_t vi =
            __riscv_vsub(__riscv_vwmul(vai, vbr, vl), __riscv_vwmul(var, vbi, vl), vl);
        vfloat32m4_t vrf = __riscv_vfmul(__riscv_vfwcvt_f(vr, vl), 1.0f / scalar, vl);
        vfloat32m4_t vif = __riscv_vfmul(__riscv_vfwcvt_f(vi, vl), 1.0f / scalar, vl);
        __riscv_vsseg2e32_v_f32m4x2(
            (float*)cVector, __riscv_vcreate_v_f32m4x2(vrf, vif), vl);
    }
}

#endif /* LV_HAVE_RVVSEG */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_u_avx512bw(lv_32fc_t* cVector,
                                                     const lv_8sc_t* aVector,
                                                     const lv_8sc_t* bVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;

    const float fInvScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(fInvScalar);

    __m512i conjugateSign = _mm512_set_epi16(
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);

    for (; number < sixteenthPoints; number++) {
        /* Convert 16 complex int8 values to 32 int16 values */
        __m512i x = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)a));
        __m512i y = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)b));

        /* Calculate ar*br + ai*bi (real part of conjugate multiply) */
        __m512i realz = _mm512_madd_epi16(x, y);

        /* Conjugate b: negate the imaginary parts (odd int16 positions).
         * Multiply by conjugateSign: +1 for real, -1 for imag.
         * _mm512_sign_epi16 does not exist, so use mullo instead. */
        y = _mm512_mullo_epi16(y, conjugateSign);

        /* Swap the order of the cr and ci values within each pair */
        y = _mm512_shufflehi_epi16(
            _mm512_shufflelo_epi16(y, _MM_SHUFFLE(2, 3, 0, 1)),
            _MM_SHUFFLE(2, 3, 0, 1));

        /* Calculate ai*br - ar*bi (imaginary part) */
        __m512i imagz = _mm512_madd_epi16(x, y);

        /* Interleave real and imaginary, convert to float, and scale.
         * realz and imagz each have 16 int32 values.
         * unpacklo/hi operates within 128-bit lanes:
         *   lo = [r0,i0,r1,i1 | r4,i4,r5,i5 | r8,i8,r9,i9 | r12,i12,r13,i13]
         *   hi = [r2,i2,r3,i3 | r6,i6,r7,i7 | r10,i10,r11,i11 | r14,i14,r15,i15]
         * We merge them into sequential complex output using permutex2var. */
        __m512i lo = _mm512_unpacklo_epi32(realz, imagz);
        __m512i hi = _mm512_unpackhi_epi32(realz, imagz);

        /* Merge lo and hi: indices 0-15 select from lo, 16-31 from hi */
        const __m512i merge_lo_idx = _mm512_set_epi32(
            23, 22, 21, 20, 7, 6, 5, 4, 19, 18, 17, 16, 3, 2, 1, 0);
        const __m512i merge_hi_idx = _mm512_set_epi32(
            31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8);

        __m512 retlo = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_lo_idx, hi)),
            invScalar);
        __m512 rethi = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_hi_idx, hi)),
            invScalar);

        _mm512_storeu_ps((float*)c, retlo);
        c += 8;
        _mm512_storeu_ps((float*)c, rethi);
        c += 8;

        a += 16;
        b += 16;
    }

    number = sixteenthPoints * 16;
    volk_8ic_x2_s32f_multiply_conjugate_32fc_generic(
        c, a, b, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */


#endif /* INCLUDED_volk_8ic_x2_s32f_multiply_conjugate_32fc_u_H */

#ifndef INCLUDED_volk_8ic_x2_s32f_multiply_conjugate_32fc_a_H
#define INCLUDED_volk_8ic_x2_s32f_multiply_conjugate_32fc_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_a_sse4_1(lv_32fc_t* cVector,
                                                  const lv_8sc_t* aVector,
                                                  const lv_8sc_t* bVector,
                                                  const float scalar,
                                                  unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m128i x, y, realz, imagz;
    __m128 ret;
    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;
    __m128i conjugateSign = _mm_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1);

    const float fInvScalar = 1.0f / scalar;
    __m128 invScalar = _mm_set_ps1(fInvScalar);

    for (; number < quarterPoints; number++) {
        // Convert into 8 bit values into 16 bit values
        x = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)a));
        y = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)b));

        // Calculate the ar*cr - ai*(-ci) portions
        realz = _mm_madd_epi16(x, y);

        // Calculate the complex conjugate of the cr + ci j values
        y = _mm_sign_epi16(y, conjugateSign);

        // Shift the order of the cr and ci values
        y = _mm_shufflehi_epi16(_mm_shufflelo_epi16(y, _MM_SHUFFLE(2, 3, 0, 1)),
                                _MM_SHUFFLE(2, 3, 0, 1));

        // Calculate the ar*(-ci) + cr*(ai)
        imagz = _mm_madd_epi16(x, y);

        // Interleave real and imaginary and then convert to float values
        ret = _mm_cvtepi32_ps(_mm_unpacklo_epi32(realz, imagz));

        // Normalize the floating point values
        ret = _mm_mul_ps(ret, invScalar);

        // Store the floating point values
        _mm_store_ps((float*)c, ret);
        c += 2;

        // Interleave real and imaginary and then convert to float values
        ret = _mm_cvtepi32_ps(_mm_unpackhi_epi32(realz, imagz));

        // Normalize the floating point values
        ret = _mm_mul_ps(ret, invScalar);

        // Store the floating point values
        _mm_store_ps((float*)c, ret);
        c += 2;

        a += 4;
        b += 4;
    }

    number = quarterPoints * 4;
    float* cFloatPtr = (float*)&cVector[number];
    const int8_t* a8Ptr = (const int8_t*)&aVector[number];
    const int8_t* b8Ptr = (const int8_t*)&bVector[number];
    for (; number < num_points; number++) {
        float aReal = (float)*a8Ptr++;
        float aImag = (float)*a8Ptr++;
        lv_32fc_t aVal = lv_cmake(aReal, aImag);
        float bReal = (float)*b8Ptr++;
        float bImag = (float)*b8Ptr++;
        lv_32fc_t bVal = lv_cmake(bReal, -bImag);
        lv_32fc_t temp = aVal * bVal;

        *cFloatPtr++ = lv_creal(temp) * fInvScalar;
        *cFloatPtr++ = lv_cimag(temp) * fInvScalar;
    }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_a_avx2(lv_32fc_t* cVector,
                                                const lv_8sc_t* aVector,
                                                const lv_8sc_t* bVector,
                                                const float scalar,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEigthPoints = num_points / 8;

    __m256i x, y, realz, imagz;
    __m256 ret, retlo, rethi;
    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;
    __m256i conjugateSign =
        _mm256_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);

    const float fInvScalar = 1.0f / scalar;
    __m256 invScalar = _mm256_set1_ps(fInvScalar);

    for (; number < oneEigthPoints; number++) {
        // Convert  8 bit values into 16 bit values
        x = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)a));
        y = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)b));

        // Calculate the ar*cr - ai*(-ci) portions
        realz = _mm256_madd_epi16(x, y);

        // Calculate the complex conjugate of the cr + ci j values
        y = _mm256_sign_epi16(y, conjugateSign);

        // Shift the order of the cr and ci values
        y = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(y, _MM_SHUFFLE(2, 3, 0, 1)),
                                   _MM_SHUFFLE(2, 3, 0, 1));

        // Calculate the ar*(-ci) + cr*(ai)
        imagz = _mm256_madd_epi16(x, y);

        // Interleave real and imaginary and then convert to float values
        retlo = _mm256_cvtepi32_ps(_mm256_unpacklo_epi32(realz, imagz));

        // Normalize the floating point values
        retlo = _mm256_mul_ps(retlo, invScalar);

        // Interleave real and imaginary and then convert to float values
        rethi = _mm256_cvtepi32_ps(_mm256_unpackhi_epi32(realz, imagz));

        // Normalize the floating point values
        rethi = _mm256_mul_ps(rethi, invScalar);

        ret = _mm256_permute2f128_ps(retlo, rethi, 0b00100000);
        _mm256_store_ps((float*)c, ret);
        c += 4;

        ret = _mm256_permute2f128_ps(retlo, rethi, 0b00110001);
        _mm256_store_ps((float*)c, ret);
        c += 4;

        a += 8;
        b += 8;
    }

    number = oneEigthPoints * 8;
    float* cFloatPtr = (float*)&cVector[number];
    const int8_t* a8Ptr = (const int8_t*)&aVector[number];
    const int8_t* b8Ptr = (const int8_t*)&bVector[number];
    for (; number < num_points; number++) {
        float aReal = (float)*a8Ptr++;
        float aImag = (float)*a8Ptr++;
        lv_32fc_t aVal = lv_cmake(aReal, aImag);
        float bReal = (float)*b8Ptr++;
        float bImag = (float)*b8Ptr++;
        lv_32fc_t bVal = lv_cmake(bReal, -bImag);
        lv_32fc_t temp = aVal * bVal;

        *cFloatPtr++ = lv_creal(temp) * fInvScalar;
        *cFloatPtr++ = lv_cimag(temp) * fInvScalar;
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_a_avx512f(lv_32fc_t* cVector,
                                                     const lv_8sc_t* aVector,
                                                     const lv_8sc_t* bVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;

    const float fInvScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(fInvScalar);

    __m256i conjugateSign =
        _mm256_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);

    for (; number < sixteenthPoints; number++) {
        /* Process first 8 complex values using AVX2 widening */
        __m256i x0 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)a));
        __m256i y0 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)b));

        __m256i realz0 = _mm256_madd_epi16(x0, y0);
        y0 = _mm256_sign_epi16(y0, conjugateSign);
        y0 = _mm256_shufflehi_epi16(
            _mm256_shufflelo_epi16(y0, _MM_SHUFFLE(2, 3, 0, 1)),
            _MM_SHUFFLE(2, 3, 0, 1));
        __m256i imagz0 = _mm256_madd_epi16(x0, y0);

        /* Process second 8 complex values using AVX2 widening */
        __m256i x1 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(a + 8)));
        __m256i y1 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)(b + 8)));

        __m256i realz1 = _mm256_madd_epi16(x1, y1);
        y1 = _mm256_sign_epi16(y1, conjugateSign);
        y1 = _mm256_shufflehi_epi16(
            _mm256_shufflelo_epi16(y1, _MM_SHUFFLE(2, 3, 0, 1)),
            _MM_SHUFFLE(2, 3, 0, 1));
        __m256i imagz1 = _mm256_madd_epi16(x1, y1);

        /* Combine into 512-bit vectors: [batch0 | batch1] */
        __m512i realz = _mm512_inserti64x4(_mm512_castsi256_si512(realz0), realz1, 1);
        __m512i imagz = _mm512_inserti64x4(_mm512_castsi256_si512(imagz0), imagz1, 1);

        /* Interleave real and imaginary, convert to float, and scale */
        __m512i lo = _mm512_unpacklo_epi32(realz, imagz);
        __m512i hi = _mm512_unpackhi_epi32(realz, imagz);

        const __m512i merge_lo_idx = _mm512_set_epi32(
            23, 22, 21, 20, 7, 6, 5, 4, 19, 18, 17, 16, 3, 2, 1, 0);
        const __m512i merge_hi_idx = _mm512_set_epi32(
            31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8);

        __m512 retlo = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_lo_idx, hi)),
            invScalar);
        __m512 rethi = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_hi_idx, hi)),
            invScalar);

        _mm512_store_ps((float*)c, retlo);
        c += 8;
        _mm512_store_ps((float*)c, rethi);
        c += 8;

        a += 16;
        b += 16;
    }

    number = sixteenthPoints * 16;
    volk_8ic_x2_s32f_multiply_conjugate_32fc_generic(
        c, a, b, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX512BW
#include <immintrin.h>

static inline void
volk_8ic_x2_s32f_multiply_conjugate_32fc_a_avx512bw(lv_32fc_t* cVector,
                                                     const lv_8sc_t* aVector,
                                                     const lv_8sc_t* bVector,
                                                     const float scalar,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    lv_32fc_t* c = cVector;
    const lv_8sc_t* a = aVector;
    const lv_8sc_t* b = bVector;

    const float fInvScalar = 1.0f / scalar;
    __m512 invScalar = _mm512_set1_ps(fInvScalar);

    __m512i conjugateSign = _mm512_set_epi16(
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);

    for (; number < sixteenthPoints; number++) {
        /* Convert 16 complex int8 values to 32 int16 values */
        __m512i x = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)a));
        __m512i y = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*)b));

        /* Calculate ar*br + ai*bi (real part of conjugate multiply) */
        __m512i realz = _mm512_madd_epi16(x, y);

        /* Conjugate b: negate the imaginary parts (odd int16 positions).
         * Multiply by conjugateSign: +1 for real, -1 for imag.
         * _mm512_sign_epi16 does not exist, so use mullo instead. */
        y = _mm512_mullo_epi16(y, conjugateSign);

        /* Swap the order of the cr and ci values within each pair */
        y = _mm512_shufflehi_epi16(
            _mm512_shufflelo_epi16(y, _MM_SHUFFLE(2, 3, 0, 1)),
            _MM_SHUFFLE(2, 3, 0, 1));

        /* Calculate ai*br - ar*bi (imaginary part) */
        __m512i imagz = _mm512_madd_epi16(x, y);

        /* Interleave real and imaginary, convert to float, and scale */
        __m512i lo = _mm512_unpacklo_epi32(realz, imagz);
        __m512i hi = _mm512_unpackhi_epi32(realz, imagz);

        /* Merge lo and hi: indices 0-15 select from lo, 16-31 from hi */
        const __m512i merge_lo_idx = _mm512_set_epi32(
            23, 22, 21, 20, 7, 6, 5, 4, 19, 18, 17, 16, 3, 2, 1, 0);
        const __m512i merge_hi_idx = _mm512_set_epi32(
            31, 30, 29, 28, 15, 14, 13, 12, 27, 26, 25, 24, 11, 10, 9, 8);

        __m512 retlo = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_lo_idx, hi)),
            invScalar);
        __m512 rethi = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_permutex2var_epi32(lo, merge_hi_idx, hi)),
            invScalar);

        _mm512_store_ps((float*)c, retlo);
        c += 8;
        _mm512_store_ps((float*)c, rethi);
        c += 8;

        a += 16;
        b += 16;
    }

    number = sixteenthPoints * 16;
    volk_8ic_x2_s32f_multiply_conjugate_32fc_generic(
        c, a, b, scalar, num_points - number);
}
#endif /* LV_HAVE_AVX512BW */


#endif /* INCLUDED_volk_8ic_x2_s32f_multiply_conjugate_32fc_a_H */
